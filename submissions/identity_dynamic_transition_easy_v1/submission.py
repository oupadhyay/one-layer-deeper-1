"""Identity-initialized dynamic-depth learned transition."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


PAD, N_MARKER, X_MARKER, T_MARKER, ANSWER_MARKER, DIGIT_BASE = 0, 2, 3, 4, 5, 7
D, HEADS, FFN, SCRATCH = 128, 4, 256, 8
MIN_MICROSTEPS, MAX_MICROSTEPS, EVAL_MICROSTEPS = 2, 5, 4
MAX_MACROSTEPS = 64


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class TransitionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_norm = nn.RMSNorm(D)
        self.self_qkv = nn.Linear(D, 3 * D, bias=False)
        self.self_out = nn.Linear(D, D, bias=False)
        self.cross_norm = nn.RMSNorm(D)
        self.cross_q = nn.Linear(D, D, bias=False)
        self.cross_kv = nn.Linear(D, 2 * D, bias=False)
        self.cross_out = nn.Linear(D, D, bias=False)
        self.ffn_norm = nn.RMSNorm(D)
        self.ffn_up = nn.Linear(D, 2 * FFN, bias=False)
        self.ffn_down = nn.Linear(FFN, D, bias=False)
        nn.init.zeros_(self.self_out.weight)
        nn.init.zeros_(self.cross_out.weight)
        nn.init.zeros_(self.ffn_down.weight)

    @staticmethod
    def attention(q: Tensor, k: Tensor, v: Tensor, key_mask: Tensor | None = None) -> Tensor:
        batch, query_length, _ = q.shape
        key_length = k.shape[1]

        def heads(value: Tensor, length: int) -> Tensor:
            return value.reshape(batch, length, HEADS, D // HEADS).transpose(1, 2)

        mask = None if key_mask is None else key_mask[:, None, None, :]
        result = F.scaled_dot_product_attention(
            heads(q, query_length), heads(k, key_length), heads(v, key_length),
            attn_mask=mask, dropout_p=0.0,
        )
        return result.transpose(1, 2).contiguous().reshape(batch, query_length, D)

    def forward(self, state: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        q, k, v = self.self_qkv(self.self_norm(state)).chunk(3, dim=-1)
        state = state + self.self_out(self.attention(q, k, v))
        q = self.cross_q(self.cross_norm(state))
        k, v = self.cross_kv(context).chunk(2, dim=-1)
        state = state + self.cross_out(self.attention(q, k, v, context_mask))
        gate, value = self.ffn_up(self.ffn_norm(state)).chunk(2, dim=-1)
        return state + self.ffn_down(F.silu(gate) * value)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.digit_embedding = nn.Embedding(10, D)
        self.place_embedding = nn.Embedding(spec.max_seq_len, D)
        self.n_role = nn.Parameter(torch.empty(D))
        self.state_role = nn.Parameter(torch.empty(D))
        self.context_role = nn.Parameter(torch.empty(D))
        self.scratch = nn.Parameter(torch.empty(SCRATCH, D))
        self.block = TransitionBlock()
        self.readout_norm = nn.RMSNorm(D)
        self.readout = nn.Linear(D, 10, bias=False)
        self.readout.weight = self.digit_embedding.weight
        for parameter in (self.n_role, self.state_role, self.context_role, self.scratch):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def fields(input_ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        is_digit = (input_ids >= DIGIT_BASE) & (input_ids < DIGIT_BASE + 10) & mask
        marker = ((input_ids == N_MARKER) | (input_ids == X_MARKER)
                  | (input_ids == T_MARKER) | (input_ids == ANSWER_MARKER))
        role = marker.long().cumsum(dim=1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same_field = role[:, :, None].eq(role[:, None, :])
        place = (same_field & (index[None, None, :] > index[None, :, None])
                 & is_digit[:, None, :]).sum(dim=2)
        digit_value = (input_ids - DIGIT_BASE).clamp(0, 9)
        t_digits = is_digit & role.eq(3)
        decimal_place = torch.pow(input_ids.new_tensor(10), place)
        steps = (digit_value * decimal_place * t_digits).sum(dim=1).clamp(0, MAX_MACROSTEPS)
        return role, place, steps

    def extract(self, role: Tensor, place: Tensor, values: Tensor, field: int) -> tuple[Tensor, Tensor]:
        slots = torch.arange(self.max_seq_len, device=role.device)
        selected = role[:, :, None].eq(field) & place[:, :, None].eq(slots)
        return (selected.to(values.dtype) * values[:, :, None]).sum(dim=1).long(), selected.any(dim=1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = (input_ids != PAD if attention_mask is None
                else attention_mask.to(device=input_ids.device, dtype=torch.bool))
        role, place, steps = self.fields(input_ids, mask)
        values = (input_ids - DIGIT_BASE).clamp(0, 9)
        n_digits, n_mask = self.extract(role, place, values, 1)
        x_digits, _ = self.extract(role, place, values, 2)
        widths = n_mask.sum(dim=1)
        width = int(widths.max().item())
        n_digits, x_digits = n_digits[:, :width], x_digits[:, :width]
        place_mask = torch.arange(width, device=input_ids.device)[None, :] < widths[:, None]
        n_digits = torch.where(place_mask, n_digits, torch.zeros_like(n_digits))
        x_digits = torch.where(place_mask, x_digits, torch.zeros_like(x_digits))
        place_code = self.place_embedding.weight[None, :width]
        immutable_n = self.digit_embedding(n_digits) + place_code + self.n_role
        state = self.digit_embedding(x_digits) + place_code + self.state_role
        hard_initial = F.one_hot(x_digits, 10).to(state.dtype)
        endpoint = torch.log(hard_initial.clamp_min(1e-7))
        if self.training:
            microsteps = int(torch.randint(
                MIN_MICROSTEPS, MAX_MICROSTEPS + 1, (), device=input_ids.device
            ).item())
        else:
            microsteps = EVAL_MICROSTEPS
        macrosteps = int(steps.max().item())
        for macrostep in range(macrosteps):
            anchor = state
            scratch = self.scratch[None].expand(input_ids.shape[0], -1, -1)
            work = torch.cat((state, scratch), dim=1)
            context = torch.cat((immutable_n + self.context_role, anchor), dim=1)
            context_mask = torch.cat((place_mask, place_mask), dim=1)
            for _ in range(microsteps):
                work = self.block(work, context, context_mask)
            transition_logits = self.readout(self.readout_norm(work[:, :width]))
            probabilities = transition_logits.softmax(dim=-1)
            hard = F.one_hot(transition_logits.argmax(dim=-1), 10).to(probabilities.dtype)
            feedback = hard + probabilities - probabilities.detach() if self.training else hard
            next_state = feedback @ self.digit_embedding.weight + place_code + self.state_role
            active = (steps > macrostep)[:, None, None]
            state = torch.where(active, next_state, state)
            endpoint = torch.where(active, transition_logits, endpoint)
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None]
        output_slot = mask.sum(dim=1)[:, None] - 1 - positions
        output_slot = torch.minimum(
            output_slot.clamp_min(0), (widths - 1).clamp_min(0)[:, None]
        )
        selected = endpoint.gather(1, output_slot[..., None].expand(-1, -1, 10))
        logits = selected.new_full((input_ids.shape[0], length, self.vocab_size), -1e4)
        logits[:, :, DIGIT_BASE:DIGIT_BASE + 10] = selected
        return logits, {
            "steps": steps,
            "widths": widths,
            "macrosteps": macrosteps,
            "microsteps": microsteps,
        }


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.02, capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 64.0, 1.0)
    )
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,
    eval_batch_size=1024,
    max_steps=None,
)
