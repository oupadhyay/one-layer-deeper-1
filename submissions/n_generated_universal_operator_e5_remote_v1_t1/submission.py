"""N-generated universal latent operator for E5 (T1-gradient training)."""

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D, HEADS, MAX_PLACES, SCRATCH, MICRO_UPDATES, MAX_STEPS = 64, 4, 16, 8, 6, 64
RANK = 8


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class WarmupConstant:
    def __init__(self, optimizer, warmup=30):
        self.optimizer, self.warmup, self.steps = optimizer, warmup, 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._set(1 / warmup)

    def _set(self, scale):
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale

    def step(self):
        self.steps += 1
        self._set(min(1.0, (self.steps + 1) / self.warmup))


class FiLMNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gamma, beta):
        normalized = F.rms_norm(x, (x.shape[-1],), self.weight)
        return normalized * (1.0 + gamma[:, None]) + beta[:, None]


class UniversalBlock(nn.Module):
    """The single tied all-to-all block used by every micro/macro update."""

    def __init__(self):
        super().__init__()
        self.attn_norm, self.ffn_norm = FiLMNorm(D), FiLMNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.attn_out = nn.Linear(D, D, bias=False)
        self.ff1, self.ff2 = nn.Linear(D, 2 * D, bias=False), nn.Linear(2 * D, D, bias=False)
        self.adapter_down = nn.Linear(D, RANK, bias=False)
        self.adapter_up = nn.Linear(RANK, D, bias=False)
        self.calls = 0

    def forward(self, state, film, adapter_gates, mask):
        self.calls += 1
        ag, ab, fg, fb = film.chunk(4, -1)
        z = self.attn_norm(state, ag, ab)
        batch, length, _ = z.shape
        q, k, v = self.qkv(z).chunk(3, -1)
        q = q.view(batch, length, HEADS, D // HEADS).transpose(1, 2)
        k = k.view(batch, length, HEADS, D // HEADS).transpose(1, 2)
        v = v.view(batch, length, HEADS, D // HEADS).transpose(1, 2)
        attention_mask = mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attended = attended.transpose(1, 2).reshape(batch, length, D)
        state = state + self.attn_out(attended)
        z = self.ffn_norm(state, fg, fb)
        state = state + self.ff2(F.gelu(self.ff1(z)))
        # A generated diagonal gate in a compact learned rank-8 residual adapter.
        state = state + self.adapter_up(self.adapter_down(z) * adapter_gates[:, None])
        return state * mask[..., None]


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len
        self.digit_embedding = nn.Embedding(10, D)
        self.n_place = nn.Embedding(MAX_PLACES, D)
        self.x_place = nn.Embedding(MAX_PLACES, D)
        self.n_role, self.x_role = nn.Parameter(torch.empty(D)), nn.Parameter(torch.empty(D))
        self.scratch_tokens = nn.Parameter(torch.empty(SCRATCH, D))
        self.n_cell = nn.GRUCell(D, D)
        self.generator = nn.Linear(D, 4 * D + RANK)
        self.block = UniversalBlock()
        self.output_norm = nn.RMSNorm(D)
        self.readout = nn.Linear(D, 10)
        for parameter in (self.n_role, self.x_role, self.scratch_tokens):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def parse(input_ids, attention_mask=None):
        """Return field roles, LSD places and T only; never reconstruct N or X."""
        valid = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        is_digit = input_ids.ge(DIGIT) & input_ids.lt(DIGIT + 10) & valid
        marker = torch.zeros_like(input_ids)
        marker = torch.where(input_ids.eq(N) & valid, 1, marker)
        marker = torch.where(input_ids.eq(X) & valid, 2, marker)
        marker = torch.where(input_ids.eq(T) & valid, 3, marker)
        marker = torch.where(input_ids.eq(ANS) & valid, 4, marker)
        active = torch.cummax(marker, 1).values
        roles = torch.where(is_digit, active, 0)
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = roles[:, :, None].eq(roles[:, None, :])
        later = index[None, None, :].gt(index[None, :, None])
        places = (same & later & is_digit[:, None]).sum(2)
        # Decimal reconstruction is intentionally confined to T loop control.
        t_digit = is_digit & roles.eq(3)
        powers = torch.pow(input_ids.new_tensor(10), places.clamp(max=18))
        steps = (((input_ids - DIGIT).clamp(0, 9) * powers) * t_digit).sum(1)
        return roles, places, steps.clamp(1, MAX_STEPS), valid

    def _fields(self, ids, roles, places):
        slots = torch.arange(MAX_PLACES, device=ids.device)
        values = (ids - DIGIT).clamp(0, 9)
        def gather(role):
            assignment = roles[:, :, None].eq(role) & places[:, :, None].eq(slots)
            digits = (values[:, :, None] * assignment).sum(1).long()
            return digits, assignment.any(1)
        return gather(1), gather(2)

    def encode_n(self, n_digits, n_mask):
        hidden = self.n_role[None].expand(n_digits.shape[0], -1)
        # LSD-first ordered, immutable encoding; padding cannot alter hidden state.
        for place in range(n_digits.shape[1]):
            token = self.digit_embedding(n_digits[:, place]) + self.n_place.weight[place]
            proposed = self.n_cell(token, hidden)
            hidden = torch.where(n_mask[:, place, None], proposed, hidden)
        generated = self.generator(hidden)
        return hidden, generated[:, :4 * D], torch.tanh(generated[:, 4 * D:])

    def macrostep(self, x_state, place_mask, film, gates):
        batch = x_state.shape[0]
        scratch = self.scratch_tokens[None].expand(batch, -1, -1)
        state = torch.cat((x_state, scratch), 1)
        mask = torch.cat((place_mask, torch.ones(batch, SCRATCH, dtype=torch.bool,
                                                device=x_state.device)), 1)
        for _ in range(MICRO_UPDATES):
            state = self.block(state, film, gates, mask)
        return state[:, :x_state.shape[1]], scratch

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        roles, places, parsed_steps, valid = self.parse(input_ids, attention_mask)
        (nd, nm), (xd, xm) = self._fields(input_ids, roles, places)
        widths = nm.sum(1)
        torch._assert((widths > 0).all() & (widths <= MAX_PLACES).all(), "N width out of range")
        width = int(widths.max().item())
        nd, nm, xd = nd[:, :width], nm[:, :width], xd[:, :width]
        place_mask = torch.arange(width, device=input_ids.device)[None] < widths[:, None]
        n_hidden, film, gates = self.encode_n(nd, nm)
        # Missing high X places are exactly the learned zero digit, then receive identity.
        x_state = self.digit_embedding(xd) + self.x_place.weight[:width] + self.x_role
        x_state = x_state * place_mask[..., None]
        effective_steps = torch.ones_like(parsed_steps) if self.training else parsed_steps
        endpoint = self.readout(self.output_norm(x_state))
        self.block.calls = 0
        resets = 0
        last_reset = None
        for macro in range(int(effective_steps.max().item())):
            updated, last_reset = self.macrostep(x_state, place_mask, film, gates)
            resets += 1
            active = effective_steps.gt(macro)[:, None, None]
            x_state = torch.where(active, F.rms_norm(updated, (D,)) + self.x_place.weight[:width]
                                  + self.x_role, x_state)
            decoded = self.readout(self.output_norm(updated))
            endpoint = torch.where(active, decoded, endpoint)
        batch, length = input_ids.shape
        logits = endpoint.new_full((batch, length, self.vocab_size), -1e4)
        answer_place = (valid.sum(1)[:, None] - 1 - torch.arange(length, device=input_ids.device)[None])
        answer_place = torch.minimum(answer_place.clamp_min(0), (widths - 1)[:, None])
        selected = endpoint.gather(1, answer_place[..., None].expand(-1, -1, 10))
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated = logits
        if self.training:
            gate = parsed_steps.eq(1).to(logits.dtype)[:, None, None]
            logits = logits.detach() + gate * (logits - logits.detach())
        return logits, {"parsed_steps": parsed_steps, "widths": widths,
                        "n_hidden": n_hidden, "generated_film": film,
                        "adapter_gates": gates, "x_state": x_state,
                        "scratch_reset": last_reset, "scratch_resets": resets,
                        "universal_block_calls": self.block.calls, "ungated_logits": ungated}


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and "embedding" not in name and "norm" not in name
         else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=1e-4, betas=(0.9, 0.95),
                                  capturable=spec.device_type == "cuda")
    return OptimizerBundle(optimizer, WarmupConstant(optimizer, 30))


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None)
