"""C0: a tied, soft-discrete recurrent candidate for ``squaring_mod``."""

import torch
from torch import nn
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, BOS, N, X, T, ANS, EOS, DIGIT = 0, 1, 2, 3, 4, 5, 6, 7
MAX_STEPS = 64


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class Transition(nn.Module):
    """The sole macro-transition; its output is again only decimal distributions."""
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 3 * d), nn.GELU(), nn.Linear(3 * d, d))
        self.readout = nn.Linear(d, 10)

    def forward(self, state, context, context_mask):
        q = self.n1(state)
        state = state + self.self_attention(q, q, q, need_weights=False)[0]
        q = self.n2(state)
        state = state + self.cross_attention(
            q, context, context, key_padding_mask=~context_mask, need_weights=False
        )[0]
        state = state + self.ff(self.n3(state))
        return self.readout(state)


class Model(nn.Module):
    """Prompt-conditioned recurrent decimal channel with no hidden macro carry."""
    def __init__(self, spec, d_model=112, heads=4):
        super().__init__()
        self.config = Config(spec)
        self.num_loops = MAX_STEPS
        self.token_embedding = nn.Embedding(spec.vocab_size, d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(spec.max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, 3 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.prompt_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.digit_embedding = nn.Parameter(torch.empty(10, d_model))
        self.place_embedding = nn.Embedding(spec.max_seq_len, d_model)
        self.role_embedding = nn.Parameter(torch.empty(d_model))
        self.transition = Transition(d_model, heads)
        nn.init.normal_(self.digit_embedding, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    @staticmethod
    def parse(input_ids, mask):
        """Return field roles, LSD places, and T using only device tensor operations."""
        is_digit = (input_ids >= DIGIT) & mask
        markers = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(markers.to(torch.long), 1) * is_digit
        same = role[:, :, None] == role[:, None, :]
        later = torch.arange(input_ids.shape[1], device=input_ids.device)[None, None, :] > torch.arange(input_ids.shape[1], device=input_ids.device)[None, :, None]
        places = (same & later & is_digit[:, None, :]).sum(2)
        t_digits = is_digit & (role == 3)
        powers = torch.pow(input_ids.new_tensor(10), places)
        steps = (((input_ids - DIGIT) * powers) * t_digits).sum(1)
        return role, places, steps

    def _prepare(self, input_ids, attention_mask):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.to(torch.bool)
        role, places, steps = self.parse(input_ids, mask)
        # N is immutable context. X initializes the recurrent state and T controls
        # the loop; neither may remain available as an endpoint shortcut.
        control = (
            (input_ids == T)
            | (role == 3)
            | (input_ids == X)
            | (role == 2)
        )
        clean = torch.where(control, torch.zeros_like(input_ids), input_ids)
        context_mask = mask & ~control
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        context = self.prompt_encoder(
            self.token_embedding(clean) + self.position_embedding(pos)[None],
            src_key_padding_mask=~context_mask,
        )
        slots = torch.arange(self.config.max_seq_len, device=input_ids.device)
        assignment = (role[:, :, None] == 2) & (places[:, :, None] == slots[None, None, :])
        values = (input_ids - DIGIT).clamp(0, 9)
        selected = torch.einsum("bls,bl->bs", assignment.to(context.dtype), values.to(context.dtype))
        present = assignment.any(1)
        selected = torch.where(present, selected, torch.zeros_like(selected)).long()
        probs = torch.nn.functional.one_hot(selected, 10).to(context.dtype)
        return mask, context_mask, context, probs, steps

    def debug_execution(self, input_ids, attention_mask=None):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.to(torch.bool)
        steps = self.parse(input_ids, mask)[2]
        active = torch.arange(MAX_STEPS, device=input_ids.device)[None] < steps[:, None]
        return {"parsed_steps": steps, "active_updates": active.sum(1), "active_mask": active}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask, context_mask, context, probs, steps = self._prepare(input_ids, attention_mask)
        places = self.place_embedding.weight[None]
        endpoint_logits = torch.log(probs.clamp_min(1e-7))
        # One scalar host synchronization chooses the batch maximum. All arithmetic,
        # state, and per-row activity remain device-resident; no endpoint is clamped.
        executed_steps = int(steps.max().clamp_max(MAX_STEPS).item())
        for iteration in range(executed_steps):
            # Canonical state is reconstructed wholesale; no transition hidden state survives.
            state = probs @ self.digit_embedding + places + self.role_embedding
            digit_logits = self.transition(state, context, context_mask)
            next_probs = (torch.softmax(digit_logits, -1) if self.training else
                          torch.nn.functional.one_hot(digit_logits.argmax(-1), 10).to(digit_logits.dtype))
            active = (steps > iteration)[:, None, None]
            probs = torch.where(active, next_probs, probs)
            endpoint_logits = torch.where(active, digit_logits, endpoint_logits)
        b, length = input_ids.shape
        logits = endpoint_logits.new_full((b, length, self.config.vocab_size), -1e4)
        valid_length = mask.sum(1)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = valid_length[:, None] - 1 - positions
        gather = endpoint_logits.gather(1, slot.clamp(0, self.config.max_seq_len - 1)[:, :, None].expand(-1, -1, 10))
        logits[:, :, DIGIT:DIGIT + 10] = gather
        auxiliary = {
            "parsed_steps": steps,
            "active_updates": steps.clamp_max(MAX_STEPS),
            "executed_macrosteps": steps.new_tensor(executed_steps),
            "digit_probabilities": probs,
        }
        return logits, auxiliary


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec: OptimizerSpec):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.02}, {"params": no_decay, "weight_decay": 0.0}],
        lr=8e-4, betas=(0.9, 0.98)
    )
    return OptimizerBundle(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 200)))


SUBMISSION = Submission(build_model, build_optimizer, batch_size=64, eval_batch_size=128, max_steps=9500)
