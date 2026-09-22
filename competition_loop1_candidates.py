"""Compact C1B-parented transition architecture screen (research only)."""

import torch
from torch import nn

from benchmark import assert_model_state
import competition_submission as c0
import competition_submission_c1b as c1b


class PairwiseLowRank(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        rank = d // 4
        self.norm = nn.LayerNorm(d)
        self.left, self.right = nn.Linear(d, rank), nn.Linear(d, rank)
        self.value, self.out = nn.Linear(d, rank), nn.Linear(rank, d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.readout = nn.Linear(d, 10)

    def forward(self, state, context, context_mask):
        x = self.norm(state)
        weights = torch.softmax(self.left(x) @ self.right(x).transpose(1, 2) / self.left.out_features**0.5, -1)
        x = state + self.out(weights @ self.value(x))
        x = x + self.cross(x, context, context, key_padding_mask=~context_mask, need_weights=False)[0]
        return self.readout(x)


class ConvCarry(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 3, padding=1, groups=d), nn.GELU(), nn.Conv1d(d, d, 1))
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.readout = nn.Linear(d, 10)

    def forward(self, state, context, context_mask):
        x = state + self.conv(self.norm(state).transpose(1, 2)).transpose(1, 2)
        x = x + self.cross(x, context, context, key_padding_mask=~context_mask, need_weights=False)[0]
        return self.readout(x)


class LSDGRU(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)
        self.context = nn.Linear(d, d)
        self.readout = nn.Linear(d, 10)

    def forward(self, state, context, context_mask):
        # Slots are already LSD-first.  h0 is rebuilt from immutable context on every call.
        pooled = (context * context_mask[..., None]).sum(1) / context_mask.sum(1, keepdim=True).clamp_min(1)
        output, _ = self.gru(state, torch.tanh(self.context(pooled))[None])
        return self.readout(output)


class GlobalSlotMLP(nn.Module):
    def __init__(self, d, heads, slots):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.slot_in, self.slot_out = nn.Linear(slots, slots * 2), nn.Linear(slots * 2, slots)
        self.channel = nn.Sequential(nn.Linear(d, 3 * d), nn.GELU(), nn.Linear(3 * d, d))
        self.context = nn.Linear(d, d)
        self.readout = nn.Linear(d, 10)

    def forward(self, state, context, context_mask):
        x = self.n1(state)
        x = state + self.slot_out(torch.nn.functional.gelu(self.slot_in(x.transpose(1, 2)))).transpose(1, 2)
        pooled = (context * context_mask[..., None]).sum(1) / context_mask.sum(1, keepdim=True).clamp_min(1)
        x = x + self.channel(self.n2(x)) + self.context(pooled)[:, None]
        return self.readout(x)


class SwiGLUCross(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up, self.gate, self.down = nn.Linear(d, 3 * d), nn.Linear(d, 3 * d), nn.Linear(3 * d, d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.readout = nn.Linear(d, 10)

    def forward(self, state, context, context_mask):
        z = self.norm(state)
        x = state + self.down(torch.nn.functional.silu(self.gate(z)) * self.up(z))
        x = x + self.cross(x, context, context, key_padding_mask=~context_mask, need_weights=False)[0]
        return self.readout(x)


TRANSITIONS = {"A": PairwiseLowRank, "B": ConvCarry, "C": LSDGRU, "D": GlobalSlotMLP, "E": SwiGLUCross}


class Model(c0.Model):
    """Frozen C0 outer loop with the C1B backward-only endpoint gate."""
    def __init__(self, spec, candidate="A", d_model=112, heads=4):
        super().__init__(spec, d_model=d_model, heads=heads)
        cls = TRANSITIONS[candidate]
        self.transition = cls(d_model, heads, spec.max_seq_len) if candidate == "D" else cls(d_model, heads)
        self.candidate = candidate

    def forward(self, input_ids, attention_mask=None):
        logits, auxiliary = super().forward(input_ids, attention_mask)
        if self.training:
            auxiliary["ungated_logits"] = logits
            scale = torch.where(auxiliary["parsed_steps"] == 1, logits.new_tensor(1.0), logits.new_tensor(0.01))
            raw = logits
            logits = raw.detach() + scale[:, None, None] * (raw - raw.detach())
        return logits, auxiliary


def build_model(spec, candidate="A", d_model=112, heads=4):
    model = Model(spec, candidate, d_model, heads)
    assert_model_state(model, spec)
    return model


BUILDERS = {name: (lambda spec, name=name: build_model(spec, name)) for name in TRANSITIONS}
build_optimizer = c1b.SUBMISSION.build_optimizer  # identity reuse is intentional
