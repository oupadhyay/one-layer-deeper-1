"""Local-only oracle diagnostic for exact CRT quotient recurrence.

This intentionally hard-codes factor discovery and CRT arithmetic. It is not a
legal competition submission; it measures the ceiling of canonical quotient
state while leaving the decimal renderer learned from random initialization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, assert_model_state

W, MAX_FACTOR = 8, 4096


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.renderer = nn.Embedding(10, 10)

    @staticmethod
    def parse(ids, mask):
        digit = ids.ge(7) & ids.lt(17) & mask
        marker = ids.eq(2).long() + 2 * ids.eq(3).long() + 3 * ids.eq(4).long()
        region = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        same = region[:, :, None].eq(region[:, None, :]) & region[:, :, None].gt(0)
        place = (same & (index[None, None, :] > index[None, :, None]) & digit[:, None, :]).sum(-1)
        value = (ids - 7).clamp(0, 9)
        steps = (value * torch.pow(ids.new_tensor(10), place) * region.eq(3)).sum(1).long()
        return region, place, steps

    @staticmethod
    def factors_and_inverses(n):
        candidates = torch.arange(2, MAX_FACTOR + 1, device=n.device, dtype=torch.long)
        divides = torch.remainder(n[:, None], candidates[None]).eq(0)
        p = candidates[divides.long().argmax(-1)]
        q = torch.div(n, p, rounding_mode="floor")
        inverse_candidates = torch.arange(1, MAX_FACTOR + 1, device=n.device, dtype=torch.long)
        inverse_p = inverse_candidates[
            torch.remainder((q % p)[:, None] * inverse_candidates[None], p[:, None]).eq(1).long().argmax(-1)
        ]
        inverse_q = inverse_candidates[
            torch.remainder((p % q)[:, None] * inverse_candidates[None], q[:, None]).eq(1).long().argmax(-1)
        ]
        return p, q, inverse_p, inverse_q

    def forward(self, ids, attention_mask=None):
        batch, length = ids.shape
        mask = ids.ne(0) if attention_mask is None else attention_mask.bool()
        region, place, steps = self.parse(ids, mask)
        nw, xw = region.eq(1).sum(1), region.eq(2).sum(1)
        if length > self.config.max_seq_len or bool(((nw < 1) | (nw > W) | (xw < 1) | (xw > W) | (steps > 64)).any()):
            raise ValueError("invalid Medium diagnostic contract")
        slots = torch.arange(W, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long()

        n_digits, x_digits = field(1), field(2)
        powers = torch.pow(ids.new_tensor(10), slots)
        n = (n_digits * powers).sum(1)
        x = (x_digits * powers).sum(1)
        p, q, inverse_p, inverse_q = self.factors_and_inverses(n)

        for tick in range(int(steps.max().item())):
            yp = torch.remainder(torch.remainder(x, p).square(), p)
            yq = torch.remainder(torch.remainder(x, q).square(), q)
            proposed = torch.remainder(yp * q * inverse_p + yq * p * inverse_q, n)
            x = torch.where(steps > tick, proposed, x)

        result_digits = torch.remainder(torch.div(x[:, None], powers[None], rounding_mode="floor"), 10)
        endpoint = self.renderer(result_digits)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (7, self.config.vocab_size - 17), value=-10000)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -10000)
        return torch.where(occupied[..., None], digits, logits), {
            "widths": nw, "x_widths": xw, "steps": steps, "macrosteps": int(steps.max().item()),
        }


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=0,
                                  capturable=spec.device_type == "cuda")
    return OptimizerBundle(optimizer)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=256, max_steps=500)
