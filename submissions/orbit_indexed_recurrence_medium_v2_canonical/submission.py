"""Local probe: learned categorical recurrence indexed by a reflection orbit."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, ORBITS, RBF, H = 4, 20000, 256, 256


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.orbit_logits = nn.Embedding(ORBITS, W * 10)
        nn.init.normal_(self.orbit_logits.weight, std=.02)
        self.centers = nn.Parameter(torch.rand(RBF))
        self.log_scales = nn.Parameter(torch.randn(RBF) * .1 + 6)
        self.smooth = nn.Sequential(nn.Linear(RBF, H), nn.GELU(), nn.Linear(H, W * 10))

    @staticmethod
    def parse(ids, mask):
        digit = ids.ge(7) & ids.lt(17) & mask
        marker = ids.eq(2).long() + 2 * ids.eq(3).long() + 3 * ids.eq(4).long()
        region = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        same = region[:, :, None].eq(region[:, None, :]) & region[:, :, None].gt(0)
        place = (same & (index[None, None, :] > index[None, :, None]) & digit[:, None, :]).sum(-1)
        value = (ids - 7).clamp(0, 9)
        powers = torch.pow(ids.new_tensor(10), place)
        steps = (value * powers * region.eq(3)).sum(1).long()
        return region, place, steps

    def forward(self, ids, attention_mask=None):
        batch, length = ids.shape
        mask = ids.ne(0) if attention_mask is None else attention_mask.bool()
        region, place, steps = self.parse(ids, mask)
        nw, xw = region.eq(1).sum(1), region.eq(2).sum(1)
        if length > self.config.max_seq_len or bool(((nw < 1) | (nw > W) | (xw < 1) | (xw > W) | (steps > 64)).any()):
            raise ValueError("invalid Easy contract")
        slots = torch.arange(W, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, np = field(1)
        x, xp = field(2)
        powers = ids.new_tensor((1, 10, 100, 1000))
        n_int = (n * powers).sum(1)
        probability = F.one_hot(x, 10).to(self.orbit_logits.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        digit_values = torch.arange(10, device=ids.device, dtype=probability.dtype)
        runs = int(steps.max().item())

        for tick in range(runs):
            x_soft = (probability * digit_values).sum(-1)
            x_hard = x_soft.round().long()
            x_int = (x_hard * powers).sum(1)
            orbit = (2 * x_int - n_int).abs().clamp_max(ORBITS - 1)
            table = self.orbit_logits(orbit)
            if self.training:
                table = table * (torch.rand(batch, 1, device=ids.device) >= .5).to(table.dtype)
            ratio = orbit.to(table.dtype) / n_int.clamp_min(1).to(table.dtype)
            rbf = torch.exp(-(ratio[:, None] - self.centers).square() * self.log_scales.exp())
            proposed = (table + self.smooth(rbf)).view(batch, W, 10)
            soft = proposed.softmax(-1)
            hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero = F.one_hot(torch.zeros_like(n), 10).to(feedback.dtype)
            confident = soft.max(-1, keepdim=True).values.ge(.5)
            feedback = torch.where(np[..., None] & confident, feedback, zero)
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, feedback, probability)
            xp = torch.where(active[:, :, 0], np, xp)

        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (7, self.config.vocab_size - 17), value=-10000)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -10000)
        return torch.where(occupied[..., None], digits, logits), {
            "widths": nw,
            "x_widths": xw,
            "steps": steps,
            "macrosteps": runs,
            "orbit": orbit,
        }


def build_model(spec: ModelSpec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, betas=(.9, .95), weight_decay=0,
                                  capturable=spec.device_type == "cuda")
    return OptimizerBundle(optimizer)


def token_training_loss(batch: TokenLossBatch):
    selected = batch.auxiliary["steps"].eq(1)
    valid = batch.valid_mask[selected]
    if not bool(valid.any()):
        return batch.logits.sum() * 0
    ce = F.cross_entropy(batch.logits[selected].transpose(1, 2), batch.labels[selected],
                         ignore_index=-100, reduction="none")
    return (ce * valid).sum() / valid.sum()


SUBMISSION = Submission(build_model, build_optimizer, batch_size=128, eval_batch_size=256,
                        max_steps=10000, token_training_loss=token_training_loss)
