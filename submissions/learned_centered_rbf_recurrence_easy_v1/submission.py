"""Categorical recurrence through a learned centered scalar invariant and RBF map."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, Q, J, H = 4, 256, 128, 128


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.radix_logits = nn.Parameter(torch.randn(Q) * 6)
        self.center_logits = nn.Parameter(torch.randn(Q))
        self.centers = nn.Parameter(torch.rand(J) * .25)
        self.log_scales = nn.Parameter(torch.randn(J) * .1 + 5)
        self.gate_logits = nn.Parameter(torch.randn(Q) * .01)
        self.hidden = nn.Sequential(nn.Linear(J, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
        self.output = nn.Linear(H, W * 10)

    def coordinate(self, digits, present):
        radix = 1 + F.softplus(self.radix_logits)
        place = torch.arange(W, device=digits.device, dtype=digits.dtype)
        weight = radix[:, None].pow(place[None, :])
        weight = weight / weight.sum(1, keepdim=True)
        return (digits * present.to(digits.dtype)) @ weight.transpose(0, 1) / 9

    def forward(self, n, np, probability, xp):
        dtype = self.output.weight.dtype
        values = torch.arange(10, device=probability.device, dtype=dtype)
        x = probability @ values
        n = n.to(dtype)
        xc = self.coordinate(x, xp)
        nc = self.coordinate(n, np)
        centered = (xc - self.center_logits.sigmoid() * nc) / nc.clamp_min(1e-4)
        invariant = centered.square()
        rbf = torch.exp(-(invariant[:, :, None] - self.centers).square() * self.log_scales.exp())
        head_state = self.hidden(rbf)
        gate = self.gate_logits.softmax(0)
        state = (head_state * gate[None, :, None]).sum(1)
        head_logits = self.output(head_state).reshape(-1, Q, W, 10)
        logits = (head_logits * gate[None, :, None, None]).sum(1)
        return logits, state


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = Transition()

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
        probability = F.one_hot(x, 10).to(self.transition.output.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        present = xp
        runs = min(4, int(steps.max().item())) if self.training else int(steps.max().item())
        state = None
        for tick in range(runs):
            proposed, state = self.transition(n, np, probability, present)
            soft = proposed.softmax(-1)
            hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, feedback, probability)
            present = torch.where(active[:, :, 0], np, present)
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
            "state": state,
        }


def build_model(spec: ModelSpec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim == 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        ({"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.}),
        lr=2e-3,
        betas=(.9, .95),
        eps=1e-8,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 16, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    selected = batch.auxiliary["steps"].eq(1)
    valid = batch.valid_mask[selected]
    if not bool(valid.any()):
        return batch.logits.sum() * 0
    ce = F.cross_entropy(batch.logits[selected].transpose(1, 2), batch.labels[selected],
                         ignore_index=-100, reduction="none")
    return (ce * valid).sum() / valid.sum()


SUBMISSION = Submission(
    build_model,
    build_optimizer,
    batch_size=128,
    eval_batch_size=256,
    max_steps=10000,
    token_training_loss=token_training_loss,
)
