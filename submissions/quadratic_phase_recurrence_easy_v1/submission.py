"""Categorical recurrence with learned quadratic phase features."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, K, D = 4, 128, 192


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMS(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        scale = (value.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt().to(value.dtype)
        return value * scale * self.weight


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_coordinate = nn.Linear(2 * W, 1)
        self.n_coordinate = nn.Linear(2 * W, 1)
        self.linear_frequency = nn.Parameter(torch.randn(K))
        self.square_frequency = nn.Parameter(torch.randn(K))
        self.condition_frequency = nn.Parameter(torch.randn(K))
        self.phase_bias = nn.Parameter(torch.randn(K))
        self.raw = nn.Linear(4 * W, D)
        self.mix = nn.Sequential(
            nn.Linear(D + 2 * K, D),
            nn.GELU(),
            RMS(D),
            nn.Linear(D, D),
            nn.GELU(),
        )
        self.query = nn.Parameter(torch.randn(W, D))
        self.output = nn.Linear(D, 10)

    def forward(self, n, np, x_probability, xp):
        dtype = self.query.dtype
        values = torch.arange(10, device=x_probability.device, dtype=dtype)
        x = x_probability @ values / 9
        n = n.to(dtype) / 9
        npf, xpf = np.to(dtype), xp.to(dtype)
        xc = self.x_coordinate(torch.cat((x, xpf), -1))
        nc = self.n_coordinate(torch.cat((n, npf), -1))
        phase = (
            xc * self.linear_frequency
            + xc.square() * self.square_frequency
            + nc * self.condition_frequency
            + self.phase_bias
        )
        raw = F.gelu(self.raw(torch.cat((n, npf, x, xpf), -1)))
        state = self.mix(torch.cat((raw, phase.sin(), phase.cos()), -1))
        hidden = state[:, None, :] + self.query[None, :, :]
        return self.output(hidden), state


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
        probability = F.one_hot(x, 10).to(self.transition.query.dtype)
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
        lr=1e-3,
        betas=(.9, .95),
        eps=1e-8,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 16, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    ce = F.cross_entropy(batch.logits.transpose(1, 2), batch.labels, ignore_index=-100, reduction="none")
    valid = batch.valid_mask
    sequence = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    weight = torch.where(batch.auxiliary["steps"].eq(1), sequence.new_tensor(4.), sequence.new_tensor(1.))
    return (sequence * weight).sum() / weight.sum()


SUBMISSION = Submission(
    build_model,
    build_optimizer,
    batch_size=128,
    eval_batch_size=256,
    max_steps=None,
    token_training_loss=token_training_loss,
)
