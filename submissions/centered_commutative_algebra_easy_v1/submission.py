"""Learned centered commutative-algebra recurrence."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, D, R, H = 4, 64, 192, 96


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


class Algebra(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, D)
        self.place = nn.Parameter(torch.randn(W, D))
        self.encode_norm = RMS(D)
        self.n_center = nn.Linear(D, D)
        self.x_center = nn.Linear(D, D)
        self.center_shift = nn.Linear(D, D)
        self.alpha = nn.Parameter(torch.randn(()))
        self.center_norm = RMS(D)
        self.state_norm = RMS(D)
        self.left = nn.Linear(D, R, bias=False)
        self.condition = nn.Linear(D, R)
        self.product = nn.Linear(R, D, bias=False)
        self.bias = nn.Linear(D, D)
        self.transition_norm = RMS(D)
        self.state_decode = nn.Linear(D, H)
        self.center_decode = nn.Linear(D, H)
        self.query = nn.Parameter(torch.randn(W, H))
        self.output = nn.Linear(H, 10)
        self.reconstruct = nn.Linear(D, W * 10)

    def encode(self, probability, present):
        token = (probability @ self.digit.weight) * self.place[None, :, :]
        return self.encode_norm((token * present[..., None]).sum(1))

    def initialize(self, n_probability, np, x_probability, xp):
        hn = self.encode(n_probability, np)
        hx = self.encode(x_probability, xp)
        center = self.center_norm(self.n_center(hn))
        state = self.state_norm(self.x_center(hx) + self.alpha * self.center_shift(hn))
        reconstruction = self.reconstruct(hx).reshape(-1, W, 10)
        return center, state, reconstruction

    def step(self, state, center):
        factors = self.left(state) * (1 + torch.tanh(self.condition(center)))
        return self.transition_norm(self.product(factors.square()) + self.bias(center))

    def decode(self, state, center):
        hidden = F.gelu(
            self.state_decode(state)[:, None, :]
            + self.center_decode(center)[:, None, :]
            + self.query[None, :, :]
        )
        return self.output(hidden)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.algebra = Algebra()

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
        n_probability = F.one_hot(n, 10).to(self.algebra.digit.weight.dtype)
        x_probability = F.one_hot(x, 10).to(self.algebra.digit.weight.dtype)
        center, state, reconstruction = self.algebra.initialize(n_probability, np, x_probability, xp)
        runs = int(steps.max().item())
        for tick in range(runs):
            proposed = self.algebra.step(state, center)
            state = torch.where((steps > tick)[:, None], proposed, state)
        end = self.algebra.decode(state, center)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(end.dtype) * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), end)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (7, self.config.vocab_size - 17), value=-10000)
        logits = end.new_full((batch, length, self.config.vocab_size), -10000)
        return torch.where(occupied[..., None], digits, logits), {
            "widths": nw,
            "x_widths": xw,
            "steps": steps,
            "macrosteps": runs,
            "state": state,
            "input_digits": x,
            "input_present": xp,
            "reconstruction": reconstruction,
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
        lr=6e-4,
        betas=(.9, .95),
        eps=1e-8,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 32, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    ce = F.cross_entropy(batch.logits.transpose(1, 2), batch.labels, ignore_index=-100, reduction="none")
    valid = batch.valid_mask
    endpoint = ((ce * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()
    auxiliary = batch.auxiliary
    reconstruction = F.cross_entropy(
        auxiliary["reconstruction"].transpose(1, 2),
        auxiliary["input_digits"],
        reduction="none",
    )
    present = auxiliary["input_present"]
    reconstruction = (reconstruction * present).sum() / present.sum().clamp_min(1)
    return endpoint + .1 * reconstruction


SUBMISSION = Submission(
    build_model,
    build_optimizer,
    batch_size=256,
    eval_batch_size=512,
    max_steps=None,
    token_training_loss=token_training_loss,
)
