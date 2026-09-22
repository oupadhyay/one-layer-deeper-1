"""Learned spectral period attention with localized soft routing."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, D, H, EXPERTS, PERIODS, HEADS = 8, 64, 128, 1024, 255, 4


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(5, D), nn.GELU(), nn.Linear(D, D), nn.GELU())
        self.period_score = nn.Linear(D, HEADS)
        self.encoder = nn.Sequential(nn.Linear(HEADS * D, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
        self.router = nn.Linear(H, EXPERTS)
        self.base = nn.Linear(H, W * 10)
        self.experts = nn.Parameter(torch.empty(EXPERTS, W * 10))
        nn.init.normal_(self.experts, std=.02)

    def forward(self, n, x):
        periods = torch.arange(2, PERIODS + 2, device=x.device, dtype=torch.float32)
        n_angle = 2 * torch.pi * n.float()[:, None] / periods[None]
        x_angle = 2 * torch.pi * x.float()[:, None] / periods[None]
        features = torch.stack((periods[None].expand_as(n_angle) / (PERIODS + 1),
                                n_angle.sin(), n_angle.cos(), x_angle.sin(), x_angle.cos()), -1)
        tokens = self.token(features.to(self.token[0].weight.dtype))
        weights = (self.period_score(tokens).float().transpose(1, 2) / .1).softmax(-1).to(tokens.dtype)
        pooled = torch.einsum("bhp,bpd->bhd", weights, tokens)
        hidden = self.encoder(pooled.flatten(1))
        dtype = hidden.dtype
        routing = (self.router(hidden).float() / .2).softmax(-1).to(dtype)
        local = routing @ self.experts
        return (self.base(hidden) + local).view(-1, W, 10), routing


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
            raise ValueError("invalid Medium contract")
        slots = torch.arange(W, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, np = field(1)
        x, xp = field(2)
        probability = F.one_hot(x, 10).to(self.transition.token[0].weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        digit_values = torch.arange(10, device=ids.device, dtype=torch.float32)
        powers = torch.pow(ids.new_tensor(10), torch.arange(W, device=ids.device)).float()
        n_int = (n * powers.long()).sum(1)
        runs = int(steps.max().item())
        routing = None

        for tick in range(runs):
            x_digits = (probability.float() * digit_values).sum(-1)
            x_int = (x_digits * powers).sum(1)
            proposed, routing = self.transition(n_int, x_int)
            soft = proposed.softmax(-1)
            hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero = F.one_hot(torch.zeros_like(n), 10).to(feedback.dtype)
            confident = soft.max(-1, keepdim=True).values.ge(.5)
            feedback = torch.where(np[..., None] & confident, feedback, zero)
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, feedback, probability)

        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (7, self.config.vocab_size - 17), value=-10000)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -10000)
        return torch.where(occupied[..., None], digits, logits), {
            "widths": nw, "x_widths": xw, "steps": steps,
            "macrosteps": runs, "routing": routing,
        }


def build_model(spec: ModelSpec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.01,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 16, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    ce = F.cross_entropy(batch.logits.transpose(1, 2), batch.labels,
                         ignore_index=-100, reduction="none")
    per_example = (ce * batch.valid_mask).sum(1) / batch.valid_mask.sum(1).clamp_min(1)
    steps = batch.auxiliary["steps"]
    losses = {depth: per_example[steps.eq(depth)].mean() for depth in (1, 2, 4)
              if bool(steps.eq(depth).any())}
    anchor = losses.get(1, per_example.mean())
    gate1 = torch.exp(-3 * anchor.detach())
    loss = anchor + gate1 * .3 * losses.get(2, anchor * 0)
    gate2 = torch.exp(-3 * losses.get(2, anchor).detach())
    loss = loss + gate1 * gate2 * .1 * losses.get(4, anchor * 0)
    routing = batch.auxiliary["routing"]
    if routing is not None:
        loss = loss + 1e-3 * EXPERTS * routing.float().mean(0).square().sum()
    return loss


SUBMISSION = Submission(build_model, build_optimizer, batch_size=128, eval_batch_size=256,
                        max_steps=50000, token_training_loss=token_training_loss)
