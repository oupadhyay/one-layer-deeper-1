"""Reflection fallback plus a learned two-character periodic memory."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, ORBITS, FREQUENCIES, MEMORIES = 4, 20000, 64, 512


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class PeriodicMemory(nn.Module):
    def __init__(self):
        super().__init__()
        # Random candidate periods span the useful input scale but remain fully learned.
        initial_periods = torch.empty(FREQUENCIES).uniform_(2, 64)
        initial_fraction = (initial_periods - 2) / 62
        self.raw_periods = nn.Parameter(torch.logit(initial_fraction.clamp(.01, .99)))
        self.head_bias = nn.Parameter(torch.randn(2, FREQUENCIES) * .5)
        self.head_phase_gain = nn.Parameter(torch.randn(2, FREQUENCIES) * .1)
        self.keys = nn.Parameter(torch.empty(MEMORIES, 2).uniform_(-1, 1))
        self.log_scales = nn.Parameter(torch.full((MEMORIES, 2), 2.0))
        self.values = nn.Parameter(torch.randn(MEMORIES, W * 10) * .02)
        self.temperature = nn.Parameter(torch.tensor(-3.0))

    def forward(self, n: torch.Tensor, x: torch.Tensor):
        periods = 2 + 62 * self.raw_periods.float().sigmoid()
        omega = 2 * math.pi / periods
        n_phase = n.float()[:, None] * omega[None]
        x_phase = x.float()[:, None] * omega[None]
        selector_logits = (
            self.head_bias.float()[None]
            + self.head_phase_gain.float()[None] * n_phase.cos()[:, None]
            + 8 * n_phase.cos()[:, None]
        )
        selectors = (selector_logits / .2).softmax(-1)
        coordinates = torch.einsum("bhk,bk->bh", selectors, x_phase.cos())
        keys = self.keys.float().tanh()
        scales = self.log_scales.float().clamp(-4, 4).exp()
        distance = ((coordinates[:, None] - keys[None]).square() * scales[None]).sum(-1)
        routing = (-distance / self.temperature.float().exp().clamp(.03, 2)).softmax(-1)
        logits = (routing.to(self.values.dtype) @ self.values).view(-1, W, 10)
        return logits, selectors, periods


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.orbit_logits = nn.Embedding(ORBITS, W * 10)
        nn.init.normal_(self.orbit_logits.weight, std=.02)
        self.periodic = PeriodicMemory()

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
        if length > self.config.max_seq_len or bool(
            ((nw < 1) | (nw > W) | (xw < 1) | (xw > W) | (steps > 64)).any()
        ):
            raise ValueError("invalid Medium contract")
        slots = torch.arange(W, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, np = field(1)
        x, _ = field(2)
        powers = ids.new_tensor((1, 10, 100, 1000))
        n_int = (n * powers).sum(1)
        probability = F.one_hot(x, 10).to(self.orbit_logits.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        digit_values = torch.arange(10, device=ids.device, dtype=probability.dtype)
        runs = int(steps.max().item())
        torus_logits = selectors = periods = None

        for tick in range(runs):
            x_soft = (probability * digit_values).sum(-1)
            x_int = (x_soft.round().long() * powers).sum(1)
            orbit = (2 * x_int - n_int).abs().clamp_max(ORBITS - 1)
            table_logits = self.orbit_logits(orbit).view(batch, W, 10)
            torus_logits, selectors, periods = self.periodic(n_int, x_int)
            confidence = table_logits.softmax(-1).amax(-1).amax(-1)
            if self.training:
                proposed = table_logits
            else:
                proposed = torch.where(
                    confidence[:, None, None].ge(.15), table_logits, torus_logits
                )
            soft = proposed.softmax(-1)
            hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero = F.one_hot(torch.zeros_like(n), 10).to(feedback.dtype)
            feedback = torch.where(np[..., None], feedback, zero)
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
            "widths": nw,
            "x_widths": xw,
            "steps": steps,
            "macrosteps": runs,
            "torus_logits": torus_logits,
            "selectors": selectors,
            "periods": periods,
            "n_int": n_int,
            "target_positions": target,
        }


def build_model(spec: ModelSpec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-2, betas=(.9, .95), weight_decay=0,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer)


def token_training_loss(batch: TokenLossBatch):
    selected = batch.auxiliary["steps"].eq(1)
    valid = batch.valid_mask[selected]
    if not bool(valid.any()):
        return batch.logits.sum() * 0
    ce = F.cross_entropy(
        batch.logits[selected].transpose(1, 2), batch.labels[selected],
        ignore_index=-100, reduction="none",
    )
    main = (ce * valid).sum() / valid.sum()
    torus = batch.auxiliary["torus_logits"][selected]
    slots = torch.arange(W, device=torus.device)
    widths = batch.auxiliary["widths"][selected]
    labels = batch.labels[selected]
    label_digits = torch.where(labels.ge(7) & labels.lt(17), labels - 7, 0)
    positions = batch.auxiliary["target_positions"][selected]
    targets = label_digits.gather(1, positions.clamp(0, labels.shape[1] - 1))
    active = slots[None] < widths[:, None]
    torus_ce = F.cross_entropy(torus.transpose(1, 2), targets, reduction="none")
    torus_loss = (torus_ce * active).sum() / active.sum().clamp_min(1)
    selectors = batch.auxiliary["selectors"][selected].float()
    periods = batch.auxiliary["periods"].float()
    n = batch.auxiliary["n_int"][selected].float()
    closure = (
        selectors * (1 - torch.cos(2 * math.pi * n[:, None] / periods[None]))[:, None]
    ).sum(-1).mean()
    diversity = (selectors[:, 0] * selectors[:, 1]).sum(-1).mean()
    return main + .5 * torus_loss + .2 * closure + .1 * diversity


SUBMISSION = Submission(
    build_model, build_optimizer, batch_size=128, eval_batch_size=256,
    max_steps=30000, token_training_loss=token_training_loss,
)
