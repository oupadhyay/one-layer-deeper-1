"""Learned pairwise digit transition with canonical closed-state supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

D, H = 64, 128


class Config:
    def __init__(self, vocab_size, max_seq_len):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMS(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, value):
        scale = (value.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        return value * scale.to(value.dtype) * self.weight


class Transition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.digit = nn.Embedding(10, D)
        self.xdigit = nn.Embedding(10, D)
        self.presence = nn.Embedding(2, D)
        self.nplace = nn.Embedding(width, D)
        self.xplace = nn.Embedding(width, D)
        self.relation = nn.Linear(3 * D, H)
        self.ninit = nn.Linear(D, H)
        self.scan_in = nn.Linear(H, 3 * H)
        self.scan_state = nn.Linear(H, 3 * H, bias=False)
        self.norm = RMS(H)
        self.out = nn.Linear(H, 10)

    def encode(self, n, present):
        width = n.shape[1]
        encoded = self.digit(n) + self.nplace.weight[:width][None] + self.presence(present.long())
        weight = present[..., None].to(encoded.dtype)
        summary = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1)
        return encoded, torch.tanh(self.ninit(summary))

    def forward(self, n_encoded, n_present, x, x_present, initial):
        width = x.shape[1]
        x_encoded = x.to(self.xdigit.weight.dtype) @ self.xdigit.weight
        x_encoded = x_encoded + self.xplace.weight[:width][None] + self.presence(x_present.long())
        batch = x.shape[0]
        n_pair = n_encoded[:, None].expand(batch, width, width, D)
        x_pair = x_encoded[:, :, None].expand(batch, width, width, D)
        delta = self.xplace.weight[:width, None] - self.nplace.weight[None, :width]
        delta = delta[None].expand(batch, width, width, D)
        pair = F.gelu(self.relation(torch.cat((n_pair, x_pair, delta), -1)))
        weight = n_present[:, None, :, None].to(pair.dtype)
        related = (pair * weight).sum(2) / weight.sum(2).clamp_min(1)

        hidden = initial
        states = []
        for item in related.unbind(1):
            ir, iz, candidate_in = self.scan_in(item).chunk(3, -1)
            hr, hz, candidate_state = self.scan_state(hidden).chunk(3, -1)
            reset = (ir + hr).sigmoid()
            update = (iz + hz).sigmoid()
            candidate = (candidate_in + reset * candidate_state).tanh()
            hidden = (1 - update) * candidate + update * hidden
            states.append(hidden)
        state = torch.stack(states, 1)
        return self.out(self.norm(state)), state


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = Transition(spec.max_seq_len)

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
        if length > self.config.max_seq_len or bool(((nw < 1) | (xw < 1) | (steps > 64)).any()):
            raise ValueError("invalid Hard contract")
        slots = torch.arange(length, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, n_present = field(1)
        x, x_present = field(2)
        n_encoded, initial = self.transition.encode(n, n_present)
        probability = F.one_hot(x, 10).to(self.transition.xdigit.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        runs = int(steps.max().item())
        state = None
        for tick in range(runs):
            proposed, state = self.transition(n_encoded, n_present, probability, x_present, initial)
            soft = proposed.softmax(-1)
            hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero = F.one_hot(torch.zeros_like(n), 10).to(feedback.dtype)
            feedback = torch.where(n_present[..., None], feedback, zero)
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, feedback, probability)
            x_present = torch.where(active[:, :, 0], n_present, x_present)

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
            "endpoint_logits": endpoint,
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
        [{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0}],
        lr=1e-3, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 8, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    selected = batch.auxiliary["steps"].le(4)
    valid = batch.valid_mask[selected]
    if not bool(valid.any()):
        return batch.logits.sum() * 0
    ce = F.cross_entropy(
        batch.logits[selected].transpose(1, 2), batch.labels[selected],
        ignore_index=-100, reduction="none",
    )
    selected_steps = batch.auxiliary["steps"][selected]
    weights = torch.where(
        selected_steps.eq(1), ce.new_tensor(1.0),
        torch.where(selected_steps.eq(2), ce.new_tensor(.1), ce.new_tensor(.005)),
    )
    main_per_example = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    main = (main_per_example * weights).sum() / weights.sum()

    labels = batch.labels[selected]
    count = valid.sum(1)
    endpoint = batch.auxiliary["endpoint_logits"][selected]
    width = endpoint.shape[1]
    slots = torch.arange(width, device=endpoint.device)
    source = (count[:, None] - 1 - slots[None]).clamp(0, labels.shape[1] - 1)
    gathered = labels.gather(1, source).sub(7).clamp(0, 9)
    canonical_target = torch.where(slots[None] < count[:, None], gathered, 0)
    active = slots[None] < batch.auxiliary["widths"][selected, None]
    canonical_ce = F.cross_entropy(endpoint.transpose(1, 2), canonical_target, reduction="none")
    canonical_per_example = (canonical_ce * active).sum(1) / active.sum(1).clamp_min(1)
    canonical = (canonical_per_example * weights).sum() / weights.sum()
    return main + canonical


SUBMISSION = Submission(
    build_model, build_optimizer, batch_size=64, eval_batch_size=256,
    max_steps=None, token_training_loss=token_training_loss,
)
