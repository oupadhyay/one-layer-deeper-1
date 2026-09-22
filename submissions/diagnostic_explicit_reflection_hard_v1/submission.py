"""Local-only diagnostic: explicit reflection arithmetic is not submittable."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

TABLE, D, H = 524_288, 64, 128


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.memory_a = nn.Embedding(TABLE, D // 2)
        self.memory_b = nn.Embedding(TABLE, D // 2)
        self.place = nn.Embedding(spec.max_seq_len, D)
        self.state_projection = nn.Linear(D, H, bias=False)
        self.place_projection = nn.Linear(D, H, bias=False)
        self.output = nn.Linear(H, 10)
        nn.init.normal_(self.memory_a.weight, std=.01)
        nn.init.normal_(self.memory_b.weight, std=.01)
        nn.init.normal_(self.place.weight, std=.02)

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
    def reflection(n, x):
        """Return little-endian decimal digits of abs(2*x - n) without int64 packing."""
        doubled = torch.zeros_like(x)
        carry = torch.zeros_like(x[:, 0])
        for slot in range(x.shape[1]):
            value = 2 * x[:, slot] + carry
            doubled[:, slot] = torch.remainder(value, 10)
            carry = torch.div(value, 10, rounding_mode="floor")

        greater = torch.zeros_like(carry, dtype=torch.bool)
        equal = torch.ones_like(greater)
        for slot in range(x.shape[1] - 1, -1, -1):
            greater = greater | (equal & doubled[:, slot].gt(n[:, slot]))
            equal = equal & doubled[:, slot].eq(n[:, slot])

        high = torch.where(greater[:, None], doubled, n)
        low = torch.where(greater[:, None], n, doubled)
        difference = torch.zeros_like(x)
        borrow = torch.zeros_like(carry)
        for slot in range(x.shape[1]):
            value = high[:, slot] - low[:, slot] - borrow
            borrowed = value.lt(0)
            difference[:, slot] = value + borrowed.long() * 10
            borrow = borrowed.long()
        return difference

    def decode(self, n, orbit):
        width = n.shape[1]
        slots = torch.arange(width, device=n.device)
        coefficient_a = torch.remainder(104729 + 13007 * slots + 97 * slots.square(), TABLE)
        coefficient_b = torch.remainder(130363 + 17011 * slots + 193 * slots.square(), TABLE)
        key_a = torch.remainder(((n + 11 * orbit) * coefficient_a).sum(1), TABLE)
        key_b = torch.remainder(((orbit + 13 * n) * coefficient_b).sum(1), TABLE)
        memory = torch.cat((self.memory_a(key_a), self.memory_b(key_b)), -1)
        state = self.state_projection(memory)[:, None]
        places = self.place_projection(self.place.weight[:width])[None]
        return self.output(F.gelu(state + places)), key_a

    def forward(self, ids, attention_mask=None):
        batch, length = ids.shape
        mask = ids.ne(0) if attention_mask is None else attention_mask.bool()
        region, place, steps = self.parse(ids, mask)
        nw, xw = region.eq(1).sum(1), region.eq(2).sum(1)
        if length > self.config.max_seq_len or bool(
            ((nw < 1) | (xw < 1) | (steps > 64)).any()
        ):
            raise ValueError("invalid arithmetic contract")
        width = length
        slots = torch.arange(width, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, np = field(1)
        x, _ = field(2)
        probability = F.one_hot(x, 10).to(self.memory_a.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        digit_values = torch.arange(10, device=ids.device, dtype=probability.dtype)
        runs = int(steps.max().item())
        key = None

        for tick in range(runs):
            x_digits = (probability * digit_values).sum(-1).round().long()
            orbit = self.reflection(n, x_digits)
            proposed, key = self.decode(n, orbit)
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
            "key": key,
            "endpoint_logits": endpoint,
        }


def build_model(spec: ModelSpec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-2, betas=(.9, .95), weight_decay=0,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer)


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
    canonical_targets = torch.where(slots[None] < count[:, None], gathered, 0)
    active = slots[None] < batch.auxiliary["widths"][selected, None]
    canonical_ce = F.cross_entropy(
        endpoint.transpose(1, 2), canonical_targets, reduction="none"
    )
    canonical_per_example = (
        (canonical_ce * active).sum(1) / active.sum(1).clamp_min(1)
    )
    canonical_loss = (canonical_per_example * weights).sum() / weights.sum()
    return main + canonical_loss


SUBMISSION = Submission(
    build_model, build_optimizer, batch_size=128, eval_batch_size=256,
    max_steps=20000, token_training_loss=token_training_loss,
)
