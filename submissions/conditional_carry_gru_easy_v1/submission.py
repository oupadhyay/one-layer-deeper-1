"""Learned directional recurrent transducer for the Easy tier."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, TokenLossBatch, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
WIDTH, D_MODEL, HIDDEN, MAX_T = 4, 96, 128, 64
NEG = -10_000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Transition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit = nn.Embedding(10, D_MODEL)
        self.presence = nn.Embedding(2, D_MODEL)
        self.n_scan = nn.GRU(D_MODEL, HIDDEN, batch_first=True)
        self.x_scan = nn.GRU(D_MODEL + HIDDEN, HIDDEN, batch_first=True)
        self.initial = nn.Linear(HIDDEN, HIDDEN)
        self.readout = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 10))

    def encode_n(self, n: Tensor, present: Tensor) -> tuple[Tensor, Tensor]:
        state = self.digit(n) + self.presence(present.long())
        encoded, hidden = self.n_scan(state)
        weight = present[..., None].to(encoded.dtype)
        context = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1.)
        return context, torch.tanh(self.initial(hidden))

    def forward(self, context: Tensor, initial: Tensor, x: Tensor,
                present: Tensor) -> tuple[Tensor, Tensor]:
        state = x.to(self.digit.weight.dtype) @ self.digit.weight
        state = state + self.presence(present.long())
        context_slots = context[:, None, :].expand(-1, WIDTH, -1)
        scanned, hidden = self.x_scan(torch.cat((state, context_slots), -1), initial)
        return self.readout(scanned), hidden


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = Transition()

    @staticmethod
    def parse(ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        marker = ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() + 3 * ids.eq(T_MARK).long()
        role = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & (index[None, None, :] > index[None, :, None]) & digit[:, None, :]).sum(-1)
        value = (ids - DIGIT_BASE).clamp(0, 9)
        steps = (value * torch.pow(ids.new_tensor(10), place) * role.eq(3)).sum(1).long()
        return role, place, steps

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        batch, length = input_ids.shape
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, steps = self.parse(input_ids, mask)
        n_width, x_width = role.eq(1).sum(1), role.eq(2).sum(1)
        invalid = (n_width < 1) | (n_width > WIDTH) | (x_width < 1) | (x_width > WIDTH) | (steps > MAX_T)
        if length > self.config.max_seq_len or bool(invalid.any()):
            raise ValueError("invalid Easy contract")
        slots = torch.arange(WIDTH, device=input_ids.device)
        values = (input_ids - DIGIT_BASE).clamp(0, 9)

        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            return (values[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, n_present = field(1)
        x, x_present = field(2)
        context, initial = self.transition.encode_n(n, n_present)
        probability = F.one_hot(x, 10).to(self.transition.digit.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        present = x_present
        runs = 3 if self.training else int(steps.max().item())
        hidden = None
        for macrostep in range(runs):
            proposed, hidden = self.transition(context, initial, probability, present)
            active = (steps > macrostep)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, proposed.softmax(-1), probability)
            present = torch.where(active[:, :, 0], n_present, present)

        target = mask.sum(1)[:, None] - 1 - slots[None]
        active_place = slots[None] < n_width[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active_place[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digit_logits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), NEG)
        return torch.where(occupied[..., None], digit_logits, logits), {
            "widths": n_width, "x_widths": x_width, "steps": steps,
            "macrosteps": runs, "hidden": hidden,
        }


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim == 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.}],
        lr=1e-3, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 8., 1.))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    loss = F.cross_entropy(batch.logits.transpose(1, 2), batch.labels,
                           ignore_index=-100, reduction="none")
    valid = batch.valid_mask
    mean = (loss * valid).sum() / valid.sum()
    preference = loss.masked_fill(~valid, NEG).softmax(1)
    return mean + .5 * (preference * loss).sum(1).mean()


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None,
                        token_training_loss=token_training_loss)
