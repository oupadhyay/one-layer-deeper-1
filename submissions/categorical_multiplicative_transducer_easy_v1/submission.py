"""Wide categorical multiplicative transducer for Seen-N specialization."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, TokenLossBatch, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
WIDTH, CATEGORIES, HIDDEN, BOTTLENECK, MAX_T = 4, 11, 4096, 256, 64
NEG = -10_000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Transition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = WIDTH * CATEGORIES
        self.n_projection = nn.Linear(features, HIDDEN)
        self.x_projection = nn.Linear(features, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)
        self.down = nn.Linear(HIDDEN, BOTTLENECK)
        self.up = nn.Linear(BOTTLENECK, HIDDEN)
        self.readout = nn.Linear(HIDDEN, WIDTH * 10)

    @staticmethod
    def categorical(digits: Tensor, present: Tensor, dtype: torch.dtype) -> Tensor:
        category = torch.where(present, digits, torch.full_like(digits, 10))
        return F.one_hot(category, CATEGORIES).to(dtype).flatten(1)

    def forward(self, n: Tensor, n_present: Tensor, x: Tensor,
                x_present: Tensor) -> tuple[Tensor, Tensor]:
        dtype = self.n_projection.weight.dtype
        n_features = self.categorical(n, n_present, dtype)
        x_features = torch.cat((x, (~x_present)[..., None].to(dtype)), -1).flatten(1)
        hidden = F.gelu(self.n_projection(n_features)) * F.gelu(self.x_projection(x_features))
        hidden = hidden + self.up(F.gelu(self.down(self.norm(hidden))))
        return self.readout(hidden).reshape(-1, WIDTH, 10), hidden


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = Transition()

    @staticmethod
    def parse(ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        marker = ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() + 3 * ids.eq(T_MARK).long()
        role = torch.cummax(marker, dim=1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(dim=-1)
        values = (ids - DIGIT_BASE).clamp(0, 9)
        steps = (values * torch.pow(ids.new_tensor(10), place) * role.eq(3)).sum(1).long()
        return role, place, steps

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, steps = self.parse(input_ids, mask)
        n_width, x_width = role.eq(1).sum(1), role.eq(2).sum(1)
        invalid = (n_width < 1) | (n_width > WIDTH) | (x_width < 1) | (x_width > WIDTH)
        if bool(invalid.any()) or bool((steps > MAX_T).any()):
            raise ValueError("invalid Easy contract")
        slots = torch.arange(WIDTH, device=input_ids.device)
        values = (input_ids - DIGIT_BASE).clamp(0, 9)

        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            return (values[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, n_present = field(1)
        x, x_present = field(2)
        probability = F.one_hot(x, 10).to(self.transition.n_projection.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        present = x_present
        runs = 3 if self.training else int(steps.max().item())
        hidden = None
        for macrostep in range(runs):
            proposed, hidden = self.transition(n, n_present, probability, present)
            active = (steps > macrostep)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, proposed.softmax(-1), probability)
            present = torch.where(active[:, :, 0], n_present, present)

        target = mask.sum(1)[:, None] - 1 - slots[None]
        active_place = slots[None] < n_width[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype)
        placement = placement * active_place[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digit_logits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), NEG)
        logits = torch.where(occupied[..., None], digit_logits, logits)
        return logits, {"widths": n_width, "x_widths": x_width, "steps": steps,
                        "macrosteps": runs, "hidden": hidden}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim == 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 16.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    losses = F.cross_entropy(batch.logits.transpose(1, 2), batch.labels,
                             ignore_index=-100, reduction="none")
    valid = batch.valid_mask.to(losses.dtype)
    horizon = torch.where(batch.auxiliary["steps"].eq(1), losses.new_tensor(4.),
                          losses.new_tensor(1.))[:, None]
    weight = valid * horizon
    return (losses * weight).sum() / weight.sum().clamp_min(1.)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=512,
                        eval_batch_size=512, max_steps=None,
                        token_training_loss=token_training_loss)
