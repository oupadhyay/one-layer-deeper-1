"""Dual-stream N-conditioned recurrent transducer for Easy E3."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
WIDTH, D_MODEL, SCAN, ATTN, MAX_T = 4, 96, 48, 32, 64
STATE_ELEMENTS, NEG = 148_138, -10_000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: Tensor) -> Tensor:
        scale = value.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt()
        return value * scale.to(value.dtype) * self.weight


class ConditionalTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, D_MODEL)
        self.presence_embedding = nn.Embedding(2, D_MODEL)
        self.n_scan = nn.GRU(D_MODEL, SCAN, batch_first=True, bidirectional=True)
        self.x_scan = nn.GRU(D_MODEL, SCAN, batch_first=True, bidirectional=True)
        self.query = nn.Linear(D_MODEL, ATTN, bias=False)
        self.key = nn.Linear(D_MODEL, ATTN, bias=False)
        self.value = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.content = nn.Linear(2 * D_MODEL, D_MODEL)
        self.film = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.norm = RMSNorm(D_MODEL)
        self.refine = nn.Linear(D_MODEL, D_MODEL)
        self.readout = nn.Linear(D_MODEL, 10)

    def encode_n(self, n: Tensor, present: Tensor) -> Tensor:
        state = self.digit_embedding(n) + self.presence_embedding(present.long())
        encoded, _ = self.n_scan(state)
        return encoded

    def forward(self, n_state: Tensor, n_present: Tensor, x: Tensor,
                x_present: Tensor) -> tuple[Tensor, Tensor]:
        dtype = self.digit_embedding.weight.dtype
        x_state = x.to(dtype) @ self.digit_embedding.weight
        x_state = x_state + self.presence_embedding(x_present.long())
        x_encoded, _ = self.x_scan(x_state)
        scores = self.query(x_encoded) @ self.key(n_state).transpose(1, 2)
        scores = scores / (ATTN ** 0.5)
        scores = scores.masked_fill(~n_present[:, None, :], NEG)
        context = scores.softmax(-1) @ self.value(n_state)
        hidden = F.gelu(self.content(torch.cat((x_encoded, context), dim=-1)))
        scale, shift = self.film(context).chunk(2, dim=-1)
        hidden = self.norm(hidden) * (1.0 + torch.tanh(scale)) + shift
        hidden = hidden + F.gelu(self.refine(hidden))
        return self.readout(hidden), hidden


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = ConditionalTransition()

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
            raise ValueError("input exceeds the Easy decimal-width or T contract")
        slots = torch.arange(WIDTH, device=input_ids.device)
        values = (input_ids - DIGIT_BASE).clamp(0, 9)

        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            return (values[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, n_present = field(1)
        x, x_present = field(2)
        n_state = self.transition.encode_n(n, n_present)
        probability = F.one_hot(x, 10).to(self.transition.digit_embedding.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        present = x_present
        runs = int(steps.max().item())
        state = None
        for macrostep in range(runs):
            proposed, state = self.transition(n_state, n_present, probability, present)
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
                        "macrosteps": runs, "state": state}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=6e-4, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 32.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None)
