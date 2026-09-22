"""One-block spatial-gating categorical register with a learned highway gate."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
WIDTH, D_MODEL, HIDDEN, MAX_T = 4, 112, 224, 64
STATE_ELEMENTS, NEG = 92_594, -10_000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class SpatialGatingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D_MODEL)
        self.expand = nn.Linear(D_MODEL, 2 * HIDDEN, bias=True)
        self.gate_norm = nn.LayerNorm(HIDDEN)
        self.spatial = nn.Linear(2 * WIDTH, 2 * WIDTH, bias=True)
        self.project = nn.Linear(HIDDEN, D_MODEL, bias=True)
        self.highway = nn.Linear(D_MODEL, D_MODEL, bias=True)
        nn.init.uniform_(self.spatial.weight, -1.25e-4, 1.25e-4)
        nn.init.ones_(self.spatial.bias)

    def forward(self, register: Tensor) -> Tensor:
        normalized = self.norm(register)
        u, v = F.gelu(self.expand(normalized)).chunk(2, dim=-1)
        gate = self.spatial(self.gate_norm(v).transpose(1, 2)).transpose(1, 2)
        update = self.project(u * gate)
        return register + torch.sigmoid(self.highway(normalized)) * update


class SGURegister(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, D_MODEL)
        self.role_embedding = nn.Embedding(2, D_MODEL)
        self.place_embedding = nn.Embedding(WIDTH, D_MODEL)
        self.presence_embedding = nn.Embedding(2, D_MODEL)
        self.block = SpatialGatingBlock()
        self.readout_norm = nn.LayerNorm(D_MODEL)
        self.readout = nn.Linear(D_MODEL, 10, bias=True)

    def make_register(self, n: Tensor, n_present: Tensor, x: Tensor, x_present: Tensor) -> Tensor:
        dtype = self.digit_embedding.weight.dtype
        n_content = self.digit_embedding(n) * n_present[..., None].to(dtype)
        x_content = (x.to(dtype) @ self.digit_embedding.weight) * x_present[..., None].to(dtype)
        place = self.place_embedding.weight[None]
        nt = n_content + self.role_embedding.weight[0] + place + self.presence_embedding(n_present.long())
        xt = x_content + self.role_embedding.weight[1] + place + self.presence_embedding(x_present.long())
        return torch.stack((nt, xt), dim=2).reshape(n.shape[0], 2 * WIDTH, D_MODEL)

    def forward(self, n: Tensor, n_present: Tensor, x: Tensor, x_present: Tensor) -> tuple[Tensor, Tensor]:
        register = self.block(self.make_register(n, n_present, x, x_present))
        return self.readout(self.readout_norm(register[:, 1::2])), register


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(); self.config = Config(spec.vocab_size, spec.max_seq_len); self.transition = SGURegister()

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
        if length > self.config.max_seq_len: raise ValueError("input sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, steps = self.parse(input_ids, mask)
        n_width, x_width = role.eq(1).sum(1), role.eq(2).sum(1)
        invalid = (n_width < 1) | (n_width > WIDTH) | (x_width < 1) | (x_width > WIDTH)
        if bool(invalid.any()): raise ValueError("N and X widths must be between one and four")
        if bool((steps > MAX_T).any()): raise ValueError("T exceeds 64")
        slots = torch.arange(WIDTH, device=input_ids.device)
        values = (input_ids - DIGIT_BASE).clamp(0, 9)

        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            return (values[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, n_present = field(1); x, x_present = field(2)
        probability = F.one_hot(x, 10).to(self.transition.digit_embedding.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log(); present = x_present
        runs = 3 if self.training else int(steps.max().item()); register = None
        for macrostep in range(runs):
            proposed, register = self.transition(n, n_present, probability, present)
            active = (steps > macrostep)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, proposed.softmax(-1), probability)
            present = torch.where(active[:, :, 0], n_present, present)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active_place = slots[None] < n_width[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active_place[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint); occupied = placement.sum(1).bool()
        digit_logits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), NEG)
        logits = torch.where(occupied[..., None], digit_logits, logits)
        return logits, {"widths": n_width, "x_widths": x_width, "steps": steps, "macrosteps": runs, "register": register}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec); actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS: raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.0}], lr=6e-4, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 32.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None)
