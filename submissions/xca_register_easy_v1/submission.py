"""One-block cross-covariance categorical register for the Easy tier."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
WIDTH, D_MODEL, HEADS, HEAD_DIM, FFN_DIM, MAX_T = 4, 96, 6, 16, 192, 64
STATE_ELEMENTS = 95_142
NEG = -10_000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class CrossCovarianceTick(nn.Module):
    """K-transpose-Q channel mixing, normalized across the eight tokens."""
    def __init__(self) -> None:
        super().__init__()
        self.xca_norm = nn.RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.temperature_parameter = nn.Parameter(torch.zeros(HEADS))
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ffn_norm = nn.RMSNorm(D_MODEL)
        self.ffn_up = nn.Linear(D_MODEL, 2 * FFN_DIM, bias=False)
        self.ffn_down = nn.Linear(FFN_DIM, D_MODEL, bias=False)

    @property
    def temperature(self) -> Tensor:
        return F.softplus(self.temperature_parameter) + 1e-6

    def channel_operator(self, q: Tensor, k: Tensor) -> Tensor:
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        covariance = k.transpose(-2, -1) @ q
        scaled = covariance * self.temperature[None, :, None, None]
        return scaled.softmax(dim=-2)

    def forward(self, register: Tensor) -> Tensor:
        batch = register.shape[0]
        q, k, v = self.qkv(self.xca_norm(register)).chunk(3, dim=-1)
        q = q.view(batch, 8, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch, 8, HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch, 8, HEADS, HEAD_DIM).transpose(1, 2)
        operator = self.channel_operator(q, k)
        mixed = (v @ operator).transpose(1, 2).contiguous().view(batch, 8, D_MODEL)
        register = register + self.out(mixed)
        gate, value = self.ffn_up(self.ffn_norm(register)).chunk(2, dim=-1)
        return register + self.ffn_down(F.silu(gate) * value)


class XCARegister(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, D_MODEL)
        self.role_embedding = nn.Embedding(2, D_MODEL)
        self.place_embedding = nn.Embedding(WIDTH, D_MODEL)
        self.presence_embedding = nn.Embedding(2, D_MODEL)
        self.tick = CrossCovarianceTick()
        self.readout_norm = nn.RMSNorm(D_MODEL)
        self.readout = nn.Linear(D_MODEL, 10, bias=False)

    def make_register(self, n: Tensor, n_present: Tensor, x: Tensor,
                      x_present: Tensor) -> Tensor:
        dtype = self.digit_embedding.weight.dtype
        n_content = self.digit_embedding(n) * n_present[..., None].to(dtype)
        x_content = (x.to(dtype) @ self.digit_embedding.weight) * x_present[..., None].to(dtype)
        place = self.place_embedding.weight[None]
        nt = n_content + self.role_embedding.weight[0] + place + self.presence_embedding(n_present.long())
        xt = x_content + self.role_embedding.weight[1] + place + self.presence_embedding(x_present.long())
        return torch.stack((nt, xt), dim=2).reshape(n.shape[0], 2 * WIDTH, D_MODEL)

    def forward(self, n: Tensor, n_present: Tensor, x: Tensor,
                x_present: Tensor) -> tuple[Tensor, Tensor]:
        register = self.tick(self.make_register(n, n_present, x, x_present))
        return self.readout(self.readout_norm(register[:, 1::2])), register


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = XCARegister()

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
        powers = torch.pow(ids.new_tensor(10), place)
        steps = (values * powers * role.eq(3)).sum(dim=1).long()
        return role, place, steps

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, steps = self.parse(input_ids, mask)
        n_width, x_width = role.eq(1).sum(1), role.eq(2).sum(1)
        if bool(((n_width < 1) | (n_width > WIDTH) | (x_width < 1) | (x_width > WIDTH)).any()):
            raise ValueError("N and X widths must be between one and four")
        if bool((steps > MAX_T).any()):
            raise ValueError("T exceeds 64")
        slots = torch.arange(WIDTH, device=input_ids.device)
        values = (input_ids - DIGIT_BASE).clamp(0, 9)

        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            return (values[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, n_present = field(1)
        x, x_present = field(2)
        probability = F.one_hot(x, 10).to(self.transition.digit_embedding.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        present = x_present
        runs = 3 if self.training else int(steps.max().item())
        register = None
        for macrostep in range(runs):
            proposed, register = self.transition(n, n_present, probability, present)
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
        full = endpoint.new_full((batch, length, self.config.vocab_size), NEG)
        full = torch.where(occupied[..., None], digit_logits, full)
        return full, {"widths": n_width, "x_widths": x_width, "steps": steps,
                      "macrosteps": runs, "register": register}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        group = decay if parameter.ndim == 2 and "embedding" not in name else no_decay
        group.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=6e-4, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 32.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None)
