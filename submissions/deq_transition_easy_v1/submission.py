"""Tied equilibrium transition over categorical, LSD-relative digit places."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
WIDTH, D, HEADS, SCRATCH, FF = 16, 96, 4, 4, 192
NEG = -10000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class Norm(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (D,), self.weight)


class Attention(nn.Module):
    def __init__(self, cross: bool):
        super().__init__()
        self.cross = cross
        if cross:
            self.q = nn.Linear(D, D, bias=False); self.kv = nn.Linear(D, 2 * D, bias=False)
        else:
            self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x: Tensor, source: Tensor, key_mask: Tensor) -> Tensor:
        b, nq, _ = x.shape; nk = source.shape[1]
        if self.cross:
            q = self.q(x); k, v = self.kv(source).chunk(2, -1)
        else:
            q, k, v = self.qkv(x).chunk(3, -1)
        def split(y: Tensor, n: int) -> Tensor:
            return y.reshape(b, n, HEADS, D // HEADS).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            split(q, nq), split(k, nk), split(v, nk),
            attn_mask=key_mask[:, None, None, :], dropout_p=0.0)
        return self.out(y.transpose(1, 2).contiguous().reshape(b, nq, D))


class EquilibriumCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2, self.n3, self.n4 = Norm(), Norm(), Norm(), Norm()
        self.self_attention = Attention(False); self.cross_attention = Attention(True)
        self.up = nn.Linear(D, 2 * FF, bias=False); self.down = nn.Linear(FF, D, bias=False)
        self.projection = nn.Linear(4 * D, D)

    def forward(self, z: Tensor, context: Tensor, state_mask: Tensor,
                context_mask: Tensor, identity: Tensor) -> Tensor:
        base = self.n1(z + identity)
        sa = self.self_attention(base, base, state_mask)
        ca = self.cross_attention(self.n2(z + sa + identity), context, context_mask)
        a, b = self.up(self.n3(z + sa + ca + identity)).chunk(2, -1)
        ff = self.down(F.silu(a) * b)
        proposal = torch.tanh(self.projection(torch.cat((self.n4(z + identity), sa, ca, ff), -1)))
        stepped = .5 * z + .5 * proposal
        return torch.where(state_mask[..., None], stepped, z)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.max_seq_len > 64: raise ValueError("max_seq_len exceeds 64")
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.digit_embedding = nn.Embedding(10, D)
        self.context_role = nn.Parameter(torch.empty(2, D))
        self.state_role = nn.Parameter(torch.empty(2, D))
        self.cell = EquilibriumCell()
        self.digit_head = nn.Linear(D, 10)
        position = torch.arange(WIDTH + SCRATCH, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        place = torch.zeros(WIDTH + SCRATCH, D)
        place[:, 0::2], place[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("place", place)
        nn.init.normal_(self.context_role, std=.02); nn.init.normal_(self.state_role, std=.02)

    @staticmethod
    def parse(ids: Tensor, mask: Tensor):
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        marker = ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() + 3 * ids.eq(T_MARK).long()
        role = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(-1)
        widths = torch.maximum(role.eq(1).sum(1), role.eq(2).sum(1))
        if bool((widths > WIDTH).any()): raise ValueError("digit width exceeds 16")
        return role, place, widths

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        batch, length = input_ids.shape
        if length > self.config.max_seq_len: raise ValueError("input sequence exceeds configured cap")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, widths = self.parse(input_ids, mask)
        slots = torch.arange(WIDTH, device=input_ids.device)
        def field(which: int):
            choose = role.eq(which)[:, :, None] & place[:, :, None].eq(slots[None, None, :])
            present = choose.any(1)
            value = (((input_ids - DIGIT_BASE).clamp(0, 9))[:, :, None] * choose).sum(1)
            return value.long(), present
        n_digit, n_present = field(1); x_digit, x_present = field(2)
        active_place = slots[None] < widths[:, None]
        q = F.one_hot(x_digit, 10).to(self.digit_embedding.weight.dtype)
        q = torch.where(x_present[..., None], q, F.one_hot(torch.zeros_like(x_digit), 10).to(q.dtype))
        # T is interpreted only as a generic decimal loop count.
        t_digit = ((input_ids - DIGIT_BASE).clamp(0, 9) * role.eq(3)).long()
        t_place = place * role.eq(3)
        counts = (t_digit * torch.pow(input_ids.new_tensor(10), t_place) * role.eq(3)).sum(1)
        state_mask = torch.cat((active_place, torch.ones(batch, SCRATCH, dtype=torch.bool, device=input_ids.device)), 1)
        identity = self.place.to(q.dtype)[None] + torch.cat((
            self.state_role[0].expand(WIDTH, -1), self.state_role[1].expand(SCRATCH, -1)), 0)[None]
        current_logits = q.clamp_min(1e-8).log()
        residual_sum = q.new_zeros(()); residual_weight = q.new_zeros(())
        max_steps = int(counts.max().item())
        for step in range(max_steps):
            residue = q @ self.digit_embedding.weight
            n_context = self.digit_embedding(n_digit) + self.context_role[0] + self.place[:WIDTH]
            x_context = residue + self.context_role[1] + self.place[:WIDTH]
            context = torch.cat((n_context, x_context), 1)
            context_mask = torch.cat((n_present, active_place), 1)
            z = identity.expand(batch, -1, -1).clone()
            with torch.no_grad():
                for _ in range(6): z = self.cell(z, context, state_mask, context_mask, identity)
            z = z.detach()
            z7 = self.cell(z, context, state_mask, context_mask, identity)
            z8 = self.cell(z7, context, state_mask, context_mask, identity)
            proposed_logits = .5 * (self.digit_head(z7[:, :WIDTH]) + self.digit_head(z8[:, :WIDTH]))
            proposed = proposed_logits.softmax(-1)
            applying = counts.gt(step)
            q = torch.where(applying[:, None, None], proposed, q)
            current_logits = torch.where(applying[:, None, None], proposed_logits, current_logits)
            residual = (z8 - z7).square().mean(-1)
            weights = applying[:, None] & state_mask
            residual_sum = residual_sum + (residual * weights).sum()
            residual_weight = residual_weight + weights.sum()
        equilibrium_residual = residual_sum / residual_weight.clamp_min(1)
        full = q.new_full((batch, length, self.config.vocab_size), NEG)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(q.dtype) * active_place[..., None]
        placed = torch.einsum("bwl,bwd->bld", placement, current_logits)
        occupied = placement.sum(1).bool()
        digit_logits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        full = torch.where(occupied[..., None], digit_logits, full)
        return full, {"equilibrium_residual": equilibrium_residual, "macrosteps": max_steps}


def token_training_loss(logits: Tensor, labels: Tensor, auxiliary: object) -> Tensor:
    ce = F.cross_entropy(logits, labels)
    return ce + .02 * auxiliary["equilibrium_residual"]


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec); assert_model_state(model, spec); return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        excluded = (parameter.ndim != 2 or "embedding" in name or "norm" in name or
                    "role" in name or name.endswith("bias") or ".n" in name)
        (no_decay if excluded else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=6e-4, betas=(.9, .95), eps=1e-8,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 32.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, token_training_loss,
                        batch_size=512, eval_batch_size=512, max_steps=None)
