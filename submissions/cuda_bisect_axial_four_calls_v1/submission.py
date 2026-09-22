"""Baseline Transformer plus four diagnostic axial-cell calls."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS = 64
D_MODEL = 128
NUM_HEADS = 4
CHANNELS = 32


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class AxialLocalCell(nn.Module):
    """One shared radius-one GRU-like update along a selected axis."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(CHANNELS)
        self.gates = nn.Linear(3 * CHANNELS, 2 * CHANNELS)
        self.candidate = nn.Linear(3 * CHANNELS, CHANNELS)

    def forward(self, workspace: Tensor, axis: str) -> Tensor:
        center = self.norm(workspace)
        if axis == "H":
            left = F.pad(center[:, :, :-1], (0, 0, 1, 0))
            right = F.pad(center[:, :, 1:], (0, 0, 0, 1))
        elif axis == "V":
            left = F.pad(center[:, :-1], (0, 0, 0, 0, 1, 0))
            right = F.pad(center[:, 1:], (0, 0, 0, 0, 0, 1))
        else:
            raise ValueError("axis must be H or V")
        local = torch.cat((center, left, right), -1)
        reset, update = self.gates(local).chunk(2, -1)
        reset, update = reset.sigmoid(), update.sigmoid()
        proposed = torch.tanh(self.candidate(
            torch.cat((center, reset * left, reset * right), -1)
        ))
        return (1 - update) * workspace + update * proposed


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # Declared after all baseline modules so seeded baseline initialization is
        # byte-for-byte unchanged.
        self.embedding = nn.Embedding(10, 16)
        self.initializer = nn.Sequential(
            nn.Linear(70, CHANNELS), nn.GELU(), nn.Linear(CHANNELS, CHANNELS)
        )
        self.cell = AxialLocalCell()

    @staticmethod
    def parse(input_ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        is_digit = (input_ids >= DIGIT) & mask
        markers = ((input_ids == N) | (input_ids == X) |
                   (input_ids == T) | (input_ids == ANS))
        role = torch.cumsum(markers.long(), 1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None] == role[:, None, :]
        places = (same & (index[None, None] > index[None, :, None]) &
                  is_digit[:, None]).sum(2)
        td = is_digit & role.eq(3)
        steps = (((input_ids - DIGIT) *
                  torch.pow(input_ids.new_tensor(10), places)) * td).sum(1)
        return role, places, steps.clamp(max=MAX_STEPS)

    def _prepare(self, ids: Tensor, attention_mask: Tensor | None):
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, places, steps = self.parse(ids, mask)
        slots = torch.arange(self.config.max_seq_len, device=ids.device)
        values = (ids - DIGIT).clamp(0, 9)

        def field(which: int) -> tuple[Tensor, Tensor]:
            assignment = role[:, :, None].eq(which) & places[:, :, None].eq(slots)
            digits = (assignment.to(values.dtype) * values[:, :, None]).sum(1)
            return digits.long(), assignment.any(1)

        nd, present = field(1)
        xd, _ = field(2)
        return nd, xd, present.sum(1), steps

    def _workspace(
        self, probabilities: Tensor, n_digits: Tensor, widths: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, width = probabilities.shape[:2]
        columns = width + 1
        state = probabilities @ self.embedding.weight
        n_state = self.embedding(n_digits)
        ei = state[:, :, None].expand(-1, -1, columns, -1)
        ni = n_state[:, :, None].expand_as(ei)
        ej = state[:, None, :].expand(-1, width, -1, -1)
        nj = n_state[:, None, :].expand_as(ej)
        zero = state.new_zeros((batch, width, 1, 16))
        ej, nj = torch.cat((ej, zero), 2), torch.cat((nj, zero), 2)
        rows = torch.arange(width, device=state.device)[None, :, None]
        cols = torch.arange(columns, device=state.device)[None, None, :]
        actual_width = widths[:, None, None]
        valid_row = rows < actual_width
        pair = cols < actual_width
        boundary = cols == actual_width
        boundary_cells = boundary.expand(-1, width, -1)[..., None]
        ej = torch.where(boundary_cells, torch.zeros_like(ej), ej)
        nj = torch.where(boundary_cells, torch.zeros_like(nj), nj)
        flags = torch.stack(
            (
                valid_row.expand(-1, -1, columns),
                pair.expand(-1, width, -1),
                boundary.expand(-1, width, -1),
                (rows == 0).expand(batch, -1, columns),
                (cols == 0).expand(batch, width, -1),
                (rows == actual_width - 1).expand(-1, -1, columns),
            ),
            -1,
        ).to(state.dtype)
        valid = valid_row & (pair | boundary)
        workspace = self.initializer(torch.cat((ei, ej, ni, nj, flags), -1))
        return workspace * valid[..., None], valid

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        nd, xd, widths, steps = self._prepare(input_ids, attention_mask)
        width = int(widths.max().item())
        nd, xd = nd[:, :width], xd[:, :width]
        probabilities = F.one_hot(xd, 10).to(self.embedding.weight.dtype)
        workspace, valid = self._workspace(probabilities, nd, widths)
        workspace = self.cell(workspace, "H") * valid[..., None]
        workspace = self.cell(workspace, "V") * valid[..., None]
        workspace = self.cell(workspace, "H") * valid[..., None]
        workspace = self.cell(workspace, "V") * valid[..., None]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.block(x, attention_mask)
        logits = self.head(self.final_norm(x))
        parsed = nd.sum() + xd.sum() + widths.sum() + steps.sum()
        parser_zero = logits.sum() * parsed.to(logits.dtype) * 0.0
        # This intentionally gives workspace parameters explicit, exactly-zero
        # diagnostic gradients without changing logits or baseline gradients.
        workspace_zero = workspace.sum().to(logits.dtype) * 0.0
        return logits + parser_zero + workspace_zero, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=64,
    eval_batch_size=128,
    max_steps=1,
)
