"""Rev2 Baseline Submission (Untied Head, batch_size=128, FFN_DIM=512)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)

# --- Architectural Hyperparameters ---
D_MODEL = 256
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS
FFN_DIM = 512
UNROLL_STEPS = 2


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class QKNormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, L, self.num_heads, HEAD_DIM)).transpose(1, 2)
        k = self.k_norm(k.view(B, L, self.num_heads, HEAD_DIM)).transpose(1, 2)
        v = v.view(B, L, self.num_heads, HEAD_DIM).transpose(1, 2)

        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (B, L):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (B, L, L):
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            mask = mask.to(device=x.device, dtype=torch.bool)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        w1, w2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(w1) * w2)


class RQSTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = QKNormAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)

        self.prefix_block = RQSTBlock(D_MODEL, NUM_HEADS, FFN_DIM)
        self.recurrent_block = RQSTBlock(D_MODEL, NUM_HEADS, FFN_DIM)
        self.suffix_block = RQSTBlock(D_MODEL, NUM_HEADS, FFN_DIM)

        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        x = self.prefix_block(x, attention_mask)

        for _ in range(UNROLL_STEPS):
            x = self.recurrent_block(x, attention_mask)

        x = self.suffix_block(x, attention_mask)
        final_logits = self.head(self.final_norm(x))
        return final_logits, None


# --- Submission Contract Export ---
def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    use_cuda = spec.device_type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        capturable=use_cuda,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=1e-5)
    return OptimizerBundle(optimizer=optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=128,
    eval_batch_size=512,
)
