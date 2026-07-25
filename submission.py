"""Medium M5 baseline: selected wider random-depth architecture."""

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

# --- Architectural Hyperparameters (R6 settings preserved) ---
D_MODEL = 384
NUM_HEADS = 12
HEAD_DIM = D_MODEL // NUM_HEADS
FFN_DIM = 768
UNROLL_STEPS = 6


class EarlyDecayWSD:
    """
    Warmup-Stable-Decay scheduler scaled to roughly 12 M5 dataset passes:
    - Step 1..100: Linear warmup (0.0 -> 1.0)
    - Step 100..5100: Stable peak LR (1.0)
    - Step 5100..7600: Cosine decay to min LR (1.0 -> 0.01)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        stable_steps: int = 5000,
        decay_steps: int = 2500,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        s = self.step_count
        w = self.warmup_steps
        st = self.stable_steps
        d = self.decay_steps

        if s <= w:
            scale = float(s) / max(1, w)
        elif s <= (w + st):
            scale = 1.0
        elif s <= (w + st + d):
            progress = float(s - w - st) / max(1, d)
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            scale = self.min_lr_ratio

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale


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

        state = self.prefix_block(x, attention_mask)

        steps = (
            int(torch.randint(2, UNROLL_STEPS + 1, (), device=input_ids.device).item())
            if self.training
            else UNROLL_STEPS
        )
        for _ in range(steps):
            state = self.recurrent_block(state, attention_mask)

        x = self.suffix_block(state, attention_mask)
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
    scheduler = EarlyDecayWSD(optimizer, warmup_steps=100, stable_steps=5000, decay_steps=2500)
    return OptimizerBundle(optimizer=optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=128,
    eval_batch_size=512,
    max_steps=7600,
)
