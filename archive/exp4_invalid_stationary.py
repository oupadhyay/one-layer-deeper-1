"""Experiment 4: Stationary-State Redesign (s_{t+1} = F_θ(s_t, c)).
Directly implements Oracle Directive 4 & Ranked Experiment 4:
- Truly stationary recurrence transition s_{t+1} = F_θ(s_t, c).
- Context matrix c contains immutable prompt representations (N and operator conditions).
- Initial state s_0 initialized once from X.
- NO step embeddings e(t), NO total T passed into transition, NO repeated re-injection of X.
- Internal LSB place features and low weight decay (0.05 / 0.0) retained.
- Rule 6 compliant (standard random initialization).
"""

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

# --- Hyperparameters ---
D_MODEL = 128
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
FFN_DIM = 256
MAX_ITERATIONS = 64
MAX_FIELD_LEN = 32

N_TOKEN_ID = 1
X_TOKEN_ID = 2
T_TOKEN_ID = 4
ANS_TOKEN_ID = 5
DIGIT_TOKEN_OFFSET = 7
DIGIT_TOKEN_COUNT = 10


class WSD_Scheduler:
    """Budget-calibrated Warmup-Stable-Decay Scheduler for 800 steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 50,
        stable_steps: int = 450,
        decay_steps: int = 300,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

        first_scale = 1.0 / max(1, self.warmup_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * first_scale

    def _lr_scale(self, step: int) -> float:
        w = self.warmup_steps
        st = self.stable_steps
        d = self.decay_steps
        if step <= w:
            return float(step) / max(1, w)
        if step <= (w + st):
            return 1.0
        if step <= (w + st + d):
            progress = float(step - w - st) / max(1, d)
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return self.min_lr_ratio

    def step(self) -> None:
        self.step_count += 1
        next_scale = self._lr_scale(self.step_count + 1)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * next_scale


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
        q = q.view(B, L, self.num_heads, HEAD_DIM).transpose(1, 2)
        k = k.view(B, L, self.num_heads, HEAD_DIM).transpose(1, 2)
        v = v.view(B, L, self.num_heads, HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

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


class StationaryRecurrentBlock(nn.Module):
    """Stationary recurrence block transforming state s by attending to immutable context c."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = QKNormAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, state: Tensor, mask: Tensor | None = None) -> Tensor:
        # Stationary state transition s_{t+1} = F_θ(s_t, c)
        x = state + self.attn(self.attn_norm(state), mask)
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)

        self.field_role_embedding = nn.Embedding(5, D_MODEL)
        self.lsd_place_embedding = nn.Embedding(MAX_FIELD_LEN, D_MODEL)

        # Standard random Gaussian initialization (Rule 6 compliant)
        for emb in [
            self.token_embedding,
            self.position_embedding,
            self.field_role_embedding,
            self.lsd_place_embedding,
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        self.prefix_block = StationaryRecurrentBlock(D_MODEL, NUM_HEADS, FFN_DIM)
        # Stationary tied recurrent transition (no step embeddings, no total T, no repeated X)
        self.stationary_block = StationaryRecurrentBlock(D_MODEL, NUM_HEADS, FFN_DIM)
        self.suffix_block = StationaryRecurrentBlock(D_MODEL, NUM_HEADS, FFN_DIM)

        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)

    def _compute_internal_lsb_features(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        B, L = input_ids.shape
        roles = torch.zeros(B, L, device=input_ids.device, dtype=torch.long)
        places = torch.zeros(B, L, device=input_ids.device, dtype=torch.long)

        for b in range(B):
            seq = input_ids[b].tolist()
            current_role = 0
            field_indices = []

            for i, tok in enumerate(seq):
                if tok == N_TOKEN_ID:
                    current_role = 1
                    field_indices = []
                elif tok == X_TOKEN_ID:
                    current_role = 2
                    field_indices = []
                elif tok == T_TOKEN_ID:
                    current_role = 3
                    field_indices = []
                elif tok == ANS_TOKEN_ID:
                    current_role = 4
                    field_indices = []
                elif DIGIT_TOKEN_OFFSET <= tok < DIGIT_TOKEN_OFFSET + DIGIT_TOKEN_COUNT:
                    roles[b, i] = current_role
                    field_indices.append(i)
                    for pos, idx in enumerate(field_indices):
                        lsd_place = (len(field_indices) - 1 - pos)
                        places[b, idx] = min(lsd_place, MAX_FIELD_LEN - 1)
                else:
                    current_role = 0
                    field_indices = []

        return roles, places

    @staticmethod
    def parse_time_steps(
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        B, L = input_ids.shape
        if attention_mask is not None and attention_mask.shape == (B, L):
            valid_tokens = attention_mask.bool()
        else:
            valid_tokens = torch.ones_like(input_ids, dtype=torch.bool)

        parsed = torch.zeros(B, device=input_ids.device, dtype=torch.long)
        reading_t = torch.zeros(B, device=input_ids.device, dtype=torch.bool)
        found_digit = torch.zeros(B, device=input_ids.device, dtype=torch.bool)
        for column in range(L):
            token = input_ids[:, column]
            valid = valid_tokens[:, column]
            starts_t = (token == T_TOKEN_ID) & valid
            is_digit_token = (
                valid
                & (token >= DIGIT_TOKEN_OFFSET)
                & (token < DIGIT_TOKEN_OFFSET + DIGIT_TOKEN_COUNT)
            )
            is_digit = reading_t & is_digit_token
            digit = token - DIGIT_TOKEN_OFFSET
            parsed = torch.where(is_digit, parsed * 10 + digit, parsed)
            found_digit = found_digit | is_digit
            reading_t = starts_t | (reading_t & is_digit_token)

        fallback = torch.full_like(parsed, 1)
        return torch.where(found_digit, parsed, fallback).clamp_(1, MAX_ITERATIONS)

    def run_stationary_recurrence_k2(
        self,
        initial_state: Tensor,
        time_steps: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        state = initial_state
        batch_t = int(time_steps.max().item())
        # Truly stationary loop: s_{t+1} = F_θ(s_t, c) with K=2 sub-steps
        for step_i in range(batch_t):
            m1 = self.stationary_block(state, attention_mask)
            updated = self.stationary_block(m1, attention_mask)

            active = (time_steps > step_i)[:, None, None]
            state = torch.where(active, updated, state)
        return state

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        roles, places = self._compute_internal_lsb_features(input_ids)
        x = x + self.field_role_embedding(roles) + self.lsd_place_embedding(places)

        # Context c and initial state s_0 computed by prefix block
        c = self.prefix_block(x, attention_mask)

        # Parse T for outer loop control ONLY
        time_steps = self.parse_time_steps(input_ids, attention_mask)
        state = self.run_stationary_recurrence_k2(c, time_steps, attention_mask)

        x = self.suffix_block(state, attention_mask)
        final_logits = self.head(self.final_norm(x))
        return final_logits, None


def custom_training_loss(logits: Tensor, targets: Tensor, auxiliary: object) -> Tensor:
    return F.cross_entropy(logits, targets, label_smoothing=0.00)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embedding" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 0.05},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_cuda = spec.device_type == "cuda"
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=2.0e-3,
        betas=(0.9, 0.98),
        capturable=use_cuda,
    )
    scheduler = WSD_Scheduler(
        optimizer=optimizer,
        warmup_steps=50,
        stable_steps=450,
        decay_steps=300,
        min_lr_ratio=0.01,
    )
    return OptimizerBundle(optimizer=optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=custom_training_loss,
    batch_size=64,
    eval_batch_size=256,
    max_steps=800,
)
