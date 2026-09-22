"""Learned local Neural-GPU arithmetic tape for the controlled E5 T1 gate.
The active candidate uses:
1. Token Constants: N=2, X=3, T=4, ANS=5, DIGIT_OFFSET=7.
2. 100% PyTorch Vectorized GPU LSB Feature Calculation:
   - Field Roles (1=N, 2=X, 3=T, 4=ANS, 0=non-digit).
   - Field-Local LSD Places (0=units, 1=tens, 2=hundreds...).
3. A 16-cell LSD-first residue/scratch/output/modulus tape.
4. One radius-2 ConvGRU cell tied across 32 microticks and all macrosteps.
5. Immutable modulus cells, reset scratch/output cells, and differentiable residue feedback.
6. One decimal decoder shared by intermediate feedback and endpoint readout.
7. Parameter Grouping & Low Decay: Matrix decay = 0.05, 1D/Norm/Embedding = 0.0.
8. Rule 6 Compliant: Standard random Gaussian initialization.
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
D_MODEL = 256
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS
FFN_DIM = 1024
USE_NEURAL_GPU = True
CELL_DIM = 128
REGISTER_PLACES = 4
SCRATCH_CELLS = 4
OUTPUT_CELLS = 4
LOCAL_RADIUS = 2
LOCAL_MICROTICKS = 32
MAX_ITERATIONS = 64
MAX_FIELD_LEN = 32
STATE_TAIL_PLACES = 8
USE_PAIRWISE_GRID = False
USE_SOFT_DIGIT_REGISTER = False
PAIRWISE_SCRATCH_SLOTS = 8

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
N_TOKEN_ID = 2
X_TOKEN_ID = 3
T_TOKEN_ID = 4
ANS_TOKEN_ID = 5
EOS_TOKEN_ID = 6
DIGIT_TOKEN_OFFSET = 7
DIGIT_TOKEN_COUNT = 10


class WSD_Scheduler:
    """Warmup-Stable-Decay scheduler for the local curriculum probe."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 300,
        stable_steps: int = 7700,
        decay_steps: int = 1500,
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


class CrossAttention(nn.Module):
    """QK-Norm Cross Attention: Queries from state s, Keys/Values from context c."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, state: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        B, L_s, _ = state.shape
        _, L_c, _ = context.shape

        q = self.q(state).view(B, L_s, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(B, L_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_c, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = None
        if context_mask is not None:
            if context_mask.shape == (B, L_c):
                attn_mask = context_mask[:, None, None, :].to(dtype=torch.bool)
            elif context_mask.shape == (B, L_s, L_c):
                attn_mask = context_mask[:, None, :, :].to(dtype=torch.bool)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_s, -1)
        return self.out(attn_out)


class QKNormSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.shape == (B, L):
                mask_bool = attention_mask[:, None, None, :].to(dtype=torch.bool)
            elif attention_mask.shape == (B, L, L):
                mask_bool = attention_mask[:, None, :, :].to(dtype=torch.bool)
            else:
                mask_bool = attention_mask.to(dtype=torch.bool)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_bool)
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
    """Stationary recurrence block transforming state s by cross-attending to immutable context c."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn_norm = RMSNorm(dim)
        self.self_attn = QKNormSelfAttention(dim, num_heads)
        self.cross_attn_norm = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(
        self,
        state: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
        state_mask: Tensor | None = None,
    ) -> Tensor:
        s = state + self.self_attn(self.self_attn_norm(state), state_mask)
        s = s + self.cross_attn(self.cross_attn_norm(s), context, context_mask)
        return s + self.ffn(self.ffn_norm(s))


class LocalConvGRUCell(nn.Module):
    """One shared neighbor-only cellular update."""

    def __init__(self, channels: int, radius: int) -> None:
        super().__init__()
        kernel_size = 2 * radius + 1
        self.radius = radius
        self.norm = RMSNorm(channels)
        self.gates = nn.Conv1d(channels, 2 * channels, 1)
        self.candidate = nn.Conv1d(
            channels, channels, kernel_size, padding=radius
        )

    def forward(self, state: Tensor) -> Tensor:
        normalized = self.norm(state).transpose(1, 2)
        update, reset = self.gates(normalized).chunk(2, dim=1)
        update = update.sigmoid()
        reset = reset.sigmoid()
        candidate = self.candidate(reset * normalized).tanh()
        next_state = (1.0 - update) * state.transpose(1, 2) + update * candidate
        return next_state.transpose(1, 2)


class SoftDigitMacroTransition(nn.Module):
    """Compose through a differentiable decimal-digit bottleneck."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.block = StationaryRecurrentBlock(dim, num_heads, ffn_dim)
        self.digit_norm = RMSNorm(dim)
        self.digit_decoder = nn.Linear(dim, DIGIT_TOKEN_COUNT, bias=False)

    def forward(
        self,
        residue: Tensor,
        modulus: Tensor,
        numeric_mask: Tensor,
        digit_basis: Tensor,
        place_identity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        state = residue
        for _ in range(6):
            state = self.block(state, modulus, numeric_mask, numeric_mask)

        digit_logits = self.digit_decoder(self.digit_norm(state))
        digit_probabilities = digit_logits.softmax(dim=-1)
        requantized = digit_probabilities @ digit_basis
        requantized = requantized + place_identity[None, :, :]
        requantized = torch.where(
            numeric_mask[:, :, None], requantized, torch.zeros_like(requantized)
        )
        return requantized, digit_logits


class PairwiseMacroTransition(nn.Module):
    """Learn one transition through a generic pairwise interaction workspace."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        self.row_embedding = nn.Embedding(REGISTER_PLACES, dim)
        self.column_embedding = nn.Embedding(REGISTER_PLACES, dim)
        self.scratch = nn.Parameter(torch.empty(PAIRWISE_SCRATCH_SLOTS, dim))
        self.workspace_block = StationaryRecurrentBlock(dim, num_heads, ffn_dim)
        self.output_place_embedding = nn.Embedding(REGISTER_PLACES, dim)
        self.output_norm = RMSNorm(dim)
        self.output_attention = CrossAttention(dim, num_heads)
        self.output_ffn_norm = RMSNorm(dim)
        self.output_ffn = SwiGLU(dim, ffn_dim)

        for embedding in (
            self.row_embedding,
            self.column_embedding,
            self.output_place_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.scratch, mean=0.0, std=0.02)

    def forward(
        self,
        residue: Tensor,
        modulus: Tensor,
        numeric_mask: Tensor,
    ) -> Tensor:
        B, P, D = residue.shape
        left = self.left(residue)[:, :, None, :]
        right = self.right(residue)[:, None, :, :]
        row = self.row_embedding.weight[None, :, None, :]
        column = self.column_embedding.weight[None, None, :, :]
        interactions = (left + right + row + column).reshape(B, P * P, D)

        pair_mask = (
            numeric_mask[:, :, None] & numeric_mask[:, None, :]
        ).reshape(B, P * P)
        scratch = self.scratch[None, :, :].expand(B, -1, -1)
        workspace = torch.cat((interactions, scratch), dim=1)
        scratch_mask = torch.ones(
            B,
            PAIRWISE_SCRATCH_SLOTS,
            device=residue.device,
            dtype=torch.bool,
        )
        workspace_mask = torch.cat((pair_mask, scratch_mask), dim=1)

        # Two tied workspace-refinement microsteps form one macro-transition.
        workspace = self.workspace_block(
            workspace, modulus, numeric_mask, workspace_mask
        )
        workspace = self.workspace_block(
            workspace, modulus, numeric_mask, workspace_mask
        )

        queries = residue + self.output_place_embedding.weight[None, :, :]
        decoded = queries + self.output_attention(
            self.output_norm(queries), workspace, workspace_mask
        )
        decoded = decoded + self.output_ffn(self.output_ffn_norm(decoded))
        return torch.where(numeric_mask[:, :, None], decoded, torch.zeros_like(decoded))


class PrefixEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = QKNormSelfAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        if USE_NEURAL_GPU:
            self.digit_embedding = nn.Embedding(DIGIT_TOKEN_COUNT, CELL_DIM)
            self.tape_place_embedding = nn.Embedding(REGISTER_PLACES, CELL_DIM)
            self.tape_region_embedding = nn.Embedding(4, CELL_DIM)
            self.scratch_constants = nn.Parameter(
                torch.empty(SCRATCH_CELLS, CELL_DIM)
            )
            self.output_constants = nn.Parameter(
                torch.empty(OUTPUT_CELLS, CELL_DIM)
            )
            self.local_cell = LocalConvGRUCell(CELL_DIM, LOCAL_RADIUS)
            self.digit_decoder = nn.Linear(
                CELL_DIM, DIGIT_TOKEN_COUNT, bias=False
            )
            for embedding in (
                self.digit_embedding,
                self.tape_place_embedding,
                self.tape_region_embedding,
            ):
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.scratch_constants, mean=0.0, std=0.02)
            nn.init.normal_(self.output_constants, mean=0.0, std=0.02)
            return

        self.register_buffer(
            "curriculum_training_step", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        if USE_PAIRWISE_GRID or USE_SOFT_DIGIT_REGISTER:
            self.register_place_embedding = nn.Embedding(REGISTER_PLACES, D_MODEL)
        else:
            self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
            self.field_role_embedding = nn.Embedding(5, D_MODEL)
            self.lsd_place_embedding = nn.Embedding(MAX_FIELD_LEN, D_MODEL)
            self.state_tail_embedding = nn.Embedding(STATE_TAIL_PLACES, D_MODEL)

        # Standard random Gaussian initialization (Rule 6 compliant)
        embeddings = [self.token_embedding]
        if USE_PAIRWISE_GRID or USE_SOFT_DIGIT_REGISTER:
            embeddings.append(self.register_place_embedding)
        else:
            embeddings.extend(
                [
                    self.position_embedding,
                    self.field_role_embedding,
                    self.lsd_place_embedding,
                    self.state_tail_embedding,
                ]
            )
        for emb in embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        if USE_SOFT_DIGIT_REGISTER:
            self.soft_digit_transition = SoftDigitMacroTransition(
                D_MODEL, NUM_HEADS, FFN_DIM
            )
        elif USE_PAIRWISE_GRID:
            self.pairwise_transition = PairwiseMacroTransition(
                D_MODEL, NUM_HEADS, FFN_DIM
            )
        else:
            self.prefix_encoder = PrefixEncoderBlock(D_MODEL, NUM_HEADS, FFN_DIM)
            self.stationary_transition = StationaryRecurrentBlock(
                D_MODEL, NUM_HEADS, FFN_DIM
            )
            self.suffix_block = PrefixEncoderBlock(D_MODEL, NUM_HEADS, FFN_DIM)

        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)

    @staticmethod
    def compute_vectorized_lsb_features(input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """100% PyTorch vectorized GPU tensor calculation of field roles and field-local LSD places.

        Roles: 0=non-digit, 1=N, 2=X, 3=T, 4=ANS
        LSD Places: 0=units, 1=tens, 2=hundreds...
        """
        B, L = input_ids.shape
        device = input_ids.device

        is_n = (input_ids == N_TOKEN_ID)
        is_x = (input_ids == X_TOKEN_ID)
        is_t = (input_ids == T_TOKEN_ID)
        is_ans = (input_ids == ANS_TOKEN_ID)

        marker_val = torch.zeros(B, L, device=device, dtype=torch.long)
        marker_val = torch.where(is_n, torch.tensor(1, device=device), marker_val)
        marker_val = torch.where(is_x, torch.tensor(2, device=device), marker_val)
        marker_val = torch.where(is_t, torch.tensor(3, device=device), marker_val)
        marker_val = torch.where(is_ans, torch.tensor(4, device=device), marker_val)

        active_role = torch.cummax(marker_val, dim=1)[0]
        is_digit = (input_ids >= DIGIT_TOKEN_OFFSET) & (input_ids < DIGIT_TOKEN_OFFSET + DIGIT_TOKEN_COUNT)
        roles = torch.where(is_digit, active_role, torch.zeros_like(active_role))

        rev_digit = torch.flip(is_digit, dims=[1])
        rev_non_digit = ~rev_digit

        cum_digits = torch.cumsum(rev_digit.long(), dim=1)
        prev_cum = torch.where(rev_non_digit, cum_digits, torch.tensor(0, device=device))
        group_start_cum = torch.cummax(prev_cum, dim=1)[0]

        rev_lsd_place = torch.where(rev_digit, cum_digits - group_start_cum - 1, torch.zeros_like(cum_digits))
        places = torch.flip(rev_lsd_place, dims=[1]).clamp(0, MAX_FIELD_LEN - 1)
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

    @staticmethod
    def curriculum_max_time_steps(training_step: int) -> int:
        if training_step <= 6000:
            return 1
        if training_step <= 8000:
            return 2
        return 3

    @staticmethod
    def apply_curriculum_gradient_gate(
        logits: Tensor,
        parsed_time_steps: Tensor,
        maximum_time_steps: int,
    ) -> Tensor:
        full_gradient = parsed_time_steps <= maximum_time_steps
        gradient_scale = torch.where(
            full_gradient,
            torch.ones_like(parsed_time_steps, dtype=logits.dtype),
            torch.full_like(parsed_time_steps, 0.01, dtype=logits.dtype),
        )
        detached = logits.detach()
        return detached + gradient_scale[:, None, None] * (logits - detached)

    @staticmethod
    def compute_answer_places(
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        B, L = input_ids.shape
        valid = (
            attention_mask.bool()
            if attention_mask is not None and attention_mask.shape == (B, L)
            else input_ids != PAD_TOKEN_ID
        )
        positions = torch.arange(L, device=input_ids.device)
        valid_lengths = valid.long().sum(dim=1)
        return (valid_lengths[:, None] - 1 - positions[None, :]).clamp_(
            0, REGISTER_PLACES - 1
        )

    def build_digit_registers(
        self,
        input_ids: Tensor,
        roles: Tensor,
        places: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, _ = input_ids.shape
        register_places = torch.arange(REGISTER_PLACES, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        zero = self.token_embedding.weight[DIGIT_TOKEN_OFFSET]
        place_embeddings = self.register_place_embedding.weight[None, :, :]

        def build(role: int) -> Tensor:
            assignment = (
                (roles == role)[:, :, None]
                & (places[:, :, None] == register_places[None, None, :])
            )
            gathered = torch.einsum(
                "blp,bld->bpd", assignment.to(token_embeddings.dtype), token_embeddings
            )
            digits = torch.where(
                assignment.any(dim=1)[:, :, None],
                gathered,
                zero[None, None, :],
            )
            return digits + place_embeddings

        n_lengths = (roles == 1).long().sum(dim=1)
        torch._assert(
            (n_lengths <= REGISTER_PLACES).all(),
            "modulus exceeds pairwise register capacity",
        )
        numeric_mask = register_places[None, :] < n_lengths[:, None]
        return build(2), build(1), numeric_mask

    def extract_lsd_digit_register(
        self,
        input_ids: Tensor,
        roles: Tensor,
        places: Tensor,
        role: int,
    ) -> Tensor:
        register_places = torch.arange(REGISTER_PLACES, device=input_ids.device)
        assignment = (
            (roles == role)[:, :, None]
            & (places[:, :, None] == register_places[None, None, :])
        )
        digit_values = (input_ids - DIGIT_TOKEN_OFFSET).clamp(0, 9)
        return torch.einsum(
            "blp,bl->bp", assignment.to(torch.long), digit_values
        )

    def build_local_tape(
        self,
        residue_content: Tensor,
        modulus_digits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = residue_content.shape[0]
        place = self.tape_place_embedding.weight
        residue = residue_content + place[None, :, :]
        modulus = self.digit_embedding(modulus_digits) + place[None, :, :]
        scratch = self.scratch_constants[None, :, :].expand(batch_size, -1, -1)
        output = self.output_constants[None, :, :].expand(batch_size, -1, -1)
        regions = self.tape_region_embedding.weight
        tape = torch.cat(
            (
                residue + regions[0],
                scratch + regions[2],
                output + regions[3],
                modulus + regions[1],
            ),
            dim=1,
        )
        return tape, tape[:, -REGISTER_PLACES:].clone()

    def run_local_macrostep(
        self,
        residue_content: Tensor,
        modulus_digits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tape, immutable_modulus = self.build_local_tape(
            residue_content, modulus_digits
        )
        for _ in range(LOCAL_MICROTICKS):
            tape = self.local_cell(tape)
            tape = torch.cat(
                (
                    tape[:, :-REGISTER_PLACES],
                    immutable_modulus,
                ),
                dim=1,
            )
        output_start = REGISTER_PLACES + SCRATCH_CELLS
        output_state = tape[:, output_start : output_start + OUTPUT_CELLS]
        digit_logits = self.digit_decoder(output_state)
        digit_probabilities = digit_logits.softmax(dim=-1)
        next_residue = digit_probabilities @ self.digit_embedding.weight
        return next_residue, digit_logits, tape

    def forward_neural_gpu(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, None]:
        roles, places = self.compute_vectorized_lsb_features(input_ids)
        parsed_time_steps = self.parse_time_steps(input_ids, attention_mask)
        residue_digits = self.extract_lsd_digit_register(
            input_ids, roles, places, role=2
        )
        modulus_digits = self.extract_lsd_digit_register(
            input_ids, roles, places, role=1
        )
        residue_content = self.digit_embedding(residue_digits)
        effective_time_steps = (
            torch.ones_like(parsed_time_steps) if self.training else parsed_time_steps
        )
        final_digit_logits = torch.zeros(
            input_ids.shape[0],
            REGISTER_PLACES,
            DIGIT_TOKEN_COUNT,
            device=input_ids.device,
            dtype=residue_content.dtype,
        )
        for step_index in range(int(effective_time_steps.max().item())):
            next_residue, digit_logits, _ = self.run_local_macrostep(
                residue_content, modulus_digits
            )
            active = effective_time_steps > step_index
            residue_content = torch.where(
                active[:, None, None], next_residue, residue_content
            )
            final_digit_logits = torch.where(
                active[:, None, None], digit_logits, final_digit_logits
            )

        answer_places = self.compute_answer_places(input_ids, attention_mask)
        selected_digit_logits = torch.gather(
            final_digit_logits,
            1,
            answer_places[:, :, None].expand(-1, -1, DIGIT_TOKEN_COUNT),
        )
        non_digit_logits = torch.full(
            (*selected_digit_logits.shape[:2], DIGIT_TOKEN_OFFSET),
            -20.0,
            device=input_ids.device,
            dtype=selected_digit_logits.dtype,
        )
        logits = torch.cat((non_digit_logits, selected_digit_logits), dim=-1)
        if self.training:
            logits = self.apply_curriculum_gradient_gate(
                logits, parsed_time_steps, maximum_time_steps=1
            )
        return logits, None

    def run_pairwise_recurrence(
        self,
        residue: Tensor,
        modulus: Tensor,
        numeric_mask: Tensor,
        time_steps: Tensor,
    ) -> Tensor:
        batch_t = int(time_steps.max().item())
        for step_i in range(batch_t):
            updated = self.pairwise_transition(residue, modulus, numeric_mask)
            residue = torch.where(
                (time_steps > step_i)[:, None, None], updated, residue
            )
        return residue

    def run_soft_digit_recurrence(
        self,
        residue: Tensor,
        modulus: Tensor,
        numeric_mask: Tensor,
        time_steps: Tensor,
    ) -> Tensor:
        digit_basis = self.token_embedding.weight[
            DIGIT_TOKEN_OFFSET : DIGIT_TOKEN_OFFSET + DIGIT_TOKEN_COUNT
        ]
        place_identity = self.register_place_embedding.weight
        batch_t = int(time_steps.max().item())
        for step_i in range(batch_t):
            updated, _ = self.soft_digit_transition(
                residue,
                modulus,
                numeric_mask,
                digit_basis,
                place_identity,
            )
            residue = torch.where(
                (time_steps > step_i)[:, None, None], updated, residue
            )
        return residue

    def run_stationary_cross_recurrence_k6(
        self,
        initial_state: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
        state_mask: Tensor,
        time_steps: Tensor,
    ) -> Tensor:
        state = initial_state
        batch_t = int(time_steps.max().item())
        for step_i in range(batch_t):
            updated = state
            for _ in range(6):
                updated = self.stationary_transition(
                    updated, context, context_mask, state_mask
                )

            active = (time_steps > step_i)[:, None, None]
            state = torch.where(active, updated, state)
        return state

    @staticmethod
    def compute_state_tail_places(
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        B, L = input_ids.shape
        valid = (
            attention_mask.bool()
            if attention_mask is not None and attention_mask.shape == (B, L)
            else input_ids != PAD_TOKEN_ID
        )
        positions = torch.arange(L, device=input_ids.device)
        valid_lengths = valid.long().sum(dim=1)
        tail_places = (valid_lengths[:, None] - 1 - positions[None, :]).clamp_(
            0, STATE_TAIL_PLACES - 1
        )
        return tail_places, valid

    def build_prompt_state(
        self,
        input_ids: Tensor,
        roles: Tensor,
        places: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build prompt features and symmetry-breaking mutable state."""
        L = input_ids.shape[1]
        positions = torch.arange(L, device=input_ids.device)
        prompt = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.field_role_embedding(roles)
            + self.lsd_place_embedding(places)
        )
        tail_places, state_mask = self.compute_state_tail_places(
            input_ids, attention_mask
        )
        state_identity = self.state_tail_embedding(tail_places)
        initial_state = torch.where(
            (roles == 2)[:, :, None], prompt + state_identity, state_identity
        )
        initial_state = torch.where(
            state_mask[:, :, None], initial_state, torch.zeros_like(initial_state)
        )
        return prompt, initial_state, state_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        if USE_NEURAL_GPU:
            return self.forward_neural_gpu(input_ids, attention_mask)

        B, L = input_ids.shape

        roles, places = self.compute_vectorized_lsb_features(input_ids)
        parsed_time_steps = self.parse_time_steps(input_ids, attention_mask)
        time_steps = parsed_time_steps
        curriculum_maximum = None
        if self.training:
            self.curriculum_training_step.add_(1)
            curriculum_maximum = self.curriculum_max_time_steps(
                int(self.curriculum_training_step.item())
            )
            time_steps = parsed_time_steps.clamp_max(curriculum_maximum)

        if USE_SOFT_DIGIT_REGISTER:
            residue, modulus, numeric_mask = self.build_digit_registers(
                input_ids, roles, places
            )
            residue = self.run_soft_digit_recurrence(
                residue, modulus, numeric_mask, time_steps
            )
            answer_places = self.compute_answer_places(input_ids, attention_mask)
            decoded = torch.gather(
                residue,
                1,
                answer_places[:, :, None].expand(-1, -1, D_MODEL),
            )
            return self.head(self.final_norm(decoded)), None

        if USE_PAIRWISE_GRID:
            residue, modulus, numeric_mask = self.build_digit_registers(
                input_ids, roles, places
            )
            residue = self.run_pairwise_recurrence(
                residue, modulus, numeric_mask, time_steps
            )
            answer_places = self.compute_answer_places(input_ids, attention_mask)
            decoded = torch.gather(
                residue,
                1,
                answer_places[:, :, None].expand(-1, -1, D_MODEL),
            )
            return self.head(self.final_norm(decoded)), None

        x, initial_state, state_mask = self.build_prompt_state(
            input_ids, roles, places, attention_mask
        )

        # Context c excludes T and X digits from context key representations
        # Context contains ONLY N digits and operator markers
        is_context_allowed = (roles == 1) | (input_ids == N_TOKEN_ID) | (input_ids == X_TOKEN_ID)
        context_mask = is_context_allowed
        if attention_mask is not None:
            context_mask = context_mask & attention_mask.bool()

        context = self.prefix_encoder(x, context_mask)

        # Time steps T parsed ONLY for outer loop count control
        state = self.run_stationary_cross_recurrence_k6(
            initial_state, context, context_mask, state_mask, time_steps
        )

        x = self.suffix_block(state, state_mask)
        final_logits = self.head(self.final_norm(x))
        if curriculum_maximum is not None:
            final_logits = self.apply_curriculum_gradient_gate(
                final_logits, parsed_time_steps, curriculum_maximum
            )
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
        warmup_steps=300,
        stable_steps=7700,
        decay_steps=1500,
        min_lr_ratio=0.01,
    )
    return OptimizerBundle(optimizer=optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=custom_training_loss,
    batch_size=64,
    eval_batch_size=256,
    max_steps=9500,
)
