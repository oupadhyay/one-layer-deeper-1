"""Generic universal recurrent Transformer candidate."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


D = 256
HEADS = 8
FFN = 768
SCRATCH = 32
REFINEMENTS = 16
MAX_SEQUENCE_LENGTH = 64
RESIDUAL_SCALE = 1.0 / math.sqrt(REFINEMENTS)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (D,), self.weight)


class Attention(nn.Module):
    def __init__(self, cross: bool = False) -> None:
        super().__init__()
        self.cross = cross
        if cross:
            self.q = nn.Linear(D, D, bias=False)
            self.kv = nn.Linear(D, 2 * D, bias=False)
        else:
            self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None,
                key_mask: Tensor | None = None) -> Tensor:
        source = x if context is None else context
        batch, query_length, _ = x.shape
        key_length = source.shape[1]
        if self.cross:
            q = self.q(x)
            k, v = self.kv(source).chunk(2, dim=-1)
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(value: Tensor, length: int) -> Tensor:
            return value.reshape(batch, length, HEADS, D // HEADS).transpose(1, 2)
        mask = None if key_mask is None else key_mask[:, None, None, :].bool()
        attended = F.scaled_dot_product_attention(
            split(q, query_length), split(k, key_length), split(v, key_length),
            attn_mask=mask, dropout_p=0.0,
        )
        return self.out(attended.transpose(1, 2).contiguous().reshape(batch, query_length, D))


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D, 2 * FFN, bias=False)
        self.down = nn.Linear(FFN, D, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class PromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm()
        self.attention = Attention()
        self.ffn_norm = RMSNorm()
        self.ffn = SwiGLU()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attention = self.attention(self.attention_norm(x), key_mask=mask)
        x = x + F.dropout(attention, p=0.1, training=self.training)
        ffn = self.ffn(self.ffn_norm(x))
        return x + F.dropout(ffn, p=0.1, training=self.training)


class RecurrentBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_norm = RMSNorm()
        self.self_attention = Attention()
        self.cross_norm = RMSNorm()
        self.cross_attention = Attention(cross=True)
        self.ffn_norm = RMSNorm()
        self.ffn = SwiGLU()

    def forward(self, state: Tensor, prompt: Tensor, prompt_mask: Tensor) -> Tensor:
        self_attention = self.self_attention(self.self_norm(state))
        state = state + RESIDUAL_SCALE * F.dropout(
            self_attention, p=0.1, training=self.training
        )
        cross_attention = self.cross_attention(self.cross_norm(state), prompt, prompt_mask)
        state = state + RESIDUAL_SCALE * F.dropout(
            cross_attention, p=0.1, training=self.training
        )
        ffn = self.ffn(self.ffn_norm(state))
        return state + RESIDUAL_SCALE * F.dropout(ffn, p=0.1, training=self.training)


def relative_sinusoid(mask: Tensor, length: int, dtype: torch.dtype) -> Tensor:
    valid_length = mask.long().sum(dim=1)
    index = torch.arange(length, device=mask.device)
    position = index[None, :] - valid_length[:, None]
    frequency = torch.exp(
        torch.arange(0, D, 2, device=mask.device, dtype=torch.float32)
        * (-math.log(10000.0) / D)
    )
    angle = position.to(torch.float32)[:, :, None] * frequency
    encoding = torch.empty(mask.shape[0], length, D, device=mask.device, dtype=torch.float32)
    encoding[:, :, 0::2] = angle.sin()
    encoding[:, :, 1::2] = angle.cos()
    return encoding.to(dtype=dtype)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        if spec.max_seq_len > MAX_SEQUENCE_LENGTH:
            raise ValueError("max_seq_len exceeds candidate sequence cap")
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D)
        self.prompt_type = nn.Parameter(torch.empty(D))
        self.output_query = nn.Parameter(torch.empty(D))
        self.output_type = nn.Parameter(torch.empty(D))
        self.scratch_tokens = nn.Parameter(torch.empty(SCRATCH, D))
        self.scratch_type = nn.Parameter(torch.empty(D))
        self.prompt_encoder = PromptEncoder()
        self.recurrent_block = RecurrentBlock()
        self.final_norm = RMSNorm()
        self.vocabulary_projection = nn.Linear(D, spec.vocab_size, bias=False)
        self.vocabulary_projection.weight = self.token_embedding.weight
        for parameter in (self.prompt_type, self.output_query, self.output_type,
                          self.scratch_tokens, self.scratch_type):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        batch, length = input_ids.shape
        if length > self.config.max_seq_len or length > MAX_SEQUENCE_LENGTH:
            raise ValueError("input sequence exceeds configured cap")
        mask = (torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None
                else attention_mask.to(device=input_ids.device, dtype=torch.bool))
        position = relative_sinusoid(mask, length, self.token_embedding.weight.dtype)
        prompt = self.token_embedding(input_ids) + self.prompt_type + position
        encoded = self.prompt_encoder(prompt, mask)
        outputs = self.output_query + self.output_type + position
        scratch = (self.scratch_tokens + self.scratch_type)[None].expand(batch, -1, -1)
        state = torch.cat((outputs, scratch), dim=1)
        for _ in range(REFINEMENTS):
            state = self.recurrent_block(state, encoded, mask)
        logits = self.vocabulary_projection(self.final_norm(state[:, :length]))
        return logits, {"iterations": REFINEMENTS, "state_length": length + SCRATCH}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and name != "token_embedding.weight":
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=8e-4, betas=(0.9, 0.95), eps=1e-8,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 32.0, 1.0)
    )
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer,
                        batch_size=512, eval_batch_size=1024, max_steps=None)
