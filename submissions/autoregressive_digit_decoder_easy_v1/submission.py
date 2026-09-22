"""Soft autoregressive digit decoder for the marker-delimited easy task."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

D, HEADS, FFN = 192, 6, 576
ENCODER_BLOCKS, DECODER_BLOCKS = 3, 2
CAP, DIGIT_BASE = 64, 7
PAD, N_MARK, X_MARK, T_MARK, ANS_MARK = 0, 2, 3, 4, 5


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (D,), self.weight)


class Attention(nn.Module):
    def __init__(self, cross: bool = False):
        super().__init__()
        self.cross = cross
        if cross:
            self.q = nn.Linear(D, D, bias=False)
            self.kv = nn.Linear(D, 2 * D, bias=False)
        else:
            self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None,
                key_mask: Tensor | None = None, causal: bool = False) -> Tensor:
        source = x if context is None else context
        b, qn, _ = x.shape; kn = source.shape[1]
        if self.cross:
            q = self.q(x); k, v = self.kv(source).chunk(2, -1)
        else:
            q, k, v = self.qkv(x).chunk(3, -1)
        def heads(t: Tensor, n: int) -> Tensor:
            return t.reshape(b, n, HEADS, D // HEADS).transpose(1, 2)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(heads(q, qn), heads(k, kn), heads(v, kn),
                                            attn_mask=mask, dropout_p=0.0,
                                            is_causal=causal and mask is None)
        return self.out(y.transpose(1, 2).contiguous().reshape(b, qn, D))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__(); self.up = nn.Linear(D, 2 * FFN, bias=False); self.down = nn.Linear(FFN, D, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.n1 = RMSNorm(); self.attn = Attention(); self.n2 = RMSNorm(); self.ffn = SwiGLU()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x), key_mask=mask)
        return x + self.ffn(self.n2(x))


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(); self.self_attn = Attention()
        self.n2 = RMSNorm(); self.cross_attn = Attention(True)
        self.n3 = RMSNorm(); self.ffn = SwiGLU()

    def forward(self, x: Tensor, prompt: Tensor, prompt_mask: Tensor) -> Tensor:
        x = x + self.self_attn(self.n1(x), causal=True)
        x = x + self.cross_attn(self.n2(x), prompt, prompt_mask)
        return x + self.ffn(self.n3(x))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.max_seq_len > CAP: raise ValueError("max_seq_len exceeds 64")
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len
        self.token_embedding = nn.Embedding(spec.vocab_size, D)
        self.role_embedding = nn.Embedding(5, D)
        self.place_embedding = nn.Embedding(CAP, D)
        self.position_embedding = nn.Embedding(CAP, D)
        self.encoder_blocks = nn.ModuleList(EncoderBlock() for _ in range(ENCODER_BLOCKS))
        self.decoder_blocks = nn.ModuleList(DecoderBlock() for _ in range(DECODER_BLOCKS))
        self.digit_embedding = nn.Parameter(torch.empty(10, D))
        self.start = nn.Parameter(torch.empty(D)); self.output_role = nn.Parameter(torch.empty(D))
        self.output_place = nn.Embedding(CAP, D); self.final_norm = RMSNorm()
        nn.init.normal_(self.digit_embedding, std=.02); nn.init.normal_(self.start, std=.02)
        nn.init.normal_(self.output_role, std=.02)

    @staticmethod
    def parse(ids: Tensor, mask: Tensor):
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        marker_role = (ids.eq(N_MARK).long() + ids.eq(X_MARK).long() * 2 +
                       ids.eq(T_MARK).long() * 3 + ids.eq(ANS_MARK).long() * 4)
        role = torch.cummax(marker_role, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(-1).clamp_max(CAP - 1)
        widths = torch.maximum((digit & role.eq(1)).sum(1), (digit & role.eq(2)).sum(1))
        return role, place, widths

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, length = input_ids.shape
        if length > self.max_seq_len or length > CAP: raise ValueError("input sequence exceeds configured cap")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, widths = self.parse(input_ids, mask)
        pos = torch.arange(length, device=input_ids.device)
        prompt = (self.token_embedding(input_ids) + self.role_embedding(role) +
                  self.place_embedding(place) + self.position_embedding(pos)[None])
        prompt = prompt * mask[..., None]
        for block in self.encoder_blocks: prompt = block(prompt, mask)
        steps = int(widths.max().item())
        prefix = self.start + self.output_role + self.output_place.weight[0]
        prefix = prefix[None, None].expand(b, 1, -1)
        generated = []
        for k in range(steps):
            state = prefix
            for block in self.decoder_blocks: state = block(state, prompt, mask)
            z = self.final_norm(state[:, -1]) @ self.digit_embedding.t()
            generated.append(z)
            if k + 1 < steps:
                feedback = z.softmax(-1) @ self.digit_embedding
                token = feedback + self.output_role + self.output_place.weight[k + 1]
                prefix = torch.cat((prefix, token[:, None]), 1)
        logits = self.token_embedding.weight.new_full((b, length, self.vocab_size), -1e4)
        if steps:
            zall = torch.stack(generated, 1)
            ks = torch.arange(steps, device=input_ids.device)
            target = mask.long().sum(1)[:, None] - 1 - ks[None]
            active = ks[None] < widths[:, None]
            placement = F.one_hot(target.clamp(0, length - 1), length).to(zall.dtype) * active[..., None]
            digit_logits = torch.einsum("bkl,bkd->bld", placement, zall)
            occupied = placement.sum(1).bool()
            full_digits = F.pad(digit_logits, (DIGIT_BASE, self.vocab_size - DIGIT_BASE - 10), value=-1e4)
            logits = torch.where(occupied[..., None], full_digits, logits)
        return logits, {"decode_widths": widths, "decoder_steps": steps,
                        "decoder_block_calls": DECODER_BLOCKS * steps}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec); assert_model_state(model, spec); return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        embedding = "embedding" in name
        (decay if parameter.ndim == 2 and not embedding else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=6e-4, betas=(.9, .95), eps=1e-8,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 32, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=512, eval_batch_size=1024, max_steps=None)
