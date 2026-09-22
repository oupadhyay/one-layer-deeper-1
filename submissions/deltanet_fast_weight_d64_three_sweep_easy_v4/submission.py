"""D64 three-sweep chunkwise DeltaNet transition for the Easy benchmark."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, ANSWER_MARK, DIGIT_BASE = 0, 2, 3, 4, 5, 7
WIDTH, D_MODEL, HEADS, HEAD_DIM, TOKENS = 4, 64, 4, 16, 12
STATE_ELEMENTS = 43_594
NEG = -10000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


def delta_chunk(q: Tensor, k: Tensor, v: Tensor, beta: Tensor,
                state: Tensor) -> tuple[Tensor, Tensor]:
    """Vectorized causal delta rule; all recurrence algebra is FP32."""
    device_type = q.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        qf = F.normalize(F.silu(q.float()), dim=-1)
        kf = F.normalize(F.silu(k.float()), dim=-1)
        vf = F.silu(v.float())
        bf = torch.sigmoid(beta.float())
        sf = state.float()
        gram = kf @ kf.transpose(-1, -2)
        lower = torch.tril(bf.unsqueeze(-1) * gram, diagonal=-1)
        eye = torch.eye(q.shape[-2], device=q.device, dtype=torch.float32)
        rhs = torch.diag_embed(bf)
        a = torch.linalg.solve_triangular(eye + lower, rhs, upper=False)
        w = a @ kf
        u = a @ vf
        residual = u - w @ sf
        next_state = sf + kf.transpose(-1, -2) @ residual
        causal = torch.tril(torch.ones_like(gram))
        output = (qf @ sf + ((qf @ kf.transpose(-1, -2)) * causal) @ residual) / math.sqrt(HEAD_DIM)
    return output, next_state


class DeltaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta_norm = nn.RMSNorm(D_MODEL)
        self.qkv_beta = nn.Linear(D_MODEL, 3 * D_MODEL + HEADS, bias=False)
        self.head_rms_scale = nn.Parameter(torch.ones(D_MODEL))
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ffn_norm = nn.RMSNorm(D_MODEL)
        self.ffn_in = nn.Linear(D_MODEL, 4 * D_MODEL, bias=False)
        self.ffn_down = nn.Linear(2 * D_MODEL, D_MODEL, bias=False)

    def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = x.shape
        fused = self.qkv_beta(self.delta_norm(x))
        q, k, v, beta = torch.split(fused, (D_MODEL, D_MODEL, D_MODEL, HEADS), -1)
        def heads(t: Tensor) -> Tensor:
            return t.view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        o, state = delta_chunk(heads(q), heads(k), heads(v), beta.transpose(1, 2), state)
        # Per-head RMS with one learned scale per channel.
        scale = self.head_rms_scale.view(HEADS, HEAD_DIM)
        o = F.rms_norm(o.to(x.dtype), (HEAD_DIM,)) * scale[None, :, None, :]
        o = o.transpose(1, 2).reshape(batch, length, D_MODEL)
        x = x + self.out(o)
        gate, value = self.ffn_in(self.ffn_norm(x)).chunk(2, -1)
        x = x + self.ffn_down(F.silu(gate) * value)
        return x, state


class DeltaNetTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, D_MODEL)
        self.role_embedding = nn.Embedding(3, D_MODEL)
        self.place_embedding = nn.Embedding(4, D_MODEL)
        self.presence_embedding = nn.Embedding(2, D_MODEL)
        self.output_query = nn.Parameter(torch.empty(4, D_MODEL))
        self.block = DeltaBlock()
        self.final_norm = nn.RMSNorm(D_MODEL)
        self.decoder = nn.Linear(D_MODEL, 10, bias=True)
        nn.init.normal_(self.output_query, std=.02)

    def forward(self, n: Tensor, n_present: Tensor, x: Tensor,
                x_present: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        b = n.shape[0]
        dtype = self.digit_embedding.weight.dtype
        place = self.place_embedding.weight.unsqueeze(0)
        n_content = self.digit_embedding(n) * n_present.unsqueeze(-1).to(dtype)
        x_content = (x.to(dtype) @ self.digit_embedding.weight) * x_present.unsqueeze(-1).to(dtype)
        nt = n_content + self.role_embedding.weight[0] + place + self.presence_embedding(n_present.long())
        xt = x_content + self.role_embedding.weight[1] + place + self.presence_embedding(x_present.long())
        qt = self.output_query.unsqueeze(0).expand(b, -1, -1) + self.role_embedding.weight[2] + place
        tokens = torch.cat((nt, xt, qt), 1)
        state = torch.zeros(b, HEADS, HEAD_DIM, HEAD_DIM, device=n.device, dtype=torch.float32)
        for _ in range(3):
            tokens, state = self.block(tokens, state)
        logits = self.decoder(self.final_norm(tokens[:, 8:12]))
        return logits, {"state": state, "tokens": tokens}


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = DeltaNetTransition()

    @staticmethod
    def parse(ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        marker = ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() + 3 * ids.eq(T_MARK).long()
        role = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(-1)
        return role, place, digit

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, _ = self.parse(input_ids, mask)
        nw, xw = role.eq(1).sum(1), role.eq(2).sum(1)
        if bool(((nw < 1) | (nw > WIDTH) | (xw < 1) | (xw > WIDTH)).any()):
            raise ValueError("N and X widths must be between one and four")
        slots = torch.arange(WIDTH, device=input_ids.device)
        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            present = selected.any(1)
            value = (((input_ids - DIGIT_BASE).clamp(0, 9))[:, :, None] * selected).sum(1)
            return value.long(), present
        n, np = field(1); x, xp = field(2)
        decimal = torch.pow(input_ids.new_tensor(10), place)
        steps = (((input_ids - DIGIT_BASE).clamp(0, 9) * decimal) * role.eq(3)).sum(1).long()
        if bool((steps > 64).any()):
            raise ValueError("T exceeds 64")
        q = F.one_hot(x, 10).to(self.transition.digit_embedding.weight.dtype)
        endpoint = q.clamp_min(1e-8).log()
        current_present = xp
        runs = 3 if self.training else int(steps.max().item())
        detail = None
        for step in range(runs):
            proposed, detail = self.transition(n, np, q, current_present)
            active = (steps > step)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            q = torch.where(active, proposed.softmax(-1), q)
            current_present = torch.where(active[:, :, 0], np, current_present)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active_place = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active_place[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        full = endpoint.new_full((b, length, self.config.vocab_size), NEG)
        full = torch.where(occupied[..., None], digits, full)
        return full, {"widths": nw, "x_widths": xw, "steps": steps,
                      "macrosteps": runs, "transition": detail}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=6e-4, betas=(.9, .95), eps=1e-8,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 32.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None)
