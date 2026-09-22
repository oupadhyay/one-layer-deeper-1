"""Recurrent Independent Mechanisms transition for the Easy E5 task."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, DIGIT_BASE = 0, 2, 3, 4, 7
PLACES, EXPERTS, WIDTH, TICKS = 4, 6, 32, 8
STATE_ELEMENTS = 14_795
NEG = -10000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class RIMTransition(nn.Module):
    """Six symmetric experts, with tied routing, update, and communication."""
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, WIDTH)
        self.presence_embedding = nn.Embedding(2, WIDTH)
        self.field_embedding = nn.Embedding(2, WIDTH)
        self.place_embedding = nn.Embedding(PLACES, WIDTH)
        self.initial_state = nn.Parameter(torch.empty(EXPERTS, WIDTH))
        self.input_q = nn.Linear(WIDTH, WIDTH, bias=False)
        self.input_k = nn.Linear(WIDTH, WIDTH, bias=False)
        self.input_v = nn.Linear(WIDTH, WIDTH, bias=False)
        self.router = nn.Linear(2 * WIDTH, 1)
        self.update = nn.GRUCell(WIDTH, WIDTH)
        self.message_q = nn.Linear(WIDTH, WIDTH, bias=False)
        self.message_k = nn.Linear(WIDTH, WIDTH, bias=False)
        self.message_v = nn.Linear(WIDTH, WIDTH, bias=False)
        self.message_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.output_queries = nn.Parameter(torch.empty(PLACES, WIDTH))
        self.digit_head = nn.Linear(WIDTH, 10)
        nn.init.normal_(self.initial_state, std=.02)
        nn.init.normal_(self.output_queries, std=.02)

    def slots(self, n: Tensor, np: Tensor, x: Tensor, xp: Tensor) -> Tensor:
        place = self.place_embedding.weight[None]
        ne = self.digit_embedding(n) * np[..., None].to(place.dtype)
        xe = (x @ self.digit_embedding.weight) * xp[..., None].to(place.dtype)
        ns = ne + self.presence_embedding(np.long()) + self.field_embedding.weight[0] + place
        xs = xe + self.presence_embedding(xp.long()) + self.field_embedding.weight[1] + place
        return torch.cat((ns, xs), 1)

    def tick(self, state: Tensor, slots: Tensor) -> tuple[Tensor, Tensor]:
        # Every expert independently attends over the complete immutable bus.
        scores = self.input_q(state) @ self.input_k(slots).transpose(1, 2) / math.sqrt(WIDTH)
        read = scores.softmax(-1) @ self.input_v(slots)
        pooled = slots.mean(1, keepdim=True).expand(-1, EXPERTS, -1)
        route_logits = self.router(torch.cat((state, pooled), -1)).squeeze(-1)
        values, indices = route_logits.topk(2, -1)
        mask = torch.zeros_like(route_logits).scatter(1, indices, values.sigmoid())
        candidate = self.update(read.reshape(-1, WIDTH), state.reshape(-1, WIDTH)).view_as(state)
        state = state + mask[..., None] * (candidate - state)
        message_score = self.message_q(state) @ self.message_k(state).transpose(1, 2) / math.sqrt(WIDTH)
        message = self.message_out(message_score.softmax(-1) @ self.message_v(state))
        return state + message, mask

    def forward(self, n: Tensor, np: Tensor, x: Tensor, xp: Tensor):
        slots = self.slots(n, np, x, xp)
        state = self.initial_state[None].expand(n.shape[0], -1, -1)
        masks = []
        for _ in range(TICKS):
            state, selected = self.tick(state, slots)
            masks.append(selected)
        score = self.output_queries[None] @ state.transpose(1, 2) / math.sqrt(WIDTH)
        decoded = score.softmax(-1) @ state
        return self.digit_head(decoded), {"router_masks": torch.stack(masks, 1), "state": state}


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(); self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = RIMTransition()

    @staticmethod
    def parse(ids: Tensor, mask: Tensor):
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        marker = ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() + 3 * ids.eq(T_MARK).long()
        role = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(-1)
        return role, place

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, length = input_ids.shape
        if length > self.config.max_seq_len: raise ValueError("input sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place = self.parse(input_ids, mask)
        nw, xw = role.eq(1).sum(1), role.eq(2).sum(1)
        if bool(((nw < 1) | (nw > 4) | (xw < 1) | (xw > 4)).any()):
            raise ValueError("N and X widths must be between one and four")
        slots = torch.arange(PLACES, device=input_ids.device)
        def field(which: int):
            chosen = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            return ((((input_ids-DIGIT_BASE).clamp(0, 9))[:, :, None] * chosen).sum(1).long(), chosen.any(1))
        n, np = field(1); x, xp = field(2)
        powers = torch.pow(input_ids.new_tensor(10), place)
        steps = (((input_ids-DIGIT_BASE).clamp(0, 9) * powers) * role.eq(3)).sum(1).long()
        if bool((steps > 64).any()): raise ValueError("T exceeds 64")
        q = F.one_hot(x, 10).to(self.transition.digit_embedding.weight.dtype)
        endpoint = q.clamp_min(1e-8).log(); current_present = xp
        runs = 3 if self.training else int(steps.max().item()); detail = None
        for step in range(runs):
            proposed, detail = self.transition(n, np, q, current_present)
            active = steps.gt(step)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            q = torch.where(active, proposed.softmax(-1), q)
            current_present = torch.where(active[:, :, 0], np, current_present)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        placement = F.one_hot(target.clamp(0, length-1), length).to(endpoint.dtype) * (slots[None] < nw[:, None])[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        digits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size-DIGIT_BASE-10), value=NEG)
        full = endpoint.new_full((b, length, self.config.vocab_size), NEG)
        full = torch.where(placement.sum(1).bool()[..., None], digits, full)
        return full, {"widths": nw, "x_widths": xw, "steps": steps, "macrosteps": runs, "transition": detail}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec); actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS: raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim == 2 and "embedding" not in name else no_decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.}], lr=6e-4, betas=(.9,.95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/32., 1.))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None)
