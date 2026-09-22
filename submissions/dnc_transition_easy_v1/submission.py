"""Shared categorical digit transition with a generic DNC working memory."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, ANS_MARK, DIGIT_BASE = 0, 2, 3, 4, 5, 7
CAP, H, E, SLOTS, WORD, READS, PLACE = 64, 128, 32, 16, 32, 2, 16
NEG = -10000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


def _content(memory: Tensor, key: Tensor, strength: Tensor) -> Tensor:
    similarity = F.cosine_similarity(memory, key.unsqueeze(1), dim=-1, eps=1e-6)
    return F.softmax(similarity * (1.0 + F.softplus(strength))[:, None], dim=-1)


class DNCTransition(nn.Module):
    """One stationary sweep; all storage bookkeeping is generic and FP32."""
    def __init__(self):
        super().__init__()
        self.n_embedding = nn.Embedding(10, E)
        self.q_embedding = nn.Parameter(torch.empty(10, E))
        self.phase_embedding = nn.Embedding(2, 8)
        self.event_mlp = nn.Sequential(nn.Linear(E * 2 + PLACE + 2 + 8, H), nn.SiLU(), nn.Linear(H, H))
        self.controller = nn.LSTMCell(H + READS * WORD, H)
        # write key/strength/erase/vector/write gate/allocation gate, then read interfaces
        self.interface = nn.Linear(H, WORD + 1 + WORD + WORD + 2 + READS * WORD + READS + READS + READS * 3)
        self.readout = nn.Sequential(nn.Linear(H + READS * WORD, H), nn.SiLU(), nn.Linear(H, 10))
        nn.init.normal_(self.q_embedding, std=.02)

    @staticmethod
    def place_features(width: int, device: torch.device) -> Tensor:
        p = torch.arange(width, device=device, dtype=torch.float32)[:, None]
        f = torch.exp(torch.arange(0, PLACE, 2, device=device, dtype=torch.float32)
                      * (-math.log(10000.0) / PLACE))
        return torch.cat((torch.sin(p * f), torch.cos(p * f)), -1)

    def forward(self, n_digit: Tensor, n_present: Tensor, q: Tensor, q_present: Tensor) -> tuple[Tensor, dict]:
        b, width, _ = q.shape
        device, dtype = q.device, self.q_embedding.dtype
        memory = torch.zeros(b, SLOTS, WORD, device=device, dtype=torch.float32)
        usage = torch.zeros(b, SLOTS, device=device, dtype=torch.float32)
        precedence = torch.zeros_like(usage)
        link = torch.zeros(b, SLOTS, SLOTS, device=device, dtype=torch.float32)
        write_weight = torch.zeros_like(usage)
        read_weight = torch.full((b, READS, SLOTS), 1.0 / SLOTS, device=device, dtype=torch.float32)
        read_vectors = torch.zeros(b, READS, WORD, device=device, dtype=torch.float32)
        h = torch.zeros(b, H, device=device, dtype=dtype)
        c = torch.zeros_like(h)
        place = self.place_features(width, device).to(dtype)
        n_emb = self.n_embedding(n_digit)
        q_emb = q.to(dtype) @ self.q_embedding
        events = []
        for phase in range(2):
            phase_vec = self.phase_embedding.weight[phase].expand(b, -1)
            for p in range(width):
                place_active = (n_present[:, p] | q_present[:, p])[:, None]
                old_h, old_c = h, c
                old_memory, old_usage = memory, usage
                old_precedence, old_link = precedence, link
                old_write_weight = write_weight
                old_read_weight, old_read_vectors = read_weight, read_vectors
                raw = torch.cat((n_emb[:, p], q_emb[:, p], place[p].expand(b, -1),
                                 n_present[:, p, None].to(dtype), q_present[:, p, None].to(dtype), phase_vec), -1)
                event = self.event_mlp(raw)
                h, c = self.controller(torch.cat((event, read_vectors.flatten(1).to(dtype)), -1), (h, c))
                z = self.interface(h).float()
                i = 0
                wkey = z[:, i:i+WORD]; i += WORD
                wstrength = z[:, i]; i += 1
                erase = torch.sigmoid(z[:, i:i+WORD]); i += WORD
                add = torch.tanh(z[:, i:i+WORD]); i += WORD
                write_gate = torch.sigmoid(z[:, i]); allocation_gate = torch.sigmoid(z[:, i+1]); i += 2
                rkeys = z[:, i:i+READS*WORD].reshape(b, READS, WORD); i += READS*WORD
                rstrength = z[:, i:i+READS]; i += READS
                free = torch.sigmoid(z[:, i:i+READS]); i += READS
                modes = F.softmax(z[:, i:i+READS*3].reshape(b, READS, 3), -1)

                retention = torch.prod(1.0 - free[:, :, None] * read_weight, dim=1)
                usage = (usage + write_weight - usage * write_weight) * retention
                sorted_usage, order = torch.sort(usage, dim=-1)
                allocation_sorted = (1.0 - sorted_usage) * torch.cumprod(
                    torch.cat((torch.ones(b, 1, device=device), sorted_usage[:, :-1]), -1), -1)
                allocation = torch.zeros_like(usage).scatter(1, order, allocation_sorted)
                content_w = _content(memory, wkey, wstrength)
                write_weight = write_gate[:, None] * (allocation_gate[:, None] * allocation +
                                                       (1.0 - allocation_gate[:, None]) * content_w)
                memory = memory * (1.0 - write_weight[:, :, None] * erase[:, None, :]) + write_weight[:, :, None] * add[:, None, :]
                old_precedence = precedence
                link = (1.0 - write_weight[:, :, None] - write_weight[:, None, :]) * link
                link = link + write_weight[:, :, None] * old_precedence[:, None, :]
                eye = torch.eye(SLOTS, device=device, dtype=torch.bool)[None]
                link = link.masked_fill(eye, 0.0)
                precedence = (1.0 - write_weight.sum(-1, keepdim=True)) * old_precedence + write_weight
                rcontent = torch.stack([_content(memory, rkeys[:, r], rstrength[:, r]) for r in range(READS)], 1)
                backward = torch.bmm(read_weight, link)
                forward = torch.bmm(read_weight, link.transpose(1, 2))
                read_weight = modes[:, :, 0, None] * backward + modes[:, :, 1, None] * rcontent + modes[:, :, 2, None] * forward
                read_vectors = torch.bmm(read_weight, memory)
                h = torch.where(place_active, h, old_h)
                c = torch.where(place_active, c, old_c)
                memory = torch.where(place_active[:, :, None], memory, old_memory)
                usage = torch.where(place_active, usage, old_usage)
                precedence = torch.where(place_active, precedence, old_precedence)
                link = torch.where(place_active[:, :, None], link, old_link)
                write_weight = torch.where(place_active, write_weight, old_write_weight)
                read_weight = torch.where(place_active[:, :, None], read_weight, old_read_weight)
                read_vectors = torch.where(place_active[:, :, None], read_vectors, old_read_vectors)
                if phase == 1:
                    events.append(self.readout(torch.cat((h, read_vectors.flatten(1).to(dtype)), -1)))
        logits = torch.stack(events, 1)
        return F.softmax(logits, -1), {"digit_logits": logits, "memory_norm": memory.norm(dim=(1, 2))}


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.max_seq_len > CAP: raise ValueError("max_seq_len exceeds 64")
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = DNCTransition()

    @staticmethod
    def parse(ids: Tensor, mask: Tensor):
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        markers = ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() + 3 * ids.eq(T_MARK).long()
        role = torch.cummax(markers, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(-1)
        widths = torch.maximum((role == 1).sum(1), (role == 2).sum(1))
        return role, place, widths

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, length = input_ids.shape
        if length > self.config.max_seq_len: raise ValueError("input sequence exceeds configured cap")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, widths = self.parse(input_ids, mask)
        width = int(widths.max().item())
        slots = torch.arange(width, device=input_ids.device)
        def field(which: int):
            select = role.eq(which)[:, :, None] & place[:, :, None].eq(slots[None, None, :])
            present = select.any(1)
            values = ((input_ids - DIGIT_BASE).clamp(0, 9)[:, :, None] * select).sum(1)
            return values.long(), present
        n_digit, n_present = field(1); x_digit, x_present = field(2)
        q = F.one_hot(x_digit, 10).to(self.transition.q_embedding.dtype)
        q = torch.where(x_present[..., None], q, F.one_hot(torch.zeros_like(x_digit), 10).to(q.dtype))
        t_select = role.eq(3)
        decimal_place = torch.pow(input_ids.new_tensor(10), place)
        t_category = (((input_ids - DIGIT_BASE).clamp(0, 9) * decimal_place) * t_select).sum(1).long()
        steps = int(t_category.max().item())
        digit_logits = q.clamp_min(1e-8).log()
        last = None
        for step in range(steps):
            proposed, last = self.transition(n_digit, n_present, q, x_present)
            active_step = (t_category > step)[:, None, None]
            q = torch.where(active_step, proposed, q)
            digit_logits = torch.where(active_step, last["digit_logits"], digit_logits)
        full = q.new_full((b, length, self.config.vocab_size), NEG)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < widths[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(q.dtype) * active[..., None]
        placed = torch.einsum("bwl,bwd->bld", placement, digit_logits)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        full = torch.where(occupied[..., None], digits, full)
        return full, {"widths": widths, "macrosteps": steps,
                      "memory_norm": None if last is None else last["memory_norm"]}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec); assert_model_state(model, spec); return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=3e-4, betas=(.9, .95), eps=1e-8,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 50.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None)
