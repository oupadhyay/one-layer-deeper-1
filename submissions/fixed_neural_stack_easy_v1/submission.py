"""Fixed eight-tick differentiable neural-stack transition for FNST-8 Easy E5."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, ANSWER_MARK, DIGIT_BASE = 0, 2, 3, 4, 5, 7
WIDTH, SLOTS, VALUE, HIDDEN = 4, 8, 64, 160
STATE_ELEMENTS = 179_821
NEG = -10000.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


def continuous_stack_step(values: Tensor, strengths: Tensor, push_value: Tensor,
                          action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Apply PUSH/POP/NOOP probabilities and return the new top-unit read.

    Slots are chronological.  Strength and read-weight arithmetic is deliberately
    FP32, including under autocast; this routine does not mutate its arguments.
    """
    old_values, old_strengths = values.float(), strengths.float()
    push, pop = action[:, 0].float(), action[:, 1].float()
    remaining = pop
    revised = []
    for slot in range(old_strengths.shape[1] - 1, -1, -1):
        removed = torch.minimum(old_strengths[:, slot], remaining)
        revised.append(old_strengths[:, slot] - removed)
        remaining = remaining - removed
    revised_strengths = torch.stack(revised[::-1], dim=1)
    new_values = torch.cat((old_values, push_value.float()[:, None, :]), dim=1)
    new_strengths = torch.cat((revised_strengths, push[:, None]), dim=1)
    # Eight operations imply at most eight chronological slots; retain newest.
    new_values, new_strengths = new_values[:, -SLOTS:], new_strengths[:, -SLOTS:]
    remaining = torch.ones_like(push)
    weights = []
    for slot in range(SLOTS - 1, -1, -1):
        weight = torch.minimum(new_strengths[:, slot], remaining)
        weights.append(weight)
        remaining = remaining - weight
    read_weights = torch.stack(weights[::-1], dim=1)
    read = (read_weights[:, :, None] * new_values).sum(dim=1)
    return new_values, new_strengths, read


class FixedNeuralStackTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, 32)
        self.role_embedding = nn.Embedding(2, 32)
        self.place_embedding = nn.Embedding(4, 16)
        self.phase_embedding = nn.Embedding(2, 16)
        self.source = nn.Linear(98, 64)
        self.source_norm = nn.RMSNorm(64)
        self.controller = nn.GRUCell(128, HIDDEN)
        self.initial_hidden = nn.Parameter(torch.empty(HIDDEN))
        self.action_head = nn.Linear(HIDDEN, 3)
        self.push_value_head = nn.Linear(HIDDEN, VALUE)
        self.emit_norm = nn.RMSNorm(HIDDEN + VALUE)
        self.emit_hidden = nn.Linear(HIDDEN + VALUE, 96)
        self.emit_head = nn.Linear(96, 10)
        nn.init.normal_(self.initial_hidden, std=.02)

    def forward(self, n: Tensor, n_present: Tensor, q: Tensor,
                x_present: Tensor) -> tuple[Tensor, dict]:
        batch = n.shape[0]
        dtype, device = self.digit_embedding.weight.dtype, n.device
        hidden = self.initial_hidden.to(dtype).expand(batch, -1)
        values = torch.zeros(batch, SLOTS, VALUE, device=device, dtype=torch.float32)
        strengths = torch.zeros(batch, SLOTS, device=device, dtype=torch.float32)
        read = torch.zeros(batch, VALUE, device=device, dtype=torch.float32)
        outputs: list[Tensor] = []
        actions: list[Tensor] = []
        for tick in range(8):
            ingest = tick < 4
            place = tick if ingest else 7 - tick
            active = (n_present[:, place] | x_present[:, place]) if ingest else (
                n_present[:, place] | x_present[:, place])
            n_emb = self.digit_embedding(n[:, place]) + self.role_embedding.weight[0]
            x_emb = q[:, place].to(dtype) @ self.digit_embedding.weight + self.role_embedding.weight[1]
            raw = torch.cat((n_emb, x_emb, self.place_embedding.weight[place].expand(batch, -1),
                             self.phase_embedding.weight[0 if ingest else 1].expand(batch, -1),
                             n_present[:, place, None].to(dtype), x_present[:, place, None].to(dtype)), -1)
            source = self.source_norm(self.source(raw))
            hidden = self.controller(torch.cat((source, read.to(dtype)), -1), hidden)
            pre_read = read
            action = F.softmax(self.action_head(hidden), -1)
            noop = F.one_hot(torch.full((batch,), 2, device=device), 3).to(action.dtype)
            action = torch.where(active[:, None], action, noop)
            values, strengths, read = continuous_stack_step(
                values, strengths, self.push_value_head(hidden), action)
            actions.append(action)
            if not ingest:
                outputs.append(self.emit_head(F.silu(self.emit_hidden(
                    self.emit_norm(torch.cat((hidden, pre_read.to(dtype)), -1))))))
        # Emission chronology is places 3,2,1,0; expose LSD-first.
        logits = torch.stack(outputs[::-1], dim=1)
        return logits, {"actions": torch.stack(actions, 1), "strengths": strengths, "read": read}


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = FixedNeuralStackTransition()

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
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, _ = self.parse(input_ids, mask)
        widths = role.eq(1).sum(1)
        x_widths = role.eq(2).sum(1)
        if bool(((widths < 1) | (widths > WIDTH)).any()) or bool((x_widths > WIDTH).any()):
            raise ValueError("FNST-8 supports at most four digits")
        slots = torch.arange(WIDTH, device=input_ids.device)
        def field(which: int) -> tuple[Tensor, Tensor]:
            selected = role.eq(which)[:, :, None] & place[:, :, None].eq(slots)
            present = selected.any(1)
            value = (((input_ids - DIGIT_BASE).clamp(0, 9))[:, :, None] * selected).sum(1)
            return value.long(), present
        n, np = field(1); x, xp = field(2)
        q = F.one_hot(x, 10).to(self.transition.digit_embedding.weight.dtype)
        decimal = torch.pow(input_ids.new_tensor(10), place)
        steps = (((input_ids - DIGIT_BASE).clamp(0, 9) * decimal) * role.eq(3)).sum(1).long()
        if bool((steps > 64).any()):
            raise ValueError("T exceeds 64")
        endpoint = q.clamp_min(1e-8).log()
        detail = None
        runs = 1 if self.training else int(steps.max().item())
        current_present = xp
        for step in range(runs):
            proposed, detail = self.transition(n, np, q, current_present)
            active = (steps > step)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            hard = F.one_hot(proposed.argmax(-1), 10).to(q.dtype).detach()
            q = torch.where(active, hard, q)
            current_present = torch.where(active[:, :, 0], np, current_present)
        # The benchmark trains transition T=1 only; other returned endpoints are gated.
        if self.training:
            endpoint = torch.where((steps == 1)[:, None, None], endpoint, endpoint.detach())
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active_place = slots[None] < widths[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active_place[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (DIGIT_BASE, self.config.vocab_size - DIGIT_BASE - 10), value=NEG)
        full = endpoint.new_full((batch, length, self.config.vocab_size), NEG)
        full = torch.where(occupied[..., None], digits, full)
        return full, {"widths": widths, "steps": steps, "macrosteps": runs, "stack": detail}


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
