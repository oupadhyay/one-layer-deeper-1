"""Fixed conditional log-space WFST for the marker-delimited easy E5 task."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

PAD, N_MARK, X_MARK, T_MARK, ANS_MARK, DIGIT_BASE = 0, 2, 3, 4, 5, 7
PLACES, STATES, MAX_T = 4, 32, 64
STATE_ELEMENTS = 52_176


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len
        self.digit_embedding = nn.Embedding(10, 32)
        self.presence_embedding = nn.Embedding(2, 32)
        self.field_embedding = nn.Embedding(2, 32)
        self.input_place_embedding = nn.Embedding(4, 32)
        self.output_place_embedding = nn.Embedding(4, 16)
        self.context = nn.Sequential(nn.Linear(256, 96), nn.GELU(), nn.Linear(96, 80))
        self.transition_gate = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 16))
        self.emission_gate = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 16))
        self.A0 = nn.Parameter(torch.empty(32, 32))
        self.U = nn.Parameter(torch.empty(32, 16)); self.V = nn.Parameter(torch.empty(32, 16))
        self.E0 = nn.Parameter(torch.empty(32, 10))
        self.Q = nn.Parameter(torch.empty(32, 16)); self.D = nn.Parameter(torch.empty(10, 16))
        self.start = nn.Linear(80, 32); self.end = nn.Linear(80, 32)
        for parameter in (self.A0, self.U, self.V, self.E0, self.Q, self.D):
            nn.init.normal_(parameter, std=.02)

    @staticmethod
    def parse(ids: Tensor, mask: Tensor):
        """Return roles, LSD places, field widths, and decimal T."""
        digit = ids.ge(DIGIT_BASE) & ids.lt(DIGIT_BASE + 10) & mask
        markers = (ids.eq(N_MARK).long() + 2 * ids.eq(X_MARK).long() +
                   3 * ids.eq(T_MARK).long() + 4 * ids.eq(ANS_MARK).long())
        role = torch.cummax(markers, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        later = index[None, None, :] > index[None, :, None]
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & later & digit[:, None, :]).sum(-1).clamp_max(PLACES - 1)
        nw = (digit & role.eq(1)).sum(1); xw = (digit & role.eq(2)).sum(1)
        values = (ids - DIGIT_BASE).clamp(0, 9)
        time = (values * torch.pow(ids.new_tensor(10), place) * role.eq(3)).sum(1)
        time = time.clamp(1, MAX_T)
        return role, place, nw, xw, time

    @staticmethod
    def _field(ids: Tensor, role: Tensor, place: Tensor, which: int) -> tuple[Tensor, Tensor]:
        keys = torch.arange(PLACES, device=ids.device)
        assignment = role.eq(which)[..., None] & place[..., None].eq(keys)
        values = (ids - DIGIT_BASE).clamp(0, 9)
        digits = (assignment * values[..., None]).sum(1)
        return digits, assignment.any(1)

    def _slots(self, n_digits: Tensor, n_present: Tensor, x_prob: Tensor,
               x_present: Tensor) -> Tensor:
        b = n_digits.shape[0]; places = torch.arange(PLACES, device=n_digits.device)
        n = (self.digit_embedding(n_digits) + self.presence_embedding(n_present.long()) +
             self.field_embedding.weight[0] + self.input_place_embedding(places))
        # Absent X has zero digit content, while its presence embedding remains explicit.
        x_digit = x_prob @ self.digit_embedding.weight
        x_digit = x_digit * x_present[..., None]
        x = (x_digit + self.presence_embedding(x_present.long()) +
             self.field_embedding.weight[1] + self.input_place_embedding(places))
        return torch.cat((n, x), 1).reshape(b, 256)

    def wfst(self, context: Tensor, width: Tensor) -> tuple[Tensor, Tensor]:
        b = context.shape[0]; place = self.output_place_embedding.weight
        conditioned = torch.cat((context[:, None].expand(-1, PLACES, -1),
                                 place[None].expand(b, -1, -1)), -1)
        tg = torch.tanh(self.transition_gate(conditioned))
        eg = torch.tanh(self.emission_gate(conditioned))
        a = self.A0[None, None] + (self.U[None, None] * tg[:, :, None]) @ self.V.T
        e = self.E0[None, None] + (self.Q[None, None] * eg[:, :, None]) @ self.D.T
        log_a, log_e = a.log_softmax(-1), e.log_softmax(-1)
        active = torch.arange(PLACES, device=context.device)[None] < width[:, None]
        # The finite representation of log(0) avoids undefined inf-inf derivatives.
        log_zero = torch.finfo(context.dtype).min
        identity = context.new_full((STATES, STATES), log_zero)
        identity.diagonal().fill_(0)
        deterministic = context.new_full((STATES, 10), log_zero); deterministic[:, 0] = 0
        log_a = torch.where(active[..., None, None], log_a, identity)
        log_e = torch.where(active[..., None, None], log_e, deterministic)
        alpha = self.start(context).log_softmax(-1)
        alphas = []
        for p in range(PLACES):
            alpha = alpha[:, :, None] + log_e[:, p]
            alphas.append(alpha)
            state = torch.logsumexp(alpha, -1)
            if p < PLACES - 1:
                alpha = torch.logsumexp(state[:, :, None] + log_a[:, p + 1], 1)
        log_z = torch.logsumexp(torch.logsumexp(alphas[-1], -1) + self.end(context), -1)
        beta = self.end(context); marginals = [None] * PLACES
        for p in range(PLACES - 1, -1, -1):
            joint = alphas[p] + beta[:, :, None]
            marginals[p] = torch.logsumexp(joint, 1) - log_z[:, None]
            if p: beta = torch.logsumexp(log_a[:, p] +
                                         torch.logsumexp(log_e[:, p] + beta[:, :, None], -1)[:, None, :], -1)
        return torch.stack(marginals, 1), log_z

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, length = input_ids.shape
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, place, nw, xw, time = self.parse(input_ids, mask)
        n_digits, n_present = self._field(input_ids, role, place, 1)
        x_digits, x_present = self._field(input_ids, role, place, 2)
        if bool(((nw < 1) | (nw > PLACES) | (xw > PLACES)).any()):
            raise ValueError("N and X widths must be in the supported range")
        x_prob = F.one_hot(x_digits, 10).to(self.digit_embedding.weight.dtype)
        width = nw
        runs = 3 if self.training else int(time.max().item())
        endpoints, logzs = [], []
        for step in range(runs):
            context = self.context(self._slots(n_digits, n_present, x_prob, x_present))
            log_marginal, log_z = self.wfst(context, width)
            endpoints.append(log_marginal); logzs.append(log_z)
            probability = log_marginal.exp()
            active = time.gt(step)[:, None, None]
            x_prob = torch.where(active, probability, x_prob)
            x_present = torch.where(active[:, :, 0], n_present, x_present)
        endpoint_stack = torch.stack(endpoints, 1)
        endpoint_index = time.clamp_max(3) - 1 if self.training else time - 1
        selected = endpoint_stack[torch.arange(b, device=input_ids.device), endpoint_index]
        logits = self.digit_embedding.weight.new_full((b, length, self.vocab_size), -1e4)
        k = torch.arange(PLACES, device=input_ids.device)
        target = mask.sum(1)[:, None] - 1 - k
        placement = F.one_hot(target.clamp(0, length - 1), length).to(logits.dtype) * (k[None] < width[:, None])[..., None]
        digit_logits = torch.bmm(placement.transpose(1, 2), selected)
        occupied = placement.sum(1).bool()
        full = F.pad(digit_logits, (DIGIT_BASE, self.vocab_size - DIGIT_BASE - 10), value=-1e4)
        logits = torch.where(occupied[..., None], full, logits)
        return logits, {"log_marginals": selected, "logZ": torch.stack(logzs, 1),
                        "soft_feedback": x_prob, "time_steps": time}


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
                                   {"params": no_decay, "weight_decay": 0.}],
                                  lr=6e-4, betas=(.9, .95), eps=1e-8,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 32, 1.))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None)
