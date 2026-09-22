"""Small role--filler tensor-product transition for Easy E5."""

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state


PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
PLACES, FILLER, ROLE, CHANNELS, MAX_STEPS = 4, 32, 16, 4, 64
MICROTICKS = 3
STATE_ELEMENTS = 14_186


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class BilinearMicrotick(nn.Module):
    """Four generic multiplicative channels made solely from A Z B maps."""

    def __init__(self):
        super().__init__()
        self.left = nn.Parameter(torch.empty(CHANNELS, 2, FILLER, FILLER))
        self.right = nn.Parameter(torch.empty(CHANNELS, 2, ROLE, ROLE))
        self.channel_gate = nn.Parameter(torch.zeros(CHANNELS, FILLER, ROLE))
        self.residual_gate = nn.Parameter(torch.zeros(FILLER, ROLE))
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)

    def kronecker(self, z, branch):
        """Parallel A Z B maps; exposed to make the mechanism auditable."""
        return torch.einsum("cij,bjk,ckl->bcil", self.left[:, branch], z,
                            self.right[:, branch])

    def forward(self, z):
        first = self.kronecker(z, 0)
        second = self.kronecker(z, 1)
        update = (first * torch.sigmoid(second + self.channel_gate[None])).sum(dim=1)
        gate = torch.sigmoid(self.residual_gate)[None]
        return z + gate * (torch.tanh(update) - z)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.digit_embedding = nn.Embedding(10, FILLER)
        self.presence_embedding = nn.Embedding(2, FILLER)
        self.field_role = nn.Embedding(2, ROLE)
        self.place_role = nn.Embedding(PLACES, ROLE)
        self.base_memory = nn.Parameter(torch.empty(FILLER, ROLE))
        self.cell = BilinearMicrotick()
        self.role_queries = nn.Parameter(torch.empty(PLACES, ROLE))
        self.decoder = nn.Linear(FILLER, 10)
        nn.init.normal_(self.base_memory, std=0.02)
        nn.init.normal_(self.role_queries, std=0.02)

    @staticmethod
    def _field_layout(input_ids, mask):
        digit = input_ids.ge(DIGIT) & input_ids.lt(DIGIT + 10) & mask
        marker = input_ids.eq(N) | input_ids.eq(X) | input_ids.eq(T) | input_ids.eq(ANS)
        field = marker.long().cumsum(1) * digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = field[:, :, None].eq(field[:, None, :])
        place = (same & (index[None, None] > index[None, :, None])
                 & digit[:, None, :]).sum(2)
        return field, place, digit

    @staticmethod
    def _raw_steps(input_ids, field, digit):
        value = (input_ids - DIGIT).clamp(0, 9)
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same_t = field[:, :, None].eq(3) & field[:, None, :].eq(3)
        place = (same_t & (index[None, None] > index[None, :, None])
                 & digit[:, None, :]).sum(2)
        powers = input_ids.new_tensor([1, 10, 100, 1000])
        selected = digit & field.eq(3) & place.lt(PLACES)
        return (value * powers[place.clamp_max(PLACES - 1)] * selected).sum(1)

    @classmethod
    def parse(cls, input_ids, mask):
        field, place, digit = cls._field_layout(input_ids, mask)
        value = (input_ids - DIGIT).clamp(0, 9)
        slots = torch.arange(PLACES, device=input_ids.device)

        def take(which):
            selected = field[:, :, None].eq(which) & place[:, :, None].eq(slots)
            return ((selected * value[:, :, None]).sum(1).long(), selected.any(1))

        n_digits, n_present = take(1)
        x_digits, x_present = take(2)
        steps = cls._raw_steps(input_ids, field, digit)
        return n_digits, n_present, x_digits, x_present, steps

    def bind(self, digits, present, field_index, probabilities=None):
        filler = (self.digit_embedding(digits) if probabilities is None
                  else probabilities @ self.digit_embedding.weight)
        filler = filler + self.presence_embedding(present.long())
        role = (self.field_role.weight[field_index][None, None]
                + self.place_role.weight[None])
        return torch.einsum("bpf,bpr->bfr", filler, role.expand(digits.shape[0], -1, -1))

    def transition(self, n_memory, probabilities, present):
        dummy = probabilities.new_zeros(probabilities.shape[:2], dtype=torch.long)
        x_memory = self.bind(dummy, present, 1, probabilities)
        z = self.base_memory[None] + n_memory + x_memory
        for _ in range(MICROTICKS):
            z = self.cell(z)
        read = torch.einsum("bfr,pr->bpf", z, self.role_queries)
        return self.decoder(read), z

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        nd, npresent, xd, xpresent, steps = self.parse(input_ids, mask)
        n_width = npresent.sum(1)
        x_width = xpresent.sum(1)
        role, _, digit = self._field_layout(input_ids, mask)
        if bool(((role.eq(1) & digit).sum(1) != n_width).any()
                or ((role.eq(2) & digit).sum(1) != x_width).any()
                or (n_width < 1).any() or (x_width < 1).any()):
            raise ValueError("N and X widths must be between one and four")
        raw_steps = self._raw_steps(input_ids, role, digit)
        if bool(((role.eq(3) & digit).sum(1) > 2).any()
                or (raw_steps > MAX_STEPS).any()):
            raise ValueError("T exceeds 64")
        steps = raw_steps
        n_memory = self.bind(nd, npresent, 0)
        probabilities = F.one_hot(xd, 10).to(self.base_memory.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7))
        macrosteps = 3 if self.training else int(steps.max().item())
        final_memory = None
        for macrostep in range(macrosteps):
            transition_logits, final_memory = self.transition(n_memory, probabilities, xpresent)
            feedback = transition_logits.softmax(-1)
            active = (steps > macrostep)[:, None, None]
            probabilities = torch.where(active, feedback, probabilities)
            endpoint = torch.where(active, transition_logits, endpoint)

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        slot = (mask.sum(1)[:, None] - 1 - positions).clamp(0, PLACES - 1)
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits = selected.new_full((*input_ids.shape, self.vocab_size), -1e4)
        logits[:, :, DIGIT:DIGIT + 10] = selected
        return logits, {"parsed_steps": steps, "macrosteps": macrosteps,
                        "microticks": macrosteps * MICROTICKS,
                        "n_memory": n_memory, "final_memory": final_memory}


def build_model(spec):
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual} != {STATE_ELEMENTS}")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        matrix = parameter.ndim == 2 and "embedding" not in name
        (decay if matrix else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
        capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 32.0, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None)
