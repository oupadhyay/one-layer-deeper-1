"""Nested low-rank scan transducer for Easy E5."""

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state


PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D, H, RANK, MAX_PLACES, MAX_STEPS = 64, 112, 32, 16, 64
STATE_ELEMENTS = 180_986


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class LowRankScanCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_norm = nn.RMSNorm(H)
        self.factor_norm = nn.RMSNorm(H)
        self.hidden_rank = nn.Linear(H, RANK, bias=False)
        self.factor_rank = nn.Linear(H, RANK, bias=False)
        self.rank_out = nn.Linear(RANK, H, bias=False)
        self.candidate_hidden = nn.Linear(H, H, bias=False)
        self.candidate_factor = nn.Linear(H, H)
        self.gate_hidden = nn.Linear(H, H, bias=False)
        self.gate_factor = nn.Linear(H, H)

    def forward(self, hidden, factor):
        normalized_hidden = self.hidden_norm(hidden)
        normalized_factor = self.factor_norm(factor)
        interaction = self.rank_out(
            self.hidden_rank(normalized_hidden) * self.factor_rank(normalized_factor)
        )
        candidate = torch.tanh(
            self.candidate_hidden(normalized_hidden)
            + self.candidate_factor(normalized_factor) + interaction
        )
        gate = torch.sigmoid(
            self.gate_hidden(normalized_hidden) + self.gate_factor(normalized_factor)
        )
        return (1.0 - gate) * hidden + gate * candidate


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.digit_embedding = nn.Embedding(10, D)
        self.n_role = nn.Parameter(torch.empty(D))
        self.x_role = nn.Parameter(torch.empty(D))
        self.pair_left = nn.Linear(2 * D, H, bias=False)
        self.pair_right = nn.Linear(2 * D, H, bias=False)
        self.inner_phase = nn.Parameter(torch.empty(H))
        self.outer_phase = nn.Parameter(torch.empty(H))
        self.inner_initial = nn.Parameter(torch.empty(H))
        self.outer_initial = nn.Parameter(torch.empty(H))
        self.contraction_cell = LowRankScanCell()
        self.output_local = nn.Linear(2 * D, H, bias=False)
        self.output_global = nn.Linear(H, H, bias=False)
        self.output_phase = nn.Parameter(torch.empty(H))
        self.output_initial = nn.Parameter(torch.empty(H))
        self.output_cell = LowRankScanCell()
        self.readout_norm = nn.RMSNorm(H)
        self.readout = nn.Linear(H, 10)
        for parameter in (
            self.n_role, self.x_role, self.inner_phase, self.outer_phase,
            self.inner_initial, self.outer_initial, self.output_phase,
            self.output_initial,
        ):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def parse(input_ids, mask):
        digit = (input_ids >= DIGIT) & (input_ids < DIGIT + 10) & mask
        marker = ((input_ids == N) | (input_ids == X)
                  | (input_ids == T) | (input_ids == ANS))
        role = torch.cumsum(marker.long(), dim=1) * digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None].eq(role[:, None, :])
        place = (same & (index[None, None] > index[None, :, None])
                 & digit[:, None]).sum(dim=2)
        values = (input_ids - DIGIT).clamp(0, 9)
        slots = torch.arange(MAX_PLACES, device=input_ids.device)

        def field(which):
            assignment = role[:, :, None].eq(which) & place[:, :, None].eq(slots)
            field_digits = (assignment.to(values.dtype) * values[:, :, None]).sum(dim=1)
            return field_digits.long(), assignment.any(dim=1)

        n_digits, n_mask = field(1)
        x_digits, _ = field(2)
        t_digits = digit & role.eq(3)
        powers = torch.pow(input_ids.new_tensor(10), place)
        steps = (values * powers * t_digits).sum(dim=1).clamp(0, MAX_STEPS)
        return n_digits, x_digits, n_mask, steps

    def prepare(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        n_digits, x_digits, n_mask, steps = self.parse(input_ids, mask)
        widths = n_mask.sum(dim=1)
        p = int(widths.max().item())
        if not 1 <= p <= MAX_PLACES:
            raise ValueError(f"active modulus places must be in 1..{MAX_PLACES}")
        place_mask = torch.arange(p, device=input_ids.device)[None] < widths[:, None]
        n_digits = torch.where(place_mask, n_digits[:, :p], torch.zeros_like(n_digits[:, :p]))
        x_digits = torch.where(place_mask, x_digits[:, :p], torch.zeros_like(x_digits[:, :p]))
        return mask, n_digits, x_digits, widths, steps, place_mask

    def transition(self, probabilities, n_digits, place_mask):
        batch, p, _ = probabilities.shape
        x_state = probabilities @ self.digit_embedding.weight + self.x_role
        n_state = self.digit_embedding(n_digits) + self.n_role
        local = torch.cat((x_state, n_state), dim=-1)
        left = self.pair_left(local)
        right = self.pair_right(local)

        inner = self.inner_initial[None, None].expand(batch, p, -1)
        for column in range(p):
            factor = left + right[:, column, None] + self.inner_phase
            candidate = self.contraction_cell(inner, factor)
            active = (place_mask & place_mask[:, column, None])[:, :, None]
            inner = torch.where(active, candidate, inner)

        outer = self.outer_initial[None].expand(batch, -1)
        for row in range(p):
            candidate = self.contraction_cell(outer, inner[:, row] + self.outer_phase)
            outer = torch.where(place_mask[:, row, None], candidate, outer)

        output = self.output_initial[None].expand(batch, -1)
        outputs = []
        for place in range(p):
            factor = (self.output_local(local[:, place])
                      + self.output_global(outer) + self.output_phase)
            candidate = self.output_cell(output, factor)
            output = torch.where(place_mask[:, place, None], candidate, output)
            outputs.append(output)
        output_state = torch.stack(outputs, dim=1)
        return self.readout(self.readout_norm(output_state)), output_state

    def forward(self, input_ids, attention_mask=None):
        mask, n_digits, x_digits, widths, steps, place_mask = self.prepare(
            input_ids, attention_mask
        )
        probabilities = F.one_hot(x_digits, 10).to(self.digit_embedding.weight.dtype)
        zero_hot = F.one_hot(torch.zeros_like(x_digits), 10).to(probabilities.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7))
        macrosteps = 1 if self.training else int(steps.max().item())
        final_output_state = None
        for macrostep in range(macrosteps):
            transition_logits, final_output_state = self.transition(
                probabilities, n_digits, place_mask
            )
            soft = transition_logits.softmax(dim=-1)
            hard = F.one_hot(soft.argmax(dim=-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            feedback = torch.where(place_mask[..., None], feedback, zero_hot)
            active = (steps > macrostep)[:, None, None]
            probabilities = torch.where(active, feedback, probabilities)
            endpoint = torch.where(active, transition_logits, endpoint)

        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = mask.sum(dim=1)[:, None] - 1 - positions
        slot = torch.minimum(slot.clamp_min(0), (widths - 1).clamp_min(0)[:, None])
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits = endpoint.new_full((*input_ids.shape, self.vocab_size), -1e4)
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated = logits
        if self.training:
            gate = steps.eq(1).to(logits.dtype)[:, None, None]
            logits = logits.detach() + gate * (logits - logits.detach())
        return logits, {
            "parsed_steps": steps,
            "widths": widths,
            "macrosteps": macrosteps,
            "inner_calls": macrosteps * probabilities.shape[1],
            "outer_calls": macrosteps * probabilities.shape[1],
            "output_calls": macrosteps * probabilities.shape[1],
            "feedback": probabilities,
            "final_output_state": final_output_state,
            "ungated_logits": ungated,
            "plain_endpoint_ce": True,
        }


def build_model(spec):
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        is_matrix = parameter.ndim == 2 and "embedding" not in name
        (decay if is_matrix else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 32.0, 1.0)
    )
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(
    build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None
)
