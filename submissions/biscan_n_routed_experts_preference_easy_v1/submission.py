"""N-routed BiScan experts with a label-free sharp and balanced routing preference."""

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS = 64
EXPERTS = 128
LOG_EXPERTS = 4.852030263919617


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class BiScan(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_cell = nn.GRUCell(64, 64)
        self.up_cell = nn.GRUCell(128, 64)

    def forward(self, state, context, width_mask):
        pair = torch.cat((state, context), -1)
        hidden = pair.new_zeros((pair.shape[0], 64))
        down = [None] * pair.shape[1]
        for place in range(pair.shape[1] - 1, -1, -1):
            proposed = self.down_cell(pair[:, place], hidden)
            hidden = torch.where(width_mask[:, place, None], proposed, hidden)
            down[place] = hidden
        hidden = pair.new_zeros((pair.shape[0], 64))
        output = []
        for place in range(pair.shape[1]):
            proposed = self.up_cell(torch.cat((pair[:, place], down[place]), -1), hidden)
            hidden = torch.where(width_mask[:, place, None], proposed, hidden)
            output.append(torch.where(width_mask[:, place, None], hidden, torch.zeros_like(hidden)))
        return torch.stack(output, 1)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.embedding = nn.Embedding(10, 32)
        self.scan = BiScan()
        self.router = nn.Linear(64, EXPERTS)
        self.expert_weight = nn.Parameter(torch.empty(EXPERTS, 64, 10))
        self.expert_bias = nn.Parameter(torch.zeros(EXPERTS, 10))
        nn.init.kaiming_uniform_(self.expert_weight, a=5 ** .5)

    @staticmethod
    def parse(input_ids, mask):
        is_digit = (input_ids >= DIGIT) & mask
        marker = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(marker.long(), 1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None] == role[:, None, :]
        places = (same & (index[None, None] > index[None, :, None]) & is_digit[:, None]).sum(2)
        t_digits = is_digit & (role == 3)
        steps = (((input_ids - DIGIT) * torch.pow(input_ids.new_tensor(10), places)) * t_digits).sum(1)
        return role, places, steps

    def _prepare(self, input_ids, attention_mask=None):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, places, steps = self.parse(input_ids, mask)
        slots = torch.arange(self.max_seq_len, device=input_ids.device)
        values = (input_ids - DIGIT).clamp(0, 9)

        def digits_for(field):
            assignment = (role[:, :, None] == field) & (places[:, :, None] == slots)
            selected = (assignment.to(values.dtype) * values[:, :, None]).sum(1)
            present = assignment.any(1)
            return torch.where(present, selected, torch.zeros_like(selected)).long(), present

        n_digits, n_present = digits_for(1)
        x_digits, _ = digits_for(2)
        width_mask = slots[None] < n_present.sum(1)[:, None]
        return mask, steps, n_digits, x_digits, width_mask

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask, steps, n_digits, x_digits, width_mask = self._prepare(input_ids, attention_mask)
        zero = torch.zeros_like(x_digits)
        state_digits = torch.where(width_mask, x_digits, zero)
        context = self.embedding(torch.where(width_mask, n_digits, zero))
        n_states = self.scan(torch.zeros_like(context), context, width_mask)
        width = width_mask[:, :, None].to(n_states.dtype)
        summary = (n_states * width).sum(1) / width.sum(1).clamp_min(1.)
        routing = torch.softmax(self.router(summary), -1)
        sample_entropy = -(routing * routing.clamp_min(1e-7).log()).sum(-1).mean()
        marginal = routing.mean(0)
        marginal_entropy = -(marginal * marginal.clamp_min(1e-7).log()).sum()
        routing_preference = (sample_entropy - marginal_entropy) / LOG_EXPERTS
        mixed_weight = (routing @ self.expert_weight.flatten(1)).view(-1, 64, 10)
        mixed_bias = routing @ self.expert_bias
        probabilities = F.one_hot(state_digits, 10).to(context.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7))
        executed = int(steps.max().clamp_max(MAX_STEPS).item())
        for iteration in range(executed):
            state = probabilities @ self.embedding.weight
            hidden = self.scan(state, context, width_mask)
            digit_logits = torch.bmm(hidden, mixed_weight) + mixed_bias[:, None]
            feedback = (torch.softmax(digit_logits, -1) if self.training else
                        F.one_hot(digit_logits.argmax(-1), 10).to(digit_logits.dtype))
            feedback = torch.where(width_mask[:, :, None], feedback,
                                   F.one_hot(zero, 10).to(feedback.dtype))
            active = (steps > iteration)[:, None, None]
            probabilities = torch.where(active, feedback, probabilities)
            endpoint = torch.where(active, digit_logits, endpoint)
        batch, length = input_ids.shape
        logits = endpoint.new_full((batch, length, self.vocab_size), -1e4)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = mask.sum(1)[:, None] - 1 - positions
        selected = endpoint.gather(1, slot.clamp(0, self.max_seq_len - 1)[:, :, None].expand(-1, -1, 10))
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated_logits = logits
        if self.training:
            scale = torch.ones_like(steps, dtype=logits.dtype)
            logits = logits.detach() + scale[:, None, None] * (logits - logits.detach())
        return logits, {"parsed_steps": steps, "active_updates": steps.clamp_max(MAX_STEPS),
                        "executed_macrosteps": steps.new_tensor(executed),
                        "digit_probabilities": probabilities, "ungated_logits": ungated_logits,
                        "initial_state_digits": state_digits,
                        "context_digits": torch.where(width_mask, n_digits, zero),
                        "width_mask": width_mask, "routing": routing,
                        "routing_preference": routing_preference}


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    if sum(parameter.numel() for parameter in model.parameters()) != 154048:
        raise RuntimeError("N-routed expert parameter count drift")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": .02},
         {"params": no_decay, "weight_decay": 0.}],
        lr=8e-4, betas=(.9, .98), capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1., (step + 1) / 20))
    return OptimizerBundle(optimizer, scheduler)


def training_loss(logits, labels, auxiliary):
    return F.cross_entropy(logits, labels) + .01 * auxiliary["routing_preference"]


SUBMISSION = Submission(build_model, build_optimizer, batch_size=64, eval_batch_size=128,
                        max_steps=None, training_loss=training_loss)
