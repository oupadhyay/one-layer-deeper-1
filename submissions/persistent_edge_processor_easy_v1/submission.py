"""Persistent learned edge-state processor for Easy E5."""

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state


PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D, RANK, HEADS, ROUNDS, MAX_PLACES, MAX_STEPS = 64, 32, 4, 4, 16, 64


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class EdgeInitializer(nn.Module):
    """Learned low-rank interactions between ordered categorical nodes."""

    def __init__(self):
        super().__init__()
        self.x_left = nn.Linear(D, RANK, bias=False)
        self.x_right = nn.Linear(D, RANK, bias=False)
        self.xn_left = nn.Linear(D, RANK, bias=False)
        self.xn_right = nn.Linear(D, RANK, bias=False)
        self.nx_left = nn.Linear(D, RANK, bias=False)
        self.nx_right = nn.Linear(D, RANK, bias=False)
        self.combine = nn.Linear(3 * RANK, D)
        self.row_place = nn.Embedding(MAX_PLACES, D)
        self.column_place = nn.Embedding(MAX_PLACES, D)

    def forward(self, x_state, n_state):
        left_x = x_state[:, :, None]
        right_x = x_state[:, None, :]
        left_n = n_state[:, :, None]
        right_n = n_state[:, None, :]
        features = torch.cat((
            self.x_left(left_x) * self.x_right(right_x),
            self.xn_left(left_x) * self.xn_right(right_n),
            self.nx_left(left_n) * self.nx_right(right_x),
        ), dim=-1)
        edge = self.combine(features)
        p = x_state.shape[1]
        return (edge + self.row_place.weight[None, :p, None]
                + self.column_place.weight[None, None, :p])


class TiedEdgeRound(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_input = nn.Linear(4 * D, D)
        self.edge_gru = nn.GRUCell(D, D)
        self.query_norm = nn.RMSNorm(D)
        self.context_norm = nn.RMSNorm(D)
        self.query = nn.Linear(D, D, bias=False)
        self.key = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.attention_out = nn.Linear(D, D, bias=False)
        self.output_gru = nn.GRUCell(D, D)
        self.output_norm = nn.RMSNorm(D)
        self.output_ffn = nn.Sequential(
            nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D)
        )

    @staticmethod
    def attend(query, key, value, context_mask):
        batch, q_length, _ = query.shape
        k_length = key.shape[1]

        def split_heads(tensor, length):
            return tensor.reshape(batch, length, HEADS, D // HEADS).transpose(1, 2)

        result = F.scaled_dot_product_attention(
            split_heads(query, q_length),
            split_heads(key, k_length),
            split_heads(value, k_length),
            attn_mask=context_mask[:, None, None],
            dropout_p=0.0,
        )
        return result.transpose(1, 2).contiguous().reshape(batch, q_length, D)

    def forward(self, edge, output, x_state, n_state, context_mask):
        batch, p, _, _ = edge.shape
        left_x = x_state[:, :, None].expand(-1, -1, p, -1)
        right_x = x_state[:, None, :].expand(-1, p, -1, -1)
        left_n = n_state[:, :, None].expand(-1, -1, p, -1)
        right_n = n_state[:, None, :].expand(-1, p, -1, -1)
        edge_input = self.edge_input(torch.cat((left_x, right_x, left_n, right_n), -1))
        edge = self.edge_gru(
            edge_input.reshape(batch * p * p, D), edge.reshape(batch * p * p, D)
        ).reshape(batch, p, p, D)
        context = torch.cat((edge.reshape(batch, p * p, D), n_state), dim=1)
        normalized = self.context_norm(context)
        message = self.attention_out(self.attend(
            self.query(self.query_norm(output)), self.key(normalized),
            self.value(normalized), context_mask,
        ))
        output = self.output_gru(
            message.reshape(batch * p, D), output.reshape(batch * p, D)
        ).reshape(batch, p, D)
        output = output + self.output_ffn(self.output_norm(output))
        return edge, output


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.digit_embedding = nn.Embedding(10, D)
        self.place_embedding = nn.Embedding(MAX_PLACES, D)
        self.n_role = nn.Parameter(torch.empty(D))
        self.x_role = nn.Parameter(torch.empty(D))
        self.output_role = nn.Parameter(torch.empty(D))
        self.edge_initializer = EdgeInitializer()
        self.edge_round = TiedEdgeRound()
        self.readout_norm = nn.RMSNorm(D)
        self.readout = nn.Linear(D, 10)
        for parameter in (self.n_role, self.x_role, self.output_role):
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
        place = self.place_embedding.weight[None, :p]
        x_state = probabilities @ self.digit_embedding.weight + place + self.x_role
        n_state = self.digit_embedding(n_digits) + place + self.n_role
        edge = self.edge_initializer(x_state, n_state)
        output = x_state + self.output_role
        edge_mask = (place_mask[:, :, None] & place_mask[:, None, :]).reshape(batch, p * p)
        context_mask = torch.cat((edge_mask, place_mask), dim=1)
        for _ in range(ROUNDS):
            edge, output = self.edge_round(
                edge, output, x_state, n_state, context_mask
            )
        logits = self.readout(self.readout_norm(output))
        return logits, edge

    def forward(self, input_ids, attention_mask=None):
        mask, n_digits, x_digits, widths, steps, place_mask = self.prepare(
            input_ids, attention_mask
        )
        probabilities = F.one_hot(x_digits, 10).to(self.digit_embedding.weight.dtype)
        zero_hot = F.one_hot(torch.zeros_like(x_digits), 10).to(probabilities.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7))
        macrosteps = 1 if self.training else int(steps.max().item())
        final_edge = None
        for macrostep in range(macrosteps):
            transition_logits, final_edge = self.transition(
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
            "edge_round_calls": macrosteps * ROUNDS,
            "edge_shape": None if final_edge is None else tuple(final_edge.shape),
            "ungated_logits": ungated,
            "plain_endpoint_ce": True,
        }


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
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
