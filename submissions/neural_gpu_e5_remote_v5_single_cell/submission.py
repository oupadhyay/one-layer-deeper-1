"""Easy E5 Neural-GPU with one homogeneous recurrent grid cell."""

import torch
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
CHANNELS, MAX_PLACES, MAX_STEPS = 48, 16, 64
STATE_ELEMENTS = 64_330


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class CGRUCell(nn.Module):
    """A generic radius-one convolutional GRU update of the entire grid."""
    def __init__(self):
        super().__init__()
        self.gates = nn.Conv2d(CHANNELS, 2 * CHANNELS, 3, padding=1)
        self.candidate = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)

    def forward(self, state):
        reset, update = self.gates(state).chunk(2, 1)
        reset, update = reset.sigmoid(), update.sigmoid()
        proposed = torch.tanh(self.candidate(reset * state))
        return (1.0 - update) * state + update * proposed


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len, self.vocab_size = spec.max_seq_len, spec.vocab_size
        self.digit_embedding = nn.Embedding(10, CHANNELS)
        self.place_embedding = nn.Embedding(MAX_PLACES, CHANNELS)
        self.row_embedding = nn.Embedding(4, CHANNELS)
        self.cell = CGRUCell()
        self.output_norm = nn.RMSNorm(CHANNELS)
        self.readout = nn.Linear(CHANNELS, 10)

    @staticmethod
    def parse(input_ids, mask):
        digit = (input_ids >= DIGIT) & (input_ids < DIGIT + 10) & mask
        marker = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(marker.long(), 1) * digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None].eq(role[:, None, :])
        place = (same & (index[None, None] > index[None, :, None]) & digit[:, None]).sum(2)
        values = (input_ids - DIGIT).clamp(0, 9)
        slots = torch.arange(MAX_PLACES, device=input_ids.device)

        def field(which):
            assignment = role[:, :, None].eq(which) & place[:, :, None].eq(slots)
            return (assignment.to(values.dtype) * values[:, :, None]).sum(1).long(), assignment.any(1)

        nd, nmask = field(1)
        xd, xmask = field(2)
        td = digit & role.eq(3)
        powers = torch.pow(input_ids.new_tensor(10), place)
        steps = ((values * powers) * td).sum(1).clamp(0, MAX_STEPS)
        return nd, xd, nmask, xmask, steps

    def prepare(self, ids, attention_mask=None):
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        return (mask, *self.parse(ids, mask))

    def initialize_grid(self, x_state, n_digits, places):
        b, p, _ = x_state.shape
        place = self.place_embedding.weight[:p][None]
        rows = self.row_embedding.weight[None, :, None, :]
        grid = place[:, None] + rows
        grid = grid.expand(b, -1, -1, -1).clone()
        grid[:, 0] = x_state
        grid[:, 1] = grid[:, 1] + self.digit_embedding(n_digits)
        return grid.permute(0, 3, 1, 2).contiguous()

    def transition(self, x_state, n_digits):
        p = x_state.shape[1]
        grid = self.initialize_grid(x_state, n_digits, p)
        for _ in range(2 * p + 4):
            grid = self.cell(grid)
        output = grid[:, :, 3].transpose(1, 2)
        normalized = self.output_norm(output)
        return normalized, self.readout(normalized), grid

    def forward(self, input_ids, attention_mask=None):
        mask, nd, xd, nmask, _xmask, steps = self.prepare(input_ids, attention_mask)
        widths = nmask.sum(1)
        p = int(widths.max().item())
        if not 1 <= p <= MAX_PLACES:
            raise ValueError(f"active modulus places must be in 1..{MAX_PLACES}")
        nd, xd = nd[:, :p], xd[:, :p]
        place = self.place_embedding.weight[None, :p]
        x_state = self.digit_embedding(xd) + place + self.row_embedding.weight[0]
        endpoint = self.readout(self.output_norm(x_state))
        macrosteps = 1 if self.training else int(steps.max().item())
        final_grid = None
        for macro in range(macrosteps):
            candidate, candidate_logits, final_grid = self.transition(x_state, nd)
            active = (steps > macro)[:, None, None]
            x_state = torch.where(active, candidate, x_state)
            endpoint = torch.where(active, candidate_logits, endpoint)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        slot = mask.sum(1)[:, None] - 1 - positions
        slot = torch.minimum(slot.clamp_min(0), (widths - 1).clamp_min(0)[:, None])
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits = endpoint.new_full((*input_ids.shape, self.vocab_size), -1e4)
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated = logits
        if self.training:
            gate = steps.eq(1).to(logits.dtype)[:, None, None]
            logits = logits.detach() + gate * (logits - logits.detach())
        calls = macrosteps * (2 * p + 4)
        auxiliary = {"parsed_steps": steps, "widths": widths, "active_places": p,
                     "macrosteps": macrosteps, "microticks_per_macrostep": 2 * p + 4,
                     "cell_calls": calls, "grid_shape": (input_ids.shape[0], 4, p, CHANNELS),
                     "scratch_resets": macrosteps, "n_reinitializations": macrosteps,
                     "ungated_logits": ungated, "x_state": x_state, "final_grid": final_grid}
        return logits, auxiliary


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
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=1e-4, betas=(.9, .95),
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min((update + 1) / 30, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=64,
                        eval_batch_size=512, max_steps=None)
