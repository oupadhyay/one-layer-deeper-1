"""Official Easy-tier package of loop13; only batch sizes and manifest step deferral differ."""

import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS = 64
CHANNELS = 32
STATE_ELEMENTS = 13162


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class AxialLocalCell(nn.Module):
    """One shared radius-one GRU-like update along a selected axis."""
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(CHANNELS)
        self.gates = nn.Linear(3 * CHANNELS, 2 * CHANNELS)
        self.candidate = nn.Linear(3 * CHANNELS, CHANNELS)

    def forward(self, workspace, axis):
        center = self.norm(workspace)
        if axis == "H":
            left = F.pad(center[:, :, :-1], (0, 0, 1, 0))
            right = F.pad(center[:, :, 1:], (0, 0, 0, 1))
        elif axis == "V":
            left = F.pad(center[:, :-1], (0, 0, 0, 0, 1, 0))
            right = F.pad(center[:, 1:], (0, 0, 0, 0, 0, 1))
        else:
            raise ValueError("axis must be H or V")
        local = torch.cat((center, left, right), -1)
        reset, update = self.gates(local).chunk(2, -1)
        reset, update = reset.sigmoid(), update.sigmoid()
        proposed = torch.tanh(self.candidate(torch.cat((center, reset * left, reset * right), -1)))
        return (1 - update) * workspace + update * proposed


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len, self.vocab_size = spec.max_seq_len, spec.vocab_size
        self.embedding = nn.Embedding(10, 16)
        self.initializer = nn.Sequential(nn.Linear(70, CHANNELS), nn.GELU(),
                                         nn.Linear(CHANNELS, CHANNELS))
        self.cell = AxialLocalCell()
        self.readout = nn.Linear(CHANNELS, 10)

    @staticmethod
    def parse(input_ids, mask):
        is_digit = (input_ids >= DIGIT) & mask
        markers = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(markers.long(), 1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None] == role[:, None, :]
        places = (same & (index[None, None] > index[None, :, None]) & is_digit[:, None]).sum(2)
        td = is_digit & role.eq(3)
        steps = (((input_ids - DIGIT) * torch.pow(input_ids.new_tensor(10), places)) * td).sum(1)
        return role, places, steps.clamp(max=MAX_STEPS)

    def _prepare(self, ids, attention_mask):
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, places, steps = self.parse(ids, mask)
        slots = torch.arange(self.max_seq_len, device=ids.device)
        values = (ids - DIGIT).clamp(0, 9)
        def field(which):
            assignment = role[:, :, None].eq(which) & places[:, :, None].eq(slots)
            digits = torch.einsum("bls,bl->bs", assignment.long(), values)
            return digits.long(), assignment.any(1)
        nd, present = field(1); xd, _ = field(2)
        return mask, steps, nd, xd, present.sum(1)

    def _workspace(self, probabilities, n_digits, widths):
        b, w = probabilities.shape[:2]; columns = w + 1
        state = probabilities @ self.embedding.weight
        ne = self.embedding(n_digits)
        ei = state[:, :, None].expand(-1, -1, columns, -1)
        ni = ne[:, :, None].expand_as(ei)
        ej = state[:, None, :].expand(-1, w, -1, -1)
        nj = ne[:, None, :].expand_as(ej)
        zero = state.new_zeros((b, w, 1, 16))
        ej, nj = torch.cat((ej, zero), 2), torch.cat((nj, zero), 2)
        rows = torch.arange(w, device=state.device)[None, :, None]
        cols = torch.arange(columns, device=state.device)[None, None, :]
        width = widths[:, None, None]
        valid_row, pair, boundary = rows < width, cols < width, cols == width
        boundary_cells = boundary.expand(-1, w, -1)[..., None]
        ej = torch.where(boundary_cells, torch.zeros_like(ej), ej)
        nj = torch.where(boundary_cells, torch.zeros_like(nj), nj)
        flags = torch.stack((valid_row.expand(-1, -1, columns), pair.expand(-1, w, -1),
                             boundary.expand(-1, w, -1), (rows == 0).expand(b, -1, columns),
                             (cols == 0).expand(b, w, -1),
                             (rows == width - 1).expand(-1, -1, columns)), -1).to(state.dtype)
        valid = valid_row & (pair | boundary)
        workspace = self.initializer(torch.cat((ei, ej, ni, nj, flags), -1))
        return workspace * valid[..., None], valid

    def debug_execution(self, input_ids, attention_mask=None):
        _, steps, _, _, widths = self._prepare(input_ids, attention_mask)
        maximum = int(steps.max())
        return {"parsed_steps": steps, "widths": widths,
                "active_macrosteps": int(steps.sum()), "workspace_resets": maximum,
                "workspace_initializations": maximum, "horizontal_cell_calls": 2 * maximum,
                "vertical_cell_calls": 2 * maximum, "vectorized_cell_calls": 4 * maximum}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask, steps, nd, xd, widths = self._prepare(input_ids, attention_mask)
        w = int(widths.max().item()); nd, xd = nd[:, :w], xd[:, :w]
        place_mask = torch.arange(w, device=input_ids.device)[None] < widths[:, None]
        probabilities = F.one_hot(xd, 10).to(self.embedding.weight.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7)); initializations = calls = 0
        axis_order = ("H", "V", "H", "V")
        for macro in range(int(steps.max().item())):
            workspace, valid = self._workspace(probabilities, nd, widths); initializations += 1
            for axis in axis_order:
                proposed = self.cell(workspace, axis); calls += 1
                active = steps > macro
                workspace = torch.where(active[:, None, None, None], proposed, workspace)
                workspace = workspace * valid[..., None]
            boundary = workspace[torch.arange(input_ids.shape[0], device=input_ids.device)[:, None],
                                 torch.arange(w, device=input_ids.device)[None].expand(input_ids.shape[0], -1),
                                 widths[:, None].expand(-1, w)]
            digit_logits = self.readout(boundary)
            soft = digit_logits.softmax(-1); hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero_digit = F.one_hot(torch.zeros_like(xd), 10).to(feedback.dtype)
            feedback = torch.where(place_mask[..., None], feedback, zero_digit)
            active = (steps > macro)[:, None, None]
            probabilities = torch.where(active, feedback, probabilities)
            endpoint = torch.where(active, digit_logits, endpoint)
        b, length = input_ids.shape
        logits = endpoint.new_full((b, length, self.vocab_size), -1e4)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = mask.sum(1)[:, None] - 1 - positions
        slot = torch.minimum(slot.clamp_min(0), (widths - 1)[:, None])
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated = logits
        if self.training:
            scale = torch.where(steps.eq(1), logits.new_tensor(1.), logits.new_tensor(.01))
            logits = logits.detach() + scale[:, None, None] * (logits - logits.detach())
        return logits, {"parsed_steps": steps, "widths": widths,
                        "digit_probabilities": probabilities, "ungated_logits": ungated,
                        "workspace_initializations": initializations, "workspace_resets": initializations,
                        "vectorized_cell_calls": calls, "axis_order": axis_order}


def build_model(spec):
    model = Model(spec); assert_model_state(model, spec)
    if sum(p.numel() for p in model.parameters()) != STATE_ELEMENTS:
        raise RuntimeError("compressed axial cellular parameter count drift")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .02},
                                   {"params": no_decay, "weight_decay": 0.}],
                                  lr=8e-4, betas=(.9, .98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / 20))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=512, eval_batch_size=512, max_steps=None)
