"""Dynamic-width axial workspace with generic bilinear digit-pair features."""
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS, CHANNELS, STATE_ELEMENTS = 64, 64, 63146

class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len

class AxialLocalCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(CHANNELS)
        self.gates = nn.Linear(3 * CHANNELS, 2 * CHANNELS)
        self.candidate = nn.Linear(3 * CHANNELS, CHANNELS)

    def forward(self, workspace, axis):
        center = self.norm(workspace)
        if axis == "H":
            left, right = F.pad(center[:, :, :-1], (0, 0, 1, 0)), F.pad(center[:, :, 1:], (0, 0, 0, 1))
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
        self.initializer = nn.Sequential(nn.Linear(118, CHANNELS), nn.GELU(), nn.Linear(CHANNELS, CHANNELS))
        self.cell = AxialLocalCell()
        self.readout = nn.Linear(CHANNELS, 10)
        self.boundary_norm = nn.RMSNorm(CHANNELS)
        self.pair_norm = nn.RMSNorm(CHANNELS)
        self.attention_q = nn.Linear(CHANNELS, CHANNELS, bias=False)
        self.attention_kv = nn.Linear(CHANNELS, 2 * CHANNELS, bias=False)
        self.attention_out = nn.Linear(CHANNELS, CHANNELS, bias=False)
        nn.init.zeros_(self.attention_out.weight)
        self.global_norm = nn.RMSNorm(CHANNELS)
        self.global_out = nn.Linear(CHANNELS, CHANNELS, bias=False)
        self.global_gate = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def parse(input_ids, mask):
        is_digit = (input_ids >= DIGIT) & mask
        markers = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(markers.long(), 1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None] == role[:, None, :]
        places = (same & (index[None, None] > index[None, :, None]) & is_digit[:, None]).sum(2)
        td = is_digit & role.eq(3)
        powers = torch.pow(input_ids.new_tensor(10), places)
        steps = (((input_ids - DIGIT) * powers) * td).sum(1)
        return role, places, steps.clamp(max=MAX_STEPS)

    def _prepare(self, ids, attention_mask):
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, places, steps = self.parse(ids, mask)
        slots = torch.arange(self.max_seq_len, device=ids.device)
        values = (ids - DIGIT).clamp(0, 9)
        def field(which):
            assignment = role[:, :, None].eq(which) & places[:, :, None].eq(slots)
            digits = (assignment.to(values.dtype) * values[:, :, None]).sum(1)
            return digits.long(), assignment.any(1)
        nd, present = field(1)
        xd, _ = field(2)
        return mask, steps, nd, xd, present.sum(1)

    def _workspace(self, probabilities, n_digits, widths):
        batch, width = probabilities.shape[:2]
        columns = width + 1
        state, n_state = probabilities @ self.embedding.weight, self.embedding(n_digits)
        ei = state[:, :, None].expand(-1, -1, columns, -1)
        ni = n_state[:, :, None].expand_as(ei)
        ej, nj = state[:, None, :].expand(-1, width, -1, -1), n_state[:, None, :].expand(-1, width, -1, -1)
        zero = state.new_zeros((batch, width, 1, 16))
        ej, nj = torch.cat((ej, zero), 2), torch.cat((nj, zero), 2)
        rows = torch.arange(width, device=state.device)[None, :, None]
        cols = torch.arange(columns, device=state.device)[None, None, :]
        actual_width = widths[:, None, None]
        valid_row, pair, boundary = rows < actual_width, cols < actual_width, cols == actual_width
        boundary_cells = boundary.expand(-1, width, -1)[..., None]
        ej, nj = torch.where(boundary_cells, torch.zeros_like(ej), ej), torch.where(boundary_cells, torch.zeros_like(nj), nj)
        flags = torch.stack((valid_row.expand(-1, -1, columns), pair.expand(-1, width, -1),
            boundary.expand(-1, width, -1), (rows == 0).expand(batch, -1, columns),
            (cols == 0).expand(batch, width, -1), (rows == actual_width - 1).expand(-1, -1, columns)), -1).to(state.dtype)
        valid = valid_row & (pair | boundary)
        products = (ei * ej, ei * nj, ni * nj)
        workspace = self.initializer(torch.cat((ei, ej, ni, nj, *products, flags), -1))
        return workspace * valid[..., None], valid

    @staticmethod
    def _boundary(workspace, widths):
        index = widths[:, None, None, None].expand(-1, workspace.shape[1], 1, workspace.shape[3])
        return workspace.gather(2, index).squeeze(2)

    def _boundary_attention(self, workspace, widths):
        boundary = self._boundary(workspace, widths)
        rows = torch.arange(workspace.shape[1], device=workspace.device)[None, :, None]
        cols = torch.arange(workspace.shape[2], device=workspace.device)[None, None, :]
        pair_mask = ((rows < widths[:, None, None]) & (cols < widths[:, None, None])).flatten(1)
        pairs = workspace.flatten(1, 2)
        query = self.attention_q(self.boundary_norm(boundary))
        key, value = self.attention_kv(self.pair_norm(pairs)).chunk(2, -1)
        scores = torch.matmul(query, key.transpose(1, 2)) * (CHANNELS ** -0.5)
        scores = scores.masked_fill(~pair_mask[:, None, :], float("-inf"))
        weights = scores.softmax(-1)
        return boundary + self.attention_out(torch.matmul(weights, value)), pair_mask, weights

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len: raise ValueError("sequence exceeds max_seq_len")
        mask, steps, nd, xd, widths = self._prepare(input_ids, attention_mask)
        width = int(widths.max().item())
        nd, xd = nd[:, :width], xd[:, :width]
        place_mask = torch.arange(width, device=input_ids.device)[None] < widths[:, None]
        probabilities = F.one_hot(xd, 10).to(self.embedding.weight.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7))
        initializations = calls = 0
        axis_order = ("H", "V", "H", "V")
        for macro in range(int(steps.max().item())):
            workspace, valid = self._workspace(probabilities, nd, widths); initializations += 1
            for axis in axis_order:
                proposed = self.cell(workspace, axis); calls += 1
                workspace = torch.where((steps > macro)[:, None, None, None], proposed, workspace) * valid[..., None]
            boundary, _, _ = self._boundary_attention(workspace, widths)
            digit_logits = self.readout(boundary)
            soft = digit_logits.softmax(-1); hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero_digit = F.one_hot(torch.zeros_like(xd), 10).to(feedback.dtype)
            feedback = torch.where(place_mask[..., None], feedback, zero_digit)
            active = (steps > macro)[:, None, None]
            probabilities, endpoint = torch.where(active, feedback, probabilities), torch.where(active, digit_logits, endpoint)
        batch, length = input_ids.shape
        logits = endpoint.new_full((batch, length, self.vocab_size), -1e4)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = torch.minimum((mask.sum(1)[:, None] - 1 - positions).clamp_min(0), (widths - 1)[:, None])
        logits[:, :, DIGIT:DIGIT + 10] = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        ungated = logits
        if self.training:
            scale = torch.where(steps.eq(1), logits.new_tensor(1.0), logits.new_tensor(0.0))
            logits = logits.detach() + scale[:, None, None] * (logits - logits.detach())
        return logits, {"parsed_steps": steps, "widths": widths, "digit_probabilities": probabilities,
            "ungated_logits": ungated, "workspace_initializations": initializations,
            "workspace_resets": initializations, "vectorized_cell_calls": calls, "axis_order": axis_order}

def build_model(spec):
    model = Model(spec); assert_model_state(model, spec)
    return model

def build_optimizer(model, spec):
    return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, capturable=spec.device_type == "cuda"))

SUBMISSION = Submission(build_model, build_optimizer, batch_size=128, eval_batch_size=256, max_steps=None)
