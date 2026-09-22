"""Width-independent digit-pair cellular candidate (single shared local cell)."""

import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle, Submission, TokenLossBatch, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS = 64
STATE_ELEMENTS = 20906


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class LocalCell(nn.Module):
    """A strictly radius-one, shared ConvGRU-like update."""
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(48)
        self.depthwise = nn.Conv2d(48, 48, 3, padding=1, groups=48)
        self.gates = nn.Conv2d(96, 96, 1)
        self.candidate = nn.Conv2d(96, 48, 1)

    def forward(self, workspace):
        x = self.norm(workspace).permute(0, 3, 1, 2)
        local = self.depthwise(x)
        reset, update = self.gates(torch.cat((x, local), 1)).chunk(2, 1)
        reset, update = reset.sigmoid(), update.sigmoid()
        proposed = torch.tanh(self.candidate(torch.cat((x, reset * local), 1)))
        return ((1 - update) * x + update * proposed).permute(0, 2, 3, 1)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len, self.vocab_size = spec.max_seq_len, spec.vocab_size
        self.embedding = nn.Embedding(10, 16)
        self.initializer = nn.Sequential(nn.Linear(70, 48), nn.GELU(), nn.Linear(48, 48))
        self.cell = LocalCell()
        self.readout = nn.Linear(48, 10)

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
            digits = (assignment.long() * values[:, :, None]).sum(1)
            return digits.long(), assignment.any(1)
        nd, present = field(1); xd, _ = field(2)
        widths = present.sum(1)
        return mask, steps, nd, xd, widths

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
        flags = torch.stack((valid_row.expand(-1,-1,columns), pair.expand(-1,w,-1),
                             boundary.expand(-1,w,-1), (rows==0).expand(b,-1,columns),
                             (cols==0).expand(b,w,-1), (rows==width-1).expand(-1,-1,columns)), -1).to(state.dtype)
        valid = valid_row & (pair | boundary)
        workspace = self.initializer(torch.cat((ei, ej, ni, nj, flags), -1))
        return workspace * valid[..., None], valid

    def debug_execution(self, input_ids, attention_mask=None):
        _, steps, _, _, widths = self._prepare(input_ids, attention_mask)
        return {"parsed_steps": steps, "widths": widths, "workspace_initializations": int(steps.max()),
                "vectorized_cell_calls": int(steps.max()) * 2 * int(widths.max()),
                "active_row_cell_calls": 2 * widths * steps}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len: raise ValueError("sequence exceeds max_seq_len")
        mask, steps, nd, xd, widths = self._prepare(input_ids, attention_mask)
        w = int(widths.max().item()); nd, xd = nd[:, :w], xd[:, :w]
        place_mask = torch.arange(w, device=input_ids.device)[None] < widths[:, None]
        probabilities = F.one_hot(xd, 10).to(self.embedding.weight.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7)); initializations = calls = 0
        for macro in range(int(steps.max().item())):
            workspace, valid = self._workspace(probabilities, nd, widths); initializations += 1
            for micro in range(2 * w):
                proposed = self.cell(workspace); calls += 1
                active = (steps > macro) & (2 * widths > micro)
                workspace = torch.where(active[:,None,None,None], proposed, workspace) * valid[...,None]
            boundary = workspace[torch.arange(input_ids.shape[0], device=input_ids.device)[:, None],
                                 torch.arange(w, device=input_ids.device)[None].expand(input_ids.shape[0],-1),
                                 widths[:,None].expand(-1,w)]
            digit_logits = self.readout(boundary)
            soft = digit_logits.softmax(-1); hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            feedback = torch.where(place_mask[...,None], feedback, F.one_hot(torch.zeros_like(xd),10).to(feedback.dtype))
            active = (steps > macro)[:,None,None]
            probabilities = torch.where(active, feedback, probabilities)
            endpoint = torch.where(active, digit_logits, endpoint)
        b, length = input_ids.shape
        logits = endpoint.new_full((b, length, self.vocab_size), -1e4)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = mask.sum(1)[:,None] - 1 - positions
        slot = torch.minimum(slot.clamp_min(0), (widths - 1)[:, None])
        selected = endpoint.gather(1, slot[...,None].expand(-1,-1,10))
        logits[:,:,DIGIT:DIGIT+10] = selected
        return logits, {"parsed_steps":steps, "widths":widths, "digit_probabilities":probabilities,
                        "workspace_initializations":initializations,
                        "vectorized_cell_calls":calls, "active_row_cell_calls":2*widths*steps}


def build_model(spec):
    model = Model(spec); assert_model_state(model, spec)
    if sum(p.numel() for p in model.parameters()) != STATE_ELEMENTS:
        raise RuntimeError("digit-pair cellular parameter count drift")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for parameter in model.parameters(): (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params":decay,"weight_decay":.02},{"params":no_decay,"weight_decay":0.}],
                                  lr=4e-4, betas=(.9,.98), capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:min(1.,(step+1)/20))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    ce = F.cross_entropy(batch.logits.transpose(1, 2), batch.labels,
                         ignore_index=-100, reduction="none")
    valid = batch.valid_mask
    sequence = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    weight = torch.where(batch.auxiliary["parsed_steps"].eq(1),
                         sequence.new_tensor(4.), sequence.new_tensor(1.))
    return (sequence * weight).sum() / weight.sum()


SUBMISSION = Submission(build_model, build_optimizer, batch_size=64, eval_batch_size=128,
                        max_steps=400, token_training_loss=token_training_loss)
