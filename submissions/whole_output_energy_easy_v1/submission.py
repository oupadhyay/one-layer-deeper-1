"""WOEL-4: whole-output energy model for Easy E5 (one to four digits)."""

import math
import torch
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_PLACES, MAX_STEPS, STATE_ELEMENTS = 4, 64, 220_832
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len
        self.input_digit_embedding = nn.Embedding(10, 48)
        self.input_place_embedding = nn.Embedding(4, 48)
        self.role_embedding = nn.Embedding(2, 48)
        self.context_norm = nn.RMSNorm(384)
        self.context_mlp = nn.Sequential(nn.Linear(384, 192), nn.GELU(), nn.Linear(192, 160))
        self.state_norm = nn.RMSNorm(160)
        self.unary_head = nn.Linear(160, 40)
        self.pair_head = nn.Linear(160, 600)
        self.global_head = nn.Linear(160, 32)
        self.candidate_digit_embedding = nn.Embedding(10, 16)
        self.candidate_place_embedding = nn.Embedding(4, 16)
        self.candidate_norm = nn.RMSNorm(64)
        self.candidate_mlp = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        numbers = torch.arange(10_000)
        catalog = torch.stack(tuple((numbers // (10 ** p)) % 10 for p in range(4)), 1)
        self.register_buffer("catalog", catalog, persistent=False)

    @staticmethod
    def parse(input_ids, mask):
        digit = (input_ids >= DIGIT) & (input_ids < DIGIT + 10) & mask
        marker = ((input_ids == N) | (input_ids == X) | (input_ids == T) |
                  (input_ids == ANS)) & mask
        role = torch.cumsum(marker.long(), 1) * digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None].eq(role[:, None, :])
        place = (same & (index[None, None] > index[None, :, None]) & digit[:, None]).sum(2)
        values = (input_ids - DIGIT).clamp(0, 9)
        slots = torch.arange(MAX_PLACES, device=input_ids.device)

        def field(which):
            assignment = role[:, :, None].eq(which) & place[:, :, None].eq(slots)
            return (assignment * values[:, :, None]).sum(1).long(), assignment.any(1)

        nd, nmask = field(1)
        xd, xmask = field(2)
        t_assignment = role.eq(3) & digit
        # T is ordinary most-significant-first decimal text; place is LSD-first.
        steps = ((values * (10 ** place)) * t_assignment).sum(1).clamp(1, MAX_STEPS)
        return nd, xd, nmask, xmask, steps

    def prepare(self, ids, attention_mask=None):
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        nd, xd, nmask, xmask, steps = self.parse(ids, mask)
        is_digit = (ids >= DIGIT) & (ids < DIGIT + 10) & mask
        markers = ((ids == N) | (ids == X) | (ids == T) | (ids == ANS)) & mask
        roles = torch.cumsum(markers.long(), 1)
        raw_n_width = (is_digit & roles.eq(1)).sum(1)
        raw_x_width = (is_digit & roles.eq(2)).sum(1)
        widths = nmask.sum(1)
        if not bool(((raw_n_width >= 1) & (raw_n_width <= 4)).all()):
            raise ValueError("modulus width must be in 1..4")
        if not bool((raw_x_width <= 4).all()):
            raise ValueError("input width must be at most 4")
        return mask, nd, xd, nmask, steps, widths

    def encode(self, nd, xd, active):
        place = self.input_place_embedding.weight[None]
        n = self.input_digit_embedding(nd) + place + self.role_embedding.weight[0]
        x = self.input_digit_embedding(xd) + place + self.role_embedding.weight[1]
        n = n * active[:, :, None]
        x = x * active[:, :, None]
        flat = torch.cat((n, x), 1).flatten(1)
        return self.state_norm(self.context_mlp(self.context_norm(flat)))

    def candidate_features(self):
        digits = self.candidate_digit_embedding(self.catalog)
        places = self.candidate_place_embedding.weight[None]
        flat = (digits + places).flatten(1)
        return self.candidate_mlp(self.candidate_norm(flat))

    def score_candidates(self, state, widths, candidate_features=None):
        # Explicit FP32 evaluation makes catalog scores and marginals exact FP32.
        state = state.float()
        unary = self.unary_head(state).reshape(-1, 4, 10)
        pair = self.pair_head(state).reshape(-1, 6, 10, 10)
        global_vector = self.global_head(state)
        catalog = self.catalog
        active = torch.arange(4, device=state.device)[None] < widths[:, None]
        scores = state.new_zeros(state.shape[0], 10_000)
        for p in range(4):
            scores = scores + unary[:, p].gather(1, catalog[:, p][None].expand(state.shape[0], -1)) * active[:, p, None]
        scores = scores / widths.float().sqrt()[:, None]
        pair_sum = torch.zeros_like(scores)
        pair_count = widths * (widths - 1) // 2
        for k, (i, j) in enumerate(PAIRS):
            chosen = pair[:, k][:, catalog[:, i], catalog[:, j]]
            pair_sum = pair_sum + chosen * ((widths > j)[:, None])
        scores = scores + pair_sum / pair_count.clamp_min(1).float().sqrt()[:, None]
        if candidate_features is None:
            candidate_features = self.candidate_features()
        scores = scores + global_vector @ candidate_features.float().T / math.sqrt(32)
        valid = (catalog[None] * (~active)[:, :, None].transpose(1, 2)).sum(2).eq(0)
        return scores.masked_fill(~valid, -torch.inf)

    def marginals(self, scores):
        grid = scores.reshape(-1, 10, 10, 10, 10)
        result = []
        for p in range(4):
            retained_axis = 4 - p
            reduced_axes = tuple(axis for axis in range(1, 5) if axis != retained_axis)
            result.append(torch.logsumexp(grid, reduced_axes))
        return torch.stack(result, 1)

    def transition(self, nd, residue, active, widths, candidate_features):
        state = self.encode(nd, residue, active)
        scores = self.score_candidates(state, widths, candidate_features)
        return self.marginals(scores), scores

    def forward(self, input_ids, attention_mask=None):
        mask, nd, residue, nmask, steps, widths = self.prepare(input_ids, attention_mask)
        endpoint = self.input_digit_embedding.weight.new_zeros(input_ids.shape[0], 4, 10)
        candidate_features = self.candidate_features()
        for macro in range(int(steps.max().item())):
            candidate_logits, _ = self.transition(
                nd, residue, nmask, widths, candidate_features
            )
            hard = torch.nn.functional.one_hot(candidate_logits.argmax(-1), 10).to(candidate_logits.dtype).detach()
            candidate = hard.argmax(-1)
            active_step = steps > macro
            residue = torch.where(active_step[:, None], candidate, residue)
            endpoint = torch.where(active_step[:, None, None], candidate_logits, endpoint)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        slot = (mask.sum(1)[:, None] - 1 - positions).clamp_min(0)
        slot = torch.minimum(slot, (widths - 1)[:, None])
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits = endpoint.new_full((*input_ids.shape, self.vocab_size), -1e4)
        logits[:, :, DIGIT:DIGIT + 10] = selected
        gate = steps.eq(1).to(logits.dtype)[:, None, None]
        logits = logits.detach() + gate * (logits - logits.detach())
        return logits, {"steps": steps, "widths": widths, "macrosteps": int(steps.max().item())}


def build_model(spec):
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": .01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=6e-4, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 32, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None)
