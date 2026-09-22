"""Width-independent reversible state-space coupling with a T1 gradient gate."""
import math
import torch
from torch import nn
from benchmark import Submission, assert_model_state
import competition_submission_c1a as c1a

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D_MODEL, HALF, MAX_STEPS = 32, 16, 64


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


def fixed_place_features(width, device, dtype):
    place = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / D_MODEL))
    result = torch.empty(width, D_MODEL, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(place * frequency)
    result[:, 1::2] = torch.cos(place * frequency)
    return result.to(dtype)


class StateSpaceMixer(nn.Module):
    """Per-place projection followed by a symmetric learned exponential kernel."""
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, HALF))
        self.decay_logit = nn.Parameter(torch.zeros(HALF))

    def kernel(self, width, active, dtype):
        positions = torch.arange(width, device=active.device)
        distance = (positions[:, None] - positions[None, :]).abs().to(dtype)
        decay = torch.sigmoid(self.decay_logit).to(dtype)
        values = decay[None, None, None, :].pow(distance[None, :, :, None])
        valid = active[:, :, None] & active[:, None, :]
        values = values * valid[:, :, :, None]
        return values / values.sum(2, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    def forward(self, state, context, active):
        projected = self.project(torch.cat((state, context), -1))
        return torch.einsum("bijc,bjc->bic", self.kernel(state.shape[1], active, state.dtype), projected)


class ReversibleTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = StateSpaceMixer()
        self.G = StateSpaceMixer()
        initial = math.atanh(.1)
        self.raw_scale_f = nn.Parameter(torch.tensor(initial))
        self.raw_scale_g = nn.Parameter(torch.tensor(initial))

    def forward(self, state, context, width_mask):
        xa, xb = state.chunk(2, -1)
        ca, cb = context.chunk(2, -1)
        ya = xa + torch.tanh(self.raw_scale_f) * self.F(xb, cb, width_mask)
        yb = xb + torch.tanh(self.raw_scale_g) * self.G(ya, ca, width_mask)
        return torch.cat((ya, yb), -1)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len, self.vocab_size = spec.max_seq_len, spec.vocab_size
        self.embedding = nn.Embedding(10, D_MODEL)
        self.transition = ReversibleTransition()
        self.readout = nn.Linear(D_MODEL, 10)

    @staticmethod
    def parse(input_ids, mask):
        is_digit = (input_ids >= DIGIT) & mask
        marker = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(marker.long(), 1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        places = ((role[:, :, None] == role[:, None, :]) &
                  (index[None, None] > index[None, :, None]) & is_digit[:, None]).sum(2)
        t_digits = is_digit & (role == 3)
        steps = (((input_ids - DIGIT) * input_ids.new_tensor(10).pow(places)) * t_digits).sum(1)
        return role, places, steps

    def _prepare(self, input_ids, attention_mask=None):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        role, places, steps = self.parse(input_ids, mask)
        slots = torch.arange(self.max_seq_len, device=input_ids.device)
        values = (input_ids - DIGIT).clamp(0, 9)
        def digits(field):
            assignment = (role[:, :, None] == field) & (places[:, :, None] == slots)
            return torch.einsum("bls,bl->bs", assignment.long(), values).long(), assignment.any(1)
        n_digits, present = digits(1)
        x_digits, _ = digits(2)
        width_mask = slots[None] < present.sum(1)[:, None]
        return mask, steps, n_digits, x_digits, width_mask

    def debug_execution(self, input_ids, attention_mask=None):
        steps = self._prepare(input_ids, attention_mask)[1].clamp_max(MAX_STEPS)
        return {"parsed_steps": steps, "active_updates": steps}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask, steps, n_digits, x_digits, width_mask = self._prepare(input_ids, attention_mask)
        n_digits = torch.where(width_mask, n_digits, torch.zeros_like(n_digits))
        x_digits = torch.where(width_mask, x_digits, torch.zeros_like(x_digits))
        place = fixed_place_features(self.max_seq_len, input_ids.device, self.embedding.weight.dtype)[None]
        context = self.embedding(n_digits) + place
        initial_state = self.embedding(x_digits) + place
        canonical_zero = self.embedding(torch.zeros_like(x_digits)) + place
        state = torch.where(width_mask[:, :, None], initial_state, canonical_zero)
        executed = int(steps.max().clamp_max(MAX_STEPS).item())
        for iteration in range(executed):
            proposed = self.transition(state, context, width_mask)
            proposed = torch.where(width_mask[:, :, None], proposed, canonical_zero)
            state = torch.where((steps > iteration)[:, None, None], proposed, state)
        endpoint = self.readout(state)
        batch, length = input_ids.shape
        logits = endpoint.new_full((batch, length, self.vocab_size), -1e4)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = mask.sum(1)[:, None] - 1 - positions
        selected = endpoint.gather(1, slot.clamp(0, self.max_seq_len - 1)[:, :, None].expand(-1, -1, 10))
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated = logits
        if self.training:
            gate = torch.where(steps == 1, logits.new_tensor(1.), logits.new_tensor(.01))
            logits = logits.detach() + gate[:, None, None] * (logits - logits.detach())
        return logits, {"ungated_logits": ungated, "state": state, "context": context,
                        "context_digits": n_digits, "initial_state": initial_state,
                        "initial_state_digits": x_digits, "width_mask": width_mask,
                        "place_features": place.squeeze(0), "parsed_steps": steps,
                        "active_updates": steps.clamp_max(MAX_STEPS),
                        "executed_macrosteps": steps.new_tensor(executed)}


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    count = sum(value.numel() for value in model.state_dict().values())
    if count != 3852:
        raise RuntimeError(f"reversible SSM state count drift: {count}")
    if count >= 15_000:
        raise RuntimeError("reversible SSM exceeds state budget")
    return model


build_optimizer = c1a.build_optimizer
SUBMISSION = Submission(build_model, build_optimizer, batch_size=64, eval_batch_size=128, max_steps=1600)
