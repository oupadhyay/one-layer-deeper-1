"""Width-independent relative-place attention state machine with a T1 gradient gate."""

import math
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import Submission, assert_model_state
import competition_submission_c1a as c1a

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS = 64
D_MODEL = 24


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


def fixed_place_features(width, device, dtype):
    """Generic sinusoidal features; prefixes do not depend on requested width."""
    place = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / D_MODEL))
    result = torch.empty(width, D_MODEL, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(place * frequency)
    result[:, 1::2] = torch.cos(place * frequency)
    return result.to(dtype)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = nn.MultiheadAttention(D_MODEL, 4, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 48), nn.GELU(), nn.Linear(48, D_MODEL))

    def forward(self, tokens, padding_mask):
        normalized = self.norm1(tokens)
        attended = self.attention(normalized, normalized, normalized,
                                  key_padding_mask=padding_mask, need_weights=False)[0]
        tokens = tokens + attended
        return tokens + self.ff(self.norm2(tokens))


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.embedding = nn.Embedding(10, D_MODEL)
        self.block = AttentionBlock()
        self.readout = nn.Linear(D_MODEL, 10)

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
        def field_digits(field):
            assignment = (role[:, :, None] == field) & (places[:, :, None] == slots)
            selected = torch.einsum("bls,bl->bs", assignment.long(), values)
            return torch.where(assignment.any(1), selected, torch.zeros_like(selected)).long(), assignment.any(1)
        n_digits, n_present = field_digits(1)
        x_digits, _ = field_digits(2)
        width_mask = slots[None] < n_present.sum(1)[:, None]
        return mask, steps, n_digits, x_digits, width_mask

    def debug_execution(self, input_ids, attention_mask=None):
        _, steps, _, _, _ = self._prepare(input_ids, attention_mask)
        active = torch.arange(MAX_STEPS, device=input_ids.device)[None] < steps[:, None]
        return {"parsed_steps": steps, "active_updates": active.sum(1), "active_mask": active}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask, steps, n_digits, x_digits, width_mask = self._prepare(input_ids, attention_mask)
        zero = torch.zeros_like(x_digits)
        state_digits = torch.where(width_mask, x_digits, zero)
        n_digits = torch.where(width_mask, n_digits, zero)
        probabilities = F.one_hot(state_digits, 10).to(self.embedding.weight.dtype)
        endpoint = torch.log(probabilities.clamp_min(1e-7))
        places = fixed_place_features(self.max_seq_len, input_ids.device, self.embedding.weight.dtype)[None]
        # Parameter-free role separation in two otherwise aligned place streams.
        role = places.new_zeros((1, 1, D_MODEL)); role[..., 0] = 1
        context = self.embedding(n_digits) + places - role
        padding = torch.cat((~width_mask, ~width_mask), 1)
        executed = int(steps.max().clamp_max(MAX_STEPS).item())
        for iteration in range(executed):
            state = probabilities @ self.embedding.weight
            tokens = torch.cat((state + places + role, context), 1)
            hidden = self.block(tokens, padding)[:, :self.max_seq_len]
            digit_logits = self.readout(hidden)
            feedback = (torch.softmax(digit_logits, -1) if self.training else
                        F.one_hot(digit_logits.argmax(-1), 10).to(digit_logits.dtype))
            canonical_zero = F.one_hot(zero, 10).to(feedback.dtype)
            feedback = torch.where(width_mask[:, :, None], feedback, canonical_zero)
            active = (steps > iteration)[:, None, None]
            probabilities = torch.where(active, feedback, probabilities)
            endpoint = torch.where(active, digit_logits, endpoint)
        b, length = input_ids.shape
        logits = endpoint.new_full((b, length, self.vocab_size), -1e4)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = mask.sum(1)[:, None] - 1 - positions
        selected = endpoint.gather(1, slot.clamp(0, self.max_seq_len - 1)[:, :, None].expand(-1, -1, 10))
        logits[:, :, DIGIT:DIGIT + 10] = selected
        ungated_logits = logits
        if self.training:
            scale = torch.where(steps == 1, logits.new_tensor(1.), logits.new_tensor(.01))
            logits = logits.detach() + scale[:, None, None] * (logits - logits.detach())
        return logits, {"parsed_steps": steps, "active_updates": steps.clamp_max(MAX_STEPS),
                        "executed_macrosteps": steps.new_tensor(executed),
                        "digit_probabilities": probabilities, "ungated_logits": ungated_logits,
                        "initial_state_digits": state_digits, "context_digits": n_digits,
                        "width_mask": width_mask, "place_features": places.squeeze(0)}


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    count = sum(p.numel() for p in model.parameters())
    if count != 5362:
        raise RuntimeError(f"relative-place attention parameter count drift: {count}")
    if count > 10_000:
        raise RuntimeError("relative-place attention exceeds state budget")
    return model


build_optimizer = c1a.build_optimizer
SUBMISSION = Submission(build_model, build_optimizer, batch_size=64, eval_batch_size=128, max_steps=1600)
