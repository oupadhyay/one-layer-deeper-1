"""Local probe: localized learned table over canonical factor quotients."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from benchmark import ModelSpec, OptimizerBundle, Submission, TokenLossBatch, assert_model_state

W, PERIODS, RESIDUES, TABLE_SIZE, FOURIER = 4, 63, 33, 131071, 256


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(TABLE_SIZE, W * 10)
        self.left_phase = nn.Embedding(PERIODS * RESIDUES, 2)
        self.right_phase = nn.Embedding(PERIODS * RESIDUES, 2)
        self.phase_decoder = nn.Linear(1 + 2 * FOURIER, W * 10)
        nn.init.normal_(self.table.weight, std=.02)

    def forward(self, n, x):
        periods = torch.arange(2, 65, device=x.device, dtype=torch.float32)
        n = n.float()
        x = x.float()
        n_remainder = torch.remainder(n[:, None], periods[None])
        factor = n_remainder.eq(0).float()
        x_remainder = torch.remainder(x[:, None], periods[None])
        x_even_remainder = torch.minimum(x_remainder, periods[None] - x_remainder)
        factor_indices = factor.topk(2, dim=-1).indices.sort(dim=-1).values
        residues = x_even_remainder.long().gather(1, factor_indices)
        key = (((factor_indices[:, 0] * PERIODS + factor_indices[:, 1]) * RESIDUES
                + residues[:, 0]) * RESIDUES + residues[:, 1])
        key = torch.remainder(key, TABLE_SIZE)
        left_key = factor_indices[:, 0] * RESIDUES + residues[:, 0]
        right_key = factor_indices[:, 1] * RESIDUES + residues[:, 1]
        left, right = self.left_phase(left_key).float(), self.right_phase(right_key).float()
        phase = torch.stack((left[:, 0] * right[:, 0] - left[:, 1] * right[:, 1],
                             left[:, 0] * right[:, 1] + left[:, 1] * right[:, 0]), -1)
        phase = F.normalize(phase, dim=-1)
        scalar = torch.remainder(torch.atan2(phase[:, 1], phase[:, 0]) / (2 * math.pi), 1.0)
        frequencies = torch.arange(1, FOURIER + 1, device=x.device, dtype=torch.float32)
        angles = 2 * math.pi * scalar[:, None] * frequencies[None]
        features = torch.cat((scalar[:, None], angles.sin(), angles.cos()), -1)
        shared = self.phase_decoder(features.to(self.phase_decoder.weight.dtype)).view(-1, W, 10)
        return self.table(key).view(-1, W, 10), shared, factor, phase


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.transition = Transition()

    @staticmethod
    def parse(ids, mask):
        digit = ids.ge(7) & ids.lt(17) & mask
        marker = ids.eq(2).long() + 2 * ids.eq(3).long() + 3 * ids.eq(4).long()
        region = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        same = region[:, :, None].eq(region[:, None, :]) & region[:, :, None].gt(0)
        place = (same & (index[None, None, :] > index[None, :, None]) & digit[:, None, :]).sum(-1)
        value = (ids - 7).clamp(0, 9)
        steps = (value * torch.pow(ids.new_tensor(10), place) * region.eq(3)).sum(1).long()
        return region, place, steps

    def forward(self, ids, attention_mask=None):
        batch, length = ids.shape
        mask = ids.ne(0) if attention_mask is None else attention_mask.bool()
        region, place, steps = self.parse(ids, mask)
        nw, xw = region.eq(1).sum(1), region.eq(2).sum(1)
        if length > self.config.max_seq_len or bool(((nw < 1) | (nw > W) | (xw < 1) | (xw > W) | (steps > 64)).any()):
            raise ValueError("invalid Medium contract")
        slots = torch.arange(W, device=ids.device)
        value = (ids - 7).clamp(0, 9)

        def field(kind):
            selected = region.eq(kind)[:, :, None] & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, np = field(1)
        x, xp = field(2)
        powers_long = ids.new_tensor((1, 10, 100, 1000))
        powers_float = powers_long.float()
        n_int = (n * powers_long).sum(1)
        probability = F.one_hot(x, 10).to(self.transition.table.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        digit_values = torch.arange(10, device=ids.device, dtype=torch.float32)
        runs = int(steps.max().item())
        resonance = None
        shared_endpoint = None
        predicted_phase = None

        for tick in range(runs):
            x_digits = (probability.float() * digit_values).sum(-1)
            x_int = (x_digits * powers_float).sum(1)
            table_proposed, shared_proposed, resonance, phase_proposal = self.transition(n_int, x_int)
            table_confident = table_proposed.softmax(-1)[:, 0].max(-1).values.ge(.5)
            proposed = table_proposed if self.training else torch.where(
                table_confident[:, None, None], table_proposed, shared_proposed)
            soft = proposed.softmax(-1)
            hard = F.one_hot(soft.argmax(-1), 10).to(soft.dtype)
            feedback = hard - soft.detach() + soft if self.training else hard
            zero = F.one_hot(torch.zeros_like(n), 10).to(feedback.dtype)
            confident = soft.max(-1, keepdim=True).values.ge(.5)
            feedback = torch.where(np[..., None] & confident, feedback, zero)
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, feedback, probability)
            shared_endpoint = torch.where(active, shared_proposed,
                                          shared_endpoint if shared_endpoint is not None else shared_proposed)
            predicted_phase = torch.where(active[:, 0], phase_proposal,
                                          predicted_phase if predicted_phase is not None else phase_proposal)

        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype) * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        shared_placed = torch.bmm(placement.transpose(1, 2), shared_endpoint)
        occupied = placement.sum(1).bool()
        digits = F.pad(placed, (7, self.config.vocab_size - 17), value=-10000)
        shared_digits = F.pad(shared_placed, (7, self.config.vocab_size - 17), value=-10000)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -10000)
        shared_logits = endpoint.new_full((batch, length, self.config.vocab_size), -10000)
        return torch.where(occupied[..., None], digits, logits), {
            "widths": nw, "x_widths": xw, "steps": steps,
            "macrosteps": runs, "resonance": resonance,
            "shared_logits": torch.where(occupied[..., None], shared_digits, shared_logits),
            "predicted_phase": predicted_phase, "modulus": n_int,
        }


def build_model(spec: ModelSpec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model, spec):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.01,
                                  capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 16, 1.0))
    return OptimizerBundle(optimizer, scheduler)


def token_training_loss(batch: TokenLossBatch):
    selected = batch.auxiliary["steps"].eq(1)
    valid = batch.valid_mask[selected]
    if not bool(valid.any()):
        return batch.logits.sum() * 0
    ce = F.cross_entropy(batch.logits[selected].transpose(1, 2), batch.labels[selected],
                         ignore_index=-100, reduction="none")
    classification = (ce * valid).sum() / valid.sum()
    shared = batch.auxiliary["shared_logits"].gather(
        1, batch.target_positions.clamp_min(0)[..., None].expand(-1, -1, batch.logits.shape[-1]))
    shared_ce = F.cross_entropy(shared[selected].transpose(1, 2), batch.labels[selected],
                                ignore_index=-100, reduction="none")
    labels = (batch.labels[selected] - 7).clamp(0, 9)
    places = valid.long().flip(1).cumsum(1).flip(1) - 1
    target_value = (labels * torch.pow(labels.new_tensor(10), places.clamp_min(0)) * valid).sum(1)
    target_angle = 2 * math.pi * target_value.float() / batch.auxiliary["modulus"][selected].float()
    target_phase = torch.stack((target_angle.cos(), target_angle.sin()), -1)
    phase_loss = F.mse_loss(batch.auxiliary["predicted_phase"][selected].float(), target_phase)
    return classification + (shared_ce * valid).sum() / valid.sum() + 10 * phase_loss


SUBMISSION = Submission(build_model, build_optimizer, batch_size=128, eval_batch_size=256,
                        max_steps=50000, token_training_loss=token_training_loss)
