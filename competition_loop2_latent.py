"""Loop-2 continuous latent-state research candidate (not a frozen submission)."""

import torch
from torch import nn

from benchmark import Submission, assert_model_state
import competition_submission as c0
import competition_submission_c1b as c1b

PAD, BOS, N, X, T, ANS, EOS, DIGIT = 0, 1, 2, 3, 4, 5, 6, 7
MAX_STEPS = 64


class LatentTransition(nn.Module):
    """Tied global slot/channel mixer with an immutable-N context residual."""

    def __init__(self, d_model, slots):
        super().__init__()
        self.slot_norm = nn.LayerNorm(d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.slot_in = nn.Linear(slots, 2 * slots)
        self.slot_out = nn.Linear(2 * slots, slots)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, 3 * d_model), nn.GELU(), nn.Linear(3 * d_model, d_model)
        )
        self.context_residual = nn.Linear(d_model, d_model)

    def forward(self, state, context, context_mask):
        mixed = self.slot_norm(state).transpose(1, 2)
        mixed = self.slot_out(torch.nn.functional.gelu(self.slot_in(mixed))).transpose(1, 2)
        pooled = (context * context_mask[..., None]).sum(1)
        pooled = pooled / context_mask.sum(1, keepdim=True).clamp_min(1).to(context.dtype)
        state = state + mixed
        return state + self.channel_mlp(self.channel_norm(state)) + self.context_residual(pooled)[:, None]


class Model(c0.Model):
    """Parse once, carry continuous latent slots, and read decimal digits once."""

    def __init__(self, spec, d_model=112, heads=4):
        super().__init__(spec, d_model=d_model, heads=heads)
        # ``heads`` remains accepted for screen/config compatibility.
        del self.transition
        self.transition = LatentTransition(d_model, spec.max_seq_len)
        self.endpoint_readout = nn.Linear(d_model, 10)

    def _prepare_latent(self, input_ids, attention_mask):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.to(torch.bool)
        role, places, steps = self.parse(input_ids, mask)
        control = (input_ids == T) | (role == 3) | (input_ids == X) | (role == 2)
        clean = torch.where(control, torch.zeros_like(input_ids), input_ids)
        context_mask = mask & ~control
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        context = self.prompt_encoder(
            self.token_embedding(clean) + self.position_embedding(positions)[None],
            src_key_padding_mask=~context_mask,
        )
        slots = torch.arange(self.config.max_seq_len, device=input_ids.device)
        assignment = (role[:, :, None] == 2) & (places[:, :, None] == slots[None, None, :])
        values = (input_ids - DIGIT).clamp(0, 9)
        selected = torch.einsum("bls,bl->bs", assignment.to(context.dtype), values.to(context.dtype))
        selected = torch.where(assignment.any(1), selected, torch.zeros_like(selected)).long()
        state = self.digit_embedding[selected] + self.place_embedding.weight[None] + self.role_embedding
        return mask, context_mask, context, state, steps

    def debug_execution(self, input_ids, attention_mask=None):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.to(torch.bool)
        steps = self.parse(input_ids, mask)[2]
        active = torch.arange(MAX_STEPS, device=input_ids.device)[None] < steps.clamp_max(MAX_STEPS)[:, None]
        return {"parsed_steps": steps, "active_updates": active.sum(1), "active_mask": active}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask, context_mask, context, state, steps = self._prepare_latent(input_ids, attention_mask)
        executed_steps = int(steps.max().clamp_max(MAX_STEPS).item())
        for iteration in range(executed_steps):
            candidate = self.transition(state, context, context_mask)
            state = torch.where((steps > iteration)[:, None, None], candidate, state)
        endpoint_logits = self.endpoint_readout(state)
        batch, length = input_ids.shape
        logits = endpoint_logits.new_full((batch, length, self.config.vocab_size), -1e4)
        valid_length = mask.sum(1)
        positions = torch.arange(length, device=input_ids.device)[None]
        slot = valid_length[:, None] - 1 - positions
        gather = endpoint_logits.gather(
            1, slot.clamp(0, self.config.max_seq_len - 1)[:, :, None].expand(-1, -1, 10)
        )
        logits[:, :, DIGIT:DIGIT + 10] = gather
        return logits, {
            "parsed_steps": steps,
            "active_updates": steps.clamp_max(MAX_STEPS),
            "executed_macrosteps": steps.new_tensor(executed_steps),
            "latent_state": state,
        }


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    return model


# Identity reuse: C1B's AdamW and exact 20-update warmup.
build_optimizer = c1b.SUBMISSION.build_optimizer

SUBMISSION = Submission(build_model, build_optimizer, batch_size=64, eval_batch_size=128, max_steps=200)
