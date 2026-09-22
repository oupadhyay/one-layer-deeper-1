"""Authorized x2 model with a training-only T1 backward-gradient gate."""

import torch

import competition_loop3_latent_x2 as parent


class Model(parent.Model):
    """Preserve x2's forward values while attenuating non-T1 logit gradients."""

    def forward(self, input_ids, attention_mask=None):
        logits, auxiliary = super().forward(input_ids, attention_mask)
        auxiliary["ungated_logits"] = logits
        if not self.training:
            return logits, auxiliary

        # Recombination is exactly value preserving.  Only the derivative through
        # logits is changed: T1 receives 1 and every other parsed horizon 0.01.
        gate = torch.where(auxiliary["parsed_steps"].eq(1), 1.0, 0.01)
        gate = gate.to(device=logits.device, dtype=logits.dtype)[:, None, None]
        detached = logits.detach()
        return detached + gate * (logits - detached), auxiliary


def build_model(spec):
    model = Model(spec)
    parent.parent.assert_model_state(model, spec)
    return model


build_optimizer = parent.build_optimizer
SUBMISSION = parent.parent.Submission(build_model, build_optimizer, batch_size=64,
                                      eval_batch_size=128, max_steps=200)
