"""Authorized x2 inner-refinement variant of the frozen loop-2 latent model."""

import competition_loop2_latent as parent


class LatentTransitionX2(parent.LatentTransition):
    """Apply the parent's exact, tied transition twice without adding state."""

    refine = parent.LatentTransition.forward

    def forward(self, state, context, context_mask):
        h1 = self.refine(state, context, context_mask)
        h2 = self.refine(h1, context, context_mask)
        return h2


class Model(parent.Model):
    """Loop-2 latent model with two continuous refinements per macrostep."""

    def __init__(self, spec, d_model=112, heads=4):
        super().__init__(spec, d_model=d_model, heads=heads)
        # Preserve the already initialized object, parameters, registration, and state keys.
        self.transition.__class__ = LatentTransitionX2

    def debug_execution(self, input_ids, attention_mask=None):
        result = super().debug_execution(input_ids, attention_mask)
        result["active_refinements"] = 2 * result["active_updates"]
        return result

    def forward(self, input_ids, attention_mask=None):
        logits, auxiliary = super().forward(input_ids, attention_mask)
        auxiliary["active_refinements"] = 2 * auxiliary["active_updates"]
        return logits, auxiliary


def build_model(spec):
    model = Model(spec)
    parent.assert_model_state(model, spec)
    return model


build_optimizer = parent.build_optimizer
SUBMISSION = parent.Submission(build_model, build_optimizer, batch_size=64,
                               eval_batch_size=128, max_steps=200)
