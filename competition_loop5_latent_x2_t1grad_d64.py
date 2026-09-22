"""Capacity-regularized d64 variant of the authorized x2 T1-gradient model."""

import competition_loop4_latent_x2_t1grad as parent


class Model(parent.Model):
    """The unchanged x2/T1-gate model with a smaller default latent width."""

    def __init__(self, spec, d_model=64, heads=4):
        super().__init__(spec, d_model=d_model, heads=heads)


def build_model(spec):
    model = Model(spec)
    parent.parent.parent.assert_model_state(model, spec)
    return model


build_optimizer = parent.build_optimizer
SUBMISSION = parent.parent.parent.Submission(build_model, build_optimizer, batch_size=64,
                                             eval_batch_size=128, max_steps=200)
