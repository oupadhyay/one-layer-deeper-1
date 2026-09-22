"""Loop-2 cheap control: Loop-1 D with the frozen C0 outer loop, ungated."""

from benchmark import Submission, assert_model_state
import competition_submission as c0
import competition_submission_c1b as c1b
from competition_loop1_candidates import GlobalSlotMLP


class Model(c0.Model):
    """C0's true-T recurrence with only its transition replaced by Loop-1 D."""

    def __init__(self, spec, d_model=112, heads=4):
        super().__init__(spec, d_model=d_model, heads=heads)
        self.transition = GlobalSlotMLP(d_model, heads, spec.max_seq_len)


def build_model(spec):
    model = Model(spec, d_model=112, heads=4)
    assert_model_state(model, spec)
    return model


# Identity (not a reimplementation): AdamW and the 20-update C1B/C1A warmup.
build_optimizer = c1b.SUBMISSION.build_optimizer

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=64,
    eval_batch_size=128,
    max_steps=200,
)
