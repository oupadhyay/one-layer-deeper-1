"""C1B local diagnostic: C1A plus T1-dominant endpoint gradients."""

import torch
from benchmark import Submission, assert_model_state
import competition_submission as c1
import competition_submission_c1a as c1a


class T1DominantModel(c1.Model):
    def forward(self, input_ids, attention_mask=None):
        logits, auxiliary = super().forward(input_ids, attention_mask=attention_mask)
        if self.training:
            auxiliary["ungated_logits"] = logits
            scale = torch.where(
                auxiliary["parsed_steps"] == 1,
                logits.new_tensor(1.0),
                logits.new_tensor(0.01),
            )
            detached = logits.detach()
            logits = detached + scale[:, None, None] * (logits - detached)
        return logits, auxiliary


def build_model(spec):
    model = T1DominantModel(spec)
    assert_model_state(model, spec)
    return model


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=c1a.build_optimizer,
    batch_size=64,
    eval_batch_size=128,
    max_steps=9500,
)
