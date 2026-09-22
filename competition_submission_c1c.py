"""C1C preflight: two tied continuous refinements per outer macrostep."""

from benchmark import Submission, assert_model_state
import competition_submission as c1
import competition_submission_c1b as c1b


class InnerRefinementX2Transition(c1.Transition):
    def refine(self, state, context, context_mask):
        query = self.n1(state)
        state = state + self.self_attention(
            query, query, query, need_weights=False
        )[0]
        query = self.n2(state)
        state = state + self.cross_attention(
            query,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )[0]
        return state + self.ff(self.n3(state))

    def forward(self, state, context, context_mask):
        state = self.refine(state, context, context_mask)
        state = self.refine(state, context, context_mask)
        return self.readout(state)


class InnerRefinementX2Model(c1b.T1DominantModel):
    def __init__(self, spec, d_model=112, heads=4):
        super().__init__(spec, d_model=d_model, heads=heads)
        self.transition.__class__ = InnerRefinementX2Transition

    def forward(self, input_ids, attention_mask=None):
        logits, auxiliary = super().forward(input_ids, attention_mask=attention_mask)
        auxiliary["active_refinements"] = auxiliary["active_updates"] * 2
        return logits, auxiliary


def build_model(spec):
    model = InnerRefinementX2Model(spec)
    assert_model_state(model, spec)
    return model


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=c1b.SUBMISSION.build_optimizer,
    batch_size=64,
    eval_batch_size=128,
    max_steps=9500,
)
