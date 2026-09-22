import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_submission as c1
import competition_submission_c1b as c1b
import competition_submission_c1c as c1c


SPEC = ModelSpec(17, 19, 500_000_000)


def rows():
    return torch.tensor([
        [c1.N, 10, 9, 10, c1.X, 8, 12, c1.T, 8],
        [c1.N, 10, 9, 10, c1.X, 15, 16, c1.T, 9],
        [c1.N, 10, 9, 10, c1.X, 11, 9, c1.T, 10],
    ])


class C1CTests(unittest.TestCase):
    def paired_models(self, d_model=112):
        torch.manual_seed(74)
        parent = c1b.build_model(SPEC)
        torch.manual_seed(74)
        candidate = c1c.InnerRefinementX2Model(SPEC, d_model=d_model, heads=4)
        return parent, candidate

    def test_state_optimizer_gate_and_source_contract_unchanged(self):
        parent, candidate = self.paired_models()
        self.assertEqual(count_model_state_elements(parent), count_model_state_elements(candidate))
        self.assertEqual(parent.state_dict().keys(), candidate.state_dict().keys())
        for key in parent.state_dict():
            self.assertTrue(torch.equal(parent.state_dict()[key], candidate.state_dict()[key]))
        self.assertIs(c1c.SUBMISSION.build_optimizer, c1b.SUBMISSION.build_optimizer)
        parent_bundle = c1b.SUBMISSION.build_optimizer(parent, OptimizerSpec(60, "cpu"))
        candidate_bundle = c1c.SUBMISSION.build_optimizer(candidate, OptimizerSpec(60, "cpu"))
        self.assertEqual(
            [group["lr"] for group in parent_bundle.optimizer.param_groups],
            [group["lr"] for group in candidate_bundle.optimizer.param_groups],
        )
        validate_submission_source(
            "competition_submission_c1c.py",
            Path("competition_submission_c1c.py").read_text(encoding="utf-8"),
            256 * 1024,
            required_filename=None,
        )

    def test_one_body_identity_counts_and_true_t_activity(self):
        _, model = self.paired_models()
        identity = id(model.transition)
        calls = []
        readouts = []
        original = model.transition.refine
        original_readout = model.transition.readout.forward
        def counted(*args, **kwargs):
            calls.append(id(model.transition))
            return original(*args, **kwargs)
        def counted_readout(*args, **kwargs):
            readouts.append(True)
            return original_readout(*args, **kwargs)
        model.transition.refine = counted
        model.transition.readout.forward = counted_readout
        _, auxiliary = model(rows())
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(readouts), 3)
        self.assertEqual(set(calls), {identity})
        self.assertEqual(auxiliary["active_refinements"].tolist(), [2, 4, 6])
        for time_steps in (1, 2, 3):
            calls.clear(); readouts.clear()
            model(rows()[time_steps - 1:time_steps])
            self.assertEqual(len(calls), 2 * time_steps)
            self.assertEqual(len(readouts), time_steps)

    def test_continuous_second_pass_and_outer_hard_boundary(self):
        _, model = self.paired_models(d_model=16)
        model.eval()
        captured_inputs, captured_outputs = [], []
        original = model.transition.refine
        def captured(state, context, context_mask):
            captured_inputs.append(state.clone())
            output = original(state, context, context_mask)
            captured_outputs.append(output.clone())
            return output
        model.transition.refine = captured
        with torch.no_grad():
            _, auxiliary = model(rows()[1:2])
        self.assertEqual(len(captured_inputs), 4)
        torch.testing.assert_close(captured_inputs[1], captured_outputs[0])
        first_logits = model.transition.readout(captured_outputs[1])
        hard = F.one_hot(first_logits.argmax(-1), 10).to(first_logits.dtype)
        expected_next = (
            hard @ model.digit_embedding
            + model.place_embedding.weight[None]
            + model.role_embedding
        )
        torch.testing.assert_close(captured_inputs[2], expected_next)
        self.assertTrue(torch.all((auxiliary["digit_probabilities"] == 0) | (auxiliary["digit_probabilities"] == 1)))

    def test_gradients_cross_both_refinements_and_gate_scales(self):
        labels = [torch.tensor([9, 9, 12]), torch.tensor([8, 10, 14]), torch.tensor([9, 16, 7])]
        for index, scale in enumerate((1.0, 0.01, 0.01)):
            _, model = self.paired_models(d_model=16)
            outputs = []
            original = model.transition.refine
            def captured(*args, **kwargs):
                output = original(*args, **kwargs)
                output.retain_grad(); outputs.append(output)
                return output
            model.transition.refine = captured
            logits, auxiliary = model(rows()[index:index + 1])
            auxiliary["ungated_logits"].retain_grad()
            F.cross_entropy(logits[0, 6:9], labels[index]).backward()
            self.assertEqual(len(outputs), 2 * (index + 1))
            self.assertTrue(all(output.grad is not None and output.grad.abs().sum() > 0 for output in outputs))
            reference = auxiliary["ungated_logits"].detach().requires_grad_(True)
            F.cross_entropy(reference[0, 6:9], labels[index]).backward()
            torch.testing.assert_close(
                auxiliary["ungated_logits"].grad,
                reference.grad * scale,
                atol=2e-7,
                rtol=2e-5,
            )

    def test_mixed_isolated_row_gradient_matches_homogeneous(self):
        _, model = self.paired_models(d_model=16)
        target = torch.tensor([9, 9, 12])
        def gradient(inputs):
            model.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            F.cross_entropy(logits[0, 6:9], target).backward()
            return torch.cat([
                parameter.grad.flatten()
                for parameter in model.parameters()
                if parameter.grad is not None
            ])
        homogeneous = gradient(rows()[0:1])
        mixed = gradient(rows())
        torch.testing.assert_close(homogeneous, mixed, atol=2e-6, rtol=2e-6)

    def test_tail_output_alignment_is_unchanged(self):
        _, model = self.paired_models(d_model=16)
        class PlaceReadout(torch.nn.Module):
            def forward(_, state):
                logits = state.new_full((*state.shape[:2], 10), -20.0)
                places = torch.arange(state.shape[1], device=state.device) % 10
                return logits.scatter(
                    2,
                    places[None, :, None].expand(state.shape[0], -1, 1),
                    20.0,
                )
        model.transition.readout = PlaceReadout()
        model.eval()
        with torch.no_grad():
            logits, _ = model(rows()[0:1])
        self.assertEqual(
            logits[0, -6:, c1.DIGIT:].argmax(-1).tolist(),
            [5, 4, 3, 2, 1, 0],
        )

    def test_mixed_homogeneous_padding_alignment_determinism_and_nonmutation(self):
        _, model = self.paired_models(d_model=16)
        model.eval(); mixed = rows()
        padded = F.pad(mixed, (0, 3))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            together = model(mixed)[0]
            separate = torch.cat([model(mixed[i:i + 1])[0] for i in range(3)])
            repeated = model(mixed)[0]
            padded_logits = model(padded, attention_mask=padded != c1.PAD)[0]
            max_length_row = torch.tensor([[
                c1.N, *([8] * 8), c1.X, *([9] * 7), c1.T, 8,
            ]])
            max_length_logits = model(max_length_row)[0]
        torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(together, repeated)
        torch.testing.assert_close(together, padded_logits[:, :mixed.shape[1]], atol=2e-6, rtol=2e-6)
        self.assertEqual(max_length_logits.shape, (1, SPEC.max_seq_len, 17))
        for key in before:
            self.assertTrue(torch.equal(before[key], model.state_dict()[key]))
        self.assertTrue((together[:, -3:].argmax(-1) >= c1.DIGIT).all())

    def test_true_label_t1_overfit_bridges_soft_to_hard(self):
        inputs = rows().clone()
        inputs[:, -1] = 8
        labels = torch.tensor([[9, 9, 12], [8, 13, 16], [8, 11, 16]])
        positions = torch.tensor([[6, 7, 8]] * 3)
        batch = torch.arange(3)[:, None]
        torch.manual_seed(9)
        model = c1c.InnerRefinementX2Model(SPEC, d_model=32, heads=4).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            selected = model(inputs)[0][batch, positions]
            loss = F.cross_entropy(selected.flatten(0, 1), labels.flatten())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad():
            hard = model(inputs)[0][batch, positions]
        self.assertTrue(hard.argmax(-1).eq(labels).all())

    def test_cpu_bfloat16_forward_backward(self):
        model = c1c.InnerRefinementX2Model(SPEC, d_model=16, heads=4).train()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, auxiliary = model(rows()[2:3])
            loss = F.cross_entropy(logits[0, 6:9], torch.tensor([9, 16, 7]))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertEqual(auxiliary["active_refinements"].item(), 6)


if __name__ == "__main__":
    unittest.main()
