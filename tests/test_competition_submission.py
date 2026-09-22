import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_submission as c


def spec(length=24):
    return ModelSpec(17, length, 20_000_000)


class CompetitionCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = c.Model(spec(), d_model=16, heads=4)

    def prompt(self, t=2, pad=0):
        row = [c.N, 8, 9, 10, 11, 12, c.X, 13, 14, 15, 16, 8, c.T, c.DIGIT + t, c.ANS, 7, 7, 7, 7, 7]
        return torch.tensor([row + [0] * pad])

    def test_source_policy_interface_shapes_and_budget(self):
        path = Path(c.__file__)
        self.assertEqual(validate_submission_source(path.name, path.read_text(), 256 * 1024, required_filename=None), path.name)
        model = c.build_model(spec(32))
        x = self.prompt()
        y, aux = self.model(x)
        self.assertEqual(y.shape, (*x.shape, 17))
        self.assertEqual(aux["active_updates"].tolist(), [2])
        self.assertEqual(aux["executed_macrosteps"].item(), 2)
        self.assertTrue((y[0, -2:].argmax(-1) >= c.DIGIT).all())
        self.assertLess(count_model_state_elements(model), 20_000_000)

    def test_width_alignment_and_length_guard(self):
        x = self.prompt()
        y, aux = self.model.eval()(x)
        # Last token maps to LSD slot zero, previous token to slot one.
        # Distinct place slots are selected in reverse prompt order.
        self.assertFalse(torch.equal(y[0, -1, 7:17], y[0, -5, 7:17]))
        self.assertEqual(aux["active_updates"].item(), 2)
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 25, dtype=torch.long))

        class PlaceDecoder(torch.nn.Module):
            def forward(_, state, context, context_mask):
                logits = state.new_full((*state.shape[:2], 10), -20.0)
                places = torch.arange(state.shape[1], device=state.device) % 10
                return logits.scatter(2, places[None, :, None].expand(state.shape[0], -1, 1), 20.0)
        self.model.transition = PlaceDecoder()
        with torch.no_grad():
            aligned, _ = self.model(self.prompt(1))
        self.assertEqual(aligned[0, -6:, c.DIGIT:].argmax(-1).tolist(), [5, 4, 3, 2, 1, 0])

    def test_exact_max_length_and_hard_whole_state_reset(self):
        model = c.Model(spec(19), d_model=16, heads=4).eval()
        row = [c.N, *([8] * 8), c.X, *([9] * 7), c.T, 9]
        self.assertEqual(len(row), 19)
        captured = []
        outputs = []
        original = model.transition.forward
        def capture(state, context, context_mask):
            captured.append(state.clone())
            output = original(state, context, context_mask)
            outputs.append(output.clone())
            return output
        model.transition.forward = capture
        with torch.no_grad():
            logits, auxiliary = model(torch.tensor([row]))
        self.assertEqual(logits.shape, (1, 19, 17))
        self.assertEqual(len(captured), 2)
        hard = torch.nn.functional.one_hot(
            outputs[0].argmax(-1), 10
        ).to(captured[0].dtype)
        expected_second = (
            hard @ model.digit_embedding
            + model.place_embedding.weight[None]
            + model.role_embedding
        )
        torch.testing.assert_close(captured[1], expected_second)
        self.assertTrue(torch.all((auxiliary["digit_probabilities"] == 0) | (auxiliary["digit_probabilities"] == 1)))

    def test_tied_exact_active_updates(self):
        rows = torch.cat([self.prompt(i) for i in (1, 2, 3)])
        d = self.model.debug_execution(rows)
        self.assertEqual(d["parsed_steps"].tolist(), [1, 2, 3])
        self.assertEqual(d["active_updates"].tolist(), [1, 2, 3])
        self.assertIs(self.model.transition, self.model.transition)

        calls = 0
        original = self.model.transition.forward
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)
        self.model.transition.forward = counted
        _, auxiliary = self.model(rows)
        self.assertEqual((calls, auxiliary["executed_macrosteps"].item()), (3, 3))

    def test_mixed_t_matches_homogeneous_calls(self):
        self.model.eval()
        mixed = torch.cat([self.prompt(i) for i in (1, 2, 3)])
        with torch.no_grad():
            together = self.model(mixed)[0]
            separate = torch.cat([self.model(self.prompt(i))[0] for i in (1, 2, 3)])
        torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-6)

    def test_soft_gradient_and_hard_eval_feedback(self):
        x = self.prompt(2)
        self.model.train()
        transition_logits = []
        original = self.model.transition.forward
        def captured(*args, **kwargs):
            logits = original(*args, **kwargs)
            logits.retain_grad()
            transition_logits.append(logits)
            return logits
        self.model.transition.forward = captured
        endpoint, aux = self.model(x)
        self.assertTrue(aux["digit_probabilities"].requires_grad)
        endpoint[..., c.DIGIT:].sum().backward()
        self.assertEqual(len(transition_logits), 2)
        self.assertTrue(all(logits.grad is not None and logits.grad.abs().sum() > 0 for logits in transition_logits))
        self.assertGreater(self.model.transition.readout.weight.grad.abs().sum().item(), 0)
        self.model.transition.forward = original
        self.model.eval()
        with torch.no_grad():
            _, aux = self.model(x)
        self.assertTrue(torch.all((aux["digit_probabilities"] == 0) | (aux["digit_probabilities"] == 1)))

    def test_padding_batch_invariance_determinism_and_nonmutation(self):
        self.model.eval()
        x, padded = self.prompt(2), self.prompt(2, 3)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            a = self.model(x)[0]
            b = self.model(torch.cat([padded, self.prompt(3, 3)]))[0]
            again = self.model(x)[0]
        torch.testing.assert_close(a, b[:1, :x.shape[1]], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(a, again)
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertTrue(torch.isfinite(a).all())

    def test_only_n_remains_numeric_context(self):
        a = self.prompt(1); different_t = self.prompt(3)
        different_x = a.clone(); different_x[0, 7:12] = torch.tensor([7, 8, 9, 10, 11])
        different_n = a.clone(); different_n[0, 1] = 11
        self.model.eval()
        with torch.no_grad():
            ca = self.model._prepare(a, None)[2]
            ct = self.model._prepare(different_t, None)[2]
            cx = self.model._prepare(different_x, None)[2]
            cn = self.model._prepare(different_n, None)[2]
        torch.testing.assert_close(ca, ct)
        torch.testing.assert_close(ca, cx)
        self.assertGreater((ca - cn).abs().max().item(), 1e-5)

    def test_exact_tiny_batch_overfit_t1_and_mixed_t(self):
        cases = [
            (
                [[c.N, 10, 9, 10, c.X, 8, 12, c.T, 8]],
                [[11, 10]],
            ),
            (
                [
                    [c.N, 10, 9, 10, c.X, 8, 12, c.T, 8],
                    [c.N, 10, 9, 10, c.X, 15, 16, c.T, 9],
                ],
                [[11, 10], [12, 9]],
            ),
        ]
        for rows, targets in cases:
            torch.manual_seed(9)
            model = c.Model(spec(12), d_model=32, heads=4).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
            inputs = torch.tensor(rows)
            labels = torch.tensor(targets)
            positions = torch.tensor([[len(row) - 2, len(row) - 1] for row in rows])
            batch = torch.arange(len(rows))[:, None]
            exact = False
            for _ in range(30):
                optimizer.zero_grad(set_to_none=True)
                logits, auxiliary = model(inputs)
                selected = logits[batch, positions]
                loss = F.cross_entropy(selected.flatten(0, 1), labels.flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                exact = selected.argmax(-1).eq(labels).all().item()
                if exact:
                    break
            self.assertTrue(exact)
            expected_t = max(row[-1] - c.DIGIT for row in rows)
            self.assertEqual(auxiliary["executed_macrosteps"].item(), expected_t)

    def test_correct_label_soft_overfit_remains_exact_with_hard_feedback(self):
        # N=323: 15^(2^1)=225, 89^(2^2)=137, 42^(2^3)=290 (mod 323).
        inputs = torch.tensor([
            [c.N, 10, 9, 10, c.X, 8, 12, c.T, 8],
            [c.N, 10, 9, 10, c.X, 15, 16, c.T, 9],
            [c.N, 10, 9, 10, c.X, 11, 9, c.T, 10],
        ])
        labels = torch.tensor([[9, 9, 12], [8, 10, 14], [9, 16, 7]])
        positions = torch.tensor([[6, 7, 8]] * 3)
        batch = torch.arange(3)[:, None]
        torch.manual_seed(9)
        model = c.Model(spec(12), d_model=32, heads=4).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            selected = model(inputs)[0][batch, positions]
            loss = F.cross_entropy(selected.flatten(0, 1), labels.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad():
            hard_selected = model(inputs)[0][batch, positions]
        self.assertTrue(hard_selected.argmax(-1).eq(labels).all())

    def test_t1_carried_gradient_matches_t1_alone(self):
        torch.manual_seed(4)
        model = c.Model(spec(12), d_model=16, heads=4).train()
        t1 = torch.tensor([[c.N, 10, 9, 10, c.X, 8, 12, c.T, 8]])
        t2 = torch.tensor([[c.N, 10, 9, 10, c.X, 15, 16, c.T, 9]])
        target = torch.tensor([11, 10])

        def first_logit_gradient(inputs):
            captured = []
            original = model.transition.forward
            def capture(*args, **kwargs):
                output = original(*args, **kwargs)
                output.retain_grad()
                captured.append(output)
                return output
            model.transition.forward = capture
            model.zero_grad(set_to_none=True)
            logits = model(inputs)[0]
            F.cross_entropy(logits[0, 7:9], target).backward()
            gradient = captured[0].grad[0].clone()
            model.transition.forward = original
            return gradient

        alone = first_logit_gradient(t1)
        carried = first_logit_gradient(torch.cat((t1, t2)))
        self.assertGreater(alone.abs().sum().item(), 0)
        torch.testing.assert_close(alone, carried, atol=1e-6, rtol=1e-6)

    def test_cpu_bfloat16_autocast_forward_backward(self):
        model = c.Model(spec(12), d_model=16, heads=4).train()
        inputs = torch.tensor([[c.N, 10, 9, 10, c.X, 8, 12, c.T, 9]])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits[0, 7:9], torch.tensor([11, 10]))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.float().abs().sum().item() for g in gradients), 0)


if __name__ == "__main__":
    unittest.main()
