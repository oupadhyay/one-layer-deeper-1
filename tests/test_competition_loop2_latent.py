import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec
import competition_loop2_latent as c


def spec(length=12):
    return ModelSpec(17, length, 20_000_000)


def rows():
    return torch.tensor([
        [c.N, 10, 9, 10, c.X, 8, 12, c.T, 8],
        [c.N, 10, 9, 10, c.X, 15, 16, c.T, 9],
        [c.N, 10, 9, 10, c.X, 11, 9, c.T, 10],
    ])


class LatentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = c.Model(spec(), d_model=16, heads=4)

    def test_context_controls_and_true_t(self):
        x = rows(); changed_x = x.clone(); changed_x[:, 5:7] = 7
        changed_t = x.clone(); changed_t[:, 8] = 8
        changed_n = x.clone(); changed_n[:, 1] = 11
        prepared = [self.model._prepare_latent(v, None)[2] for v in (x, changed_x, changed_t, changed_n)]
        torch.testing.assert_close(prepared[0], prepared[1]); torch.testing.assert_close(prepared[0], prepared[2])
        self.assertGreater((prepared[0] - prepared[3]).abs().max().item(), 0)
        d = self.model.debug_execution(x)
        self.assertEqual(d["parsed_steps"].tolist(), [1, 2, 3])
        self.assertEqual(d["active_updates"].tolist(), [1, 2, 3])

    def test_shared_body_endpoint_and_continuous_carry(self):
        captured = []
        original = self.model.transition.forward
        def capture(state, *args):
            captured.append(state.clone()); return original(state, *args)
        self.model.transition.forward = capture
        _, aux = self.model(rows()[1:2])
        self.assertEqual(len(captured), 2)
        prepared = self.model._prepare_latent(rows()[1:2], None)
        torch.testing.assert_close(captured[1], original(captured[0], prepared[2], prepared[1]))
        self.assertFalse(torch.all((captured[1] == 0) | (captured[1] == 1)))
        self.assertIs(self.model.transition, self.model.transition)
        self.assertEqual(sum(isinstance(m, torch.nn.Linear) and m.out_features == 10 for m in self.model.modules()), 1)
        self.assertEqual(aux["executed_macrosteps"].item(), 2)

    def test_mixed_homogeneous_train_eval_and_nonmutation(self):
        x = rows(); before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train(); train = self.model(x)[0].detach()
        self.model.eval()
        with torch.no_grad():
            evaluation = self.model(x)[0]
            separate = torch.cat([self.model(x[i:i + 1])[0] for i in range(3)])
            again = self.model(x)[0]
        torch.testing.assert_close(train, evaluation); torch.testing.assert_close(evaluation, separate, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(evaluation, again)
        for key, value in self.model.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))

    def test_alignment_padding_max_length_and_t64(self):
        class Place(torch.nn.Module):
            def forward(_, state):
                out = state.new_full((*state.shape[:2], 10), -20)
                place = torch.arange(state.shape[1], device=state.device) % 10
                return out.scatter(2, place[None, :, None].expand(state.shape[0], -1, 1), 20)
        self.model.endpoint_readout = Place(); x = rows()[:1]
        logits, _ = self.model(x)
        self.assertEqual(logits[0, -6:, c.DIGIT:].argmax(-1).tolist(), [5, 4, 3, 2, 1, 0])
        self.assertTrue(torch.isfinite(logits).all()); self.assertTrue((logits[..., :c.DIGIT] == -1e4).all())
        padded = F.pad(x, (0, 3)); torch.testing.assert_close(logits, self.model(padded)[0][:, :9])
        with self.assertRaises(ValueError): self.model(torch.ones(1, 13, dtype=torch.long))
        t64 = torch.tensor([[c.N, 8, c.X, 8, c.T, c.DIGIT + 6, c.DIGIT + 4]])
        output, aux = self.model(t64); self.assertTrue(torch.isfinite(output).all()); self.assertEqual(aux["active_updates"].item(), 64)

    def test_all_macrostep_gradients_and_cpu_bf16(self):
        outputs = []; original = self.model.transition.forward
        def capture(*args):
            out = original(*args); out.retain_grad(); outputs.append(out); return out
        self.model.transition.forward = capture
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _ = self.model(rows()[2:]); loss = F.cross_entropy(logits[0, 6:9].reshape(-1, 17), torch.tensor([9, 16, 7]))
        loss.backward()
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(v.grad is not None and v.grad.float().abs().sum() > 0 for v in outputs))
        self.assertTrue(torch.isfinite(loss))

    def test_true_label_tiny_overfit_same_weight_eval(self):
        x = rows(); labels = torch.tensor([[9, 9, 12], [8, 10, 14], [9, 16, 7]])
        model = c.Model(spec(), d_model=32, heads=4).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        batch = torch.arange(3)[:, None]; positions = torch.tensor([[6, 7, 8]] * 3)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True); selected = model(x)[0][batch, positions]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward(); optimizer.step()
            if selected.argmax(-1).eq(labels).all(): break
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad(): self.assertTrue(model(x)[0][batch, positions].argmax(-1).eq(labels).all())


if __name__ == "__main__": unittest.main()
