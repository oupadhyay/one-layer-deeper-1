import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

PATH = Path(__file__).resolve().parents[1] / "submissions/sgu_latent_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("sgu_latent_easy_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class SGULatentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112); self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_exact_state_and_latent_topology(self):
        self.assertEqual(count_model_state_elements(self.model), 79_938)
        self.assertFalse(any("highway" in n for n, _ in self.model.named_parameters()))
        tr = self.model.transition
        n = torch.tensor([[1, 2, 0, 0]]); present = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        x = tr.initial_x(torch.tensor([[3, 4, 0, 0]]), present)
        captured = []
        hook = tr.block.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
        _, transformed = tr(n, present, x); tr(n, present, transformed); hook.remove()
        torch.testing.assert_close(captured[0][:, :4], captured[1][:, :4])
        torch.testing.assert_close(captured[1][:, 4:], transformed)

    def test_parser_alignment_t64_and_rejections(self):
        ids = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
        _, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual((steps.item(), place[0,1:5].tolist()), (64, [3,2,1,0]))
        self.model.eval(); out, info = self.model(ids)
        self.assertEqual((out.shape, info["macrosteps"]), ((1,14,17), 64))
        self.assertTrue((out[0,8:10,7:17] > -10_000).all())
        for bad in ([[2,8,9,10,11,12,3,7,4,7,5]], [[2,8,3,7,4,13,12,5]]):
            with self.assertRaises(ValueError): self.model(torch.tensor(bad))

    def test_calls_inactive_feedback_purity_determinism_and_gradients(self):
        ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0], [2,11,3,12,13,4,10,5,0,0]])
        calls = []; hook = self.model.transition.register_forward_hook(lambda *args: calls.append(args[2][1].detach().clone()))
        self.model.train(); a, _ = self.model(ids); hook.remove()
        self.assertEqual(len(calls), 3); self.assertFalse(torch.equal(calls[0], calls[1]))
        b, _ = self.model(ids.flip(0)); torch.testing.assert_close(a[0], b[1]); torch.testing.assert_close(a[1], b[0])
        c, _ = self.model(ids); torch.testing.assert_close(a, c)
        zero = torch.tensor([[2,8,3,9,4,7,5]])
        self.model.eval(); _, info = self.model(zero)
        expected = self.model.transition.initial_x(torch.tensor([[2,0,0,0]]), torch.tensor([[1,0,0,0]], dtype=torch.bool))
        torch.testing.assert_close(info["latent"], expected)
        self.model.train(); a[:,:,7:17].sum().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name); self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_source_validation(self):
        source = PATH.read_text(); self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        for forbidden in ("remainder", "cross_entropy", "argmax", "cache", "conv1d", "router"):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__": unittest.main()
