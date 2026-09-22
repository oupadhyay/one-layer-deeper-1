import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/identity_dynamic_transition_easy_v1/submission.py"
LOADER = importlib.util.spec_from_file_location("identity_dynamic_transition_easy_v1", PATH)
candidate = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(candidate)


def spec(length=16):
    return ModelSpec(17, length, 500_000_000)


class IdentityDynamicTransitionEasyV1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_source_state_optimizer_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        for forbidden in ("from submissions", ".backward(", "autograd", "training_loop",
                          "multiply", "remainder", "quotient", "carry"):
            self.assertNotIn(forbidden, source.lower())
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 234_624)
        self.assertLess(count_model_state_elements(model), 500_000_000)
        bundle = candidate.build_optimizer(model, OptimizerSpec(60.0, "cpu"))
        self.assertIsInstance(bundle.optimizer, torch.optim.AdamW)
        self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual(bundle.optimizer.defaults["weight_decay"], .02)
        self.assertIsInstance(bundle.scheduler, torch.optim.lr_scheduler.LambdaLR)
        self.assertAlmostEqual(bundle.scheduler.lr_lambdas[0](0), 1 / 64)
        submission = candidate.SUBMISSION
        self.assertEqual((submission.batch_size, submission.eval_batch_size, submission.max_steps),
                         (512, 1024, None))

    def test_block_is_exact_identity_at_initialization(self):
        block = candidate.TransitionBlock().eval()
        state = torch.randn(3, 7, candidate.D)
        context = torch.randn(3, 5, candidate.D)
        mask = torch.ones(3, 5, dtype=torch.bool)
        with torch.no_grad():
            output = block(state, context, mask)
        torch.testing.assert_close(output, state, rtol=0, atol=0)
        for name in ("self_out.weight", "cross_out.weight", "ffn_down.weight"):
            self.assertEqual(torch.count_nonzero(dict(block.named_parameters())[name]).item(), 0)

    def test_alignment_macrosteps_dynamic_depth_and_eval_determinism(self):
        model = candidate.build_model(spec()).eval()
        # N=1234, X=567, T=3, then answer positions.
        ids = torch.tensor([[2, 8, 9, 10, 11, 3, 12, 13, 14, 4, 10, 5, 0, 0, 0, 0]])
        mask = torch.tensor([[1] * 12 + [0] * 4])
        with torch.no_grad():
            first, info = model(ids, mask)
            second, again = model(ids, mask)
        self.assertEqual(first.shape, (1, 16, 17))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual((info["steps"].item(), info["widths"].item()), (3, 4))
        self.assertEqual((info["macrosteps"], info["microsteps"]), (3, 4))
        self.assertEqual(info.keys(), again.keys())
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        model.train()
        seen = set()
        for seed in range(20):
            torch.manual_seed(seed)
            _, train_info = model(ids, mask)
            seen.add(train_info["microsteps"])
        self.assertEqual(seen, {2, 3, 4, 5})

    def test_gradients_padding_batch_purity_and_max_t(self):
        model = candidate.build_model(spec()).train()
        ids = torch.tensor([
            [2, 8, 9, 3, 10, 11, 4, 8, 5, 0, 0, 0, 0, 0, 0, 0],
            [2, 9, 10, 3, 11, 12, 4, 9, 5, 0, 0, 0, 0, 0, 0, 0],
        ])
        mask = torch.tensor([[1] * 9 + [0] * 7] * 2)
        logits, _ = model(ids, mask)
        loss = F.cross_entropy(logits[:, 8, :], torch.tensor([8, 9]))
        loss.backward()
        for name in ("block.self_out.weight", "block.cross_out.weight", "block.ffn_down.weight",
                     "readout_norm.weight", "digit_embedding.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

        model.eval()
        changed_padding = ids.clone()
        changed_padding[:, 9:] = 16
        with torch.no_grad():
            normal, _ = model(ids, mask)
            padded, _ = model(changed_padding, mask)
            reversed_batch, _ = model(ids.flip(0), mask.flip(0))
        torch.testing.assert_close(normal, padded, rtol=0, atol=0)
        torch.testing.assert_close(normal[0], reversed_batch[1], rtol=0, atol=2e-5)

        max_t = torch.tensor([[2, 8, 3, 9, 4, 13, 11, 5, 0, 0, 0, 0, 0, 0, 0, 0]])
        max_mask = torch.tensor([[1] * 8 + [0] * 8])
        with torch.no_grad():
            output, info = model(max_t, max_mask)
        self.assertEqual(info["steps"].item(), 64)
        self.assertEqual(info["macrosteps"], 64)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
