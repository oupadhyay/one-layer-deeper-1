"""Independent one-control optimization regressions for the locked C21 winner."""
import importlib.util
from pathlib import Path
import unittest

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "submissions/sgu_cross_c21_easy_v1/submission.py"
CASES = {
    "sgu_opt_lr4_easy_v1": (79_682, 224, 4e-4, 32, 256, .01, 1.25e-4),
    "sgu_opt_lr8_easy_v1": (79_682, 224, 8e-4, 32, 256, .01, 1.25e-4),
    "sgu_opt_warm16_easy_v1": (79_682, 224, 6e-4, 16, 256, .01, 1.25e-4),
    "sgu_opt_warm64_easy_v1": (79_682, 224, 6e-4, 64, 256, .01, 1.25e-4),
    "sgu_opt_batch128_easy_v1": (79_682, 224, 6e-4, 32, 128, .01, 1.25e-4),
    "sgu_opt_batch512_easy_v1": (79_682, 224, 6e-4, 32, 512, .01, 1.25e-4),
    "sgu_opt_nodecay_easy_v1": (79_682, 224, 6e-4, 32, 256, 0., 1.25e-4),
    "sgu_opt_h168_easy_v1": (60_642, 168, 6e-4, 32, 256, .01, 1.25e-4),
    "sgu_opt_h280_easy_v1": (98_722, 280, 6e-4, 32, 256, .01, 1.25e-4),
    "sgu_opt_init1e3_easy_v1": (79_682, 224, 6e-4, 32, 256, .01, 1e-3),
}

def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return path, module

class SGUOptimizationTests(unittest.TestCase):
    def test_exact_state_optimizer_scheduler_submission_topology_and_init(self):
        for name, (state, hidden, lr, warm, batch, decay, init) in CASES.items():
            with self.subTest(name=name):
                path, module = load(name); torch.manual_seed(31)
                model = module.build_model(ModelSpec(17, 24, 250_000)); block = model.transition.block
                self.assertEqual((module.STATE_ELEMENTS, count_model_state_elements(model)), (state, state))
                self.assertLess(state, 250_000); self.assertEqual((module.D_MODEL, module.HIDDEN), (112, hidden))
                self.assertEqual(block.expand.weight.shape, (2 * hidden, 112))
                self.assertEqual(block.project.weight.shape, (112, hidden))
                self.assertEqual(block.gate_norm.weight.shape, (hidden,))
                self.assertEqual((block.place_operator.shape, block.role_operator.shape, block.spatial_bias.shape),
                                 ((4, 4), (4, 2, 2), (4, 2)))
                self.assertTrue(torch.equal(block.spatial_bias, torch.ones(4, 2)))
                self.assertLessEqual(block.place_operator.abs().max().item(), init)
                self.assertLessEqual(block.role_operator.abs().max().item(), init)
                bundle = module.build_optimizer(model, OptimizerSpec(1, "cpu"))
                self.assertEqual(bundle.optimizer.defaults["lr"], lr)
                self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
                self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups], [decay, 0.])
                self.assertAlmostEqual(bundle.scheduler.get_last_lr()[0], lr / warm)
                self.assertEqual((module.SUBMISSION.batch_size, module.SUBMISSION.eval_batch_size,
                                  module.SUBMISSION.max_steps), (batch, 512, None))
                source = path.read_text()
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                for excluded in ("sgu_cross_c21_easy_v1", "submissions.", "sys.path", "cross_entropy"):
                    self.assertNotIn(excluded, source)

    def test_each_source_has_only_its_declared_delta_and_parent_is_locked(self):
        parent = PARENT.read_text()
        self.assertIn("STATE_ELEMENTS, NEG = 79_682", parent)
        replacements = {
            "sgu_opt_lr4_easy_v1": [("lr=6e-4", "lr=4e-4")],
            "sgu_opt_lr8_easy_v1": [("lr=6e-4", "lr=8e-4")],
            "sgu_opt_warm16_easy_v1": [("/ 32.0", "/ 16.0")],
            "sgu_opt_warm64_easy_v1": [("/ 32.0", "/ 64.0")],
            "sgu_opt_batch128_easy_v1": [("batch_size=256", "batch_size=128")],
            "sgu_opt_batch512_easy_v1": [("batch_size=256", "batch_size=512")],
            "sgu_opt_nodecay_easy_v1": [('"weight_decay": .01', '"weight_decay": 0.0')],
            "sgu_opt_h168_easy_v1": [("224, 64", "168, 64"), ("79_682", "60_642")],
            "sgu_opt_h280_easy_v1": [("224, 64", "280, 64"), ("79_682", "98_722")],
            "sgu_opt_init1e3_easy_v1": [("-1.25e-4, 1.25e-4", "-1e-3, 1e-3")],
        }
        for name, changes in replacements.items():
            expected = parent
            for old, new in changes: expected = expected.replace(old, new)
            actual = (ROOT / "submissions" / name / "submission.py").read_text()
            self.assertEqual(actual.split('"""', 2)[2], expected.split('"""', 2)[2], name)

    def test_parser_t64_loop_parity_determinism_permutation_and_finite_gradients(self):
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name); torch.manual_seed(7)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                p = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
                _, place, steps = model.parse(p, p.ne(0))
                self.assertEqual((steps.item(), place[0,1:5].tolist()), (64, [3,2,1,0]))
                model.eval(); out, info = model(p)
                self.assertEqual((out.shape, info["macrosteps"]), (torch.Size([1,14,17]), 64))
                ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0],[2,11,3,12,13,4,10,5,0,0]])
                model.train(); a, info = model(ids); b, _ = model(ids); q, _ = model(ids.flip(0))
                self.assertEqual(info["macrosteps"], 3); torch.testing.assert_close(a, b)
                torch.testing.assert_close(a, q.flip(0))
                model.eval(); e, info = model(ids); self.assertEqual(info["macrosteps"], 3)
                torch.testing.assert_close(a, e)
                model.train(); a[:, :, 7:17].sum().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad); self.assertTrue(torch.isfinite(parameter.grad).all())

if __name__ == "__main__":
    unittest.main()
