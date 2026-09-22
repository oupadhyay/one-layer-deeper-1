import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "sgu_amlp_rms_easy_v1": (88_450, "rms", "amlp"),
    "sgu_amlp_conv_easy_v1": (89_346, "conv", "amlp"),
    "sgu_highway_rms_easy_v1": (92_146, "rms", "highway"),
    "sgu_highway_conv_easy_v1": (93_042, "conv", "highway"),
}


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class SGUChildrenTests(unittest.TestCase):
    def test_contract_architecture_delta_and_validation(self):
        for name, (count, child, parent) in CASES.items():
            with self.subTest(name=name):
                path, module = load(name)
                torch.manual_seed(112)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                self.assertEqual(count_model_state_elements(model), count)
                self.assertLess(count, 250_000)
                self.assertEqual((module.SUBMISSION.batch_size, module.SUBMISSION.eval_batch_size,
                                  module.SUBMISSION.max_steps), (256, 512, None))
                bundle = module.build_optimizer(model, OptimizerSpec(1, "cpu"))
                self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
                self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
                block = model.transition.block
                if parent == "amlp":
                    self.assertTrue(hasattr(block, "qkv")); self.assertTrue(hasattr(block, "attention_out"))
                    self.assertFalse(hasattr(block, "highway"))
                else:
                    self.assertTrue(hasattr(block, "highway")); self.assertFalse(hasattr(block, "qkv"))
                source = path.read_text()
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                lower = source.lower()
                for forbidden in ("remainder", "cross_entropy", "argmax", "cache", "router",
                                  "scaled_dot_product", "covariance", "scan"):
                    self.assertNotIn(forbidden, lower)
                if child == "rms":
                    self.assertNotIn("LayerNorm", source)
                    self.assertEqual(sum(isinstance(x, module.RMSNorm) for x in model.modules()), 3)
                    self.assertFalse(hasattr(model.transition, "local"))
                else:
                    local = model.transition.local
                    self.assertEqual((local.kernel_size, local.padding, local.groups), ((3,), (1,), 112))
                    self.assertTrue(torch.count_nonzero(local.weight).item() == 0)
                    self.assertTrue(torch.count_nonzero(local.bias).item() == 0)
                    self.assertEqual(sum(isinstance(x, torch.nn.LayerNorm) for x in model.modules()), 3)

    def test_parser_alignment_t64_permutation_parity_and_gradients(self):
        parse_ids = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
        ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0], [2,11,3,12,13,4,10,5,0,0]])
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name); torch.manual_seed(112)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                _, place, steps = model.parse(parse_ids, parse_ids.ne(0))
                self.assertEqual((steps.item(), place[0,1:5].tolist()), (64, [3,2,1,0]))
                model.eval(); out64, info = model(parse_ids)
                self.assertEqual((out64.shape, info["macrosteps"]), ((1,14,17),64))
                self.assertTrue((out64[0,8:10,7:17] > -10_000).all())
                model.train(); a, train_info = model(ids); b, _ = model(ids.flip(0))
                self.assertEqual(train_info["macrosteps"], 3)
                torch.testing.assert_close(a[0], b[1]); torch.testing.assert_close(a[1], b[0])
                model.eval(); c, eval_info = model(ids)
                self.assertEqual(eval_info["macrosteps"], 3); torch.testing.assert_close(a, c)
                model.train(); a[:, :, 7:17].sum().backward()
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)


if __name__ == "__main__":
    unittest.main()
