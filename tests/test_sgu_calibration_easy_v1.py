"""Shared regression tests for the six gate-calibrated SGU submissions."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "c11": (79_464, "rms", "place_role"),
    "c12": (79_453, "rms", "second_order"),
    "c21": (79_688, "gate_ln", "place_role"),
    "c33": (79_564, "input_ln", "einsum"),
    "c34": (79_580, "input_ln", "role_place"),
    "c44": (79_468, "eps", "role_place"),
}


def load(code):
    name = f"sgu_cal_{code}_easy_v1"
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def manual_block(block, register, topology):
    """Reconstruct the complete calibrated SGU equation independently."""
    u, v = F.gelu(block.expand(block.norm(register))).chunk(2, dim=-1)
    v = block.gate_norm(v).reshape(register.shape[0], 4, 2, 224)
    if topology == "place_role":
        place = torch.einsum("op,bprh->borh", block.place_operator, v)
        role = torch.einsum("por,bprh->bpoh", block.role_operator, v)
    elif topology == "second_order":
        place = block.place_operator(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        role = block.role_operator(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        interaction = block.place_operator(role.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        place = place + block.alpha * interaction
    elif topology == "einsum":
        place = torch.einsum("op,bprh->borh", block.place_operator, v)
        role = torch.einsum("or,bprh->bpoh", block.role_operator, v)
    else:
        place = torch.einsum("rop,bprh->borh", block.place_operator, v)
        role = torch.einsum("or,bprh->bpoh", block.role_operator, v)
    gate = place + role + block.spatial_bias[None, :, :, None]
    gate = gate * block.place_gain[None, :, None, None]
    gate = gate * block.role_gain[None, None, :, None]
    return register + block.project(u * gate.reshape_as(u))


class SGUCalibrationTests(unittest.TestCase):
    def test_exact_state_topology_gains_contract_validation_and_exclusions(self):
        for code, (count, norm, topology) in CASES.items():
            with self.subTest(code=code):
                path, module = load(code)
                torch.manual_seed(112)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                block = model.transition.block
                self.assertEqual(module.STATE_ELEMENTS, count)
                self.assertEqual(count_model_state_elements(model), count)
                self.assertLess(count, 250_000)
                self.assertEqual((module.SUBMISSION.batch_size, module.SUBMISSION.eval_batch_size,
                                  module.SUBMISSION.max_steps), (256, 512, None))
                bundle = module.build_optimizer(model, OptimizerSpec(1, "cpu"))
                self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
                self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
                self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)

                expected_ln = {"gate_ln": [False, True, False],
                               "input_ln": [True, False, False]}.get(norm, [False] * 3)
                norms = [block.norm, block.gate_norm, model.transition.readout_norm]
                self.assertEqual([isinstance(x, torch.nn.LayerNorm) for x in norms], expected_ln)
                for item, is_ln in zip(norms, expected_ln):
                    if not is_ln:
                        self.assertIsInstance(item, module.RMSNorm)
                        self.assertEqual(item.eps, 1e-5 if norm == "eps" else 1e-6)

                self.assertEqual(tuple(block.place_gain.shape), (4,))
                self.assertEqual(tuple(block.role_gain.shape), (2,))
                torch.testing.assert_close(block.place_gain, torch.ones(4))
                torch.testing.assert_close(block.role_gain, torch.ones(2))
                torch.testing.assert_close(block.spatial_bias, torch.ones(4, 2))
                if topology == "place_role":
                    self.assertEqual((block.place_operator.shape, block.role_operator.shape),
                                     (torch.Size([4, 4]), torch.Size([4, 2, 2])))
                elif topology == "second_order":
                    self.assertEqual((block.place_operator.weight.shape, block.role_operator.weight.shape),
                                     (torch.Size([4, 4]), torch.Size([2, 2])))
                    self.assertEqual(block.alpha.shape, torch.Size([]))
                    self.assertEqual(block.alpha.item(), 0)
                elif topology == "einsum":
                    self.assertEqual((block.place_operator.shape, block.role_operator.shape),
                                     (torch.Size([4, 4]), torch.Size([2, 2])))
                else:
                    self.assertEqual((block.place_operator.shape, block.role_operator.shape),
                                     (torch.Size([2, 4, 4]), torch.Size([2, 2])))

                source = path.read_text()
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                for forbidden in ("cross_entropy", "argmax", "remainder", "router", "cache",
                                  "scaled_dot_product", "conv1d", "covariance", "scan"):
                    self.assertNotIn(forbidden, source.lower())

    def test_manual_calibration_equations_match_forward(self):
        for code, (_, _, topology) in CASES.items():
            with self.subTest(code=code):
                _, module = load(code)
                torch.manual_seed(23)
                block = module.SpatialGatingBlock()
                register = torch.randn(2, 8, 112)
                torch.testing.assert_close(block(register), manual_block(block, register, topology))

    def test_parser_alignment_t64_parity_determinism_permutation_and_gradients(self):
        parsed = torch.tensor([[2, 8, 9, 10, 11, 3, 12, 7, 4, 13, 11, 5, 0, 0]])
        ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0], [2,11,3,12,13,4,10,5,0,0]])
        for code in CASES:
            with self.subTest(code=code):
                _, module = load(code)
                torch.manual_seed(7)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                _, place, steps = model.parse(parsed, parsed.ne(0))
                self.assertEqual((steps.item(), place[0, 1:5].tolist()), (64, [3, 2, 1, 0]))
                model.eval()
                out64, info = model(parsed)
                self.assertEqual((out64.shape, info["macrosteps"]), ((1, 14, 17), 64))
                self.assertTrue((out64[0, 8:10, 7:17] > -10_000).all())

                model.train()
                first, info = model(ids)
                repeated, _ = model(ids)
                permuted, _ = model(ids.flip(0))
                self.assertEqual(info["macrosteps"], 3)
                torch.testing.assert_close(first, repeated)
                torch.testing.assert_close(first, permuted.flip(0))
                model.eval()
                evaluated, info = model(ids)
                self.assertEqual(info["macrosteps"], 3)
                torch.testing.assert_close(first, evaluated)

                model.train()
                first[:, :, 7:17].sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(torch.isfinite(model.transition.block.place_gain.grad).all())
                self.assertTrue(torch.isfinite(model.transition.block.role_gain.grad).all())


if __name__ == "__main__":
    unittest.main()
