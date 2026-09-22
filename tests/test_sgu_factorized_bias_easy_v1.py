"""Round-5 Easy E5 factorized spatial-bias regression tests."""
import importlib.util
from pathlib import Path
import unittest

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "sgu_bias_c21_easy_v1": ("sgu_cross_c21_easy_v1", 79_680, False),
    "sgu_bias_diagcalc12_easy_v1": ("sgu_diag_calc12_easy_v1", 79_459, True),
    "sgu_bias_diagcalc33_easy_v1": ("sgu_diag_calc33_easy_v1", 79_570, True),
}


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class SGUFactorizedBiasTests(unittest.TestCase):
    def test_exact_state_topology_parent_parity_and_preservation(self):
        for name, (parent_name, count, calibrated) in CASES.items():
            with self.subTest(name=name):
                path, child = load(name); _, parent = load(parent_name)
                torch.manual_seed(91); cm = child.build_model(ModelSpec(17, 24, 250_000))
                torch.manual_seed(91); pm = parent.build_model(ModelSpec(17, 24, 250_000))
                self.assertEqual(count_model_state_elements(cm), count)
                self.assertEqual(child.STATE_ELEMENTS, count); self.assertLess(count, 250_000)
                block = cm.transition.block
                self.assertFalse(hasattr(block, "spatial_bias"))
                self.assertEqual(tuple(block.place_bias.shape), (4,))
                self.assertEqual(tuple(block.role_bias.shape), (2,))
                torch.testing.assert_close(block.place_bias, torch.full((4,), .5))
                torch.testing.assert_close(block.role_bias, torch.full((2,), .5))
                self.assertEqual(hasattr(block, "diagonal"), calibrated)
                self.assertEqual(hasattr(block, "place_gain"), calibrated)
                if calibrated:
                    torch.testing.assert_close(block.diagonal, torch.zeros(4, 2))
                    torch.testing.assert_close(block.place_gain, torch.ones(4))
                    torch.testing.assert_close(block.role_gain, torch.ones(2))
                child_state = {k: v for k, v in cm.state_dict().items()
                               if not k.endswith((".place_bias", ".role_bias"))}
                parent_state = {k: v for k, v in pm.state_dict().items()
                                if not k.endswith(".spatial_bias")}
                self.assertEqual(child_state.keys(), parent_state.keys())
                for key in child_state:
                    torch.testing.assert_close(child_state[key], parent_state[key])
                ids = torch.tensor([[2, 8, 9, 3, 10, 4, 10, 5, 0]])
                cm.eval(); pm.eval()
                torch.testing.assert_close(cm(ids)[0], pm(ids)[0])
                self.assertEqual((child.SUBMISSION.batch_size, child.SUBMISSION.eval_batch_size,
                                  child.SUBMISSION.max_steps), (256, 512, None))
                bundle = child.build_optimizer(cm, OptimizerSpec(1, "cpu"))
                self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
                self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
                source = path.read_text()
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                self.assertNotIn("cross_entropy", source.lower())
                self.assertNotIn("from submissions", source)
                self.assertNotIn("spatial_bias", source)

    def test_bias_oracle_and_gradients(self):
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name); torch.manual_seed(3)
                block = module.SpatialGatingBlock()
                block.place_bias.data.copy_(torch.tensor([-.3, -.1, .2, .4]))
                block.role_bias.data.copy_(torch.tensor([.7, -.2]))
                register = torch.randn(2, 8, module.D_MODEL)
                captured = {}
                handle = block.project.register_forward_pre_hook(
                    lambda _m, args: captured.setdefault("product", args[0].detach().clone()))
                block(register); handle.remove()
                u, v = torch.nn.functional.gelu(block.expand(block.norm(register))).chunk(2, -1)
                v = block.gate_norm(v).reshape(2, 4, 2, module.HIDDEN)
                if isinstance(block.place_operator, torch.nn.Linear):
                    pg = block.place_operator(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    rg = block.role_operator(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
                    gate = pg + rg + block.alpha * block.place_operator(
                        rg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                else:
                    pg = torch.einsum("op,bprh->borh", block.place_operator, v)
                    equation = "por,bprh->bpoh" if block.role_operator.ndim == 3 else "or,bprh->bpoh"
                    gate = pg + torch.einsum(equation, block.role_operator, v)
                if hasattr(block, "diagonal"):
                    gate = gate + block.diagonal[None, :, :, None] * v
                gate = gate + block.place_bias[None, :, None, None]
                gate = gate + block.role_bias[None, None, :, None]
                if hasattr(block, "place_gain"):
                    gate = gate * block.place_gain[None, :, None, None]
                    gate = gate * block.role_gain[None, None, :, None]
                torch.testing.assert_close(captured["product"], u * gate.reshape_as(u))
                block.zero_grad(); block(register).square().mean().backward()
                for bias in (block.place_bias, block.role_bias):
                    self.assertIsNotNone(bias.grad)
                    self.assertTrue(torch.isfinite(bias.grad).all())
                    self.assertGreater(bias.grad.abs().sum().item(), 0)

    def test_parser_t64_determinism_permutation_and_all_finite_gradients(self):
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name); torch.manual_seed(7)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                p = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
                _, place, steps = model.parse(p, p.ne(0))
                self.assertEqual((steps.item(), place[0,1:5].tolist()), (64,[3,2,1,0]))
                model.eval(); out, info = model(p)
                self.assertEqual(out.shape, (1,14,17)); self.assertEqual(info["macrosteps"], 64)
                ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0],
                                    [2,11,3,12,13,4,10,5,0,0]])
                model.train(); a, info = model(ids); b, _ = model(ids); q, _ = model(ids.flip(0))
                self.assertEqual(info["macrosteps"], 3)
                torch.testing.assert_close(a, b); torch.testing.assert_close(a, q.flip(0))
                a[:, :, 7:17].sum().backward()
                for key, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)


if __name__ == "__main__":
    unittest.main()
