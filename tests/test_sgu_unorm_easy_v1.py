"""Round-6 Easy E5 activation-branch normalization regression tests."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F
from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "sgu_unorm_c21_easy_v1": ("sgu_cross_c21_easy_v1", 79_906),
    "sgu_unorm_diagcalc12_easy_v1": ("sgu_diag_calc12_easy_v1", 79_685),
}


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class SGUUNormTests(unittest.TestCase):
    def test_exact_delta_state_topology_source_and_initial_difference(self):
        for name, (parent_name, count) in CASES.items():
            with self.subTest(name=name):
                path, child = load(name)
                _, parent = load(parent_name)
                torch.manual_seed(31)
                cm = child.build_model(ModelSpec(17, 24, 250_000))
                torch.manual_seed(31)
                pm = parent.build_model(ModelSpec(17, 24, 250_000))
                self.assertEqual(count_model_state_elements(cm), count)
                self.assertEqual(child.STATE_ELEMENTS, count)
                self.assertLess(count, 250_000)
                block = cm.transition.block
                self.assertIsInstance(block.u_norm, child.RMSNorm)
                self.assertEqual(tuple(block.u_norm.weight.shape), (child.HIDDEN,))
                self.assertEqual(block.u_norm.eps, 1e-6)
                torch.testing.assert_close(block.u_norm.weight, torch.ones(child.HIDDEN))
                common = {k: v for k, v in cm.state_dict().items() if k != "transition.block.u_norm.weight"}
                self.assertEqual(common.keys(), pm.state_dict().keys())
                for key, value in common.items():
                    torch.testing.assert_close(value, pm.state_dict()[key])
                register = torch.randn(2, 8, child.D_MODEL)
                self.assertFalse(torch.allclose(block(register), pm.transition.block(register)))
                source = path.read_text()
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                self.assertNotIn("from submissions", source)
                self.assertNotIn("cross_entropy", source.lower())
                self.assertNotIn("transformer", source.lower())
                self.assertEqual((child.SUBMISSION.batch_size, child.SUBMISSION.eval_batch_size,
                                  child.SUBMISSION.max_steps), (256, 512, None))

    def test_u_norm_product_oracle_and_gradients(self):
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name)
                torch.manual_seed(9)
                block = module.SpatialGatingBlock()
                register = torch.randn(2, 8, module.D_MODEL)
                captured = {}
                hook = block.project.register_forward_pre_hook(
                    lambda _m, args: captured.setdefault("product", args[0].clone()))
                block(register)
                hook.remove()
                u, v = F.gelu(block.expand(block.norm(register))).chunk(2, -1)
                v = block.gate_norm(v).reshape(2, 4, 2, module.HIDDEN)
                if isinstance(block.place_operator, torch.nn.Linear):
                    pg = block.place_operator(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    rg = block.role_operator(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
                    inter = block.place_operator(rg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    gate = pg + rg + block.alpha * inter + block.diagonal[None, :, :, None] * v
                    gate = gate + block.spatial_bias[None, :, :, None]
                    gate = gate * block.place_gain[None, :, None, None] * block.role_gain[None, None, :, None]
                else:
                    pg = torch.einsum("op,bprh->borh", block.place_operator, v)
                    rg = torch.einsum("por,bprh->bpoh", block.role_operator, v)
                    gate = pg + rg + block.spatial_bias[None, :, :, None]
                expected = block.u_norm(u) * gate.reshape_as(u)
                torch.testing.assert_close(captured["product"], expected)
                block.zero_grad()
                block(register).square().mean().backward()
                self.assertIsNotNone(block.u_norm.weight.grad)
                self.assertTrue(torch.isfinite(block.u_norm.weight.grad).all())
                self.assertGreater(block.u_norm.weight.grad.abs().sum().item(), 0)

    def test_parser_t64_parity_determinism_permutation_and_finite_gradients(self):
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name)
                torch.manual_seed(17)
                model = module.build_model(ModelSpec(17, 24, 250_000))
                p = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
                _, place, steps = model.parse(p, p.ne(0))
                self.assertEqual((steps.item(), place[0, 1:5].tolist()), (64, [3,2,1,0]))
                model.eval()
                out, info = model(p)
                self.assertEqual(out.shape, (1, 14, 17))
                self.assertEqual(info["macrosteps"], 64)
                ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0],
                                    [2,11,3,12,13,4,10,5,0,0]])
                model.train()
                a, info = model(ids)
                b, _ = model(ids)
                q, _ = model(ids.flip(0))
                self.assertEqual(info["macrosteps"], 3)
                torch.testing.assert_close(a, b)
                torch.testing.assert_close(a, q.flip(0))
                a[:, :, 7:17].sum().backward()
                for key, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)


if __name__ == "__main__":
    unittest.main()
