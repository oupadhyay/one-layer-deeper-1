"""Round-4 Easy E5 diagonal-bypass package regression tests."""
import importlib.util
from pathlib import Path
import unittest

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "sgu_diag_c21_easy_v1": ("sgu_cross_c21_easy_v1", 79_690, False),
    "sgu_diag_c11_easy_v1": ("sgu_cross_c11_easy_v1", 79_466, False),
    "sgu_diag_calc12_easy_v1": ("sgu_cal_c12_easy_v1", 79_461, True),
    "sgu_diag_calc33_easy_v1": ("sgu_cal_c33_easy_v1", 79_572, True),
}


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class SGUDiagonalTests(unittest.TestCase):
    def test_parent_preservation_contract_source_and_initial_parity(self):
        for name, (parent_name, count, calibrated) in CASES.items():
            with self.subTest(name=name):
                path, child = load(name); _, parent = load(parent_name)
                torch.manual_seed(91); cm = child.build_model(ModelSpec(17, 24, 250_000))
                torch.manual_seed(91); pm = parent.build_model(ModelSpec(17, 24, 250_000))
                self.assertEqual(count_model_state_elements(cm), count)
                self.assertEqual(child.STATE_ELEMENTS, count); self.assertLess(count, 250_000)
                self.assertEqual((child.SUBMISSION.batch_size, child.SUBMISSION.eval_batch_size,
                                  child.SUBMISSION.max_steps), (256, 512, None))
                bundle = child.build_optimizer(cm, OptimizerSpec(1, "cpu"))
                self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
                self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
                self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
                diagonal = cm.transition.block.diagonal
                self.assertEqual(tuple(diagonal.shape), (4, 2))
                torch.testing.assert_close(diagonal, torch.zeros(4, 2))
                child_state = {k: v for k, v in cm.state_dict().items() if not k.endswith(".diagonal")}
                self.assertEqual(child_state.keys(), pm.state_dict().keys())
                for key, value in child_state.items():
                    torch.testing.assert_close(value, pm.state_dict()[key])
                ids = torch.tensor([[2, 8, 9, 3, 10, 4, 10, 5, 0]])
                cm.eval(); pm.eval()
                torch.testing.assert_close(cm(ids)[0], pm(ids)[0])
                block = cm.transition.block
                self.assertEqual(hasattr(block, "place_gain"), calibrated)
                self.assertEqual(hasattr(block, "role_gain"), calibrated)
                source = path.read_text()
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                self.assertNotIn("cross_entropy", source.lower())
                self.assertNotIn("from submissions", source)

    def test_diagonal_oracle_and_gradient(self):
        for name in CASES:
            with self.subTest(name=name):
                _, module = load(name); torch.manual_seed(3)
                block = module.SpatialGatingBlock(); register = torch.randn(2, 8, module.D_MODEL)
                block.diagonal.data.uniform_(-.3, .3)
                captured = {}
                handle = block.project.register_forward_pre_hook(
                    lambda _m, args: captured.setdefault("product", args[0].detach().clone()))
                block(register); handle.remove()
                u, v = torch.nn.functional.gelu(block.expand(block.norm(register))).chunk(2, -1)
                v = block.gate_norm(v).reshape(2, 4, 2, module.HIDDEN)
                if isinstance(block.place_operator, torch.nn.Linear):
                    pg = block.place_operator(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    rg = block.role_operator(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
                    gate = pg + rg + block.alpha * block.place_operator(rg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                else:
                    pg = torch.einsum("op,bprh->borh", block.place_operator, v)
                    equation = "por,bprh->bpoh" if block.role_operator.ndim == 3 else "or,bprh->bpoh"
                    gate = pg + torch.einsum(equation, block.role_operator, v)
                gate = gate + block.diagonal[None, :, :, None] * v + block.spatial_bias[None, :, :, None]
                if hasattr(block, "place_gain"):
                    gate = gate * block.place_gain[None, :, None, None] * block.role_gain[None, None, :, None]
                torch.testing.assert_close(captured["product"], u * gate.reshape_as(u))
                block.zero_grad(); block(register).square().mean().backward()
                self.assertIsNotNone(block.diagonal.grad)
                self.assertTrue(torch.isfinite(block.diagonal.grad).all())
                self.assertGreater(block.diagonal.grad.abs().sum().item(), 0)

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
                ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0],[2,11,3,12,13,4,10,5,0,0]])
                model.train(); a, info = model(ids); b, _ = model(ids); q, _ = model(ids.flip(0))
                self.assertEqual(info["macrosteps"], 3)
                torch.testing.assert_close(a, b); torch.testing.assert_close(a, q.flip(0))
                a[:, :, 7:17].sum().backward()
                for key, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)


if __name__ == "__main__":
    unittest.main()
