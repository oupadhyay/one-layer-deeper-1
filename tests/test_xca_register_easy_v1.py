import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements

PATH = Path(__file__).resolve().parents[1] / "submissions/xca_register_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("xca_register_easy_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class XCARegisterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_state_optimizer_and_parser(self):
        self.assertEqual(count_model_state_elements(self.model), 95_142)
        bundle = m.build_optimizer(self.model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
        ids = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
        role, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual((steps.item(), place[0, 1:5].tolist()), (64, [3,2,1,0]))
        self.model.eval(); out, info = self.model(ids)
        self.assertEqual((out.shape, info["macrosteps"]), ((1,14,17), 64))

    def test_interleaving_operator_oracle_and_gradient(self):
        tr = self.model.transition
        present = torch.tensor([[1,1,0,0]], dtype=torch.bool)
        reg = tr.make_register(torch.tensor([[1,2,0,0]]), present,
            torch.nn.functional.one_hot(torch.tensor([[3,4,0,0]]), 10).float(), present)
        self.assertEqual(reg.shape, (1,8,96))
        q = torch.randn(2,6,8,16, requires_grad=True); k = torch.randn_like(q, requires_grad=True)
        got = tr.tick.channel_operator(q, k)
        oracle = ((torch.nn.functional.normalize(k, dim=-2).transpose(-2,-1) @
                   torch.nn.functional.normalize(q, dim=-2)) *
                  tr.tick.temperature[None,:,None,None]).softmax(-2)
        torch.testing.assert_close(got, oracle)
        got.sum().backward(); self.assertIsNotNone(q.grad); self.assertIsNotNone(k.grad)

    def test_ticks_parity_permutation_determinism_and_all_gradients(self):
        ids = torch.tensor([[2,8,9,3,10,4,10,5,0], [2,11,3,12,13,4,9,5,0]])
        calls = []
        hook = self.model.transition.tick.register_forward_hook(lambda *args: calls.append(1))
        self.model.train(); a, _ = self.model(ids); hook.remove()
        self.assertEqual(len(calls), 3)
        b, _ = self.model(ids.flip(0)); torch.testing.assert_close(a[0], b[1]); torch.testing.assert_close(a[1], b[0])
        c, _ = self.model(ids); torch.testing.assert_close(a, c)
        self.model.eval(); d, _ = self.model(ids); torch.testing.assert_close(a, d)
        self.model.train(); a[:,:,7:17].sum().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_rejections_and_source(self):
        for bad in ([[2,8,9,10,11,12,3,7,4,7,5]], [[2,8,3,7,4,13,12,5]]):
            with self.assertRaises(ValueError): self.model(torch.tensor(bad))
        source = PATH.read_text().lower()
        for forbidden in ("einsum", "remainder", "cache", "cross_entropy", "argmax", "conv1d"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__": unittest.main()
