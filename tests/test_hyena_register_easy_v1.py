import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements

PATH = Path(__file__).resolve().parents[1] / "submissions/hyena_register_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("hyena_register_easy_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class HyenaRegisterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_state_architecture_initialization_and_optimizer(self):
        self.assertEqual(count_model_state_elements(self.model), 105226)
        tick = self.model.transition.tick
        self.assertEqual((tick.conv.kernel_size, tick.conv.padding, tick.conv.groups), ((15,), (7,), 192))
        torch.testing.assert_close(tick.skip, torch.ones_like(tick.skip))
        bundle = m.build_optimizer(self.model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)

    def test_parser_alignment_t64_and_rejections(self):
        ids = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
        role, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual(steps.item(), 64)
        self.assertEqual(place[0, 1:5].tolist(), [3, 2, 1, 0])
        self.model.eval(); out, info = self.model(ids, ids.ne(0))
        self.assertEqual(info["macrosteps"], 64)
        self.assertEqual(out.shape, (1, 14, 17))
        for bad in ([[2,8,9,10,11,12,3,7,4,7,5]], [[2,8,3,7,4,13,12,5]]):
            with self.assertRaises(ValueError): self.model(torch.tensor(bad))

    def test_register_interleaving_absence_and_zero(self):
        tr = self.model.transition
        n = torch.tensor([[1, 2, 0, 0]]); present = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        x = torch.nn.functional.one_hot(torch.tensor([[3, 0, 0, 0]]), 10).float()
        xp = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        reg = tr.make_register(n, present, x, xp)
        expected_n0 = tr.digit_embedding(n[:, 0]) + tr.role_embedding.weight[0] + tr.place_embedding.weight[0] + tr.presence_embedding.weight[1]
        torch.testing.assert_close(reg[:, 0], expected_n0)
        self.assertFalse(torch.equal(reg[:, 3], reg[:, 5]))
        self.assertEqual(reg.shape, (1, 8, 128))

    def test_convolution_oracle_and_single_tick(self):
        tick = self.model.transition.tick
        x = torch.randn(2, 8, 128)
        calls = []
        hook = tick.register_forward_hook(lambda *args: calls.append(1))
        self.model.train()
        ids = torch.tensor([[2,8,3,9,4,10,5,0], [2,9,3,8,4,8,5,0]])
        self.model(ids); hook.remove()
        self.assertEqual(len(calls), 3)
        z = tick.norm(x); q, k, v = tick.qkv(z).chunk(3, -1); r = k*v
        oracle = torch.nn.functional.conv1d(r.transpose(1,2), tick.conv.weight, padding=7, groups=192).transpose(1,2)
        expected = x + tick.out(q * (oracle + tick.skip*r))
        torch.testing.assert_close(tick(x), expected)

    def test_soft_transition_parity_purity_determinism_and_gradients(self):
        ids = torch.tensor([[2,8,9,3,10,4,10,5,0], [2,11,3,12,13,4,9,5,0]])
        self.model.train(); a, _ = self.model(ids)
        b, _ = self.model(ids.flip(0)); torch.testing.assert_close(a[0], b[1]); torch.testing.assert_close(a[1], b[0])
        c, _ = self.model(ids); torch.testing.assert_close(a, c)
        self.model.eval(); d, _ = self.model(ids)
        torch.testing.assert_close(a, d)
        self.model.train(); loss = a[:, :, 7:17].sum(); loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_source_has_no_alternate_mechanisms(self):
        source = PATH.read_text().lower()
        for forbidden in ("einsum", "matmul", "remainder", "cache", "cross_entropy", "argmax"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__": unittest.main()
