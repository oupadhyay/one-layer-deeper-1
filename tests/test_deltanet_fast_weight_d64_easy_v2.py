import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements

PATH = Path(__file__).resolve().parents[1] / "submissions/deltanet_fast_weight_d64_easy_v2/submission.py"
loader = importlib.util.spec_from_file_location("deltanet_d64_easy", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


def sequential(q, k, v, beta, state):
    q = torch.nn.functional.normalize(torch.nn.functional.silu(q), dim=-1)
    k = torch.nn.functional.normalize(torch.nn.functional.silu(k), dim=-1)
    v, beta = torch.nn.functional.silu(v), beta.sigmoid()
    outputs = []
    for i in range(q.shape[-2]):
        read = torch.einsum("...k,...kv->...v", k[..., i, :], state)
        error = v[..., i, :] - read
        state = state + k[..., i, :].unsqueeze(-1) @ (beta[..., i, None] * error).unsqueeze(-2)
        outputs.append(torch.einsum("...k,...kv->...v", q[..., i, :], state) / (16 ** .5))
    return torch.stack(outputs, -2), state


class DeltaNetTests(unittest.TestCase):
    def test_chunk_oracle_outputs_state_and_gradients(self):
        torch.manual_seed(4)
        values = [torch.randn(2, 3, 12, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        beta = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
        state = torch.randn(2, 3, 16, 16, dtype=torch.float64, requires_grad=True)
        oracle = sequential(*values, beta, state)
        # Contract algebra is FP32, so compare against an FP32 sequential oracle.
        copied = [x.detach().float().requires_grad_() for x in (*values, beta, state)]
        expected = sequential(*copied)
        actual = m.delta_chunk(*copied)
        torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)
        weights = [torch.randn_like(x) for x in actual]
        ga = torch.autograd.grad(sum((x*w).sum() for x,w in zip(actual, weights)), copied)
        ge = torch.autograd.grad(sum((x*w).sum() for x,w in zip(expected, weights)), copied)
        for a, e in zip(ga, ge): torch.testing.assert_close(a, e, atol=8e-5, rtol=8e-5)
        self.assertEqual(oracle[0].dtype, torch.float64)

    def test_state_optimizer_tying_and_reset(self):
        model = m.build_model(ModelSpec(17, 20, 1_000_000))
        self.assertEqual(count_model_state_elements(model), 43594)
        self.assertEqual(len([x for x in model.modules() if isinstance(x, m.DeltaBlock)]), 1)
        bundle = m.build_optimizer(model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        optimized = {id(p) for group in bundle.optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.parameters()})
        n = torch.randint(0, 10, (2, 4)); present = torch.ones(2, 4, dtype=torch.bool)
        x = torch.nn.functional.one_hot(n, 10).float()
        model.eval()
        with torch.no_grad():
            a, da = model.transition(n, present, x, present)
            b, db = model.transition(n, present, x, present)
        torch.testing.assert_close(a, b); torch.testing.assert_close(da["state"], db["state"])

    def test_parser_alignment_and_macrosteps(self):
        # N=1234, X=56, T=3; answer positions are immediately before ANS.
        ids = torch.tensor([[2,8,9,10,11,3,12,13,4,10,5,0,0,0]])
        model = m.build_model(ModelSpec(17, 14, 1_000_000))
        model.train(); out, info = model(ids, ids.ne(0))
        self.assertEqual((info["widths"].item(), info["x_widths"].item(), info["steps"].item()), (4,2,3))
        self.assertEqual(info["macrosteps"], 3)
        self.assertTrue(torch.isfinite(out[:, 7:11, 7:17]).all())
        out[:, 7:11, 7:17].sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_t64_eval_is_deterministic_and_pure(self):
        ids = torch.tensor([[2, 8, 3, 9, 4, 13, 11, 5]])  # N=1, X=2, T=64
        model = m.build_model(ModelSpec(17, 8, 1_000_000)).eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            a, ai = model(ids)
            b, bi = model(ids)
        self.assertEqual(ai["macrosteps"], 64)
        self.assertEqual(bi["macrosteps"], 64)
        torch.testing.assert_close(a, b)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_source_is_self_contained_and_d64_only(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertNotIn("deltanet_fast_weight_easy_v1", source)
        self.assertIn("D_MODEL, HEADS, HEAD_DIM, TOKENS = 4, 64, 4, 16, 12", source)


if __name__ == "__main__": unittest.main()
