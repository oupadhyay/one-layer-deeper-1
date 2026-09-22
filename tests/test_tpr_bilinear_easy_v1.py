import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/tpr_bilinear_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("tpr_bilinear_easy_v1", PATH)
c = importlib.util.module_from_spec(loader)
loader.loader.exec_module(c)


def spec(): return ModelSpec(17, 20, 250_000)


def row(n=(1, 2, 3, 4), x=(5, 6), t=(3,), length=20):
    v = [c.N, *(d + c.DIGIT for d in n), c.X, *(d + c.DIGIT for d in x),
         c.T, *(d + c.DIGIT for d in t)]
    return torch.tensor([v + [c.PAD] * (length - len(v))])


class TPRBilinearTests(unittest.TestCase):
    def test_contract_parser_state_optimizer_and_t64(self):
        validate_submission_source(PATH.name, PATH.read_text(), 256 * 1024, required_filename=None)
        model = c.build_model(spec())
        nd, np, xd, xp, steps = model.parse(row(t=(6, 4)), row(t=(6, 4)).ne(0))
        self.assertEqual(nd.tolist(), [[4, 3, 2, 1]])
        self.assertEqual(xd.tolist(), [[6, 5, 0, 0]])
        self.assertEqual((np.tolist(), xp.tolist(), steps.item()),
                         ([[True]*4], [[True, True, False, False]], 64))
        self.assertEqual(count_model_state_elements(model), c.STATE_ELEMENTS)
        self.assertEqual(c.STATE_ELEMENTS, 14186)
        bundle = c.build_optimizer(model, OptimizerSpec(10, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
        self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups], [.01, 0.])
        model.eval()
        with torch.no_grad(): _, info = model(row(t=(6, 4)))
        self.assertEqual((info["macrosteps"], info["microticks"]), (64, 192))
        with self.assertRaisesRegex(ValueError, "widths"):
            model(row(n=(1, 2, 3, 4, 5)))
        with self.assertRaisesRegex(ValueError, "T exceeds"):
            model(row(t=(6, 5)))

    def test_outer_binding_absence_zero_and_kronecker_oracles(self):
        model = c.build_model(spec())
        digits = torch.tensor([[0, 1, 0, 0]])
        present = torch.tensor([[True, True, False, False]])
        bound = model.bind(digits, present, 0)
        oracle = sum(torch.outer(model.digit_embedding(digits[0, p])
                                 + model.presence_embedding(present[0, p].long()),
                                 model.field_role.weight[0] + model.place_role.weight[p])
                     for p in range(4))[None]
        torch.testing.assert_close(bound, oracle)
        self.assertFalse(torch.equal(model.presence_embedding(torch.tensor(0)),
                                     model.presence_embedding(torch.tensor(1))))
        z = torch.randn(2, 32, 16, requires_grad=True)
        got = model.cell.kronecker(z, 0)
        expected = torch.stack([model.cell.left[k, 0] @ z @ model.cell.right[k, 0]
                                for k in range(4)], 1)
        torch.testing.assert_close(got, expected)
        torch.autograd.grad(got.square().sum(), z)[0]

    def test_tying_purity_gradients_and_determinism(self):
        torch.manual_seed(9)
        model = c.build_model(spec()).train()
        batch = torch.cat([row(), row(n=(8, 7), x=(9,), t=(2,))])
        calls = []
        hook = model.cell.register_forward_hook(lambda *_: calls.append(1))
        out, info = model(batch)
        hook.remove()
        self.assertEqual((len(calls), info["macrosteps"], info["microticks"]), (9, 3, 9))
        out[..., c.DIGIT:].sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        model.eval()
        with torch.no_grad():
            together = model(batch)[0]
            separate = torch.cat([model(batch[:1])[0], model(batch[1:])[0]])
            again = model(batch)[0]
        torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(together, again, atol=0, rtol=0)


if __name__ == "__main__": unittest.main()
