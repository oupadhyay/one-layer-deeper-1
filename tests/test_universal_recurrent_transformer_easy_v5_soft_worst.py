import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SoftWorstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v4 = load("universal_recurrent_transformer_easy_v4_d128")
        cls.v5 = load("universal_recurrent_transformer_easy_v5_soft_worst")

    def test_loss_formula_mask_gradients_and_example_weighting(self):
        logits = torch.tensor([
            [[2., 0., -1.], [0., 1., 0.], [99., -99., 2.]],
            [[1., 0., 0.], [4., 1., 0.], [-50., 50., 0.]],
        ], requires_grad=True)
        labels = torch.tensor([[0, 1, 999], [2, 999, 999]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        loss = self.v5.token_training_loss(SimpleNamespace(
            logits=logits, labels=labels, valid_mask=mask))
        valid_losses = [
            F.cross_entropy(logits[0, :2], labels[0, :2], reduction="none"),
            F.cross_entropy(logits[1, :1], labels[1, :1], reduction="none"),
        ]
        expected = torch.stack([
            .5 * x.mean() + .25 * (torch.logsumexp(x / .5, 0) - x.new_tensor(x.numel()).log())
            for x in valid_losses
        ]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad[mask]).all())
        self.assertGreater(logits.grad[mask].abs().sum().item(), 0)
        self.assertEqual(logits.grad[~mask].abs().sum().item(), 0)

    def test_architecture_optimizer_contract_and_source(self):
        spec = ModelSpec(17, 8, 500_000_000)
        torch.manual_seed(1)
        old = self.v4.build_model(spec)
        torch.manual_seed(1)
        new = self.v5.build_model(spec)
        self.assertEqual(count_model_state_elements(new), 499_072)
        self.assertEqual({k: v.shape for k, v in old.state_dict().items()},
                         {k: v.shape for k, v in new.state_dict().items()})
        for module, model in ((self.v4, old), (self.v5, new)):
            optimizer = module.build_optimizer(model, OptimizerSpec(10., "cpu")).optimizer
            self.assertEqual((optimizer.defaults["lr"], optimizer.defaults["betas"],
                              optimizer.defaults["eps"]), (8e-4, (0.9, .95), 1e-8))
            self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [.05, 0.])
        calls = []
        hook = new.recurrent_block.register_forward_hook(lambda *args: calls.append(1))
        output, auxiliary = new(torch.tensor([[1, 2, 3, 4]]), torch.ones(1, 4))
        hook.remove()
        self.assertEqual(output.shape, (1, 4, 17))
        self.assertEqual((len(calls), auxiliary), (16, {"iterations": 16, "state_length": 36}))
        path = ROOT / "submissions/universal_recurrent_transformer_easy_v5_soft_worst/submission.py"
        source = path.read_text(encoding="utf-8")
        validate_submission_source(path.name, source, 256 * 1024)
        for forbidden in ("from submissions", "TokenLossBatch", "training_loop", "backward("):
            self.assertNotIn(forbidden, source)

    def test_legacy_fallback(self):
        self.assertIs(self.v5.SUBMISSION.training_loss, self.v5.legacy_training_loss)
        logits = torch.randn(4, 7)
        labels = torch.arange(4)
        torch.testing.assert_close(self.v5.legacy_training_loss(logits, labels, None),
                                   F.cross_entropy(logits, labels))


if __name__ == "__main__":
    unittest.main()
