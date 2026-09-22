import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "universal_recurrent_transformer_easy_v1",
    "universal_recurrent_transformer_easy_v2_label_smoothing",
    "universal_recurrent_transformer_easy_v3_dropout",
    "universal_recurrent_transformer_easy_v4_d128",
)


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


MODULES = {name: load(name) for name in NAMES}


def model_spec(length=8):
    return ModelSpec(17, length, 500_000_000)


class UniversalRecurrentTransformerEasyVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_sources_are_legal_self_contained_and_generic(self):
        forbidden = ("N_TOKEN", "X_TOKEN", "T_TOKEN", "DIGIT", "parse_", "solver",
                     "arithmetic", "hard_feedback", "generated_data", "training_loop",
                     "from submissions")
        for name in NAMES:
            path = ROOT / "submissions" / name / "submission.py"
            source = path.read_text(encoding="utf-8")
            validate_submission_source(path.name, source, 256 * 1024)
            for text in forbidden:
                self.assertNotIn(text, source, (name, text))

    def test_controlled_mechanisms_and_exact_counts(self):
        models = {name: module.build_model(model_spec()) for name, module in MODULES.items()}
        counts = {name: count_model_state_elements(model) for name, model in models.items()}
        self.assertEqual(counts[NAMES[0]], 1_981_184)
        self.assertEqual(counts[NAMES[1]], counts[NAMES[0]])
        self.assertEqual(counts[NAMES[2]], counts[NAMES[0]])
        self.assertEqual(counts[NAMES[3]], 499_072)
        self.assertLess(counts[NAMES[3]], counts[NAMES[0]])
        self.assertIsNotNone(MODULES[NAMES[1]].SUBMISSION.training_loss)
        for name in (NAMES[0], NAMES[2], NAMES[3]):
            self.assertIsNone(MODULES[name].SUBMISSION.training_loss)
        logits = torch.randn(7, 17, requires_grad=True)
        labels = torch.arange(7)
        loss = MODULES[NAMES[1]].training_loss(logits, labels, None)
        expected = F.cross_entropy(logits, labels, label_smoothing=0.1)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, expected)

    def test_dropout_modes_gradients_and_shared_recurrence(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        for name in (NAMES[2], NAMES[3]):
            module, model = MODULES[name], MODULES[name].build_model(model_spec())
            calls = []
            hook = model.recurrent_block.register_forward_hook(lambda *args: calls.append(1))
            model.train()
            first, _ = model(ids)
            second, _ = model(ids)
            hook.remove()
            self.assertEqual(len(calls), 32)
            self.assertEqual(sum(isinstance(m, module.RecurrentBlock) for m in model.modules()), 1)
            F.cross_entropy(first.reshape(-1, 17), targets.reshape(-1)).backward()
            for parameter_name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, parameter_name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, parameter_name)
            if name == NAMES[2]:
                self.assertFalse(torch.equal(first, second))
            model.eval()
            with torch.no_grad():
                eval_first, _ = model(ids)
                eval_second, _ = model(ids)
            torch.testing.assert_close(eval_first, eval_second, rtol=0, atol=0)

    def test_contract_optimizer_padding_shape_and_cap(self):
        for name, module in MODULES.items():
            model = module.build_model(model_spec()).eval()
            submission = module.SUBMISSION
            self.assertEqual((submission.batch_size, submission.eval_batch_size,
                              submission.max_steps), (512, 1024, None))
            bundle = module.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
            optimizer = bundle.optimizer
            self.assertEqual((optimizer.defaults["lr"], optimizer.defaults["betas"],
                              optimizer.defaults["eps"]), (8e-4, (0.9, 0.95), 1e-8))
            self.assertEqual([group["weight_decay"] for group in optimizer.param_groups],
                             [0.05, 0.0])
            grouped = [p for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
            mask = torch.tensor([[1, 1, 1, 0]])
            with torch.no_grad():
                a, _ = model(torch.tensor([[1, 2, 3, 4]]), mask)
                b, _ = model(torch.tensor([[1, 2, 3, 9]]), mask)
            self.assertEqual(a.shape, (1, 4, 17))
            torch.testing.assert_close(a, b, rtol=0, atol=2e-6)
            with self.assertRaises(ValueError):
                module.build_model(model_spec(65))


if __name__ == "__main__":
    unittest.main()
