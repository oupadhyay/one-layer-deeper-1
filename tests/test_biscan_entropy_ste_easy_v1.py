import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "biscan_entropy_ste_easy_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("biscan_entropy_ste", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(module, t, length=20):
    values = [module.N, 10, 9, module.X, 13, 8, module.T,
              *(module.DIGIT + int(d) for d in str(t))]
    return torch.tensor([values + [module.PAD] * (length - len(values))])


class BiScanEntropySteTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.model = self.module.build_model(ModelSpec(17, 20, 20_000_000))

    def test_source_state_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        self.assertEqual(count_model_state_elements(self.model), 63_178)

    def test_training_feedback_is_hard_with_finite_gradients(self):
        self.model.train()
        logits, auxiliary = self.model(row(self.module, 3))
        probabilities = auxiliary["digit_probabilities"]
        torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))
        self.assertTrue(torch.equal(probabilities, probabilities.round()))
        logits[:, 9:11, 7:].sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
