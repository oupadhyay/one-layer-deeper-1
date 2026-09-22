import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "categorical_multiplicative_transducer_easy_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("categorical_multiplicative", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CategoricalMultiplicativeTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.model = self.module.build_model(ModelSpec(17, 20, 500_000_000))

    def test_source_and_state(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        count = count_model_state_elements(self.model)
        self.assertGreater(count, 2_000_000)
        self.assertLess(count, 5_000_000)

    def test_end_to_end_gradients(self):
        ids = torch.tensor([[2, 10, 9, 3, 13, 8, 4, 8, 0, 0, 0, 0]])
        logits, _ = self.model(ids)
        F.cross_entropy(logits[0, 6:8, 7:], torch.tensor([1, 2])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))

    def test_n_and_x_are_not_reconstructed_as_integers(self):
        source = PATH.read_text(encoding="utf-8")
        parse_body = source[source.index("def parse"):source.index("def forward")]
        self.assertNotIn("n_value", parse_body)
        self.assertNotIn("x_value", parse_body)


if __name__ == "__main__":
    unittest.main()
