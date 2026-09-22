import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, TokenLossBatch, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "biscan_entropy_t1_weight4_easy_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("biscan_entropy_t1_weight4", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BiScanEntropyT1Weight4Tests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_source_state_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        model = self.module.build_model(ModelSpec(17, 20, 20_000_000))
        self.assertEqual(count_model_state_elements(model), 63_178)
        self.assertEqual((self.module.SUBMISSION.batch_size,
                          self.module.SUBMISSION.eval_batch_size), (64, 128))

    def test_t1_tokens_receive_fourfold_weight(self):
        logits = torch.tensor([[[3., 0.], [0., 3.]], [[1., 0.], [0., 1.]]], requires_grad=True)
        labels = torch.tensor([[0, 1], [0, 1]])
        valid = torch.ones_like(labels, dtype=torch.bool)
        entropy = logits.new_zeros(())
        batch = TokenLossBatch(logits, labels, valid, torch.zeros_like(labels),
                               {"parsed_steps": torch.tensor([1, 2]),
                                "feedback_entropy": entropy})
        actual = self.module.token_training_loss(batch)
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        expected = (4 * losses[0].sum() + losses[1].sum()) / 10
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
