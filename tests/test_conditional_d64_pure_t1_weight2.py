import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, TokenLossBatch, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "conditional_d64_pure_t1_weight2_easy_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("conditional_d64_pure_t1_weight2", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PureT1WeightTests(unittest.TestCase):
    def test_contract_gradients_and_exact_weighted_reduction(self):
        module = load()
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        model = module.build_model(ModelSpec(17, 64, 500_000_000))
        self.assertEqual(count_model_state_elements(model), 40_970)
        ids = torch.tensor([
            [2, 8, 9, 10, 11, 3, 12, 7, 4, 8, 5, 0, 0],
            [2, 8, 9, 10, 11, 12, 13, 3, 12, 7, 4, 9, 5],
        ])
        logits, auxiliary = model(ids)
        self.assertEqual(auxiliary["steps"].tolist(), [1, 2])
        labels = torch.full(ids.shape, -100)
        labels[0, 7:11] = torch.tensor([7, 8, 9, 10])
        labels[1, 7:13] = torch.tensor([7, 8, 9, 10, 11, 12])
        valid = labels.ne(-100)
        batch = TokenLossBatch(logits, labels, valid, None, auxiliary)
        loss = module.token_training_loss(batch)
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
        focal = ce * (1 - torch.exp(-ce)).pow(0.5)
        expected = (2 * (focal[0] * valid[0]).sum() + (focal[1] * valid[1]).sum()) / (2 * valid[0].sum() + valid[1].sum())
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNone(module.SUBMISSION.max_steps)
        self.assertEqual(module.SUBMISSION.batch_size, 256)


if __name__ == "__main__":
    unittest.main()
