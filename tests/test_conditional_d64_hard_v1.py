import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, TokenLossBatch, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "conditional_d64_hard_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("conditional_d64_hard_v1", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConditionalD64HardTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.model = self.module.build_model(ModelSpec(17, 64, 500_000_000))

    def test_source_state_and_submission_contract(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        self.assertEqual(count_model_state_elements(self.model), 40_970)
        self.assertEqual(self.module.SUBMISSION.batch_size, 256)
        self.assertEqual(self.module.SUBMISSION.eval_batch_size, 512)
        self.assertIsNone(self.module.SUBMISSION.max_steps)

    def test_dynamic_width_alignment_max_t_and_gradients(self):
        ids = torch.tensor([
            [2, 8, 9, 10, 11, 3, 12, 7, 4, 8, 5, 0, 0, 0, 0, 0],
            [2, 8, 9, 10, 11, 12, 13, 3, 12, 7, 8, 9, 4, 13, 11, 5],
        ])
        logits, auxiliary = self.model(ids)
        self.assertEqual(logits.shape, (2, 16, 17))
        self.assertEqual(auxiliary["widths"].tolist(), [4, 6])
        self.assertEqual(auxiliary["x_widths"].tolist(), [2, 4])
        self.assertEqual(auxiliary["steps"].tolist(), [1, 64])
        self.assertEqual(auxiliary["macrosteps"], 64)
        labels = torch.full(ids.shape, -100)
        labels[0, 7:11] = torch.tensor([7, 8, 9, 10])
        labels[1, 10:16] = torch.tensor([7, 8, 9, 10, 11, 12])
        valid = labels.ne(-100)
        loss = self.module.token_training_loss(TokenLossBatch(
            logits=logits,
            labels=labels,
            valid_mask=valid,
            target_positions=None,
            auxiliary=auxiliary,
        ))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ))

    def test_evaluation_is_deterministic_and_pure(self):
        ids = torch.tensor([[2, 8, 9, 10, 11, 12, 13, 3, 12, 7, 8, 9, 4, 10, 5, 0]])
        self.model.eval()
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        with torch.no_grad():
            first, _ = self.model(ids)
            second, _ = self.model(ids)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(before[name], value) for name, value in self.model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
