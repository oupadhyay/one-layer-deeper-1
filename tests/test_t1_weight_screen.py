import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, TokenLossBatch, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "conditional_t1_weight2_easy_v1",
    "conditional_t1_weight4_easy_v1",
    "conditional_t1_weight8_easy_v1",
    "conditional_t1_weight16_easy_v1",
    "conditional_t1_weight4_plain_easy_v1",
    "conditional_t1_weight4_d64_easy_v1",
)


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class T1WeightScreenTests(unittest.TestCase):
    def test_dynamic_contract_gradients_and_loss_weighting(self):
        for name in NAMES:
            with self.subTest(name=name):
                path, module = load(name)
                source = path.read_text(encoding="utf-8")
                self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
                model = module.build_model(ModelSpec(17, 64, 500_000_000))
                self.assertLess(count_model_state_elements(model), 500_000_000)
                ids = torch.tensor([
                    [2, 8, 9, 10, 11, 3, 12, 7, 4, 8, 5, 0, 0],
                    [2, 8, 9, 10, 11, 12, 13, 3, 12, 7, 4, 9, 5],
                ])
                logits, auxiliary = model(ids)
                self.assertEqual(auxiliary["steps"].tolist(), [1, 2])
                labels = torch.full(ids.shape, -100)
                labels[:, -1] = 7
                valid = labels.ne(-100)
                loss = module.token_training_loss(TokenLossBatch(
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
                    for parameter in model.parameters()
                ))
                self.assertIsNone(module.SUBMISSION.max_steps)
                self.assertEqual(module.SUBMISSION.batch_size, 256)


if __name__ == "__main__":
    unittest.main()
