import importlib.util
from pathlib import Path
import unittest

import torch
from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "sgu_relaxed_feedback_easy_v1": (79_682, 24),
    "sgu_toeplitz_easy_v1": (79_669, 24),
    "sgu_attention_easy_v1": (86_834, 24),
    "sgu_biscan_easy_v1": (135_346, 24),
    "sgu_medium_relaxed_v1": (80_033, 28),
}


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class DepthGeneralizationCampaignTests(unittest.TestCase):
    def test_fixed_width_candidates(self):
        ids = torch.tensor([[2, 8, 9, 10, 11, 3, 12, 7, 4, 13, 11, 5, 0, 0]])
        for name, (state, max_len) in CASES.items():
            with self.subTest(name=name):
                path, module = load(name)
                self.assertEqual(validate_submission_source(path.name, path.read_text(), 256_000), "submission.py")
                model = module.build_model(ModelSpec(17, max_len, 250_000))
                self.assertEqual(count_model_state_elements(model), state)
                model.eval()
                logits, info = model(ids)
                self.assertEqual(info["macrosteps"], 64)
                model.train()
                logits, _ = model(ids)
                logits[:, :, 7:].sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_hard_candidate_is_dynamic_width(self):
        path, module = load("dynamic_biscan_hard_v1")
        self.assertEqual(validate_submission_source(path.name, path.read_text(), 256_000), "submission.py")
        model = module.build_model(ModelSpec(17, 64, 500_000_000))
        self.assertEqual(count_model_state_elements(model), 134_880)
        for ids in (
            torch.tensor([[2, 8, 9, 10, 11, 3, 12, 7, 4, 10, 5, 0]]),
            torch.tensor([[2, 8, 9, 10, 11, 12, 13, 3, 12, 7, 8, 9, 4, 10, 5, 0]]),
        ):
            model.eval()
            logits, info = model(ids)
            self.assertEqual(logits.shape[:2], ids.shape)
            self.assertEqual(info["macrosteps"], 3)


if __name__ == "__main__":
    unittest.main()
