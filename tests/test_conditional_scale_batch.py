import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
MEDIUM = (
    "conditional_medium_focal05_lr9_v1",
    "conditional_medium_plain_lr9_v1",
    "conditional_medium_focal10_lr9_v1",
    "conditional_medium_focal05_lr6_v1",
    "conditional_medium_focal05_batch128_v1",
    "conditional_medium_focal05_d96_v1",
    "conditional_medium_focal05_lr10_v1",
    "conditional_medium_focal05_warm32_v1",
    "conditional_medium_focal05_batch512_v1",
    "conditional_medium_focal05_d128_v1",
)
HARD = "conditional_dynamic_hard_v1"


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class ConditionalScaleBatchTests(unittest.TestCase):
    def check_model(self, name, max_len, widths):
        path, module = load(name)
        source = path.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(path.name, source, 256_000), "submission.py")
        model = module.build_model(ModelSpec(17, max_len, 500_000_000))
        self.assertLess(count_model_state_elements(model), 500_000_000)
        for width in widths:
            row = [2] + [8] * width + [3] + [9] * width + [4, 9, 5]
            ids = torch.tensor([row])
            first, info = model(ids)
            second, _ = model(ids)
            torch.testing.assert_close(first, second)
            self.assertEqual(info["widths"].item(), width)
            self.assertEqual(first.shape, (1, len(row), 17))
        model.zero_grad()
        first[:, :, 7:].sum().backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))
        self.assertEqual(module.SUBMISSION.max_steps, None)

    def test_medium_packages(self):
        for name in MEDIUM:
            with self.subTest(name=name):
                self.check_model(name, 32, range(1, 6))

    def test_hard_package_is_dynamic_width(self):
        self.check_model(HARD, 64, (1, 4, 7, 10))


if __name__ == "__main__":
    unittest.main()
