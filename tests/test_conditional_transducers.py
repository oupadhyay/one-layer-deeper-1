import importlib.util
from pathlib import Path
import unittest

import torch
from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "conditional_cross_transducer_easy_v1": (148_138, 256),
    "conditional_cross_transducer_easy_v2_shared": (43_018, 512),
    "conditional_global_transducer_easy_v1": (40_970, 512),
    "conditional_global_transducer_easy_v2_batch256": (40_970, 256),
    "conditional_opt_lr3_easy_v1": (40_970, 256),
    "conditional_opt_lr4_easy_v1": (40_970, 256),
}


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


class ConditionalTransducerTests(unittest.TestCase):
    def test_contract_alignment_determinism_and_gradients(self):
        ids = torch.tensor([
            [2, 8, 9, 10, 11, 3, 12, 7, 4, 9, 5, 0, 0],
            [2, 11, 12, 13, 14, 3, 8, 9, 4, 9, 5, 0, 0],
        ])
        for name, (state, batch_size) in CASES.items():
            with self.subTest(name=name):
                path, module = load(name)
                self.assertEqual(validate_submission_source(path.name, path.read_text(), 256_000), "submission.py")
                model = module.build_model(ModelSpec(17, 24, 250_000))
                self.assertEqual(count_model_state_elements(model), state)
                self.assertEqual(module.SUBMISSION.batch_size, batch_size)
                model.train()
                first, info = model(ids)
                second, _ = model(ids)
                permuted, _ = model(ids.flip(0))
                self.assertEqual(info["macrosteps"], 2)
                torch.testing.assert_close(first, second)
                torch.testing.assert_close(first, permuted.flip(0))
                model.eval()
                evaluated, _ = model(ids)
                torch.testing.assert_close(first, evaluated)
                model.train()
                first[:, :, 7:].sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
