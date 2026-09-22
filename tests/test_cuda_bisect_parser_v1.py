import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "submissions/baseline_adamw/submission.py"
V3_PATH = ROOT / "submissions/compressed_axial_e5_large_v3_cuda_safe/submission.py"
BISECT_PATH = ROOT / "submissions/cuda_bisect_parser_v1/submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("cuda_bisect_baseline", BASE_PATH)
v3 = load("cuda_bisect_v3", V3_PATH)
bisect = load("cuda_bisect_parser", BISECT_PATH)


class CudaBisectParserV1Tests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(17, 16, 1_000_000)
        self.inputs = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 8, 0, 0, 0],
            [2, 16, 7, 8, 9, 3, 10, 11, 12, 13, 4, 10],
        ])

    def models(self):
        torch.manual_seed(712)
        old = baseline.build_model(self.spec)
        torch.manual_seed(712)
        new = bisect.build_model(self.spec)
        return old, new

    def test_baseline_state_logits_gradients_and_optimizer_match(self):
        old, new = self.models()
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for key in old.state_dict():
            torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
        old_logits, _ = old(self.inputs)
        new_logits, _ = new(self.inputs)
        torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
        targets = torch.tensor([8, 9])
        F.cross_entropy(old_logits[:, -1], targets).backward()
        F.cross_entropy(new_logits[:, -1], targets).backward()
        for old_parameter, new_parameter in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0)
        old_opt = baseline.build_optimizer(old, OptimizerSpec(1600, "cpu")).optimizer
        new_opt = bisect.build_optimizer(new, OptimizerSpec(1600, "cpu")).optimizer
        for field in ("lr", "betas", "weight_decay", "capturable"):
            self.assertEqual(old_opt.param_groups[0][field], new_opt.param_groups[0][field])

    def test_parser_matches_v3_on_mixed_inputs(self):
        mask = self.inputs.ne(0)
        for actual, expected in zip(bisect.Model.parse(self.inputs, mask),
                                    v3.Model.parse(self.inputs, mask)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        new = bisect.build_model(self.spec)
        reference = v3.build_model(self.spec)
        prepared = new._prepare(self.inputs, None)
        v3_prepared = reference._prepare(self.inputs, None)
        for actual, expected in zip(prepared, (v3_prepared[2], v3_prepared[3],
                                               v3_prepared[4], v3_prepared[1])):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_source_and_submission_contract(self):
        source = BISECT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("einsum", source)
        self.assertIn("(assignment.to(values.dtype) * values[:, :, None]).sum(1)", source)
        validate_submission_source(BISECT_PATH.name, source, 256 * 1024, required_filename=None)
        self.assertEqual((bisect.SUBMISSION.batch_size, bisect.SUBMISSION.eval_batch_size,
                          bisect.SUBMISSION.max_steps), (64, 128, 1))
        self.assertEqual(bisect.MAX_STEPS, 64)


if __name__ == "__main__":
    unittest.main()
