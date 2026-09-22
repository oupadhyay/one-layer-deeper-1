import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "submissions/compressed_axial_e5_large_v2_safe_batch/submission.py"
V3_PATH = ROOT / "submissions/compressed_axial_e5_large_v3_cuda_safe/submission.py"


def load(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


v2 = load("compressed_axial_e5_large_v2", V2_PATH)
v3 = load("compressed_axial_e5_large_v3", V3_PATH)


class CompressedAxialE5LargeV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ModelSpec(17, 16, 100_000)
        cls.inputs = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 8, 0, 0, 0],
            [2, 16, 7, 8, 9, 3, 10, 11, 12, 13, 4, 10],
        ])

    def models(self):
        torch.manual_seed(831)
        old = v2.build_model(self.spec)
        state = old.state_dict()
        torch.manual_seed(19)
        new = v3.build_model(self.spec)
        new.load_state_dict(state)
        return old, new

    def test_parser_and_exact_field_extraction(self):
        old, new = self.models()
        mask = self.inputs.ne(0)
        old_parse = old.parse(self.inputs, mask)
        new_parse = new.parse(self.inputs, mask)
        for actual, expected in zip(new_parse, old_parse):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(new_parse[0].tolist(), [
            [0, 1, 1, 1, 0, 2, 2, 0, 3, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 0, 3],
        ])
        self.assertEqual(new_parse[1].tolist(), [
            [0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 3, 2, 1, 0, 0, 3, 2, 1, 0, 0, 0],
        ])
        self.assertEqual(new_parse[2].tolist(), [1, 3])
        old_prepared = old._prepare(self.inputs, None)
        new_prepared = new._prepare(self.inputs, None)
        for actual, expected in zip(new_prepared, old_prepared):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(new_prepared[2].tolist()[:2], [[3, 2, 3] + [0] * 13, [2, 1, 0, 9] + [0] * 12])
        self.assertEqual(new_prepared[3].tolist()[:2], [[5, 1] + [0] * 14, [6, 5, 4, 3] + [0] * 12])
        self.assertEqual(new_prepared[2].dtype, torch.long)
        self.assertEqual(new_prepared[3].dtype, torch.long)

    def test_outputs_auxiliaries_and_gradients_match(self):
        old, new = self.models()
        old.train(); new.train()
        old_logits, old_aux = old(self.inputs)
        new_logits, new_aux = new(self.inputs)
        torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
        self.assertEqual(set(new_aux), set(old_aux))
        for key in new_aux:
            if torch.is_tensor(new_aux[key]):
                torch.testing.assert_close(new_aux[key], old_aux[key], rtol=0, atol=0)
            else:
                self.assertEqual(new_aux[key], old_aux[key])
        targets = torch.tensor([8, 9])
        F.cross_entropy(old_logits[:, -1], targets).backward()
        F.cross_entropy(new_logits[:, -1], targets).backward()
        for old_parameter, new_parameter in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0)

    def test_contract_optimizer_submission_and_source(self):
        old, new = self.models()
        self.assertEqual(count_model_state_elements(new), 13162)
        self.assertEqual(count_model_state_elements(new), count_model_state_elements(old))
        old_bundle = v2.build_optimizer(old, OptimizerSpec(1600, "cpu"))
        new_bundle = v3.build_optimizer(new, OptimizerSpec(1600, "cpu"))
        for old_group, new_group in zip(old_bundle.optimizer.param_groups, new_bundle.optimizer.param_groups):
            self.assertEqual(new_group["weight_decay"], old_group["weight_decay"])
            self.assertEqual(new_group["lr"], old_group["lr"])
            self.assertEqual(new_group["betas"], old_group["betas"])
        self.assertEqual(new_bundle.scheduler.state_dict(), old_bundle.scheduler.state_dict())
        for field in ("batch_size", "eval_batch_size", "max_steps"):
            self.assertEqual(getattr(v3.SUBMISSION, field), getattr(v2.SUBMISSION, field))
        self.assertEqual((v3.SUBMISSION.batch_size, v3.SUBMISSION.eval_batch_size, v3.SUBMISSION.max_steps),
                         (64, 128, None))
        source = V3_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.einsum", source)
        self.assertIn("digits = (assignment.to(values.dtype) * values[:, :, None]).sum(1)", source)
        validate_submission_source(V3_PATH.name, source, 256 * 1024, required_filename=None)


if __name__ == "__main__":
    unittest.main()
