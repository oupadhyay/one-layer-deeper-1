import ast
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "submissions/compressed_axial_e5_large_v3_cuda_safe/submission.py"
V4_PATH = ROOT / "submissions/compressed_axial_e5_smoke_v4/submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load("compressed_axial_e5_large_v3_cuda_safe", V3_PATH)
v4 = load("compressed_axial_e5_smoke_v4", V4_PATH)


class CompressedAxialE5SmokeV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ModelSpec(17, 16, 100_000)

    def test_only_documentation_and_step_limits_differ(self):
        old = ast.parse(V3_PATH.read_text(encoding="utf-8"))
        new = ast.parse(V4_PATH.read_text(encoding="utf-8"))
        old.body[0] = new.body[0]  # The required diagnostic-only module docstring.

        def normalize(tree):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "MAX_STEPS" for t in node.targets):
                        node.value = ast.Constant(value=0)
                    if any(isinstance(t, ast.Name) and t.id == "SUBMISSION" for t in node.targets):
                        for keyword in node.value.keywords:
                            if keyword.arg == "max_steps":
                                keyword.value = ast.Constant(value=0)
            return ast.dump(tree, include_attributes=False)

        self.assertEqual(normalize(old), normalize(new))
        self.assertEqual(v3.MAX_STEPS, 64)
        self.assertEqual(v4.MAX_STEPS, 3)
        self.assertIsNone(v3.SUBMISSION.max_steps)
        self.assertEqual(v4.SUBMISSION.max_steps, 1)
        self.assertEqual((v4.SUBMISSION.batch_size, v4.SUBMISSION.eval_batch_size), (64, 128))

    def test_state_parser_and_finite_forward_backward(self):
        model = v4.build_model(self.spec)
        self.assertEqual(count_model_state_elements(model), 13_162)
        # N 123 X 45 T 64: the little-endian field parser reads 64 and clamps it.
        inputs = torch.tensor([[2, 10, 9, 8, 3, 11, 12, 4, 13, 11, 0, 0]])
        parsed = model.parse(inputs, inputs.ne(0))[2]
        self.assertEqual(parsed.tolist(), [3])
        model.train()
        logits, auxiliary = model(inputs)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(auxiliary["parsed_steps"].tolist(), [3])
        loss = F.cross_entropy(logits[:, -1], torch.tensor([8]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_source_policy_cuda_parser_and_no_einsum(self):
        source = V4_PATH.read_text(encoding="utf-8")
        validate_submission_source(V4_PATH.name, source, 256 * 1024)
        self.assertNotIn("einsum", source)
        self.assertIn("digits = (assignment.to(values.dtype) * values[:, :, None]).sum(1)", source)
        self.assertIn("diagnostic", ast.get_docstring(ast.parse(source)).lower())


if __name__ == "__main__":
    unittest.main()
