import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "submissions/cuda_bisect_parser_v1/submission.py"
WORKSPACE_PATH = ROOT / "submissions/cuda_bisect_workspace_v1/submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("workspace_bisect_baseline", BASE_PATH)
candidate = load("workspace_bisect_candidate", WORKSPACE_PATH)


class CudaBisectWorkspaceV1Tests(unittest.TestCase):
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
        new = candidate.build_model(self.spec)
        return old, new

    def test_baseline_logits_and_shared_gradients_are_exact(self):
        old, new = self.models()
        old_state = old.state_dict()
        new_state = new.state_dict()
        self.assertEqual(set(old_state), set(new_state) - {
            "embedding.weight", "initializer.0.weight", "initializer.0.bias",
            "initializer.2.weight", "initializer.2.bias",
        })
        for name, value in old_state.items():
            torch.testing.assert_close(new_state[name], value, rtol=0, atol=0)
        old_logits, _ = old(self.inputs)
        new_logits, _ = new(self.inputs)
        torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
        targets = torch.tensor([8, 9])
        F.cross_entropy(old_logits[:, -1], targets).backward()
        F.cross_entropy(new_logits[:, -1], targets).backward()
        old_parameters = dict(old.named_parameters())
        new_parameters = dict(new.named_parameters())
        for name, parameter in old_parameters.items():
            torch.testing.assert_close(new_parameters[name].grad, parameter.grad, rtol=0, atol=0)
        for name in new_parameters.keys() - old_parameters.keys():
            gradient = new_parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertEqual(torch.count_nonzero(gradient).item(), 0, name)

    def test_workspace_shape_masking_and_boundary(self):
        model = candidate.build_model(self.spec)
        probabilities = F.one_hot(torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), 10).float()
        n_digits = torch.tensor([[3, 2, 3, 0], [9, 0, 0, 1]])
        workspace, valid = model._workspace(probabilities, n_digits, torch.tensor([3, 4]))
        self.assertEqual(workspace.shape, (2, 4, 5, candidate.CHANNELS))
        self.assertEqual(valid.shape, (2, 4, 5))
        self.assertEqual(torch.count_nonzero(workspace[~valid]).item(), 0)
        self.assertTrue(valid[0, :3, :4].all())
        self.assertFalse(valid[0, 3].any())
        self.assertFalse(valid[0, :, 4].any())
        self.assertTrue(valid[1, :, 4].all())

        captured = {}
        hook = model.initializer[0].register_forward_pre_hook(
            lambda _module, args: captured.setdefault("features", args[0].detach())
        )
        model._workspace(probabilities, n_digits, torch.tensor([3, 4]))
        hook.remove()
        features = captured["features"]
        # At each row's boundary column, j-side state and N embeddings are zero.
        self.assertEqual(torch.count_nonzero(features[0, :3, 3, 16:32]).item(), 0)
        self.assertEqual(torch.count_nonzero(features[0, :3, 3, 48:64]).item(), 0)

    def test_state_source_compile_and_submission_contract(self):
        model = candidate.build_model(self.spec)
        self.assertLessEqual(count_model_state_elements(model), 1_000_000)
        source = WORKSPACE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("einsum", source)
        self.assertNotIn("AxialLocalCell", source)
        self.assertNotIn("readout", source)
        self.assertIn("workspace.sum().to(logits.dtype) * 0.0", source)
        validate_submission_source(WORKSPACE_PATH.name, source, 256 * 1024,
                                   required_filename=None)
        py_compile.compile(str(WORKSPACE_PATH), doraise=True)
        self.assertEqual((candidate.SUBMISSION.batch_size,
                          candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 128, 1))


if __name__ == "__main__":
    unittest.main()
