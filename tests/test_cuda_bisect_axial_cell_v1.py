import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "submissions/cuda_bisect_workspace_v1/submission.py"
CANDIDATE_PATH = ROOT / "submissions/cuda_bisect_axial_cell_v1/submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("axial_bisect_baseline", BASE_PATH)
candidate = load("axial_bisect_candidate", CANDIDATE_PATH)


class CudaBisectAxialCellV1Tests(unittest.TestCase):
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
        added = {
            "cell.norm.weight", "cell.gates.weight", "cell.gates.bias",
            "cell.candidate.weight", "cell.candidate.bias",
        }
        self.assertEqual(set(old_state), set(new_state) - added)
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

    def test_exactly_one_horizontal_call_shape_and_masking(self):
        model = candidate.build_model(self.spec)
        probabilities = F.one_hot(torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), 10).float()
        n_digits = torch.tensor([[3, 2, 3, 0], [9, 0, 0, 1]])
        calls = []
        hook = model.cell.register_forward_hook(
            lambda _module, args, output: calls.append((args[1], output.detach()))
        )
        model(self.inputs)
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "H")
        nd, xd, widths, _ = model._prepare(self.inputs, None)
        width = int(widths.max())
        probabilities = F.one_hot(xd[:, :width], 10).float()
        workspace, valid = model._workspace(probabilities, nd[:, :width], widths)
        masked = model.cell(workspace, "H") * valid[..., None]
        self.assertEqual(masked.shape, (2, width, width + 1, candidate.CHANNELS))
        self.assertEqual(torch.count_nonzero(masked[~valid]).item(), 0)

    def test_state_source_compile_and_submission_contract(self):
        model = candidate.build_model(self.spec)
        self.assertLessEqual(count_model_state_elements(model), 1_000_000)
        source = CANDIDATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("einsum", source)
        self.assertNotIn("readout", source)
        self.assertNotIn("for macro", source)
        self.assertNotIn("gather(", source)
        self.assertNotIn("feedback", source)
        self.assertEqual(source.count('self.cell(workspace, "H")'), 1)
        self.assertNotIn('self.cell(workspace, "V")', source)
        self.assertIn("horizontal.sum().to(logits.dtype) * 0.0", source)
        validate_submission_source(CANDIDATE_PATH.name, source, 256 * 1024,
                                   required_filename=None)
        py_compile.compile(str(CANDIDATE_PATH), doraise=True)
        self.assertEqual((candidate.SUBMISSION.batch_size,
                          candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 128, 1))


if __name__ == "__main__":
    unittest.main()
