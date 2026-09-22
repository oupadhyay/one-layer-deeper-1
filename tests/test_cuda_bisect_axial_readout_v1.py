import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "submissions/cuda_bisect_axial_four_calls_v1/submission.py"
CANDIDATE_PATH = ROOT / "submissions/cuda_bisect_axial_readout_v1/submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("axial_readout_baseline", BASE_PATH)
candidate = load("axial_readout_candidate", CANDIDATE_PATH)


class CudaBisectAxialReadoutV1Tests(unittest.TestCase):
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

    def test_boundary_gather_uses_each_example_width_and_readout_shape(self):
        model = candidate.build_model(self.spec)
        workspace = torch.arange(2 * 4 * 6 * candidate.CHANNELS).reshape(
            2, 4, 6, candidate.CHANNELS).float()
        widths = torch.tensor([2, 5])
        boundary = model._boundary(workspace, widths)
        torch.testing.assert_close(boundary[0], workspace[0, :, 2])
        torch.testing.assert_close(boundary[1], workspace[1, :, 5])
        self.assertEqual(boundary.shape, (2, 4, candidate.CHANNELS))
        self.assertEqual(model.readout(boundary).shape, (2, 4, 10))

    def test_baseline_logits_shared_gradients_and_explicit_added_grads_are_exact(self):
        old, new = self.models()
        old_state, new_state = old.state_dict(), new.state_dict()
        self.assertEqual(set(new_state) - set(old_state), {"readout.weight", "readout.bias"})
        for name, value in old_state.items():
            torch.testing.assert_close(new_state[name], value, rtol=0, atol=0)
        old_logits, _ = old(self.inputs)
        new_logits, _ = new(self.inputs)
        torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
        targets = torch.tensor([8, 9])
        F.cross_entropy(old_logits[:, -1], targets).backward()
        F.cross_entropy(new_logits[:, -1], targets).backward()
        old_parameters, new_parameters = dict(old.named_parameters()), dict(new.named_parameters())
        for name, parameter in old_parameters.items():
            torch.testing.assert_close(new_parameters[name].grad, parameter.grad, rtol=0, atol=0)
        added_prefixes = ("embedding.", "initializer.", "cell.", "readout.")
        for name, parameter in new_parameters.items():
            if name.startswith(added_prefixes):
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_exact_h_v_h_v(self):
        model = candidate.build_model(self.spec)
        calls = []
        hook = model.cell.register_forward_pre_hook(lambda _module, args: calls.append(args[1]))
        model(self.inputs)
        hook.remove()
        self.assertEqual(calls, ["H", "V", "H", "V"])

    def test_state_source_compile_and_submission_contract(self):
        model = candidate.build_model(self.spec)
        self.assertLessEqual(count_model_state_elements(model), 1_000_000)
        source = CANDIDATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("einsum", source)
        for forbidden in ("for macro", "feedback", "endpoint", "DIGIT:DIGIT", "gradient horizon"):
            self.assertNotIn(forbidden, source)
        self.assertEqual(source.count('self.cell(workspace, "H")'), 2)
        self.assertEqual(source.count('self.cell(workspace, "V")'), 2)
        self.assertIn("workspace.gather(2, index)", source)
        self.assertIn("nn.Linear(CHANNELS, 10)", source)
        self.assertIn("readout_logits.sum().to(logits.dtype) * 0.0", source)
        validate_submission_source(CANDIDATE_PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(CANDIDATE_PATH), doraise=True)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 128, 1))


if __name__ == "__main__":
    unittest.main()
