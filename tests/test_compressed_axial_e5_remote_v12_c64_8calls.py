import importlib.util
import py_compile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PROVEN = ROOT / "submissions/compressed_axial_e5_remote_v10_c64_lr1e4/submission.py"
PATH = ROOT / "submissions/compressed_axial_e5_remote_v12_c64_8calls/submission.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load(PATH, "compressed_axial_e5_remote_v12_c64_8calls")


class CompressedAxialE5RemoteV12C64EightCallsTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(17, 20, 1_000_000)

    @staticmethod
    def inputs(step):
        step_tokens = [7 + int(digit) for digit in str(step)]
        return torch.tensor([[2, 8, 9, 10, 3, 11, 12, 4, *step_tokens, 5, 5, 5]])

    def test_source_diff_is_only_axis_order_literal(self):
        old = '("H", "V", "H", "V")'
        new = '("H", "V", "H", "V", "H", "V", "H", "V")'
        proven = PROVEN.read_text(encoding="utf-8")
        actual = PATH.read_text(encoding="utf-8")
        self.assertEqual(proven.count(old), 1)
        self.assertEqual(actual, proven.replace(old, new))

    def test_validation_purity_state_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(PATH), doraise=True)
        model = candidate.build_model(self.spec)
        self.assertEqual(count_model_state_elements(model), 46_634)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model(self.inputs(1))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        optimizer = candidate.build_optimizer(model, OptimizerSpec(10.0, "cpu")).optimizer
        group = optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"]),
                         (1e-4, (0.9, 0.95), 0.1))
        self.assertEqual(candidate.MAX_STEPS, 64)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))

    def test_exact_eight_call_order_by_instrumentation(self):
        model = candidate.build_model(self.spec)
        axes = []
        original = model.cell.forward

        def record(workspace, axis):
            axes.append(axis)
            return original(workspace, axis)

        with patch.object(model.cell, "forward", side_effect=record):
            _, info = model(self.inputs(1))
        expected = ["H", "V", "H", "V", "H", "V", "H", "V"]
        self.assertEqual(axes, expected)
        self.assertEqual(list(info["axis_order"]), expected)
        self.assertEqual(info["vectorized_cell_calls"], 8)

    def test_max_t_512_calls_and_finite_eval(self):
        model = candidate.build_model(self.spec).eval()
        with torch.no_grad():
            logits, info = model(self.inputs(64))
        self.assertEqual(info["workspace_initializations"], 64)
        self.assertEqual(info["vectorized_cell_calls"], 512)
        self.assertTrue(torch.isfinite(logits).all())

    def test_finite_gradients_and_t1_only_gate(self):
        model = candidate.build_model(self.spec).train()
        for step in (1, 2, 3):
            model.zero_grad(set_to_none=True)
            logits, _ = model(self.inputs(step))
            logits[:, -3:, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if step != 1:
                    self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)


if __name__ == "__main__":
    unittest.main()
