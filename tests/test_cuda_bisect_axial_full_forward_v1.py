import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/cuda_bisect_axial_full_forward_v1/submission.py"


def load():
    spec = importlib.util.spec_from_file_location("axial_full_forward", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load()


class CudaBisectAxialFullForwardV1Tests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(17, 20, 1_000_000)

    @staticmethod
    def inputs(step):
        # N=123, X=45, T=step, then three output-query positions.
        return torch.tensor([[2, 8, 9, 10, 3, 11, 12, 4, 7 + step, 5, 5, 5]])

    def test_legality_alignment_purity_and_state(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(PATH), doraise=True)
        self.assertNotIn("einsum", source)
        self.assertEqual(candidate.MAX_STEPS, 3)
        model = candidate.build_model(self.spec)
        self.assertEqual(count_model_state_elements(model), candidate.STATE_ELEMENTS)
        role, places, steps = model.parse(self.inputs(3), self.inputs(3).ne(0))
        self.assertEqual(steps.item(), 3)
        self.assertEqual(places[0, 1:4].tolist(), [2, 1, 0])
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model(self.inputs(1))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_all_max_t_forward_backward_gradients_and_execution(self):
        for step in (1, 2, 3):
            with self.subTest(step=step):
                torch.manual_seed(99)
                model = candidate.build_model(self.spec)
                logits, info = model(self.inputs(step))
                self.assertEqual(logits.shape, (1, 12, 17))
                self.assertEqual(info["workspace_initializations"], step)
                self.assertEqual(info["vectorized_cell_calls"], 4 * step)
                self.assertEqual(info["axis_order"], ("H", "V", "H", "V"))
                logits[:, -3:, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_output_digit_placement_horizon_scaling_and_determinism(self):
        torch.manual_seed(123)
        model = candidate.build_model(self.spec)
        ids = self.inputs(2)
        logits1, info1 = model(ids)
        logits2, info2 = model(ids)
        torch.testing.assert_close(logits1, logits2, rtol=0, atol=0)
        torch.testing.assert_close(info1["digit_probabilities"], info2["digit_probabilities"], rtol=0, atol=0)
        self.assertTrue(torch.all(logits1[:, :, :candidate.DIGIT] == -1e4))
        self.assertTrue(torch.all(logits1[:, :, candidate.DIGIT + 10:] == -1e4))
        # Forward values survive the gradient-only 0.01 horizon gate exactly.
        torch.testing.assert_close(logits1, info1["ungated_logits"], rtol=0, atol=0)

    def test_exact_baseline_optimizer_and_submission_contract(self):
        model = candidate.build_model(self.spec)
        cpu = candidate.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
        self.assertIsInstance(cpu.optimizer, torch.optim.AdamW)
        self.assertIsNone(cpu.scheduler)
        self.assertEqual(len(cpu.optimizer.param_groups), 1)
        group = cpu.optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"], group["capturable"]),
                         (1e-3, (0.9, 0.95), 0.1, False))
        cuda = candidate.build_optimizer(model, OptimizerSpec(10.0, "cuda"))
        self.assertTrue(cuda.optimizer.param_groups[0]["capturable"])
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 128, 1))


if __name__ == "__main__":
    unittest.main()
