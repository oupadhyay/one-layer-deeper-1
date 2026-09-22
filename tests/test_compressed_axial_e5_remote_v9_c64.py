import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PROVEN = ROOT / "submissions/compressed_axial_e5_remote_v8_t1only/submission.py"
PATH = ROOT / "submissions/compressed_axial_e5_remote_v9_c64/submission.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load(PATH, "compressed_axial_e5_remote_v9_c64")
proven = load(PROVEN, "compressed_axial_e5_remote_v8_t1only_reference")


class CompressedAxialE5RemoteV9C64Tests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(17, 20, 1_000_000)

    @staticmethod
    def inputs(step):
        step_tokens = [7 + int(digit) for digit in str(step)]
        return torch.tensor([[2, 8, 9, 10, 3, 11, 12, 4, *step_tokens, 5, 5, 5]])

    def test_exact_promotion_source_diff(self):
        proven_source = PROVEN.read_text(encoding="utf-8")
        candidate_source = PATH.read_text(encoding="utf-8")
        expected = proven_source.replace(
            "MAX_STEPS, CHANNELS, STATE_ELEMENTS = 64, 32, 13162",
            "MAX_STEPS, CHANNELS, STATE_ELEMENTS = 64, 64, 46634",
        )
        self.assertEqual(candidate_source, expected)
        changed = [(left, right) for left, right in zip(proven_source.splitlines(), candidate_source.splitlines())
                   if left != right]
        self.assertEqual(changed, [
            ("MAX_STEPS, CHANNELS, STATE_ELEMENTS = 64, 32, 13162",
             "MAX_STEPS, CHANNELS, STATE_ELEMENTS = 64, 64, 46634")])

    def test_legality_alignment_purity_state_and_cuda_safe_source(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(PATH), doraise=True)
        self.assertNotIn("einsum", source)
        model = candidate.build_model(self.spec)
        self.assertEqual(count_model_state_elements(model), 46_634)
        self.assertLess(count_model_state_elements(model), 250_000)
        ids = self.inputs(1)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model(ids)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_max_t_forward_and_eval(self):
        model = candidate.build_model(self.spec)
        ids = self.inputs(64)
        model.eval()
        with torch.no_grad():
            logits, info = model(ids)
        self.assertEqual(logits.shape, (1, ids.shape[1], 17))
        self.assertEqual(info["parsed_steps"].item(), 64)
        self.assertEqual(info["workspace_initializations"], 64)
        self.assertEqual(info["vectorized_cell_calls"], 256)
        self.assertTrue(torch.isfinite(logits).all())

    def test_batch512_t1_t2_t3_forward(self):
        model = candidate.build_model(self.spec)
        ids = torch.cat([self.inputs(step) for step in (1, 2, 3)] * 170 +
                        [self.inputs(1), self.inputs(2)])
        self.assertEqual(ids.shape[0], 512)
        logits, info = model(ids)
        self.assertEqual(logits.shape, (512, ids.shape[1], 17))
        self.assertEqual(info["parsed_steps"].bincount(minlength=4).tolist(), [0, 171, 171, 170])
        self.assertEqual(info["workspace_initializations"], 3)

    def test_batch_representative_gradients_and_gradient_gate(self):
        torch.manual_seed(1234)
        model = candidate.build_model(self.spec)
        model.train()
        for step in (1, 2, 3):
            ids = self.inputs(step)
            model.zero_grad(set_to_none=True)
            logits, _ = model(ids)
            logits[:, -3:, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                if step == 1:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_exact_optimizer_and_official_submission_contract(self):
        model = candidate.build_model(self.spec)
        cpu = candidate.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
        self.assertIsInstance(cpu.optimizer, torch.optim.AdamW)
        self.assertIsNone(cpu.scheduler)
        self.assertEqual(len(cpu.optimizer.param_groups), 1)
        group = cpu.optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"], group["capturable"]),
                         (3e-4, (0.9, 0.95), 0.1, False))
        cuda = candidate.build_optimizer(model, OptimizerSpec(10.0, "cuda"))
        self.assertTrue(cuda.optimizer.param_groups[0]["capturable"])
        self.assertEqual(candidate.MAX_STEPS, 64)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))


if __name__ == "__main__":
    unittest.main()
