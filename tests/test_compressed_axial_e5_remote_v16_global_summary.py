import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions/compressed_axial_e5_remote_v10_c64_lr1e4/submission.py"
PATH = ROOT / "submissions/compressed_axial_e5_remote_v16_global_summary/submission.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load(PATH, "compressed_axial_e5_remote_v16_global_summary")
baseline = load(BASE, "compressed_axial_e5_remote_v10_reference")


class GlobalSummaryTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(17, 20, 1_000_000)

    @staticmethod
    def inputs(step=1):
        tokens = [7 + int(d) for d in str(step)]
        return torch.tensor([[2, 8, 9, 10, 3, 11, 12, 4, *tokens, 5, 5, 5]])

    def test_init_parity_zero_weights_and_state_count(self):
        torch.manual_seed(91)
        old = baseline.build_model(self.spec)
        torch.manual_seed(92)
        new = candidate.build_model(self.spec)
        shared = {name: value for name, value in old.state_dict().items()
                  if name in new.state_dict()}
        new.load_state_dict(shared, strict=False)
        self.assertEqual(torch.count_nonzero(new.summary_out.weight).item(), 0)
        old.eval(), new.eval()
        with torch.no_grad():
            expected, expected_info = old(self.inputs(3))
            actual, actual_info = new(self.inputs(3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_info.keys(), expected_info.keys())
        self.assertEqual(count_model_state_elements(new), 54_954)
        self.assertEqual(new.summary_norm.weight.numel(), 128)
        self.assertEqual(new.summary_out.weight.numel(), 8_192)

    def test_masked_summaries_use_every_pair_but_no_padding_or_boundary(self):
        model = candidate.build_model(self.spec)
        workspace = torch.arange(2 * 3 * 4 * 64, dtype=torch.float32).reshape(2, 3, 4, 64)
        widths = torch.tensor([2, 3])
        row, glob = model._summaries(workspace, widths)
        torch.testing.assert_close(row[0, :2], workspace[0, :2, :2].mean(1))
        torch.testing.assert_close(glob[0, 0], workspace[0, :2, :2].mean((0, 1)))
        changed = workspace.clone()
        changed[0, :, 2:] += 1e6  # boundary and padded columns
        changed[0, 2, :] += 1e6   # padded row
        row2, glob2 = model._summaries(changed, widths)
        torch.testing.assert_close(row2[0, :2], row[0, :2])
        torch.testing.assert_close(glob2[0], glob[0])
        for i in range(2):
            for j in range(2):
                probe = workspace.clone()
                probe[0, i, j, 0] += 1
                probe_global = model._summaries(probe, widths)[1]
                self.assertNotEqual(probe_global[0, 0, 0].item(), glob[0, 0, 0].item())

    def test_gradient_reaches_zero_initialized_summary_projection(self):
        model = candidate.build_model(self.spec).train()
        logits, _ = model(self.inputs())
        target = torch.full(logits.shape[:2], candidate.DIGIT, dtype=torch.long)
        F.cross_entropy(logits.reshape(-1, 17), target.reshape(-1)).backward()
        self.assertIsNotNone(model.summary_out.weight.grad)
        self.assertTrue(torch.isfinite(model.summary_out.weight.grad).all())
        self.assertGreater(torch.count_nonzero(model.summary_out.weight.grad).item(), 0)

    def test_validation_purity_finite_max_t_and_no_arithmetic_routing(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(PATH), doraise=True)
        summary_source = source[source.index("    def _summaries"):source.index("    def forward")]
        for forbidden in ("DIGIT", "diag", "%", "remainder", "places", "role"):
            self.assertNotIn(forbidden, summary_source)
        model = candidate.build_model(self.spec).eval()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            logits, info = model(self.inputs(64))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual((info["workspace_initializations"], info["vectorized_cell_calls"]), (64, 256))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_optimizer_and_submission_unchanged(self):
        model = candidate.build_model(self.spec)
        bundle = candidate.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
        group = bundle.optimizer.param_groups[0]
        self.assertIsInstance(bundle.optimizer, torch.optim.AdamW)
        self.assertIsNone(bundle.scheduler)
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"], group["capturable"]),
                         (1e-4, (0.9, 0.95), 0.1, False))
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))


if __name__ == "__main__":
    unittest.main()
