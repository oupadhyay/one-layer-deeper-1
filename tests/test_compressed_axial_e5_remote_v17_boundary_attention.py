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
PATH = ROOT / "submissions/compressed_axial_e5_remote_v17_boundary_attention/submission.py"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

candidate, baseline = load(PATH, "v17_boundary_attention"), load(BASE, "v10_reference")

class BoundaryAttentionTests(unittest.TestCase):
    def setUp(self): self.spec = ModelSpec(17, 20, 1_000_000)
    @staticmethod
    def inputs(step=1):
        return torch.tensor([[2, 8, 9, 10, 3, 11, 12, 4, *[7 + int(x) for x in str(step)], 5, 5, 5]])

    def test_exact_init_parity_and_state(self):
        torch.manual_seed(1); old = baseline.build_model(self.spec)
        torch.manual_seed(2); new = candidate.build_model(self.spec)
        new.load_state_dict({k: v for k, v in old.state_dict().items() if k in new.state_dict()}, strict=False)
        self.assertEqual(count_model_state_elements(new), 63146)
        self.assertEqual(sum(p.numel() for n, p in new.named_parameters() if n.startswith(("boundary_norm", "pair_norm", "attention_"))), 16512)
        self.assertEqual(torch.count_nonzero(new.attention_out.weight), 0)
        old.eval(); new.eval()
        with torch.no_grad(): a, _ = old(self.inputs(3)); b, _ = new(self.inputs(3))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_pair_mask(self):
        model = candidate.build_model(self.spec)
        workspace = torch.randn(2, 3, 4, 64)
        _, mask, weights = model._boundary_attention(workspace, torch.tensor([2, 3]))
        expected = torch.tensor([[1,1,0,0, 1,1,0,0, 0,0,0,0],
                                 [1,1,1,0, 1,1,1,0, 1,1,1,0]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.equal(weights.masked_select(~mask[:, None]), torch.zeros_like(weights.masked_select(~mask[:, None]))))
        self.assertTrue((weights.masked_select(mask[:, None].expand_as(weights)) > 0).all())

    def test_ce_gradient_and_no_routing(self):
        model = candidate.build_model(self.spec).train(); logits, _ = model(self.inputs())
        F.cross_entropy(logits.reshape(-1, 17), torch.full(logits.shape[:2], candidate.DIGIT).reshape(-1)).backward()
        self.assertTrue(torch.isfinite(model.attention_out.weight.grad).all())
        self.assertGreater(torch.count_nonzero(model.attention_out.weight.grad), 0)
        source = PATH.read_text(encoding="utf-8")
        section = source[source.index("    def _boundary_attention"):source.index("    def forward")]
        for forbidden in ("DIGIT", "diag", "%", "remainder", "places", "role", "arithmetic"):
            self.assertNotIn(forbidden, section)

    def test_validation_max_t_purity_finite_gradients(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(PATH), doraise=True)
        model = candidate.build_model(self.spec).eval(); before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad(): logits, info = model(self.inputs(64))
        self.assertTrue(torch.isfinite(logits).all()); self.assertEqual((info["workspace_initializations"], info["vectorized_cell_calls"]), (64, 256))
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        model.train(); model.zero_grad(); model(self.inputs())[0].sum().backward()
        for n, p in model.named_parameters(): self.assertTrue(torch.isfinite(p.grad).all(), n)

    def test_optimizer_submission_exact(self):
        model = candidate.build_model(self.spec); bundle = candidate.build_optimizer(model, OptimizerSpec(1, "cpu")); group = bundle.optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"], group["capturable"]), (1e-4, (0.9, .95), .1, False))
        self.assertIsNone(bundle.scheduler)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size, candidate.SUBMISSION.max_steps), (512, 512, None))

if __name__ == "__main__": unittest.main()
