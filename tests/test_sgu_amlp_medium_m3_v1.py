import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

PATH = Path(__file__).resolve().parents[1] / "submissions/sgu_amlp_medium_m3_v1/submission.py"
loader = importlib.util.spec_from_file_location("sgu_amlp_medium_m3_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class SGUAMLPMediumM3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112)
        self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_exact_topology_optimizer_and_contract(self):
        self.assertEqual(count_model_state_elements(self.model), 89_048)
        block = self.model.transition.block
        self.assertEqual((m.WIDTH, m.D_MODEL, m.HIDDEN, m.MAX_T), (5, 112, 224, 64))
        self.assertEqual(tuple(self.model.transition.place_embedding.weight.shape), (5, 112))
        self.assertEqual((block.spatial.in_features, block.spatial.out_features), (10, 10))
        self.assertEqual((block.qkv.in_features, block.qkv.out_features), (112, 48))
        self.assertEqual((block.attention_out.in_features, block.attention_out.out_features), (16, 224))
        bundle = m.build_optimizer(self.model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
        self.assertEqual((m.SUBMISSION.batch_size, m.SUBMISSION.eval_batch_size, m.SUBMISSION.max_steps), (256, 512, None))

    def test_parser_five_place_alignment_and_widths(self):
        rows = []
        for width in range(1, 6):
            rows.append([2] + list(range(8, 8 + width)) + [3] + list(range(8, 8 + width)) + [4, 9, 5])
        for row, width in zip(rows, range(1, 6)):
            ids = torch.tensor([row])
            out, info = self.model(ids)
            self.assertEqual((info["widths"].item(), info["x_widths"].item()), (width, width))
            self.assertEqual((out[0, -width:, 7:17] > m.NEG).sum().item(), 10 * width)
        ids = torch.tensor([[2,8,9,10,11,12,3,12,11,10,9,8,4,9,5]])
        _, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual((steps.item(), place[0, 1:6].tolist()), (2, [4, 3, 2, 1, 0]))
        for bad in ([2,8,9,10,11,12,13,3,8,4,9,5], [2,8,3,8,9,10,11,12,13,4,9,5]):
            with self.assertRaisesRegex(ValueError, "one and five"):
                self.model(torch.tensor([bad]))

    def test_actual_t_train_eval_t64_permutation_and_determinism(self):
        ids = torch.tensor([[2,8,3,9,4,9,5,0], [2,9,3,10,4,9,5,0]])
        self.model.train(); a, info = self.model(ids)
        self.assertEqual(info["macrosteps"], 2)
        b, _ = self.model(ids[[1, 0]])
        torch.testing.assert_close(a, b[[1, 0]])
        self.model.eval(); c, detail = self.model(ids)
        self.assertEqual(detail["macrosteps"], 2)
        torch.testing.assert_close(a, c)
        d, _ = self.model(ids)
        torch.testing.assert_close(c, d)
        _, detail = self.model(torch.tensor([[2,8,3,9,4,13,11,5]]))
        self.assertEqual(detail["macrosteps"], 64)

    def test_finite_nonzero_gradients_source_and_exclusions(self):
        ids = torch.tensor([[2,8,9,3,10,11,4,9,5], [2,9,10,3,11,12,4,9,5]])
        logits, _ = self.model(ids)
        logits[:, :, 7:17].sum().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        source = PATH.read_text()
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        lower = source.lower()
        for forbidden in ("highway", "conv1d", "conv2d", "rmsnorm", "cross_entropy", "argmax", "remainder", "cache", "carry", "lookup", "synthetic"):
            self.assertNotIn(forbidden, lower)


if __name__ == "__main__":
    unittest.main()
