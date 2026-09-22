import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

PATH = Path(__file__).resolve().parents[1] / "submissions/sgu_amlp_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("sgu_amlp_easy_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class SGUAMLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112)
        self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_exact_state_api_init_and_optimizer(self):
        self.assertEqual(count_model_state_elements(self.model), 88_898)
        block = self.model.transition.block
        self.assertEqual((block.expand.in_features, block.expand.out_features), (112, 448))
        self.assertEqual((block.spatial.in_features, block.spatial.out_features), (8, 8))
        self.assertEqual((block.qkv.in_features, block.qkv.out_features, block.qkv.bias), (112, 48, None))
        self.assertEqual((block.attention_out.in_features, block.attention_out.out_features, block.attention_out.bias), (16, 224, None))
        self.assertTrue((block.spatial.weight.abs() <= 1.25e-4).all())
        torch.testing.assert_close(block.spatial.bias, torch.ones(8))
        bundle = m.build_optimizer(self.model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
        self.assertEqual((m.SUBMISSION.batch_size, m.SUBMISSION.eval_batch_size, m.SUBMISSION.max_steps), (256, 512, None))

    def test_parser_alignment_t64_tail_and_rejections(self):
        ids = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
        role, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual((steps.item(), place[0,1:5].tolist()), (64, [3,2,1,0]))
        self.model.eval(); out, info = self.model(ids, ids.ne(0))
        self.assertEqual((out.shape, info["macrosteps"]), ((1,14,17), 64))
        self.assertTrue((out[0, 8:10, 7:17] > -10_000).all())
        for bad in ([[2,8,9,10,11,12,3,7,4,7,5]], [[2,8,3,7,4,13,12,5]]):
            with self.assertRaises(ValueError): self.model(torch.tensor(bad))

    def test_register_order_soft_embedding_absence_and_sgu_oracle(self):
        tr = self.model.transition
        n = torch.tensor([[1,2,0,0]]); np = torch.tensor([[1,1,0,0]], dtype=torch.bool)
        xp = torch.tensor([[1,1,0,0]], dtype=torch.bool)
        x = torch.nn.functional.one_hot(torch.tensor([[3,4,0,0]]), 10).float()
        reg = tr.make_register(n, np, x, xp)
        expected = tr.digit_embedding(n[:,0]) + tr.role_embedding.weight[0] + tr.place_embedding.weight[0] + tr.presence_embedding.weight[1]
        torch.testing.assert_close(reg[:,0], expected)
        absent = tr.role_embedding.weight[0] + tr.place_embedding.weight[2] + tr.presence_embedding.weight[0]
        torch.testing.assert_close(reg[:,4], absent[None])
        z = torch.randn(2,8,112)
        block = tr.block; u, v = torch.nn.functional.gelu(block.expand(block.norm(z))).chunk(2,-1)
        q, k, value = block.qkv(z).chunk(3, -1)
        dynamic = block.attention_out(((q @ k.transpose(1,2) / 4).softmax(-1)) @ value)
        oracle = z + block.project(u * (block.spatial(block.gate_norm(v).transpose(1,2)).transpose(1,2) + dynamic))
        torch.testing.assert_close(block(z), oracle)

    def test_calls_parity_purity_permutation_determinism_and_gradients(self):
        ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0], [2,11,3,12,13,4,10,5,0,0]])
        calls = []
        hook = self.model.transition.block.register_forward_hook(lambda *args: calls.append(1))
        self.model.train(); a, _ = self.model(ids); hook.remove(); self.assertEqual(len(calls), 3)
        b, _ = self.model(ids.flip(0)); torch.testing.assert_close(a[0], b[1]); torch.testing.assert_close(a[1], b[0])
        c, _ = self.model(ids); torch.testing.assert_close(a, c)
        self.model.eval(); d, _ = self.model(ids); torch.testing.assert_close(a, d)
        self.model.train(); a[:,:,7:17].sum().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_source_validation_and_exclusions(self):
        source = PATH.read_text()
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        lower = source.lower()
        for forbidden in ("scaled_dot_product", "covariance", "conv1d", "scan", "router", "remainder", "cross_entropy", "argmax", "cache"):
            self.assertNotIn(forbidden, lower)


if __name__ == "__main__": unittest.main()
