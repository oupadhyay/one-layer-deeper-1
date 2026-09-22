import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

PATH = Path(__file__).resolve().parents[1] / "submissions/sgu_highway_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("sgu_highway_easy_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class SGUHighwayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112); self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_exact_state_delta_and_highway_oracle(self):
        self.assertEqual(count_model_state_elements(self.model), 92_594)
        block = self.model.transition.block
        self.assertEqual((block.highway.in_features, block.highway.out_features, block.highway.bias.numel()), (112,112,112))
        z = torch.randn(2,8,112); normalized = block.norm(z)
        u, v = torch.nn.functional.gelu(block.expand(normalized)).chunk(2,-1)
        update = block.project(u * block.spatial(block.gate_norm(v).transpose(1,2)).transpose(1,2))
        torch.testing.assert_close(block(z), z + torch.sigmoid(block.highway(normalized)) * update)

    def test_parser_alignment_t64_and_rejections(self):
        ids = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
        _, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual((steps.item(), place[0,1:5].tolist()), (64,[3,2,1,0]))
        self.model.eval(); out, info = self.model(ids)
        self.assertEqual((out.shape, info["macrosteps"]), ((1,14,17),64))
        for bad in ([[2,8,9,10,11,12,3,7,4,7,5]], [[2,8,3,7,4,13,12,5]]):
            with self.assertRaises(ValueError): self.model(torch.tensor(bad))

    def test_calls_soft_feedback_purity_determinism_and_gradients(self):
        ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0], [2,11,3,12,13,4,10,5,0,0]])
        calls = []; hook = self.model.transition.register_forward_pre_hook(lambda _, args: calls.append(args[2].detach().clone()))
        self.model.train(); a, _ = self.model(ids); hook.remove()
        self.assertEqual(len(calls),3); torch.testing.assert_close(calls[1].sum(-1), torch.ones_like(calls[1].sum(-1)))
        self.assertTrue(((calls[1] > 0) & (calls[1] < 1)).all())
        b, _ = self.model(ids.flip(0)); torch.testing.assert_close(a[0],b[1]); torch.testing.assert_close(a[1],b[0])
        c, _ = self.model(ids); torch.testing.assert_close(a,c)
        self.model.train(); a[:,:,7:17].sum().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad,name); self.assertTrue(torch.isfinite(parameter.grad).all(),name)

    def test_source_validation(self):
        source = PATH.read_text(); self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        for forbidden in ("remainder", "cross_entropy", "argmax", "cache", "conv1d", "router"):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__": unittest.main()
