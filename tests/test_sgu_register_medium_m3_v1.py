import importlib.util
import unittest
from pathlib import Path

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

PATH = Path(__file__).resolve().parents[1] / "submissions/sgu_register_medium_m3_v1/submission.py"
loader = importlib.util.spec_from_file_location("sgu_register_medium_m3_v1", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)


class SGURegisterMediumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112)
        self.model = m.build_model(ModelSpec(17, 24, 250_000))

    def test_exact_state_topology_init_and_api_optimizer(self):
        self.assertEqual(count_model_state_elements(self.model), 80_088)
        block = self.model.transition.block
        self.assertEqual((m.WIDTH, m.D_MODEL, m.HIDDEN, m.MAX_T), (5, 112, 224, 64))
        self.assertEqual(tuple(self.model.transition.place_embedding.weight.shape), (5, 112))
        self.assertEqual((block.spatial.in_features, block.spatial.out_features), (10, 10))
        self.assertTrue((block.spatial.weight.abs() <= 1.25e-4).all())
        torch.testing.assert_close(block.spatial.bias, torch.ones(10))
        bundle = m.build_optimizer(self.model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual(bundle.scheduler.get_last_lr()[0], 6e-4 / 32)
        self.assertEqual((m.SUBMISSION.batch_size, m.SUBMISSION.eval_batch_size, m.SUBMISSION.max_steps), (256, 512, None))

    def test_m3_widths_lsd_parser_alignment_and_rejections(self):
        rows = [
            [2,8,9,10,11,3,11,10,9,8,4,8,5,0,0,0,0],
            [2,12,13,14,15,3,12,11,10,9,4,8,5,0,0,0,0],
            [2,8,13,10,15,11,3,12,11,10,9,8,4,8,5,0,0],
        ]
        out, info = self.model(torch.tensor(rows))
        self.assertEqual(info["widths"].tolist(), [4, 4, 5])
        self.assertEqual((out[0, 9:13, 7:17] > m.NEG).sum().item(), 40)
        self.assertEqual((out[2, 10:15, 7:17] > m.NEG).sum().item(), 50)
        ids = torch.tensor([[2,8,9,10,11,12,3,12,11,10,9,8,4,15,5,0,0]])
        role, place, steps = self.model.parse(ids, ids.ne(0))
        self.assertEqual((steps.item(), place[0,1:6].tolist()), (8, [4,3,2,1,0]))
        bad = [
            [2,8,9,10,11,12,13,3,8,4,9,5],
            [2,8,3,8,9,10,11,12,13,4,9,5],
            [2,8,3,8,4,13,12,5],
        ]
        for row in bad:
            with self.assertRaises(ValueError): self.model(torch.tensor([row]))
        with self.assertRaises(ValueError): self.model(torch.tensor([[2,8,3,8,4,9,5] + [0] * 18]))

    def test_calls_masking_isolation_permutation_and_parity(self):
        ids = torch.tensor([
            [2,8,3,9,4,9,5,0,0],
            [2,9,3,10,4,11,5,0,0],
            [2,10,3,11,4,15,5,0,0],
        ])
        calls = []
        hook = self.model.transition.block.register_forward_hook(lambda *args: calls.append(1))
        self.model.train(); a, info = self.model(ids); hook.remove()
        self.assertEqual((len(calls), info["macrosteps"]), (8, 8))
        calls = []
        hook = self.model.transition.block.register_forward_hook(lambda *args: calls.append(1))
        self.model(ids[:1]); hook.remove()
        self.assertEqual(len(calls), 2)
        b, _ = self.model(ids[[2,0,1]])
        torch.testing.assert_close(a, b[[1,2,0]])
        self.model.eval(); c, _ = self.model(ids)
        torch.testing.assert_close(a, c)
        one, _ = self.model(ids[:1]); torch.testing.assert_close(a[0], one[0])
        sixty_four = torch.tensor([[2,8,3,9,4,13,11,5]])
        _, detail = self.model(sixty_four); self.assertEqual(detail["macrosteps"], 64)

    def test_register_interleaving_immutable_n_and_soft_feedback(self):
        tr = self.model.transition
        n = torch.tensor([[1,2,3,4,5]]); np = torch.ones(1,5,dtype=torch.bool)
        p = torch.nn.functional.one_hot(torch.tensor([[5,4,3,2,1]]), 10).float()
        reg = tr.make_register(n, np, p, np)
        for place in range(5):
            nt = tr.digit_embedding(n[:,place]) + tr.role_embedding.weight[0] + tr.place_embedding.weight[place] + tr.presence_embedding.weight[1]
            torch.testing.assert_close(reg[:,2*place], nt)
        seen = []
        hook = tr.register_forward_pre_hook(lambda _, args: seen.append(tuple(x.detach().clone() for x in args)))
        ids = torch.tensor([[2,8,9,3,10,11,4,10,5]])
        self.model(ids); hook.remove()
        self.assertEqual(len(seen), 3)
        for args in seen[1:]: torch.testing.assert_close(args[0], seen[0][0])
        self.assertTrue(torch.allclose(seen[1][2].sum(-1), torch.ones_like(seen[1][2].sum(-1))))
        self.assertTrue(((seen[1][2] > 0) & (seen[1][2] < 1)).any())

    def test_gradients_purity_determinism_and_source(self):
        ids = torch.tensor([[2,8,3,9,4,9,5,0], [2,9,3,10,4,11,5,0], [2,10,3,11,4,15,5,0]])
        before = {k: v.clone() for k,v in self.model.state_dict().items()}
        a, _ = self.model(ids); b, _ = self.model(ids)
        torch.testing.assert_close(a, b)
        for k,v in self.model.state_dict().items(): torch.testing.assert_close(v, before[k])
        a[:,:,7:17].sum().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        source = PATH.read_text()
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        lower = source.lower()
        for forbidden in ("cross_entropy", "argmax", "remainder", "cache", "carry", "lookup", "synthetic"):
            self.assertNotIn(forbidden, lower)


if __name__ == "__main__": unittest.main()
