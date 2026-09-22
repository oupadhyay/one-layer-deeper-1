import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/whole_output_energy_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("whole_output_energy_easy_v1", PATH)
c = importlib.util.module_from_spec(loader)
loader.loader.exec_module(c)


def spec(length=16):
    return ModelSpec(17, length, 500_000_000)


def prompt(n=(1, 2), x=(3, 4), t=(1,), length=16):
    tokens = [2] + [7 + d for d in n] + [3] + [7 + d for d in x] + [4] + [7 + d for d in t] + [5]
    return torch.tensor([tokens + [0] * (length - len(tokens))]), torch.tensor([[1] * len(tokens) + [0] * (length-len(tokens))])


class WholeOutputEnergyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_contract_source_state_optimizer(self):
        source = PATH.read_text()
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        for word in ("remainder", "carry", "custom_training_loss", ".backward("):
            self.assertNotIn(word, source.lower())
        model = c.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 220832)
        self.assertNotIn("catalog", model.state_dict())
        bundle = c.build_optimizer(model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 6e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups], [.01, 0])
        self.assertAlmostEqual(bundle.scheduler.lr_lambdas[0](0), 1/32)
        self.assertEqual((c.SUBMISSION.batch_size, c.SUBMISSION.eval_batch_size, c.SUBMISSION.max_steps), (256, 512, None))

    def test_parser_widths_and_rejection(self):
        model = c.build_model(spec(20))
        ids, mask = prompt((1,2,3,4), (5,6,7), (6,4), 20)
        _, nd, xd, nmask, steps, widths = model.prepare(ids, mask)
        self.assertEqual(nd.tolist(), [[4,3,2,1]])
        self.assertEqual(xd.tolist(), [[7,6,5,0]])
        self.assertEqual((widths.item(), steps.item()), (4,64))
        bad, badmask = prompt((1,2,3,4,5), (1,), (1,), 20)
        with self.assertRaises(ValueError): model.prepare(bad, badmask)

    def test_catalog_marginal_matches_bruteforce_width_two(self):
        model = c.build_model(spec())
        ids, mask = prompt()
        _, nd, xd, nm, _, widths = model.prepare(ids, mask)
        scores = model.score_candidates(model.encode(nd, xd, nm), widths)
        marginal = model.marginals(scores)
        brute = torch.full((4,10), -torch.inf)
        for p in range(4):
            for d in range(10):
                brute[p,d] = torch.logsumexp(scores[0, model.catalog[:,p] == d], 0)
        torch.testing.assert_close(marginal[0], brute)
        self.assertTrue(torch.isneginf(scores[0, 100:]).all())

    def test_gradients_alignment_gate_feedback_and_modes(self):
        model = c.build_model(spec()).train()
        ids, mask = prompt(t=(1,))
        logits, info = model(ids, mask)
        loss = F.cross_entropy(logits[:, mask.sum()-1], torch.tensor([7]))
        loss.backward()
        for name in ("pair_head.weight", "global_head.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        train = logits.detach()
        model.eval()
        with torch.no_grad(): eval_logits, _ = model(ids, mask)
        torch.testing.assert_close(train, eval_logits, rtol=0, atol=0)
        self.assertEqual(info["macrosteps"], 1)

        model.train(); model.zero_grad()
        ids2, mask2 = prompt(t=(2,))
        out2, _ = model(ids2, mask2)
        self.assertEqual(out2.requires_grad, True)
        out2.sum().backward()
        self.assertEqual(sum((p.grad.abs().sum().item() if p.grad is not None else 0) for p in model.parameters()), 0)

    def test_t64_determinism_immutability_and_batch_purity(self):
        model = c.build_model(spec()).eval()
        ids, mask = prompt(t=(6,4))
        nd_before = model.prepare(ids, mask)[1].clone()
        with torch.no_grad():
            a, info = model(ids, mask); b, _ = model(ids, mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(nd_before, model.prepare(ids, mask)[1], rtol=0, atol=0)
        self.assertEqual(info["macrosteps"], 64)
        batch_ids = torch.cat((ids, prompt((9,), (2,), (1,))[0]))
        batch_mask = torch.cat((mask, prompt((9,), (2,), (1,))[1]))
        with torch.no_grad(): mixed, _ = model(batch_ids, batch_mask)
        torch.testing.assert_close(a[0], mixed[0], rtol=0, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
