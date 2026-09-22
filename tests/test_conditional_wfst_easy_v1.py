import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submissions.conditional_wfst_easy_v1 import submission as s


def prompt(n="12", x="3", t="1"):
    tokens = [1, 2] + [7 + int(c) for c in n] + [3] + [7 + int(c) for c in x]
    tokens += [4] + [7 + int(c) for c in t] + [5]
    return tokens


def batch(rows):
    length = max(map(len, rows)); ids = torch.zeros(len(rows), length, dtype=torch.long)
    for i, row in enumerate(rows): ids[i, :len(row)] = torch.tensor(row)
    return ids, ids.ne(0)


class ConditionalWFSTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = s.Model(ModelSpec(17, 64, 250000))

    def test_exact_state_and_optimizer(self):
        self.assertEqual(count_model_state_elements(self.model), s.STATE_ELEMENTS)
        bundle = s.build_optimizer(self.model, OptimizerSpec(60.0, "cpu"))
        opt = bundle.optimizer
        self.assertEqual(opt.defaults["betas"], (.9, .95)); self.assertEqual(opt.defaults["eps"], 1e-8)
        self.assertEqual(s.SUBMISSION.batch_size, 256); self.assertEqual(s.SUBMISSION.eval_batch_size, 512)

    def test_parser_mixed_width_lsd_and_t64(self):
        ids, mask = batch([prompt("12", "3", "1"), prompt("7", "456", "64")])
        role, place, nw, xw, t = self.model.parse(ids, mask)
        self.assertEqual((nw.tolist(), xw.tolist(), t.tolist()), ([2, 1], [1, 3], [1, 64]))
        self.assertEqual(place[0, role[0].eq(1)].tolist(), [1, 0])
        self.assertEqual(place[1, role[1].eq(2)].tolist(), [2, 1, 0])

    def test_marginals_normalized_finite_and_inactive(self):
        c = torch.randn(2, 80)
        marginal, z = self.model.wfst(c, torch.tensor([1, 4]))
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.allclose(marginal.exp().sum(-1), torch.ones(2, 4), atol=2e-5))
        self.assertTrue(torch.equal(marginal[0, 1:].argmax(-1), torch.zeros(3, dtype=torch.long)))

    def test_batch_purity_and_train_eval_t1(self):
        one, one_mask = batch([prompt("12", "3", "1")])
        mixed, mixed_mask = batch([prompt("12", "3", "1"), prompt("9876", "1234", "3")])
        self.model.eval(); a = self.model(one, one_mask)[1]["log_marginals"]
        b = self.model(mixed, mixed_mask)[1]["log_marginals"][:1]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.model.train(); train = self.model(one, one_mask)[1]["log_marginals"]
        self.model.eval(); evaluate = self.model(one, one_mask)[1]["log_marginals"]
        self.assertTrue(torch.allclose(train, evaluate, atol=1e-6))

    def test_mixed_endpoint_gradients_and_soft_feedback(self):
        ids, mask = batch([prompt("12", "3", str(t)) for t in (1, 2, 3)])
        self.model.train(); logits, aux = self.model(ids, mask)
        positions = mask.sum(1) - 1
        loss = F.cross_entropy(logits[torch.arange(3), positions], torch.tensor([8, 9, 10]))
        loss.backward()
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        soft = aux["soft_feedback"]
        self.assertTrue(((soft > 0) & (soft < 1)).any())

    def test_t64_deterministic_finite_no_mutation(self):
        ids, mask = batch([prompt("1234", "5678", "64")]); self.model.eval()
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        with torch.no_grad():
            a = self.model(ids, mask); b = self.model(ids, mask)
        self.assertTrue(torch.isfinite(a[0]).all()); self.assertTrue(torch.isfinite(a[1]["logZ"]).all())
        self.assertTrue(torch.equal(a[0], b[0]))
        for n, p in self.model.named_parameters(): self.assertTrue(torch.equal(p, before[n]))


if __name__ == "__main__":
    unittest.main()
