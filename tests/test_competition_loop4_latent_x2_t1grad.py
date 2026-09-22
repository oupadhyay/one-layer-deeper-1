import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
import competition_loop3_latent_x2 as parent
import competition_loop4_latent_x2_t1grad as candidate


def spec(length=13): return ModelSpec(17, length, 20_000_000)


ROWS = torch.tensor([[2, 10, 9, 10, 3, 15, 16, 4, 8],
                     [2, 10, 9, 10, 3, 8, 12, 4, 9],
                     [2, 10, 9, 10, 3, 11, 9, 4, 10]])


class T1GradientGate(unittest.TestCase):
    def test_identity_state_optimizer_and_values(self):
        torch.manual_seed(74); a = parent.Model(spec())
        torch.manual_seed(74); b = candidate.Model(spec())
        self.assertEqual(count_model_state_elements(b), 223493)
        for k, v in a.state_dict().items():
            self.assertEqual(v.numpy().tobytes(), b.state_dict()[k].numpy().tobytes())
        self.assertIs(candidate.build_optimizer, parent.build_optimizer)
        oa = parent.build_optimizer(a, OptimizerSpec(60, "cpu"))
        ob = candidate.build_optimizer(b, OptimizerSpec(60, "cpu"))
        self.assertEqual(oa.optimizer.state_dict(), ob.optimizer.state_dict())
        self.assertEqual(oa.scheduler.state_dict(), ob.scheduler.state_dict())
        for training in (True, False):
            a.train(training); b.train(training)
            la, aa = a(ROWS); lb, ab = b(ROWS)
            self.assertTrue(torch.equal(la, lb)); self.assertTrue(torch.equal(lb, ab["ungated_logits"]))
            for key in ("parsed_steps", "active_updates", "active_refinements", "latent_state"):
                self.assertTrue(torch.equal(aa[key], ab[key]))

    def _grads(self, rows, gated):
        torch.manual_seed(31)
        model = (candidate.Model if gated else parent.Model)(spec(), d_model=16).train()
        logits, aux = model(rows)
        pre = aux.get("ungated_logits", logits); pre.retain_grad()
        loss = pre.new_tensor(0.)
        # Use the returned logits for the actual loss; pre is retained only to inspect its gradient.
        returned = model(rows)[0] if False else logits
        for i in range(len(rows)): loss = loss + F.cross_entropy(returned[i, -1:], torch.tensor([5]))
        loss.backward()
        return pre.grad, [p.grad.clone() for p in model.parameters()]

    def test_isolated_and_mixed_gradient_ratios(self):
        for i, ratio in enumerate((1.0, .01, .01)):
            gp, pp = self._grads(ROWS[i:i + 1], False)
            gc, pc = self._grads(ROWS[i:i + 1], True)
            self.assertGreater(sum(x.abs().sum() for x in pc), 0)
            torch.testing.assert_close(gc, gp * ratio, atol=2e-7, rtol=2e-5)
            for x, y in zip(pc, pp): torch.testing.assert_close(x, y * ratio, atol=2e-7, rtol=2e-5)
        # Row independence makes a mixed batch equal the sum of homogeneous isolated losses.
        _, mixed = self._grads(ROWS, True)
        isolated = [self._grads(ROWS[i:i + 1], True)[1] for i in range(3)]
        for got, parts in zip(mixed, zip(*isolated)):
            torch.testing.assert_close(got, sum(parts), atol=3e-6, rtol=3e-5)

    def test_bf16_nonmutation_and_t64(self):
        torch.manual_seed(12); model = candidate.Model(spec(), d_model=16)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.autocast("cpu", dtype=torch.bfloat16): self.assertTrue(torch.isfinite(model(ROWS)[0]).all())
        t64 = torch.tensor([[2, 8, 3, 8, 4, 13, 11]])
        self.assertEqual(model(t64)[1]["active_refinements"].item(), 128)
        for k, v in model.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))

    def test_t1_capacity(self):
        x = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8],
                          [2, 10, 9, 10, 3, 15, 16, 4, 8],
                          [2, 10, 9, 10, 3, 11, 9, 4, 8]])
        # Vocabulary tokens encode decimal digits; target residues are 225,169,149.
        labels = torch.tensor([[9, 9, 12], [8, 13, 16], [8, 11, 16]])
        pos = torch.tensor([[6, 7, 8]] * 3); batch = torch.arange(3)[:, None]
        torch.manual_seed(74); model = candidate.Model(spec(), d_model=32).train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(100):
            opt.zero_grad(set_to_none=True); selected = model(x)[0][batch, pos]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward(); opt.step()
            if selected.argmax(-1).eq(labels).all(): break
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad(): self.assertTrue(model(x)[0][batch, pos].argmax(-1).eq(labels).all())


if __name__ == "__main__": unittest.main()
