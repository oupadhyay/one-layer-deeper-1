import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
import competition_loop4_latent_x2_t1grad as parent
import competition_loop5_latent_x2_t1grad_d64 as candidate


def spec(length=13): return ModelSpec(17, length, 20_000_000)


ROWS = torch.tensor([[2, 10, 9, 10, 3, 15, 16, 4, 8],
                     [2, 10, 9, 10, 3, 8, 12, 4, 9],
                     [2, 10, 9, 10, 3, 11, 9, 4, 10]])


class D64CapacityRegularization(unittest.TestCase):
    def test_state_seed_optimizer_and_parent_equivalence(self):
        torch.manual_seed(74); explicit = parent.Model(spec(), d_model=64)
        torch.manual_seed(74); built = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(built), 75_797)
        torch.manual_seed(74); repeat = candidate.build_model(spec())
        for key, value in explicit.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), built.state_dict()[key].numpy().tobytes())
            self.assertEqual(value.numpy().tobytes(), repeat.state_dict()[key].numpy().tobytes())
        self.assertIs(candidate.build_optimizer, parent.build_optimizer)
        oa = parent.build_optimizer(explicit, OptimizerSpec(60, "cpu"))
        ob = candidate.build_optimizer(built, OptimizerSpec(60, "cpu"))
        self.assertEqual(oa.optimizer.state_dict(), ob.optimizer.state_dict())
        self.assertEqual(oa.scheduler.state_dict(), ob.scheduler.state_dict())

        for training in (False, True):
            explicit.train(training); built.train(training)
            la, aa = explicit(ROWS); lb, ab = built(ROWS)
            self.assertTrue(torch.equal(la, lb))
            for key in ("parsed_steps", "active_updates", "active_refinements",
                        "latent_state", "ungated_logits"):
                self.assertTrue(torch.equal(aa[key], ab[key]))
            explicit.zero_grad(); built.zero_grad()
            la.sum().backward(); lb.sum().backward()
            for pa, pb in zip(explicit.parameters(), built.parameters()):
                torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)

    def test_gate_is_unchanged(self):
        for row, ratio in zip(ROWS, (1.0, .01, .01)):
            torch.manual_seed(9); model = candidate.Model(spec(), d_model=16).train()
            logits, aux = model(row[None]); aux["ungated_logits"].retain_grad()
            logits.sum().backward()
            torch.testing.assert_close(aux["ungated_logits"].grad,
                                       torch.full_like(logits, ratio), atol=1e-7, rtol=1e-6)

    def test_t1_d32_capacity(self):
        x = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8],
                          [2, 10, 9, 10, 3, 15, 16, 4, 8],
                          [2, 10, 9, 10, 3, 11, 9, 4, 8]])
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

    def test_bf16_t64_and_nonmutation(self):
        torch.manual_seed(12); model = candidate.Model(spec(), d_model=16)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.isfinite(model(ROWS)[0]).all())
        t64 = torch.tensor([[2, 8, 3, 8, 4, 13, 11]])
        self.assertEqual(model(t64)[1]["active_refinements"].item(), 128)
        for key, value in model.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__": unittest.main()
