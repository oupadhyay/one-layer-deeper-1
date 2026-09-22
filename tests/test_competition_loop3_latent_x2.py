import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
import competition_loop2_latent as parent
import competition_loop3_latent_x2 as candidate


def spec(length=13):
    return ModelSpec(17, length, 20_000_000)


def rows():
    return torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8],
                         [2, 10, 9, 10, 3, 15, 16, 4, 9],
                         [2, 10, 9, 10, 3, 11, 9, 4, 10]])


class LatentX2Preflight(unittest.TestCase):
    def test_frozen_identity_state_and_optimizer(self):
        torch.manual_seed(74); a = parent.Model(spec())
        torch.manual_seed(74); b = candidate.Model(spec())
        self.assertEqual(count_model_state_elements(a), 223493)
        self.assertEqual(count_model_state_elements(b), 223493)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        for key in a.state_dict():
            self.assertEqual(a.state_dict()[key].shape, b.state_dict()[key].shape)
            self.assertEqual(a.state_dict()[key].numpy().tobytes(), b.state_dict()[key].numpy().tobytes())
        self.assertIs(candidate.build_optimizer, parent.build_optimizer)
        oa = parent.build_optimizer(a, OptimizerSpec(60, "cpu"))
        ob = candidate.build_optimizer(b, OptimizerSpec(60, "cpu"))
        self.assertEqual(oa.optimizer.state_dict(), ob.optimizer.state_dict())
        self.assertEqual([g["lr"] for g in oa.optimizer.param_groups], [g["lr"] for g in ob.optimizer.param_groups])
        self.assertEqual(oa.scheduler.state_dict(), ob.scheduler.state_dict())

    def test_exact_refinement_carry_readout_and_gradients(self):
        torch.manual_seed(9); model = candidate.Model(spec(), d_model=16)
        calls, outputs, readouts = [], [], []
        original = model.transition.refine
        def spy(state, context, mask):
            calls.append(state.detach().clone()); out = original(state, context, mask)
            out.retain_grad(); outputs.append(out); return out
        model.transition.refine = spy
        model.endpoint_readout.register_forward_hook(lambda *args: readouts.append(1))
        logits, aux = model(rows())
        F.cross_entropy(logits[2, 6:9], torch.tensor([9, 16, 7])).backward()
        self.assertEqual(len(calls), 6); self.assertEqual(len(readouts), 1)
        self.assertEqual(aux["executed_macrosteps"].item(), 3)
        self.assertEqual(aux["active_refinements"].tolist(), [2, 4, 6])
        for i in range(0, 6, 2): self.assertTrue(torch.equal(calls[i + 1], outputs[i].detach()))
        self.assertTrue(all(o.grad is not None and torch.isfinite(o.grad).all() and o.grad.abs().sum() > 0 for o in outputs))

    def test_invariance_modes_bf16_t64_and_nonmutation(self):
        torch.manual_seed(12); model = candidate.Model(spec(), d_model=16)
        x = rows(); before = {k: v.clone() for k, v in model.state_dict().items()}
        model.train(); mixed = model(x)[0].detach(); model.eval()
        with torch.no_grad():
            evaluation = model(x)[0]; separate = torch.cat([model(r[None])[0] for r in x])
            padded = model(F.pad(x, (0, 3)))[0][:, :9]; again = model(x)[0]
        torch.testing.assert_close(mixed, evaluation); torch.testing.assert_close(evaluation, separate, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(evaluation, padded); torch.testing.assert_close(evaluation, again)
        for k, v in model.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))
        with torch.autocast("cpu", dtype=torch.bfloat16): self.assertTrue(torch.isfinite(model(x)[0]).all())
        t64 = torch.tensor([[2, 8, 3, 8, 4, 13, 11]])
        out, aux = model(t64); self.assertTrue(torch.isfinite(out).all()); self.assertEqual(aux["active_refinements"].item(), 128)

    def test_true_label_capacity(self):
        torch.manual_seed(74); model = candidate.Model(spec(), d_model=32).train()
        x = rows(); labels = torch.tensor([[9, 9, 12], [8, 10, 14], [9, 16, 7]])
        positions = torch.tensor([[6, 7, 8]] * 3); batch = torch.arange(3)[:, None]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True); selected = model(x)[0][batch, positions]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward(); optimizer.step()
            with torch.no_grad(): selected = model(x)[0][batch, positions]
            if selected.argmax(-1).eq(labels).all(): break
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad(): self.assertTrue(model(x)[0][batch, positions].argmax(-1).eq(labels).all())


if __name__ == "__main__": unittest.main()
