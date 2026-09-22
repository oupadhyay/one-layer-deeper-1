import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop9_direct_transformer as c
import competition_submission_c1a as c1a


def spec(length=13):
    return ModelSpec(17, length, 20_000_000)


def row(t=1, x=(8, 12), n=(10, 9, 10), padding=0):
    td = [c.DIGIT + t] if t < 10 else [c.DIGIT + t // 10, c.DIGIT + t % 10]
    return torch.tensor([[c.N, *n, c.X, *x, c.T, *td] + [c.PAD] * padding])


class DirectTransformerTests(unittest.TestCase):
    def test_inventory_seed_optimizer_and_source(self):
        torch.manual_seed(74)
        a = c.build_model(spec())
        torch.manual_seed(74)
        b = c.Model(spec())
        self.assertEqual(count_model_state_elements(a), 800138)
        self.assertEqual(len(a.blocks), 4)
        self.assertEqual([type(x).__name__ for x in a.modules()].count("MultiheadAttention"), 4)
        for key, value in a.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), b.state_dict()[key].numpy().tobytes())
        self.assertIs(c.build_optimizer, c1a.build_optimizer)
        oa = c.build_optimizer(a, OptimizerSpec(60, "cpu"))
        ob = c.build_optimizer(b, OptimizerSpec(60, "cpu"))
        self.assertEqual(oa.optimizer.state_dict(), ob.optimizer.state_dict())
        self.assertEqual(oa.scheduler.state_dict(), ob.scheduler.state_dict())
        source = Path(c.__file__).read_text(encoding="utf-8")
        validate_submission_source(Path(c.__file__).name, source, 256 * 1024,
                                   required_filename=None)

    def test_direct_calls_t_is_input_and_output_alignment(self):
        torch.manual_seed(7)
        model = c.Model(spec()).eval()
        calls = [0, 0, 0, 0]
        hooks = [block.register_forward_hook(
            lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1)
        ) for index, block in enumerate(model.blocks)]
        output1, aux1 = model(row(1))
        output3, aux3 = model(row(3))
        for hook in hooks:
            hook.remove()
        self.assertEqual(calls, [2, 2, 2, 2])
        self.assertEqual(aux1["parsed_steps"].item(), 1)
        self.assertEqual(aux3["parsed_steps"].item(), 3)
        self.assertEqual(aux1["direct_mapping_calls"].item(), 1)
        self.assertFalse(torch.equal(output1, output3))
        torch.testing.assert_close(output1[0, -1, c.DIGIT:],
                                   aux1["output_slot_logits"][0, 0])
        torch.testing.assert_close(output1[0, -2, c.DIGIT:],
                                   aux1["output_slot_logits"][0, 1])

    def test_invariance_bf16_and_gradients(self):
        torch.manual_seed(17)
        model = c.Model(spec()).eval()
        mixed = torch.cat((row(1), row(2), row(3)))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            together = model(mixed)[0]
            separate = torch.cat([model(row(t))[0] for t in (1, 2, 3)])
            again = model(mixed)[0]
        torch.testing.assert_close(together, separate, atol=3e-6, rtol=3e-6)
        self.assertTrue(torch.equal(together, again))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        padded = row(2, padding=3)
        other = row(3, padding=3)
        with torch.no_grad():
            single = model(row(2))[0]
            batched = model(torch.cat((padded, other)))[0]
        torch.testing.assert_close(single, batched[:1, :single.shape[1]],
                                   atol=3e-6, rtol=3e-6)
        model.train().zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, _ = model(row(3))
            loss = F.cross_entropy(output[0, -2:, c.DIGIT:], torch.tensor([1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_mixed_horizon_capacity(self):
        x = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8],
                          [2, 10, 9, 10, 3, 15, 16, 4, 9],
                          [2, 10, 9, 10, 3, 11, 9, 4, 10]])
        labels = torch.tensor([[9, 9, 12], [8, 10, 14], [9, 16, 7]])
        positions = torch.tensor([[6, 7, 8]] * 3)
        batch = torch.arange(3)[:, None]
        torch.manual_seed(74)
        model = c.Model(spec()).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            selected = model(x)[0][batch, positions]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward()
            optimizer.step()
            if selected.argmax(-1).eq(labels).all():
                break
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad():
            self.assertTrue(model(x)[0][batch, positions].argmax(-1).eq(labels).all())


if __name__ == "__main__":
    unittest.main()
