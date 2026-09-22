import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop6_biscan_t1grad as c
import competition_submission_c1a as c1a


def spec(length=13):
    return ModelSpec(17, length, 20_000_000)


def row(t=1, x=(8, 12), n=(10, 9, 10), padding=0):
    t_tokens = [c.DIGIT + t] if t < 10 else [c.DIGIT + t // 10, c.DIGIT + t % 10]
    return torch.tensor([[c.N, *n, c.X, *x, c.T, *t_tokens] + [c.PAD] * padding])


class CanonicalBiScanTests(unittest.TestCase):
    def test_structure_seed_width_independence_optimizer_and_source(self):
        torch.manual_seed(74); a = c.Model(spec(13))
        torch.manual_seed(74); repeat = c.Model(spec(13))
        torch.manual_seed(74); wider = c.Model(spec(24))
        self.assertEqual(count_model_state_elements(a), 63_178)
        self.assertEqual(set(dict(a.named_modules())), {
            "", "embedding", "scan", "scan.down_cell", "scan.up_cell", "readout"
        })
        self.assertEqual(a.embedding.weight.shape, (10, 32))
        self.assertEqual((a.scan.down_cell.input_size, a.scan.up_cell.input_size), (64, 128))
        self.assertEqual(a.readout.weight.shape, (10, 64))
        for key, value in a.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), repeat.state_dict()[key].numpy().tobytes())
            self.assertEqual(value.numpy().tobytes(), wider.state_dict()[key].numpy().tobytes())
        self.assertIs(c.build_optimizer, c1a.build_optimizer)
        oa = c.build_optimizer(a, OptimizerSpec(60, "cpu"))
        ob = c.build_optimizer(repeat, OptimizerSpec(60, "cpu"))
        self.assertEqual(oa.optimizer.state_dict(), ob.optimizer.state_dict())
        self.assertEqual(oa.scheduler.state_dict(), ob.scheduler.state_dict())
        source = Path(c.__file__).read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(Path(c.__file__).name, source, 256 * 1024,
                                                    required_filename=None), Path(c.__file__).name)

    def test_lsd_fields_context_separation_width_and_t_only_loop_count(self):
        model = c.Model(spec()).eval()
        base = row(1)
        changed_t = row(3)
        changed_x = row(1, x=(9, 16))
        changed_n = row(1, n=(10, 9, 11))
        with torch.no_grad():
            _, a = model(base); _, at = model(changed_t); _, ax = model(changed_x); _, an = model(changed_n)
        self.assertEqual(a["context_digits"][0, :4].tolist(), [3, 2, 3, 0])
        self.assertEqual(a["initial_state_digits"][0, :4].tolist(), [5, 1, 0, 0])
        self.assertEqual(a["width_mask"][0, :5].tolist(), [True, True, True, False, False])
        self.assertTrue(torch.equal(a["context_digits"], at["context_digits"]))
        self.assertTrue(torch.equal(a["context_digits"], ax["context_digits"]))
        self.assertTrue(torch.equal(a["initial_state_digits"], at["initial_state_digits"]))
        self.assertFalse(torch.equal(a["initial_state_digits"], ax["initial_state_digits"]))
        self.assertFalse(torch.equal(a["context_digits"], an["context_digits"]))
        self.assertEqual((a["executed_macrosteps"].item(), at["executed_macrosteps"].item()), (1, 3))
        out_width = ~a["width_mask"]
        expected_zero = F.one_hot(torch.zeros_like(a["initial_state_digits"]), 10).to(
            a["digit_probabilities"].dtype)
        self.assertTrue(torch.equal(a["digit_probabilities"][out_width], expected_zero[out_width]))

    def test_place_order_shared_cells_readout_count_and_hidden_reset(self):
        model = c.Model(spec(5)).eval()
        state = torch.arange(5.0).view(1, 5, 1).expand(-1, -1, 32)
        context = torch.zeros_like(state)
        width = torch.ones(1, 5, dtype=torch.bool)
        down_places, up_places, down_h, up_h = [], [], [], []
        down = model.scan.down_cell.forward
        up = model.scan.up_cell.forward
        def capture_down(value, hidden):
            down_places.append(int(value[0, 0])); down_h.append(hidden.clone()); return down(value, hidden)
        def capture_up(value, hidden):
            up_places.append(int(value[0, 0])); up_h.append(hidden.clone()); return up(value, hidden)
        model.scan.down_cell.forward = capture_down
        model.scan.up_cell.forward = capture_up
        model.scan(state, context, width)
        self.assertEqual(down_places, [4, 3, 2, 1, 0])
        self.assertEqual(up_places, [0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(down_h[0], torch.zeros_like(down_h[0])))
        self.assertTrue(torch.equal(up_h[0], torch.zeros_like(up_h[0])))
        self.assertIs(model.scan.down_cell, model.scan.down_cell)
        self.assertIs(model.scan.up_cell, model.scan.up_cell)

        model = c.Model(spec()).eval()
        calls = {"scan": 0, "readout": 0}
        hs = [model.scan.register_forward_hook(lambda *_: calls.__setitem__("scan", calls["scan"] + 1)),
              model.readout.register_forward_hook(lambda *_: calls.__setitem__("readout", calls["readout"] + 1))]
        model(row(3))
        for hook in hs: hook.remove()
        self.assertEqual(calls, {"scan": 3, "readout": 3})

    def test_soft_and_hard_whole_state_feedback_and_active_gradients(self):
        for training in (True, False):
            torch.manual_seed(7); model = c.Model(spec()).train(training)
            scan_inputs, readouts = [], []
            scan = model.scan.forward
            def capture_scan(state, context, width):
                scan_inputs.append(state.clone()); return scan(state, context, width)
            def capture_readout(_, __, output):
                output.retain_grad() if output.requires_grad else None
                readouts.append(output)
            model.scan.forward = capture_scan
            hook = model.readout.register_forward_hook(capture_readout)
            logits, aux = model(row(3))
            hook.remove()
            self.assertEqual(len(scan_inputs), 3)
            first_probs = (torch.softmax(readouts[0], -1) if training else
                           F.one_hot(readouts[0].argmax(-1), 10).to(readouts[0].dtype))
            first_probs = torch.where(aux["width_mask"][:, :, None], first_probs,
                                      F.one_hot(torch.zeros_like(aux["initial_state_digits"]), 10).to(first_probs.dtype))
            torch.testing.assert_close(scan_inputs[1], first_probs @ model.embedding.weight, atol=0, rtol=0)
            if not training:
                self.assertTrue(((aux["digit_probabilities"] == 0) |
                                 (aux["digit_probabilities"] == 1)).all())
            else:
                selected = logits[0, -2:, c.DIGIT:]
                F.cross_entropy(selected, torch.tensor([1, 2])).backward()
                self.assertTrue(all(value.grad is not None and value.grad.abs().sum() > 0
                                    for value in readouts))

    def test_gradient_gate_and_mixed_homogeneous_isolation(self):
        for horizon, scale in ((1, 1.0), (2, .01), (3, .01)):
            torch.manual_seed(11); model = c.Model(spec()).train()
            output, aux = model(row(horizon)); ungated = aux["ungated_logits"]
            target = torch.tensor([1, 2])
            gated_loss = F.cross_entropy(output[0, -2:, c.DIGIT:], target)
            plain_loss = F.cross_entropy(ungated[0, -2:, c.DIGIT:], target)
            gated = torch.autograd.grad(gated_loss, ungated, retain_graph=True)[0]
            plain = torch.autograd.grad(plain_loss, ungated, retain_graph=True)[0]
            torch.testing.assert_close(gated, plain * scale, atol=2e-8, rtol=2e-6)

            torch.manual_seed(11); gated_model = c.Model(spec()).train()
            torch.manual_seed(11); plain_model = c.Model(spec()).train()
            gated_out = gated_model(row(horizon))[0]
            _, plain_aux = plain_model(row(horizon))
            F.cross_entropy(gated_out[0, -2:, c.DIGIT:], target).backward()
            F.cross_entropy(plain_aux["ungated_logits"][0, -2:, c.DIGIT:], target).backward()
            for ga, pa in zip(gated_model.parameters(), plain_model.parameters()):
                torch.testing.assert_close(ga.grad, pa.grad * scale, atol=2e-8, rtol=2e-5)

        torch.manual_seed(13); model = c.Model(spec()).train()
        mixed = torch.cat((row(1), row(2), row(3)))
        target = torch.tensor([1, 2])
        model.zero_grad(); mixed_loss = F.cross_entropy(model(mixed)[0][1, -2:, c.DIGIT:], target)
        mixed_loss.backward(); mixed_grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(); alone_loss = F.cross_entropy(model(row(2))[0][0, -2:, c.DIGIT:], target)
        alone_loss.backward()
        for mixed_grad, parameter in zip(mixed_grads, model.parameters()):
            torch.testing.assert_close(mixed_grad, parameter.grad, atol=2e-7, rtol=2e-6)

    def test_mixed_padding_batch_variable_width_max_length_t64_and_purity(self):
        torch.manual_seed(17); model = c.Model(spec()).eval()
        mixed = torch.cat((row(1), row(2), row(3)))
        with torch.no_grad():
            together = model(mixed)[0]
            separate = torch.cat([model(row(t))[0] for t in (1, 2, 3)])
        torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-6)

        padded = row(2, padding=3)
        other = row(3, padding=3)
        with torch.no_grad():
            single = model(row(2))[0]
            batched = model(torch.cat((padded, other)))[0]
        torch.testing.assert_close(single, batched[:1, :single.shape[1]], atol=2e-6, rtol=2e-6)

        exact = torch.tensor([[c.N, 8, 10, 9, 10, c.X, 8, 9, 10, 11, c.T, 13, 11]])
        self.assertEqual(exact.shape[1], 13)
        self.assertEqual(model(exact)[0].shape, (1, 13, 17))
        with self.assertRaises(ValueError): model(torch.cat((exact, torch.zeros(1, 1, dtype=torch.long)), 1))

        before = {key: value.clone() for key, value in model.state_dict().items()}
        t64 = row(64)
        first = model(t64)
        again = model(t64)
        self.assertEqual(first[1]["active_updates"].item(), 64)
        self.assertTrue(torch.equal(first[0], again[0]))
        for key, value in model.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))

    def test_bf16_finite_and_t1_capacity(self):
        torch.manual_seed(19); model = c.Model(spec()).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _ = model(row(3))
            loss = F.cross_entropy(logits[0, -2:, c.DIGIT:], torch.tensor([1, 2]))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients and all(torch.isfinite(value).all() for value in gradients))

        inputs = torch.tensor([[2,10,9,10,3,8,12,4,8], [2,10,9,10,3,15,16,4,8],
                               [2,10,9,10,3,11,9,4,8]])
        labels = torch.tensor([[9,9,12], [8,13,16], [8,11,16]])
        positions, batch = torch.tensor([[6,7,8]] * 3), torch.arange(3)[:, None]
        torch.manual_seed(74); model = c.Model(spec()).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True); selected = model(inputs)[0][batch, positions]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward(); optimizer.step()
            if selected.argmax(-1).eq(labels).all(): break
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad(): selected = model(inputs)[0][batch, positions]
        self.assertTrue(selected.argmax(-1).eq(labels).all())


if __name__ == "__main__": unittest.main()
