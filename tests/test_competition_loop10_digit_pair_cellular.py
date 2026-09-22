import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop10_digit_pair_cellular as candidate


def spec(length=13):
    return ModelSpec(17, length, 100_000)


def row(n=(3, 2, 3), x=(1, 2, 3), t=(1,), padding=0):
    return torch.tensor([[candidate.N, *(d + candidate.DIGIT for d in n),
                          candidate.X, *(d + candidate.DIGIT for d in x),
                          candidate.T, *(d + candidate.DIGIT for d in t)] +
                         [candidate.PAD] * padding])


class DigitPairCellularTests(unittest.TestCase):
    def test_state_optimizer_source_and_width_contract(self):
        torch.manual_seed(74)
        model = candidate.build_model(spec())
        torch.manual_seed(74)
        repeat = candidate.build_model(spec())
        torch.manual_seed(74)
        wide = candidate.build_model(spec(32))
        self.assertEqual(count_model_state_elements(model), 20906)
        self.assertLess(count_model_state_elements(model), 100_000)
        self.assertEqual(set(model.state_dict()), set(wide.state_dict()))
        for key, value in model.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), repeat.state_dict()[key].numpy().tobytes())
            self.assertEqual(value.shape, wide.state_dict()[key].shape)
        self.assertFalse(any("place" in key or "position" in key for key in model.state_dict()))
        bundle = candidate.build_optimizer(model, OptimizerSpec(1600, "cpu"))
        owned = [id(parameter) for group in bundle.optimizer.param_groups
                 for parameter in group["params"]]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(len(owned), len(list(model.parameters())))
        self.assertEqual([group["weight_decay"] for group in bundle.optimizer.param_groups], [.02, 0.])
        self.assertEqual(bundle.optimizer.defaults["lr"], 8e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .98))
        self.assertAlmostEqual(bundle.scheduler.get_last_lr()[0], 4e-5)
        source_path = Path(candidate.__file__)
        source = source_path.read_text(encoding="utf-8")
        validate_submission_source(source_path.name, source, 256 * 1024,
                                   required_filename=None)
        for forbidden in ("competition_submission", "MultiheadAttention", "prod_table",
                          "i+j", "remainder", "torch.load"):
            self.assertNotIn(forbidden, source)

    def test_parse_workspace_sharing_boundaries_locality_and_counts(self):
        model = candidate.build_model(spec()).train()
        inputs = torch.cat((row(n=(3, 2, 3), x=(1, 2, 3), t=(1,), padding=2),
                            row(n=(9, 0, 0, 1), x=(1, 2, 3, 4), t=(3,))))
        init_calls = cell_calls = readout_calls = 0
        workspaces = []
        def init_hook(_module, _inputs, output):
            nonlocal init_calls
            init_calls += 1
            workspaces.append(output)
        def cell_hook(_module, _inputs, _output):
            nonlocal cell_calls
            cell_calls += 1
        def readout_hook(_module, _inputs, _output):
            nonlocal readout_calls
            readout_calls += 1
        hooks = [model.initializer.register_forward_hook(init_hook),
                 model.cell.register_forward_hook(cell_hook),
                 model.readout.register_forward_hook(readout_hook)]
        _, aux = model(inputs)
        for hook in hooks:
            hook.remove()
        self.assertEqual(aux["parsed_steps"].tolist(), [1, 3])
        self.assertEqual(aux["widths"].tolist(), [3, 4])
        self.assertEqual(aux["active_row_cell_calls"].tolist(), [6, 24])
        self.assertEqual(init_calls, 3)
        self.assertEqual(cell_calls, 24)
        self.assertEqual(readout_calls, 3)
        self.assertEqual(len({id(value) for value in workspaces}), 3)
        probabilities = F.one_hot(torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), 10).float()
        workspace, valid = model._workspace(
            probabilities, torch.tensor([[3, 2, 3, 0], [9, 0, 0, 1]]),
            torch.tensor([3, 4]),
        )
        self.assertEqual(torch.count_nonzero(workspace[~valid]).item(), 0)
        self.assertTrue(valid[0, :3, 3].all())
        self.assertFalse(valid[0, :, 4].any())
        self.assertTrue(valid[1, :4, 4].all())
        torch.manual_seed(1)
        cell = candidate.LocalCell().eval()
        original = torch.zeros(1, 7, 8, 48)
        changed = original.clone(); changed[0, 3, 4, 0] = 1
        with torch.no_grad():
            delta = (cell(changed) - cell(original)).abs().sum(-1)[0]
        outside = torch.ones_like(delta, dtype=torch.bool); outside[2:5, 3:6] = False
        self.assertEqual(torch.count_nonzero(delta[outside]).item(), 0)

    def test_feedback_gradients_gate_and_invariance(self):
        torch.manual_seed(7)
        model = candidate.build_model(spec()).train()
        readouts = []
        def capture(_module, _inputs, output):
            output.retain_grad(); readouts.append(output)
        hook = model.readout.register_forward_hook(capture)
        logits, aux = model(row(t=(3,)))
        hook.remove()
        self.assertTrue(torch.equal(aux["digit_probabilities"],
                                   aux["digit_probabilities"].round()))
        F.cross_entropy(logits[0, -3:, candidate.DIGIT:], torch.tensor([1, 2, 3])).backward()
        self.assertEqual(len(readouts), 3)
        self.assertTrue(all(value.grad is not None and value.grad.abs().sum() > 0
                            for value in readouts))
        for horizon, scale in ((1, 1.), (2, .01), (3, .01)):
            torch.manual_seed(9)
            gated_model = candidate.build_model(spec()).train()
            gated, _ = gated_model(row(t=(horizon,)))
            F.cross_entropy(gated[0, -3:, candidate.DIGIT:], torch.tensor([1, 2, 3])).backward()
            torch.manual_seed(9)
            plain_model = candidate.build_model(spec()).train()
            _, plain_aux = plain_model(row(t=(horizon,)))
            F.cross_entropy(plain_aux["ungated_logits"][0, -3:, candidate.DIGIT:],
                            torch.tensor([1, 2, 3])).backward()
            for gated_parameter, plain_parameter in zip(gated_model.parameters(),
                                                         plain_model.parameters()):
                torch.testing.assert_close(gated_parameter.grad, plain_parameter.grad * scale,
                                           atol=2e-8, rtol=3e-5)
        torch.manual_seed(17)
        model = candidate.build_model(spec()).eval()
        narrow = row(n=(3, 2, 3), x=(1, 2, 3), t=(2,), padding=2)
        wide = row(n=(9, 0, 0, 1), x=(1, 2, 3, 4), t=(3,))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            together = model(torch.cat((narrow, wide)))[0]
            separate = torch.cat((model(narrow)[0], model(wide)[0]))
            again = model(torch.cat((narrow, wide)))[0]
        torch.testing.assert_close(together, separate, atol=3e-6, rtol=3e-6)
        self.assertTrue(torch.equal(together, again))
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in model.state_dict().items()))

    def test_bfloat16_t64_max_length_and_capacity(self):
        model = candidate.build_model(spec()).to(torch.bfloat16).train()
        model.zero_grad(set_to_none=True)
        output, aux = model(row(t=(6, 4)))
        loss = F.cross_entropy(output[0, -3:, candidate.DIGIT:], torch.tensor([1, 2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(aux["parsed_steps"].item(), 64)
        self.assertEqual(aux["active_row_cell_calls"].item(), 384)
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        exact = row(n=(9, 0, 0, 1), x=(1, 2, 3, 4), t=(3,))
        self.assertEqual(candidate.build_model(spec(12))(exact)[0].shape, (1, 12, 17))
        with self.assertRaises(ValueError):
            candidate.build_model(spec(11))(exact)
        x = torch.tensor([[2,10,9,10,3,8,12,4,8],
                          [2,10,9,10,3,15,16,4,8],
                          [2,10,9,10,3,11,9,4,8]])
        labels = torch.tensor([[9,9,12], [8,13,16], [8,11,16]])
        positions = torch.tensor([[6,7,8]] * 3); batch = torch.arange(3)[:, None]
        torch.manual_seed(74)
        model = candidate.build_model(spec()).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _update in range(1, 101):
            optimizer.zero_grad(set_to_none=True)
            selected = model(x)[0][batch, positions]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward()
            optimizer.step()
            if selected.argmax(-1).eq(labels).all():
                break
        self.assertTrue(selected.argmax(-1).eq(labels).all())
        model.eval()
        with torch.no_grad():
            evaluated = model(x)[0][batch, positions]
        self.assertTrue(evaluated.argmax(-1).eq(labels).all())
        self.assertLessEqual(_update, 100)


if __name__ == "__main__":
    unittest.main()
