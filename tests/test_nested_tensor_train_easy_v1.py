import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/nested_tensor_train_easy_v1/submission.py"
LOADER = importlib.util.spec_from_file_location("nested_tensor_train_easy_v1", PATH)
candidate = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(candidate)


def spec(length=20):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2, 3, 4), x=(5, 6), t=(1,), length=20):
    values = [candidate.N, *(v + candidate.DIGIT for v in n), candidate.X,
              *(v + candidate.DIGIT for v in x), candidate.T,
              *(v + candidate.DIGIT for v in t)]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class NestedTensorTrainTests(unittest.TestCase):
    def test_contract_state_source_and_optimizer(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        for forbidden in ("from submissions", ".backward(", "training_loop", "remainder",
                          "quotient", "carry", "anti_diagonal", "lookup", "conv"):
            self.assertNotIn(forbidden, source.lower())
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), candidate.STATE_ELEMENTS)
        self.assertLess(candidate.STATE_ELEMENTS, 250_000)
        bundle = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
        parameters = [id(p) for group in bundle.optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameters), len(set(parameters)))
        self.assertEqual(set(parameters), {id(p) for p in model.parameters()})
        self.assertEqual(bundle.optimizer.defaults["lr"], 3e-4)
        self.assertEqual([group["weight_decay"] for group in bundle.optimizer.param_groups],
                         [0.01, 0.0])
        self.assertEqual((candidate.SUBMISSION.batch_size,
                          candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (256, 512, None))

    def test_parser_scan_order_sharing_feedback_and_gradients(self):
        model = candidate.build_model(spec()).train()
        mixed = torch.cat((row(), row(n=(8, 7), x=(9,), t=(3,))))
        _, nd, xd, widths, steps, place_mask = model.prepare(mixed)
        self.assertEqual(nd[0, :4].tolist(), [4, 3, 2, 1])
        self.assertEqual(xd[0, :4].tolist(), [6, 5, 0, 0])
        self.assertEqual((widths.tolist(), steps.tolist()), ([4, 2], [1, 3]))
        calls = []
        hook = model.contraction_cell.register_forward_hook(
            lambda _m, _i, _o: calls.append(tuple(_i[0].shape))
        )
        logits, info = model(mixed)
        hook.remove()
        self.assertEqual(calls, [(2, 4, 112)] * 4 + [(2, 112)] * 4)
        self.assertEqual((info["inner_calls"], info["outer_calls"],
                          info["output_calls"]), (4, 4, 4))
        self.assertIs(model.contraction_cell, model.contraction_cell)
        logits[0, :, candidate.DIGIT:].sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        model.zero_grad(set_to_none=True)
        model(row(t=(3,)))[0].sum().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))
        self.assertTrue(torch.equal(info["feedback"], info["feedback"].round()))

    def test_capacity_batch_purity_determinism_and_max_t(self):
        torch.manual_seed(74)
        model = candidate.build_model(spec()).train()
        ids = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 8],
            [2, 10, 9, 10, 3, 15, 16, 4, 8],
            [2, 10, 9, 10, 3, 11, 9, 4, 8],
        ])
        labels = torch.tensor([[9, 9, 12], [8, 13, 16], [8, 11, 16]])
        positions = torch.tensor([[6, 7, 8]] * 3)
        batch = torch.arange(3)[:, None]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for update in range(1, 101):
            optimizer.zero_grad(set_to_none=True)
            selected = model(ids)[0][batch, positions]
            F.cross_entropy(selected.flatten(0, 1), labels.flatten()).backward()
            optimizer.step()
            if selected.argmax(-1).eq(labels).all():
                break
        self.assertTrue(selected.argmax(-1).eq(labels).all(), update)

        model.eval()
        narrow = row(n=(3, 2), x=(1,), t=(1,))
        wide = row(n=(1, 2, 3, 4), x=(5, 6), t=(1,))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            together = model(torch.cat((narrow, wide)))[0]
            separate = torch.cat((model(narrow)[0], model(wide)[0]))
            again = model(torch.cat((narrow, wide)))[0]
            maximum, info = model(row(n=(3, 2), x=(1,), t=(6, 4)))
        torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(together, again, atol=0, rtol=0)
        self.assertEqual((info["parsed_steps"].item(), info["macrosteps"],
                          info["inner_calls"], info["output_calls"]), (64, 64, 128, 128))
        self.assertTrue(torch.isfinite(maximum).all())
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
