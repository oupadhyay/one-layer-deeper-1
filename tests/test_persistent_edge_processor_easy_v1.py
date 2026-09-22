import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/persistent_edge_processor_easy_v1/submission.py"
LOADER = importlib.util.spec_from_file_location("persistent_edge_processor_easy_v1", PATH)
candidate = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(candidate)


def spec(length=20):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2, 3, 4), x=(5, 6), t=(1,), length=20):
    values = [candidate.N, *(v + candidate.DIGIT for v in n), candidate.X,
              *(v + candidate.DIGIT for v in x), candidate.T,
              *(v + candidate.DIGIT for v in t)]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class PersistentEdgeProcessorTests(unittest.TestCase):
    def test_contract_source_state_and_optimizer(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        for forbidden in ("from submissions", ".backward(", "training_loop", "remainder",
                          "quotient", "carry", "anti_diagonal", "lookup"):
            self.assertNotIn(forbidden, source.lower())
        model = candidate.build_model(spec())
        state = count_model_state_elements(model)
        self.assertLess(state, 250_000)
        self.assertGreater(state, 100_000)
        bundle = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 3e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual([group["weight_decay"] for group in bundle.optimizer.param_groups],
                         [0.01, 0.0])
        self.assertAlmostEqual(bundle.scheduler.get_last_lr()[0], 3e-4 / 32)
        self.assertEqual((candidate.SUBMISSION.batch_size,
                          candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (256, 512, None))

    def test_parser_alignment_edge_rounds_and_t1_gate(self):
        model = candidate.build_model(spec()).train()
        mixed = torch.cat((row(), row(n=(8, 7), x=(9,), t=(3,))))
        mask, nd, xd, widths, steps, place_mask = model.prepare(mixed)
        self.assertEqual(nd[0, :4].tolist(), [4, 3, 2, 1])
        self.assertEqual(xd[0, :4].tolist(), [6, 5, 0, 0])
        self.assertEqual((widths.tolist(), steps.tolist()), ([4, 2], [1, 3]))
        self.assertEqual(place_mask.sum(1).tolist(), [4, 2])
        logits, info = model(mixed)
        self.assertEqual((info["macrosteps"], info["edge_round_calls"]), (1, 4))
        self.assertEqual(info["edge_shape"], (2, 4, 4, 64))
        logits[0, :, candidate.DIGIT:].sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.zero_grad(set_to_none=True)
        model(row(t=(3,)))[0].sum().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_capacity_determinism_batch_purity_and_max_t(self):
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
        with torch.no_grad():
            together = model(torch.cat((narrow, wide)))[0]
            separate = torch.cat((model(narrow)[0], model(wide)[0]))
            again = model(torch.cat((narrow, wide)))[0]
        torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(together, again, atol=0, rtol=0)

        with torch.no_grad():
            output, info = model(row(n=(3, 2), x=(1,), t=(6, 4)))
        self.assertEqual((info["parsed_steps"].item(), info["macrosteps"],
                          info["edge_round_calls"]), (64, 64, 256))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
