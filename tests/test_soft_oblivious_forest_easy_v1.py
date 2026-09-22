import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/soft_oblivious_forest_easy_v1/submission.py"
SPEC = importlib.util.spec_from_file_location("soft_forest", PATH)
forest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forest)


def row(t=(3,), length=20):
    values = [forest.N, 8, 9, 10, 11, forest.X, 12, 13, forest.T,
              *(forest.DIGIT + value for value in t)]
    return torch.tensor([values + [0] * (length - len(values))])


class SoftObliviousForestTests(unittest.TestCase):
    def test_contract_paths_gradients_feedback_alignment_and_purity(self):
        self.assertEqual(validate_submission_source(PATH.name, PATH.read_text(), 256_000), "submission.py")
        model = forest.build_model(ModelSpec(17, 20, 100_000))
        self.assertEqual(count_model_state_elements(model), forest.STATE_ELEMENTS)
        self.assertLess(forest.STATE_ELEMENTS, 100_000)
        self.assertEqual((forest.TREES, forest.DEPTH, forest.LEAVES), (16, 5, 32))
        self.assertLess(model.forest.split_bias.abs().max(), .01)
        n, np, x, xp, steps = model.parse(row(), row().ne(0))
        self.assertEqual(steps.item(), 3)
        probability = torch.nn.functional.one_hot(x, 10).float()
        logits, details = model.forest(n, np, probability, xp)
        self.assertEqual(logits.shape, (1, 4, 10))
        self.assertEqual(details["contexts"].shape, (1, 4, 64))
        self.assertEqual(details["paths"].shape, (1, 4, 16, 32))
        torch.testing.assert_close(details["paths"].sum(-1), torch.ones(1, 4, 16))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(model.forest.leaf.grad).all())
        self.assertTrue((model.forest.leaf.grad.abs().sum(-1) > 0).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

        model.zero_grad(set_to_none=True)
        model.train()
        output, info = model(row())
        self.assertEqual((info["steps"].item(), info["macrosteps"]), (3, 3))
        output[:, :, forest.DIGIT:].sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            first, detail = model(row((6, 4)))
            second, _ = model(row((6, 4)))
        self.assertEqual((detail["steps"].item(), detail["macrosteps"]), (64, 64))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertEqual(forest.SUBMISSION.batch_size, 256)
        self.assertIsNone(forest.SUBMISSION.max_steps)


if __name__ == "__main__":
    unittest.main()
