import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/hierarchical_probabilistic_circuit_easy_v1/submission.py"
SPEC = importlib.util.spec_from_file_location("hpc", PATH)
hpc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hpc)


def row(t=(3,), length=20):
    values = [hpc.N, 8, 9, 10, 11, hpc.X, 12, 13, hpc.T,
              *(hpc.DIGIT + value for value in t)]
    return torch.tensor([values + [0] * (length - len(values))])


class CircuitTests(unittest.TestCase):
    def test_contract_tree_feedback_gradients_and_purity(self):
        self.assertEqual(validate_submission_source(PATH.name, PATH.read_text(), 256_000), "submission.py")
        model = hpc.build_model(ModelSpec(17, 20, 100_000))
        self.assertEqual(count_model_state_elements(model), hpc.STATE_ELEMENTS)
        self.assertLess(hpc.STATE_ELEMENTS, 100_000)
        self.assertEqual(sum(1 for module in model.modules() if isinstance(module, hpc.SharedComposer)), 1)
        self.assertEqual(model.circuit.output_queries.shape, (4, 10, 16))
        n, np, x, xp, _ = model.parse(row(), row().ne(0))
        leaves = model.circuit.leaf_potentials(n, np, torch.nn.functional.one_hot(x, 10).float(), xp)
        self.assertEqual(leaves.shape, (1, 8, 16))
        self.assertGreater(leaves[:, 4:].std(), 0)
        model.train()
        output, info = model(row())
        self.assertEqual((info["steps"].item(), info["macrosteps"]), (3, 3))
        output[:, :, 7:].sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            first, detail = model(row((6, 4)))
            second, _ = model(row((6, 4)))
        self.assertEqual((detail["steps"].item(), detail["macrosteps"]), (64, 64))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertEqual(hpc.SUBMISSION.batch_size, 256)
        self.assertIsNone(hpc.SUBMISSION.max_steps)


if __name__ == "__main__":
    unittest.main()
