import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "submissions" / "biscan_m3_remote_v3_compositional" / "submission.py"
PATH = ROOT / "submissions" / "biscan_entropy_preference_easy_v1" / "submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(module, n=(3, 2), x=(6, 1), t=2, length=20):
    values = [module.N, *(module.DIGIT + d for d in n), module.X,
              *(module.DIGIT + d for d in x), module.T,
              *(module.DIGIT + int(d) for d in str(t))]
    return torch.tensor([values + [module.PAD] * (length - len(values))])


class BiScanEntropyPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.parent = load("biscan_entropy_parent", PARENT_PATH)
        self.module = load("biscan_entropy_preference", PATH)
        self.spec = ModelSpec(17, 20, 20_000_000)

    def test_source_state_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        model = self.module.build_model(self.spec)
        self.assertEqual(count_model_state_elements(model), 63_178)
        self.assertEqual((self.module.SUBMISSION.batch_size,
                          self.module.SUBMISSION.eval_batch_size,
                          self.module.SUBMISSION.max_steps), (64, 128, None))

    def test_eval_predictions_are_identical_to_parent(self):
        torch.manual_seed(91)
        parent = self.parent.Model(self.spec).eval()
        candidate = self.module.Model(self.spec).eval()
        candidate.load_state_dict(parent.state_dict())
        ids = torch.cat((row(self.module, t=1), row(self.module, n=(9, 0, 4, 3),
                                                    x=(1, 2, 3, 4), t=64)))
        with torch.no_grad():
            expected, _ = parent(ids)
            actual, auxiliary = candidate(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(auxiliary["entropy_count"].item(), 252.)

    def test_preference_is_bounded_label_free_and_differentiable(self):
        model = self.module.Model(self.spec).train()
        ids = torch.cat((row(self.module, t=1), row(self.module, n=(9, 0, 4, 3),
                                                    x=(1, 2, 3, 4), t=3)))
        logits, auxiliary = model(ids)
        self.assertGreaterEqual(auxiliary["feedback_entropy"].item(), 0.)
        self.assertLessEqual(auxiliary["feedback_entropy"].item(), 1.00001)
        labels = torch.tensor([1, 2, 3, 4])
        endpoint = F.cross_entropy(logits[1, 9:13, 7:], labels)
        total = endpoint + .01 * auxiliary["feedback_entropy"]
        preference_gradient = torch.autograd.grad(auxiliary["feedback_entropy"],
                                                  model.readout.weight, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(preference_gradient).all())
        self.assertGreater(preference_gradient.abs().sum().item(), 0.)
        total.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_t1_has_no_preference(self):
        model = self.module.Model(self.spec).train()
        _, auxiliary = model(row(self.module, t=1))
        self.assertEqual(auxiliary["entropy_count"].item(), 0.)
        self.assertEqual(auxiliary["feedback_entropy"].item(), 0.)


if __name__ == "__main__":
    unittest.main()
