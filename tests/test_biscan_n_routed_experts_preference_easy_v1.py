import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "biscan_n_routed_experts_preference_easy_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("biscan_n_routed_experts_preference", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(module, n=(3, 2), x=(6, 1), t=2, length=20):
    values = [module.N, *(module.DIGIT + d for d in n), module.X,
              *(module.DIGIT + d for d in x), module.T,
              *(module.DIGIT + int(d) for d in str(t))]
    return torch.tensor([values + [module.PAD] * (length - len(values))])


class BiScanNRoutedExpertsPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.model = self.module.build_model(ModelSpec(17, 20, 20_000_000))

    def test_source_state_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        self.assertEqual(count_model_state_elements(self.model), 154_048)
        self.assertEqual((self.module.SUBMISSION.batch_size,
                          self.module.SUBMISSION.eval_batch_size,
                          self.module.SUBMISSION.max_steps), (64, 128, None))

    def test_preference_is_bounded_label_free_and_differentiable(self):
        ids = torch.cat((row(self.module, n=(8,), x=(2,), t=1),
                         row(self.module, n=(9, 0, 4, 3), x=(1, 2, 3, 4), t=3)))
        logits, auxiliary = self.model(ids)
        preference = auxiliary["routing_preference"]
        self.assertGreaterEqual(preference.item(), -1.00001)
        self.assertLessEqual(preference.item(), 1.00001)
        gradient = torch.autograd.grad(preference, self.model.router.weight, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.)
        loss = F.cross_entropy(logits[1, 9:13, 7:], torch.tensor([1, 2, 3, 4])) + .01 * preference
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))

    def test_eval_is_deterministic_and_pure(self):
        ids = row(self.module, n=(9, 0, 4, 3), x=(1, 2, 3, 4), t=3)
        self.model.eval()
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        with torch.no_grad():
            first, _ = self.model(ids)
            second, _ = self.model(ids)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(before[name], value)
                            for name, value in self.model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
