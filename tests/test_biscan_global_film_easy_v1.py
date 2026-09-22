import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions" / "biscan_global_film_easy_v1" / "submission.py"


def load():
    spec = importlib.util.spec_from_file_location("biscan_global_film_easy_v1", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(module, n=(3, 2), x=(6, 1), t=2, length=20):
    values = [module.N, *(module.DIGIT + d for d in n), module.X,
              *(module.DIGIT + d for d in x), module.T,
              *(module.DIGIT + int(d) for d in str(t))]
    return torch.tensor([values + [module.PAD] * (length - len(values))])


class BiScanGlobalFilmEasyTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.model = self.module.build_model(ModelSpec(17, 20, 20_000_000))

    def test_contract_state_and_single_mechanism(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertEqual(validate_submission_source(PATH.name, source, 256_000), "submission.py")
        self.assertEqual(count_model_state_elements(self.model), 75_658)
        self.assertEqual((self.module.SUBMISSION.batch_size,
                          self.module.SUBMISSION.eval_batch_size,
                          self.module.SUBMISSION.max_steps), (64, 128, None))
        self.assertEqual(set(dict(self.model.named_modules())), {
            "", "embedding", "scan", "scan.down_cell", "scan.up_cell",
            "value", "film", "norm", "readout",
        })

    def test_alignment_max_t_and_all_gradients(self):
        ids = torch.cat((row(self.module, n=(8,), x=(2,), t=1),
                         row(self.module, n=(9, 0, 4, 3), x=(1, 2, 3, 4), t=64)))
        logits, auxiliary = self.model(ids)
        self.assertEqual(logits.shape, (2, 20, 17))
        self.assertEqual(auxiliary["parsed_steps"].tolist(), [1, 64])
        self.assertEqual(auxiliary["executed_macrosteps"].item(), 64)
        self.assertEqual(auxiliary["width_mask"].sum(1).tolist(), [1, 4])
        self.assertEqual(auxiliary["context_digits"][1, :4].tolist(), [3, 4, 0, 9])
        target = torch.tensor([1, 2, 3, 4])
        loss = F.cross_entropy(logits[1, 9:13, 7:], target)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))

    def test_deterministic_pure_eval_and_optimizer(self):
        ids = row(self.module, n=(9, 0, 4, 3), x=(1, 2, 3, 4), t=3)
        self.model.eval()
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        with torch.no_grad():
            first, _ = self.model(ids)
            second, _ = self.model(ids)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(before[name], value)
                            for name, value in self.model.state_dict().items()))
        bundle = self.module.build_optimizer(self.model, OptimizerSpec(600, "cpu"))
        self.assertEqual([(group["lr"], group["weight_decay"]) for group in bundle.optimizer.param_groups],
                         [(4e-5, .02), (4e-5, 0.)])


if __name__ == "__main__":
    unittest.main()
