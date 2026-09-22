import hashlib
import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions/biscan_e1_remote_v2_batch64/submission.py"
PATH = ROOT / "submissions/biscan_m3_remote_v3_compositional/submission.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load("biscan_e1_v2_for_m3_test", BASE)
candidate = load("biscan_m3_remote_v3_compositional", PATH)


def model_spec(length=20):
    return ModelSpec(17, length, 20_000_000)


def row(n=(3, 2), x=(6, 1), t=2, length=20):
    values = [candidate.N, *(candidate.DIGIT + d for d in n), candidate.X,
              *(candidate.DIGIT + d for d in x), candidate.T,
              *(candidate.DIGIT + int(d) for d in str(t))]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class BiScanM3RemoteV3CompositionalTests(unittest.TestCase):
    def test_exact_one_line_diff(self):
        source = BASE.read_text(encoding="utf-8")
        old = "            scale = torch.where(steps == 1, logits.new_tensor(1.), logits.new_tensor(.01))"
        new = "            scale = torch.ones_like(steps, dtype=logits.dtype)"
        self.assertEqual(source.count(old), 1)
        self.assertEqual(PATH.read_text(encoding="utf-8"), source.replace(old, new))

    def test_forward_parity_and_soft_train_hard_eval(self):
        torch.manual_seed(80)
        previous = base.Model(model_spec())
        actual = candidate.Model(model_spec())
        actual.load_state_dict(previous.state_dict())
        ids = torch.cat((row(n=(4,), x=(7,), t=1), row(n=(3, 2, 1), x=(6, 5, 4), t=2)))
        for training in (True, False):
            previous.train(training)
            actual.train(training)
            expected, old_aux = previous(ids)
            result, aux = actual(ids)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            for key in ("parsed_steps", "active_updates", "digit_probabilities",
                        "initial_state_digits", "context_digits", "width_mask"):
                torch.testing.assert_close(aux[key], old_aux[key], rtol=0, atol=0)
            if training:
                self.assertTrue(((aux["digit_probabilities"] > 0) &
                                 (aux["digit_probabilities"] < 1)).any())
            else:
                self.assertTrue(((aux["digit_probabilities"] == 0) |
                                 (aux["digit_probabilities"] == 1)).all())

    def test_t2_gradient_is_exactly_100x_previous_gate(self):
        torch.manual_seed(81)
        previous = base.Model(model_spec()).train()
        actual = candidate.Model(model_spec()).train()
        actual.load_state_dict(previous.state_dict())
        ids = row(t=2)
        target = torch.tensor([1, 2])
        old_logits, old_aux = previous(ids)
        new_logits, new_aux = actual(ids)
        old_grad = torch.autograd.grad(F.cross_entropy(old_logits[0, -2:, 7:], target),
                                       old_aux["ungated_logits"])[0]
        new_grad = torch.autograd.grad(F.cross_entropy(new_logits[0, -2:, 7:], target),
                                       new_aux["ungated_logits"])[0]
        torch.testing.assert_close(new_grad, old_grad * 100, rtol=2e-6, atol=2e-8)

    def test_mixed_width_parser_max_t_and_purity(self):
        model = candidate.Model(model_spec()).eval()
        ids = torch.cat((row(n=(8,), x=(2,), t=2),
                         row(n=(9, 0, 4, 3), x=(1, 2, 3, 4), t=99)))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            logits, aux = model(ids)
        self.assertEqual(aux["parsed_steps"].tolist(), [2, 99])
        self.assertEqual(aux["active_updates"].tolist(), [2, 64])
        self.assertEqual(aux["executed_macrosteps"].item(), 64)
        self.assertEqual(aux["width_mask"].sum(1).tolist(), [1, 4])
        self.assertEqual(aux["context_digits"][1, :4].tolist(), [3, 4, 0, 9])
        self.assertTrue(torch.isfinite(logits).all())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_state_optimizer_submission_and_validation(self):
        model = candidate.build_model(model_spec())
        self.assertEqual(count_model_state_elements(model), 63_178)
        self.assertEqual(set(dict(model.named_modules())),
                         {"", "embedding", "scan", "scan.down_cell", "scan.up_cell", "readout"})
        bundle = candidate.build_optimizer(model, OptimizerSpec(600, "cpu"))
        self.assertEqual([(g["lr"], g["betas"], g["weight_decay"], g["capturable"])
                          for g in bundle.optimizer.param_groups],
                         [(4e-5, (.9, .98), .02, False), (4e-5, (.9, .98), 0., False)])
        self.assertEqual(bundle.scheduler.lr_lambdas[0](0), .05)
        self.assertEqual(bundle.scheduler.lr_lambdas[0](19), 1.)
        self.assertEqual(candidate.MAX_STEPS, 64)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 128, None))
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)

    def test_source_hash_is_reportable(self):
        self.assertEqual(len(hashlib.sha256(PATH.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
