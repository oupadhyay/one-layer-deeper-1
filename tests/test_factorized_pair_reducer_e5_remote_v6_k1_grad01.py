import difflib
import hashlib
import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v2_k1/submission.py"
V5_PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v5_k1_plain_ce/submission.py"
PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v6_k1_grad01/submission.py"


def load(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


v2 = load("factorized_pair_reducer_v2_t1only", V2_PATH)
v5 = load("factorized_pair_reducer_v5_plain_ce", V5_PATH)
candidate = load("factorized_pair_reducer_v6_grad01", PATH)


def spec(length=16):
    return ModelSpec(17, length, 250_000)


def row(module, step, padding=4):
    return torch.tensor([[
        module.N, 10, 9, 10, module.X, 8, 9, 10, module.T,
        *(int(d) + module.DIGIT for d in str(step)), *([0] * padding),
    ]])


class FactorizedPairReducerGrad01Tests(unittest.TestCase):
    def models(self, *modules):
        result = []
        for module in modules:
            torch.manual_seed(74)
            result.append(module.build_model(spec()))
        return result

    def test_exact_source_diff_is_only_non_t1_gradient_scale(self):
        changes = [line for line in difflib.ndiff(
            V2_PATH.read_text().splitlines(), PATH.read_text().splitlines()
        ) if line[:2] in ("- ", "+ ")]
        self.assertEqual(changes, [
            "-             scale = torch.where(steps.eq(1), logits.new_tensor(1.0), logits.new_tensor(0.0))",
            "+             scale = torch.where(steps.eq(1), logits.new_tensor(1.0), logits.new_tensor(0.1))",
        ])

    def test_forward_parity_at_identical_state(self):
        old, new = self.models(v2, candidate)
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for key in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[key], new.state_dict()[key]), key)
        for training in (False, True):
            old.train(training); new.train(training)
            for step in (1, 2, 3, 64):
                old_logits, old_aux = old(row(candidate, step, padding=3))
                new_logits, new_aux = new(row(candidate, step, padding=3))
                self.assertTrue(torch.equal(old_logits, new_logits))
                self.assertTrue(torch.equal(old_aux["ungated_logits"], new_aux["ungated_logits"]))

    def test_t1_identical_and_t2_t3_are_point_one_plain_ce_gradients(self):
        old, plain, new = self.models(v2, v5, candidate)
        for model in (old, plain, new):
            model.train()
        for step in (1, 2, 3):
            for model in (old, plain, new):
                model.zero_grad(set_to_none=True)
                logits = model(row(candidate, step))[0]
                logits[:, :, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
            for ((name_old, parameter_old), (name_plain, parameter_plain),
                 (name_new, parameter_new)) in zip(
                    old.named_parameters(), plain.named_parameters(), new.named_parameters()):
                self.assertEqual((name_old, name_plain), (name_new, name_new))
                self.assertIsNotNone(parameter_new.grad, name_new)
                self.assertTrue(torch.isfinite(parameter_new.grad).all(), name_new)
                if step == 1:
                    self.assertTrue(torch.equal(parameter_old.grad, parameter_new.grad), name_new)
                    self.assertTrue(torch.equal(parameter_plain.grad, parameter_new.grad), name_new)
                else:
                    expected = parameter_plain.grad * 0.1
                    finite = torch.isfinite(expected) & torch.isfinite(parameter_new.grad)
                    self.assertTrue(torch.allclose(parameter_new.grad[finite], expected[finite],
                                                   rtol=2e-5, atol=2e-7), name_new)

    def test_state_purity_optimizer_max_t_and_validation(self):
        model, = self.models(candidate)
        self.assertEqual(count_model_state_elements(model), 64_714)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        model.eval()
        ids = row(candidate, 64, padding=3)
        first, aux = model(ids)
        second, second_aux = model(ids)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        self.assertEqual((aux["reducer_calls"], aux["scratch_resets"]), (64, 64))
        self.assertEqual(second_aux["parsed_steps"].item(), candidate.MAX_STEPS)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, model.max_seq_len + 1), dtype=torch.long))
        bundle = candidate.build_optimizer(model, OptimizerSpec(1600, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 1e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual([group["weight_decay"] for group in bundle.optimizer.param_groups], [.1])
        self.assertIsNone(bundle.scheduler)
        self.assertEqual((candidate.D, candidate.MICROPHASES, candidate.MAX_STEPS), (64, 1, 64))
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))
        source = PATH.read_text()
        validate_submission_source(PATH.name, source, 256 * 1024)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         hashlib.sha256(PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
