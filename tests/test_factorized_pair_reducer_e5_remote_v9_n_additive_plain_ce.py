import difflib
import hashlib
import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
V8_PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v8_n_additive/submission.py"
PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v9_n_additive_plain_ce/submission.py"


def load(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


v8 = load("factorized_pair_reducer_e5_remote_v8_n_additive_parity", V8_PATH)
candidate = load("factorized_pair_reducer_e5_remote_v9_n_additive_plain_ce", PATH)


def spec(length=13):
    return ModelSpec(17, length, 250_000)


def row(module, t, padding=0):
    return torch.tensor([[
        module.N, 10, 9, 10, module.X, 8, 9, 10, module.T,
        *(d + module.DIGIT for d in t), *([0] * padding),
    ]])


class FactorizedPairReducerPlainCETests(unittest.TestCase):
    def paired_models(self):
        torch.manual_seed(74)
        old = v8.build_model(spec(16))
        torch.manual_seed(74)
        new = candidate.build_model(spec(16))
        return old, new

    def test_exact_source_diff_is_only_gradient_scale(self):
        old = V8_PATH.read_text().splitlines()
        new = PATH.read_text().splitlines()
        changes = [line for line in difflib.ndiff(old, new) if line[:2] in ("- ", "+ ")]
        self.assertEqual(changes, [
            "-             scale = torch.where(steps.eq(1), logits.new_tensor(1.0), logits.new_tensor(0.0))",
            "+             scale = logits.new_ones(steps.shape)",
        ])

    def test_forward_parity_at_shared_state_for_all_horizons(self):
        old, new = self.paired_models()
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for key in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[key], new.state_dict()[key]), key)
        for training in (False, True):
            old.train(training); new.train(training)
            for step in (1, 2, 3):
                ids = row(candidate, (step,), padding=4)
                old_logits, old_aux = old(ids)
                new_logits, new_aux = new(ids)
                self.assertTrue(torch.equal(old_logits, new_logits))
                self.assertTrue(torch.equal(old_aux["ungated_logits"], new_aux["ungated_logits"]))

    def test_plain_ce_gradients_and_t1_gradient_parity(self):
        old, new = self.paired_models()
        old.train(); new.train()
        for step in (1, 2, 3):
            old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
            ids = row(candidate, (step,), padding=4)
            old(ids)[0][:, :, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
            new(ids)[0][:, :, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
            for (old_name, old_parameter), (new_name, new_parameter) in zip(
                    old.named_parameters(), new.named_parameters()):
                self.assertEqual(old_name, new_name)
                self.assertIsNotNone(new_parameter.grad)
                self.assertTrue(torch.isfinite(new_parameter.grad).all(), new_name)
                if step == 1:
                    self.assertTrue(torch.equal(old_parameter.grad, new_parameter.grad), new_name)
            if step > 1:
                self.assertGreater(sum(torch.count_nonzero(p.grad).item() for p in new.parameters()), 0)
                self.assertEqual(sum(torch.count_nonzero(p.grad).item() for p in old.parameters()), 0)
                plain_grads = {name: parameter.grad.clone()
                               for name, parameter in new.named_parameters()}
                new.zero_grad(set_to_none=True)
                ungated = new(ids)[1]["ungated_logits"]
                ungated[:, :, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
                for name, parameter in new.named_parameters():
                    self.assertTrue(torch.equal(plain_grads[name], parameter.grad), name)

    def test_state_calls_purity_optimizer_and_validation(self):
        torch.manual_seed(74)
        model = candidate.build_model(spec(16))
        self.assertEqual(count_model_state_elements(model), 68_874)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        model.eval()
        ids = row(candidate, (3,), padding=4)
        first, aux = model(ids)
        second, second_aux = model(ids)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        self.assertEqual(aux["reducer_calls"], 3)
        self.assertEqual(aux["scratch_resets"], 3)
        self.assertEqual(second_aux["parsed_steps"].item(), 3)
        self.assertTrue(aux["plain_endpoint_ce"])
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
