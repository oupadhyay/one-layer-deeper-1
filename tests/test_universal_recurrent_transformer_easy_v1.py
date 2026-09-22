import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/universal_recurrent_transformer_easy_v1/submission.py"


def load():
    module_spec = importlib.util.spec_from_file_location("universal_recurrent_transformer_easy_v1", PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


candidate = load()


def spec(length=13):
    return ModelSpec(vocab_size=17, max_seq_len=length,
                     maximum_model_state_elements=500_000_000)


class UniversalRecurrentTransformerEasyV1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_source_is_self_contained_compilable_and_generic(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        for forbidden in (
            "N_TOKEN", "X_TOKEN", "T_TOKEN", "DIGIT", "parse_", "field_role",
            "number", "arithmetic", "argmax", "hard_feedback", "output_routing",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("from submissions", source)

    def test_exact_stable_state_count_and_sequence_cap(self):
        first = candidate.build_model(spec(13))
        second = candidate.build_model(spec(64))
        self.assertEqual(count_model_state_elements(first), 1_981_184)
        self.assertEqual(count_model_state_elements(second), 1_981_184)
        self.assertLess(count_model_state_elements(first), 500_000_000)
        with self.assertRaises(ValueError):
            candidate.build_model(spec(65))
        ids = torch.randint(0, 17, (1, 64))
        with torch.no_grad():
            logits, info = second(ids)
        self.assertEqual(logits.shape, (1, 64, 17))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(info, {"iterations": 16, "state_length": 96})

    def test_optimizer_scheduler_and_contract(self):
        model = candidate.build_model(spec())
        bundle = candidate.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
        optimizer = bundle.optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 8e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [.05, 0.0])
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
        decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
        for name, parameter in model.named_parameters():
            self.assertEqual(id(parameter) in decay_ids,
                             parameter.ndim == 2 and name != "token_embedding.weight", name)
        self.assertIsInstance(bundle.scheduler, torch.optim.lr_scheduler.LambdaLR)
        self.assertAlmostEqual(bundle.scheduler.lr_lambdas[0](0), 1 / 32)
        self.assertEqual(bundle.scheduler.lr_lambdas[0](31), 1.0)
        cuda = candidate.build_optimizer(model, OptimizerSpec(10.0, "cuda"))
        self.assertTrue(all(g["capturable"] for g in cuda.optimizer.param_groups))
        submission = candidate.SUBMISSION
        self.assertEqual((submission.batch_size, submission.eval_batch_size, submission.max_steps),
                         (512, 1024, None))
        self.assertIsNone(submission.training_loss)

    def test_output_shape_token_sensitivity_padding_and_batch_purity(self):
        model = candidate.build_model(spec()).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 9, 5]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            logits, _ = model(ids, mask)
            changed, _ = model(ids.flip(0), mask)
        self.assertEqual(logits.shape, (2, 5, 17))
        self.assertFalse(torch.equal(logits[0], logits[1]))
        torch.testing.assert_close(logits[0], changed[1], rtol=0, atol=2e-5)

        padded_a = torch.tensor([[1, 2, 3, 11, 12]])
        padded_b = torch.tensor([[1, 2, 3, 7, 8]])
        padding_mask = torch.tensor([[1, 1, 1, 0, 0]])
        with torch.no_grad():
            a, _ = model(padded_a, padding_mask)
            b, _ = model(padded_b, padding_mask)
        torch.testing.assert_close(a, b, rtol=0, atol=2e-6)

    def test_shared_block_calls_gradients_and_mode_purity(self):
        model = candidate.build_model(spec())
        calls = []
        handle = model.recurrent_block.register_forward_hook(lambda *args: calls.append(1))
        ids = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        logits, _ = model(ids)
        handle.remove()
        self.assertEqual(len(calls), 16)
        self.assertEqual(sum(isinstance(m, candidate.RecurrentBlock) for m in model.modules()), 1)
        F.cross_entropy(logits.reshape(-1, 17), targets.reshape(-1)).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

        before = {name: value.clone() for name, value in model.state_dict().items()}
        model.train()
        train_logits, train_info = model(ids)
        model.eval()
        with torch.no_grad():
            eval_logits, eval_info = model(ids)
            again, _ = model(ids)
        torch.testing.assert_close(train_logits, eval_logits, rtol=0, atol=0)
        torch.testing.assert_close(eval_logits, again, rtol=0, atol=0)
        self.assertEqual(train_info, eval_info)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
