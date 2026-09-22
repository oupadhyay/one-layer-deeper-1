import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/universal_recurrent_transformer_easy_v7_discrete_bottleneck/submission.py"
LOADER = importlib.util.spec_from_file_location("urt_v7_discrete", PATH)
candidate = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(candidate)


def spec(length=13):
    return ModelSpec(17, length, 500_000_000)


class DiscreteBottleneckTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_source_state_optimizer_and_contract(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        for forbidden in ("from submissions", "training_loop", ".backward(",
                          "generated_data", "parse_", "arithmetic"):
            self.assertNotIn(forbidden, source)
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 515_584)
        bundle = candidate.build_optimizer(model, OptimizerSpec(60.0, "cpu"))
        self.assertEqual((bundle.optimizer.defaults["lr"], bundle.optimizer.defaults["betas"],
                          bundle.optimizer.defaults["eps"]), (8e-4, (.9, .95), 1e-8))
        self.assertEqual([group["weight_decay"] for group in bundle.optimizer.param_groups],
                         [.05, 0.0])
        grouped = [parameter for group in bundle.optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual({id(parameter) for parameter in grouped}, {id(parameter) for parameter in model.parameters()})
        self.assertIs(candidate.SUBMISSION.training_loss, candidate.training_loss)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 1024, None))

    def test_hard_st_path_and_collapsed_usage_penalty(self):
        model = candidate.build_model(spec())
        hidden = torch.zeros(2, 3, candidate.D)
        model.train()
        quantized_train, counts, hard_counts = model.quantize(hidden)
        model.eval()
        quantized_eval, _, _ = model.quantize(hidden)
        expected = model.codebook[0].expand_as(quantized_train)
        torch.testing.assert_close(quantized_train, expected)
        torch.testing.assert_close(quantized_eval, expected)
        self.assertEqual(counts[0].item(), 6)
        self.assertEqual(hard_counts[0].item(), 6)
        self.assertEqual(torch.count_nonzero(counts[1:]).item(), 0)
        mean_usage = counts / counts.sum()
        penalty = .01 * candidate.K * ((mean_usage - 1 / candidate.K) ** 2).sum()
        self.assertAlmostEqual(penalty.item(), .63, places=6)
        endpoint = torch.randn(5, 17, requires_grad=True)
        labels = torch.arange(5)
        loss = candidate.training_loss(endpoint, labels, {"usage_loss": penalty})
        torch.testing.assert_close(loss, F.cross_entropy(endpoint, labels) + penalty)

    def test_recurrence_gradients_padding_and_eval_purity(self):
        model = candidate.build_model(spec()).train()
        calls = []
        handle = model.recurrent_block.register_forward_hook(lambda *args: calls.append(1))
        ids = torch.tensor([[1, 2, 3, 4]])
        logits, auxiliary = model(ids)
        handle.remove()
        self.assertEqual((logits.shape, len(calls), auxiliary["iterations"]), ((1, 4, 17), 16, 16))
        self.assertGreaterEqual(auxiliary["occupancy"].item(), 1)
        self.assertLessEqual(auxiliary["occupancy"].item(), candidate.K)
        candidate.training_loss(
            logits.reshape(-1, 17), torch.tensor([2, 3, 4, 5]), auxiliary
        ).backward()
        for name in ("selector.weight", "codebook", "recurrent_block.self_attention.qkv.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

        model.eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        mask = torch.tensor([[1, 1, 1, 0]])
        with torch.no_grad():
            first, _ = model(torch.tensor([[1, 2, 3, 4]]), mask)
            second, _ = model(torch.tensor([[1, 2, 3, 9]]), mask)
            again, _ = model(torch.tensor([[1, 2, 3, 4]]), mask)
        torch.testing.assert_close(first, second, rtol=0, atol=2e-6)
        torch.testing.assert_close(first, again, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
