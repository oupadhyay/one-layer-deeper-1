import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SAMCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v4 = load("universal_recurrent_transformer_easy_v4_d128")
        cls.sam = load("universal_recurrent_transformer_easy_v6_sam")

    def test_source_architecture_output_and_gradients(self):
        path = ROOT / "submissions/universal_recurrent_transformer_easy_v6_sam/submission.py"
        source = path.read_text(encoding="utf-8")
        validate_submission_source(path.name, source, 256 * 1024)
        for forbidden in ("from submissions", "training_loss", ".backward(",
                          "generated_data", "parse_", "optimizer.step("):
            self.assertNotIn(forbidden, source)
        spec = ModelSpec(17, 8, 500_000_000)
        old, new = self.v4.build_model(spec), self.sam.build_model(spec)
        self.assertEqual(count_model_state_elements(new), 499_072)
        self.assertEqual({k: v.shape for k, v in old.state_dict().items()},
                         {k: v.shape for k, v in new.state_dict().items()})
        self.assertIsNone(self.sam.SUBMISSION.training_loss)
        calls = []
        hook = new.recurrent_block.register_forward_hook(lambda *args: calls.append(1))
        logits, _ = new(torch.tensor([[1, 2, 3, 4]]))
        hook.remove()
        self.assertEqual((logits.shape, len(calls)), ((1, 4, 17), 16))
        F.cross_entropy(logits.flatten(0, 1), torch.tensor([2, 3, 4, 5])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in new.parameters()))

    def test_direct_sam_state_machine(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = self.sam.SAMAdamW([parameter], lr=0.1, betas=(0.0, 0.0),
                                      eps=1e-8, weight_decay=0.0)
        callback = self.sam.SAMBetweenBackwardPasses(optimizer)
        parameter.grad = torch.tensor([3.0, 4.0])
        original = parameter.detach().clone()
        callback(SimpleNamespace(pass_index=1))
        torch.testing.assert_close(parameter, original)
        callback(SimpleNamespace(pass_index=0))
        e_w = optimizer.state[parameter]["e_w"]
        self.assertEqual(e_w.device, parameter.device)
        torch.testing.assert_close(e_w.norm(), torch.tensor(0.05))
        torch.testing.assert_close(parameter, original + e_w)
        parameter.grad = torch.tensor([2.0, -1.0])
        control_parameter = torch.nn.Parameter(original.clone())
        control_parameter.grad = parameter.grad.clone()
        control = torch.optim.AdamW([control_parameter], lr=0.1, betas=(0.0, 0.0),
                                    eps=1e-8, weight_decay=0.0)
        control.step()
        optimizer.step()
        torch.testing.assert_close(parameter, control_parameter)
        self.assertNotIn("e_w", optimizer.state[parameter])

    def test_optimizer_settings_fallback_and_current_api_wiring(self):
        model = self.sam.build_model(ModelSpec(17, 8, 500_000_000))
        fallback = self.sam.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
        self.assertIs(type(fallback.optimizer), torch.optim.AdamW)
        self.assertEqual([g["weight_decay"] for g in fallback.optimizer.param_groups],
                         [0.05, 0.0])
        self.assertEqual((fallback.optimizer.defaults["lr"],
                          fallback.optimizer.defaults["betas"],
                          fallback.optimizer.defaults["eps"]),
                         (8e-4, (0.9, 0.95), 1e-8))
        self.assertAlmostEqual(fallback.scheduler.lr_lambdas[0](31), 1.0)

        @dataclass(frozen=True)
        class CurrentBundle:
            optimizer: object
            scheduler: object = None
            backward_passes_per_step: int = 1
            between_backward_passes: object = None
            should_reuse_batch: object = None

        stale = self.sam.OptimizerBundle
        try:
            self.sam.OptimizerBundle = CurrentBundle
            current = self.sam.build_optimizer(model, OptimizerSpec(10.0, "cpu"))
        finally:
            self.sam.OptimizerBundle = stale
        self.assertIsInstance(current.optimizer, self.sam.SAMAdamW)
        self.assertEqual(current.backward_passes_per_step, 2)
        self.assertIsNone(current.should_reuse_batch)
        self.assertIs(current.between_backward_passes.optimizer, current.optimizer)


if __name__ == "__main__":
    unittest.main()
