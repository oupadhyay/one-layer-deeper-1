from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from benchmark.s0_research import C1PointerControl, TrackAEvaluator, run_c1_g1
from data.s0_perm_v1 import EXPECTED_COUNTS, audit_s0_perm_v1, generate_s0_perm_v1


def _digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class S0PermV1Tests(unittest.TestCase):
    def test_normative_generation_audits_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            config = generate_s0_perm_v1(first, seed=12345)
            generate_s0_perm_v1(second, seed=12345)
            self.assertEqual(config["split_counts"], EXPECTED_COUNTS)
            self.assertEqual(_digest(Path(first)), _digest(Path(second)))
            report = audit_s0_perm_v1(first)
            self.assertTrue(report["passed"])
            self.assertEqual(report["oracle_accuracy"], 1.0)
            self.assertEqual(report["c6_expected_exact"], {"M16": 1 / 16, "M24": 1 / 24})
            self.assertEqual(report["entropy_bits"]["test_context"]["y"], 4.0)
            self.assertEqual(report["depth_training_edge_overlap_fraction"], 0.0)
            self.assertEqual(report["artifact_hashes_verified"], 11)

    def test_c1_shape_and_gradient(self) -> None:
        model = C1PointerControl(width=16)
        logits = model(torch.tensor([[3, 0], [1, 1]]), torch.tensor([list(range(16))] * 2))
        self.assertEqual(logits.shape, (2, 2, 10))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_track_a_shared_step_t_invisibility_and_scratch_reset(self) -> None:
        class ScratchModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.step = ScratchStep()
                self.context_calls = 0
                self.encoded: list[torch.Tensor] = []

            def encode_context(self, table: torch.Tensor) -> torch.Tensor:
                self.context_calls += 1
                return table

            def encode_state(self, digits: torch.Tensor) -> torch.Tensor:
                self.encoded.append(digits.clone())
                return torch.cat((digits.float(), torch.zeros(len(digits), 1)), 1)

            def readout(self, state: torch.Tensor) -> torch.Tensor:
                logits = torch.full((len(state), 2, 10), -100.0)
                logits[:, 0, 4] = 100.0
                logits[:, 1, 0] = 100.0
                return logits

        class ScratchStep(nn.Module):
            def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
                result = state.clone()
                result[:, 2] += 1  # forbidden scratch would accumulate without reset
                return result

        model = ScratchModel()
        evaluator = TrackAEvaluator(model)
        identity = id(model.step)
        prediction, scores = evaluator.run(torch.tensor([[3, 0]]), torch.tensor([list(range(16))]), 3)
        self.assertEqual(id(model.step), identity)
        self.assertEqual(model.context_calls, 1)
        self.assertEqual(len(scores), 3)
        self.assertEqual(prediction.tolist(), [[4, 0]])
        self.assertEqual([value.tolist() for value in model.encoded],
                         [[[3, 0]], [[4, 0]], [[4, 0]], [[4, 0]]])

        class LeakyStep(nn.Module):
            def forward(self, state: torch.Tensor, context: torch.Tensor, step_index: int) -> torch.Tensor:
                return state
        model.step = LeakyStep()
        with self.assertRaises(TypeError):
            TrackAEvaluator(model)

    def test_c2_horizon_one_conformance_and_step_purity(self) -> None:
        torch.manual_seed(75)
        model = C1PointerControl(width=16).eval()
        state = torch.tensor([[3, 0], [1, 1]])
        table = torch.tensor([list(range(15, -1, -1)), list(range(16))])
        parameters_before = {name: value.detach().clone() for name, value in model.named_parameters()}
        buffers_before = {name: value.detach().clone() for name, value in model.named_buffers()}

        context = model.encode_context(table)
        direct = model.readout(model.step(model.encode_state(state), context))
        repeated = model.readout(model.step(model.encode_state(state), model.encode_context(table)))
        prediction, scores = TrackAEvaluator(model).run(state, table, horizon=1)

        self.assertTrue(torch.equal(direct, repeated))
        self.assertTrue(torch.equal(direct, scores[0]))
        self.assertTrue(torch.equal(prediction, direct.argmax(-1)))
        self.assertEqual(tuple(inspect.signature(model.step.forward).parameters), ("state", "context"))
        self.assertEqual(sum(1 for module in model.modules() if module is model.step), 1)
        self.assertTrue(all(torch.equal(value, parameters_before[name]) for name, value in model.named_parameters()))
        self.assertTrue(all(torch.equal(value, buffers_before[name]) for name, value in model.named_buffers()))

    def test_deterministic_short_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Tiny complete tables are enough to exercise loading/training/reporting.
            for split, count in (("train", 32), ("test_state", 16), ("test_context", 16)):
                with (root / f"{split}.jsonl").open("w", encoding="utf-8") as output:
                    for index in range(count):
                        perm = list(range(16))
                        x = index % 16
                        output.write(__import__("json").dumps({"x0": x, "y": x, "perm": perm}) + "\n")
            first = run_c1_g1(root, root / "first.jsonl", smoke=True)
            second = run_c1_g1(root, root / "second.jsonl", smoke=True)
            for result in (first, second):
                result.pop("elapsed_seconds")
            self.assertEqual(first, second)
            self.assertEqual(first["config"]["seed"], 74)
            self.assertEqual(len((root / "first.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
