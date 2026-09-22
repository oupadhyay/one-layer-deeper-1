from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from benchmark.s0_depth import consecutive_certificate, evaluate_rung, verify_frozen_inputs


def _row(trajectory: list[int], permutation: list[int] | None = None) -> dict:
    permutation = permutation or list(range(16))
    return {"M": 16, "T": len(trajectory) - 1, "x0": trajectory[0],
            "y": trajectory[-1], "trajectory": trajectory, "perm": permutation}


class SyntheticStep(nn.Module):
    def __init__(self, transition: dict[int, int]) -> None:
        super().__init__()
        self.transition = transition

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        values = state[:, 0].long()
        return torch.tensor([self.transition[int(value)] for value in values], device=state.device).float()[:, None]


class SyntheticTransducer(nn.Module):
    def __init__(self, transition: dict[int, int]) -> None:
        super().__init__()
        self.step = SyntheticStep(transition)
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("calls_forbidden", torch.tensor(0))

    def encode_context(self, table: torch.Tensor) -> torch.Tensor:
        return table

    def encode_state(self, digits: torch.Tensor) -> torch.Tensor:
        return (digits[:, 0] + 10 * digits[:, 1]).float()[:, None]

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        values = state[:, 0].long()
        logits = torch.full((len(values), 2, 10), -100.0, device=state.device)
        logits[torch.arange(len(values)), 0, values.remainder(10)] = 100.0
        logits[torch.arange(len(values)), 1, values.div(10, rounding_mode="floor")] = 100.0
        return logits


class S0DepthTests(unittest.TestCase):
    def test_perfect_transducer_certifies_every_toy_rung(self) -> None:
        model = SyntheticTransducer({value: (value + 1) % 16 for value in range(16)})
        results = []
        for horizon in (1, 2, 4, 8, 16, 32, 64):
            trajectory = [step % 16 for step in range(horizon + 1)]
            results.append(evaluate_rung(model, [_row(trajectory)], horizon, torch.device("cpu")))
        self.assertEqual(consecutive_certificate(results), 64)

    def test_lucky_endpoint_does_not_repair_intermediate_error(self) -> None:
        # Oracle: 0 -> 1 -> 2. Model: 0 -> 3 -> 2.
        model = SyntheticTransducer({0: 3, 3: 2, 1: 2, 2: 2})
        result = evaluate_rung(model, [_row([0, 1, 2])], 2, torch.device("cpu"))
        self.assertEqual(result["endpoint_correct"], 1)
        self.assertEqual(result["trajectory_correct"], 0)
        self.assertEqual(result["first_error_step_histogram"]["1"], 1)

    def test_invalid_decode_is_counted_and_sticky(self) -> None:
        model = SyntheticTransducer({0: 99, 99: 1, 1: 1})
        result = evaluate_rung(model, [_row([0, 1, 1])], 2, torch.device("cpu"))
        self.assertEqual(result["invalid_decode_count"], 1)
        self.assertEqual(result["trajectory_correct"], 0)
        self.assertEqual(result["first_error_step_histogram"]["1"], 1)

    def test_certificate_is_consecutive_and_cannot_skip_failure(self) -> None:
        results = [{"T": horizon, "records": 1, "endpoint_correct": 1, "trajectory_correct": 1}
                   for horizon in (1, 2, 4, 8, 16, 32, 64)]
        results[2]["trajectory_correct"] = 0
        self.assertEqual(consecutive_certificate(results), 2)

    def test_teacher_forced_uses_true_prior_and_free_running_uses_prediction(self) -> None:
        model = SyntheticTransducer({0: 3, 3: 4, 1: 2, 2: 2})
        result = evaluate_rung(model, [_row([0, 1, 2])], 2, torch.device("cpu"))
        self.assertEqual(result["free_running_per_step"][1]["correct"], 0)
        self.assertEqual(result["teacher_forced_per_step"][1]["correct"], 1)
        self.assertEqual(result["free_running_per_step"][0]["correct"], 0)

    def test_hash_mismatch_aborts_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "source.py").write_text("original", encoding="utf-8")
            (root / "checkpoint.pt").write_bytes(b"not a checkpoint")
            (root / "data" / "dataset_config.json").write_text("{}", encoding="utf-8")
            (root / "data" / "artifact_manifest.json").write_text(
                json.dumps({"files": {"dataset_config.json": "wrong"}}), encoding="utf-8")
            provenance = {
                "checkpoint_sha256": "wrong", "dataset_config_sha256": "wrong",
                "dataset_manifest_sha256": "wrong", "source_sha256": {"source.py": "wrong"},
                "dataset_manifest": {"files": {"dataset_config.json": "wrong"}},
            }
            path = root / "provenance.json"
            path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "before inference"):
                verify_frozen_inputs(root, root / "data", root / "checkpoint.pt", path)

    def test_eval_is_deterministic_and_does_not_mutate_model(self) -> None:
        model = SyntheticTransducer({value: value for value in range(100)}).eval()
        before = copy.deepcopy(model.state_dict())
        row = _row([4, 4])
        first = evaluate_rung(model, [row], 1, torch.device("cpu"))
        second = evaluate_rung(model, [row], 1, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertEqual(id(model.step), id(model.step))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
