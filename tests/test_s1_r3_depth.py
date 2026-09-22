from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from benchmark.s1_c1 import S1GRUTransducer
from benchmark.s1_r3_depth import (
    JsonlParseMonitor, RUNGS, audit_depth_populations, consecutive_certificate, evaluate_rung,
    load_depth_rows, verify_hash,
)


def row(trajectory: list[int], addend: int = 1, carries: list[int] | None = None) -> dict:
    return {"T": len(trajectory) - 1, "a": addend, "x0": trajectory[0],
            "trajectory": trajectory, "carry_lengths": carries or [0] * (len(trajectory) - 1)}


class Step(nn.Module):
    def __init__(self, mapping: dict[int, int], nonfinite: bool = False) -> None:
        super().__init__()
        self.mapping, self.nonfinite = mapping, nonfinite
        self.first_hidden_inputs: list[torch.Tensor] = []

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        values = (state[:, :, 0] * state.new_tensor([1, 10, 100, 1000])).sum(1).long()
        outputs = torch.tensor([self.mapping[int(value)] for value in values], device=state.device)
        result = torch.zeros(len(values), 4, 1, device=state.device)
        for place in range(4):
            result[:, place, 0] = outputs.div(10 ** place, rounding_mode="floor").remainder(10)
        if self.nonfinite:
            result[0, 0, 0] = float("nan")
        return result


class Synthetic(nn.Module):
    def __init__(self, mapping: dict[int, int], nonfinite: bool = False) -> None:
        super().__init__()
        self.step = Step(mapping, nonfinite)
        self.context_calls = 0

    def encode_context(self, values: torch.Tensor) -> torch.Tensor:
        self.context_calls += 1
        return values.float()

    def encode_state(self, values: torch.Tensor) -> torch.Tensor:
        return values.float().unsqueeze(-1)

    def readout(self, states: torch.Tensor) -> torch.Tensor:
        values = states[:, :, 0].long().clamp(0, 9)
        logits = torch.full((len(states), 4, 10), -100.0)
        logits.scatter_(2, values.unsqueeze(-1), 100.0)
        if torch.isnan(states).any():
            logits[torch.isnan(states).any((1, 2))] = float("nan")
        return logits


def repeated_rows(item: dict) -> list[dict]:
    return [dict(item) for _ in range(512)]


class S1R3DepthTests(unittest.TestCase):
    def test_perfect_transducer_and_context_once(self) -> None:
        model = Synthetic({value: (value + 1) % 10000 for value in range(10000)})
        result = evaluate_rung(model, repeated_rows(row([0, 1, 2])), 2, torch.device("cpu"))
        self.assertEqual(result["endpoint_correct"], 512)
        self.assertEqual(result["trajectory_correct"], 512)
        self.assertEqual(result["context_encode_calls"], 1)
        self.assertEqual(model.context_calls, 1)

    def test_lucky_endpoint_cannot_repair_intermediate_error(self) -> None:
        model = Synthetic({0: 3, 3: 2, 1: 2, 2: 2})
        result = evaluate_rung(model, repeated_rows(row([0, 1, 2])), 2, torch.device("cpu"))
        self.assertEqual(result["endpoint_correct"], 512)
        self.assertEqual(result["trajectory_correct"], 0)
        self.assertEqual(result["first_error_histogram"]["1"], 512)

    def test_nonfinite_is_sticky(self) -> None:
        model = Synthetic({0: 1, 1: 2, 2: 2}, nonfinite=True)
        result = evaluate_rung(model, repeated_rows(row([0, 1, 2])), 2, torch.device("cpu"))
        self.assertGreater(result["invalid_nonfinite_count"], 0)
        self.assertLess(result["trajectory_correct"], 512)

    def test_no_skipped_rung_certificate(self) -> None:
        results = []
        for horizon in RUNGS:
            steps = [{"correct": 512}] * horizon
            results.append({"T": horizon, "records": 512, "endpoint_correct": 512,
                            "trajectory_correct": 512, "invalid_nonfinite_count": 0,
                            "teacher_forced_per_step": steps})
        results[1]["trajectory_correct"] = 511
        self.assertEqual(consecutive_certificate(results), 2)

    def test_teacher_and_free_use_different_priors_after_error(self) -> None:
        model = Synthetic({0: 3, 3: 4, 1: 2, 2: 2, 4: 4})
        result = evaluate_rung(model, repeated_rows(row([0, 1, 2])), 2, torch.device("cpu"))
        self.assertEqual(result["free_running_per_step"][1]["correct"], 0)
        self.assertEqual(result["teacher_forced_per_step"][1]["correct"], 512)

    def test_real_step_hard_reset_starts_from_zero_h0_and_has_no_t_argument(self) -> None:
        model = S1GRUTransducer().eval()
        seen: list[torch.Tensor] = []
        original = model.cell.forward

        def record(inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            seen.append(hidden.detach().clone())
            return original(inputs, hidden)

        model.cell.forward = record  # type: ignore[method-assign]
        state = model.encode_state(torch.tensor([[9, 9, 9, 9]]))
        context = model.encode_context(torch.tensor([[1, 0, 0, 0]]))
        model.step(state, context)
        model.step(state, context)
        self.assertTrue(torch.equal(seen[0], torch.zeros_like(seen[0])))
        self.assertTrue(torch.equal(seen[4], torch.zeros_like(seen[4])))
        self.assertEqual(tuple(__import__("inspect").signature(model.step.forward).parameters), ("state", "context"))

    def test_nested_edge_and_carry_audit(self) -> None:
        roots = []
        # Synthetic 512 unique length-64 segments across distinct addends.
        for index in range(512):
            addend = index + 1
            trajectory = list(range(65))
            roots.append(row(trajectory, addend, [0] * 64))
        populations = {h: [row(item["trajectory"][:h + 1], item["a"], [0] * h) for item in roots]
                       for h in RUNGS}
        # This fixture intentionally has a different carry total, proving strict rejection.
        with self.assertRaises(AssertionError):
            audit_depth_populations(populations)

    def test_normative_nested_edge_carry_audit_and_parsed_paths(self) -> None:
        root = Path(__file__).parents[1] / "data/generated/s1_addcarry_v1"
        with JsonlParseMonitor() as monitor:
            populations = {horizon: load_depth_rows(root, horizon) for horizon in RUNGS}
        self.assertEqual(monitor.opened, [f"depth_t_{horizon}.jsonl" for horizon in RUNGS])
        report = audit_depth_populations(populations)
        self.assertEqual(report["unique_directed_edges"], 32768)
        self.assertEqual(report["carry_totals"], {0: 18053, 1: 13250, 2: 1341, 3: 117, 4: 7})
        self.assertEqual(report["within_trajectory_repeats"], 0)

    def test_hash_mismatch_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "pre-inference"):
                verify_hash(path, "0" * 64, "fixture")

    def test_deterministic_nonmutating_evaluation(self) -> None:
        model = Synthetic({value: value for value in range(10000)}).eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rows = repeated_rows(row([4, 4]))
        first = evaluate_rung(model, rows, 1, torch.device("cpu"))
        second = evaluate_rung(model, rows, 1, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
