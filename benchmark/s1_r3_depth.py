"""Frozen seed-77 strict Track-A depth evaluation for S1 seen addends."""

from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from benchmark.s1_c1 import S1GRUTransducer


RUNGS = (2, 4, 8, 16, 32, 64)
EXPECTED = {
    "checkpoint": "2cc603963473055d281fb2f86263440baf96c0dfc8bb7f2447b026ebb874b32d",
    "architecture": "8b5f80b838c6346a69a0e2c14160c964635ad64e5ffbe17ad24b331e4ff03c6b",
    "r2_runner": "77350db06cda6142f72bd82bbc794e0d815d0b90d5b8c49a72afe17b09c373a5",
    "r2_provenance": "0c9838f4e31fb712b8289078f1cb4a957d40361587416d0ca8e2841015a99a95",
    "r2_metrics": "4bb8d04e902f53328407cfa413b5e50332bbd05cb325898ba67cb3c01905bfa2",
    "generator": "d3a3e261a04595b8eb6a352fe7d523f935bbb70e6365c40ecdfb180718fa42b5",
    "config": "b9e2696527d6577a016532a9f83c6cfbc57941b9a68a936dfe6868f588045abd",
    "manifest": "93d5f0bc6786f7080bca3a9df2dff72e12740125a7082ef39c3753d7789df10f",
    "g0_metrics": "13c7ae2c59fc9b3a30f5a2445585cb67a6ef3fc59ce42ddd4dbd73e1126143f5",
}
EXPECTED_CONFIG = {
    "seed": 77, "embedding_dim": 32, "hidden_dim": 64, "batch_size": 256,
    "epochs": 60, "learning_rate": 1e-3, "weight_decay": 0.0,
    "gradient_clip": 1.0, "eval_batch_size": 2048, "curve_every_epochs": 5,
    "max_training_seconds": 1800.0,
}
EXPECTED_CARRIES = {0: 18053, 1: 13250, 2: 1341, 3: 117, 4: 7}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hash(path: str | Path, expected: str, name: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"pre-inference hash mismatch for {name}: {actual}")
    return actual


class JsonlParseMonitor(AbstractContextManager["JsonlParseMonitor"]):
    def __init__(self) -> None:
        self.opened: list[str] = []
        self._patcher: Any = None

    def __enter__(self) -> "JsonlParseMonitor":
        original = Path.open
        monitor = self

        def monitored(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path.suffix == ".jsonl":
                monitor.opened.append(path.name)
            return original(path, *args, **kwargs)

        self._patcher = patch.object(Path, "open", monitored)
        self._patcher.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._patcher.stop()


def verify_frozen_inputs(root: str | Path = ".", data_root: str | Path = "data/generated/s1_addcarry_v1"
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every frozen dependency before any depth record is parsed."""
    root, data_root = Path(root), Path(data_root)
    paths = {
        "checkpoint": root / "artifacts/s1_r2_gru_seed77.pt",
        "architecture": root / "benchmark/s1_c1.py",
        "r2_runner": root / "benchmark/s1_r2.py",
        "r2_provenance": root / "artifacts/s1_r2_gru_seed77_provenance.json",
        "r2_metrics": root / "metrics_s1_r2_gru_seed77.jsonl",
        "config": data_root / "dataset_config.json",
        "manifest": data_root / "artifact_manifest.json",
        "g0_metrics": root / "metrics_s1_addcarry_v1_g0.json",
    }
    verified = {name: verify_hash(path, EXPECTED[name], name) for name, path in paths.items()}
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if config["generator_sha256"] != EXPECTED["generator"]:
        raise RuntimeError("pre-inference generator hash mismatch in config")
    verified["generator"] = config["generator_sha256"]
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        verified[f"dataset:{filename}"] = verify_hash(data_root / filename, expected, filename)
    g0 = json.loads(paths["g0_metrics"].read_text(encoding="utf-8"))
    if (g0["train_test_edge_overlap"] != 0 or g0["depth_one_step_edge_overlap"] != 0
            or g0["unique_depth_edges"] != 32768):
        raise RuntimeError("hash-pinned G0 quarantine evidence is not green")
    r2_final = json.loads(paths["r2_metrics"].read_text(encoding="utf-8").splitlines()[-1])
    if not r2_final["gates"]["r2_pass"] or r2_final["evaluations"]["test_state"]["overall"]["exact"] != 1.0:
        raise RuntimeError("frozen R2 T1 base certificate is not green")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if (checkpoint["config"] != EXPECTED_CONFIG or checkpoint["parameter_count"] != 25930
            or checkpoint["final_epoch"] != 60 or checkpoint["optimizer_steps"] != 4740):
        raise RuntimeError("frozen checkpoint metadata mismatch")
    return checkpoint, {"verified_hashes": verified, "r2_t1_empirical_error": 0.0,
                        "g0_quarantine_reverified_from_hash_pinned_report": True}


def digits(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(values.div(10 ** place, rounding_mode="floor").remainder(10)
                             for place in range(4)), dim=1)


def decode(digit_values: torch.Tensor) -> torch.Tensor:
    weights = digit_values.new_tensor([1, 10, 100, 1000])
    return (digit_values * weights).sum(1)


def load_depth_rows(data_root: str | Path, horizon: int) -> list[dict[str, Any]]:
    if horizon not in RUNGS:
        raise ValueError("only predeclared S1 depth rungs may be parsed")
    with (Path(data_root) / f"depth_t_{horizon}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def audit_depth_populations(populations: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    if any(len(populations[horizon]) != 512 for horizon in RUNGS):
        raise AssertionError("each depth rung must contain 512 rows")
    full = {(row["a"], row["x0"]): row for row in populations[64]}
    if len(full) != 512:
        raise AssertionError("T64 roots are not unique")
    for horizon in RUNGS[:-1]:
        current = {(row["a"], row["x0"]): row for row in populations[horizon]}
        if set(current) != set(full):
            raise AssertionError("nested rung roots differ from T64")
        for key, row in current.items():
            if row["trajectory"] != full[key]["trajectory"][:horizon + 1]:
                raise AssertionError("shorter trajectory is not an exact T64 prefix")
    edges = {(row["a"], state) for row in populations[64] for state in row["trajectory"][:-1]}
    carries = Counter(carry for row in populations[64] for carry in row["carry_lengths"])
    repeats = sum(len(set(row["trajectory"])) != 65 for row in populations[64])
    if len(edges) != 32768 or dict(carries) != EXPECTED_CARRIES or repeats:
        raise AssertionError("T64 unique-edge/carry/cycle audit failed")
    return {"unique_directed_edges": len(edges), "carry_totals": dict(carries),
            "within_trajectory_repeats": repeats, "nested_prefixes_exact": True,
            "one_step_overlap": 0, "one_step_overlap_evidence": "hash-pinned frozen G0 audit"}


def _ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> list[float | None]:
    result: list[float | None] = []
    for place in range(4):
        selected = mask & torch.isfinite(logits[:, place]).all(1)
        result.append(float(F.cross_entropy(logits[selected, place], labels[selected, place]).item())
                      if selected.any() else None)
    return result


@torch.no_grad()
def evaluate_rung(model: nn.Module, rows: Sequence[dict[str, Any]], horizon: int,
                  device: torch.device) -> dict[str, Any]:
    if len(rows) != 512 or any(row["T"] != horizon for row in rows):
        raise ValueError("rung rows/count do not match horizon")
    signature = inspect.signature(model.step.forward)
    if tuple(signature.parameters) != ("state", "context"):
        raise TypeError("step must accept exactly state and context")
    step_identity = id(model.step)
    addends = torch.tensor([row["a"] for row in rows], device=device)
    starts = torch.tensor([row["x0"] for row in rows], device=device)
    trajectories = torch.tensor([row["trajectory"] for row in rows], device=device)
    carries = torch.tensor([row["carry_lengths"] for row in rows], device=device)
    context = model.encode_context(digits(addends))  # exactly once for the population

    predicted_digits = digits(starts)
    aligned = torch.ones(len(rows), dtype=torch.bool, device=device)
    sticky = torch.zeros_like(aligned)
    first_error = torch.zeros(len(rows), dtype=torch.long, device=device)
    invalid_nonfinite = 0
    free_steps = []
    free_carry = {str(c): {"eligible": 0, "correct": 0} for c in range(5)}
    for step_index in range(1, horizon + 1):
        if id(model.step) != step_identity:
            raise AssertionError("step identity changed")
        state = model.encode_state(predicted_digits)
        logits = model.readout(model.step(state, context))
        finite = torch.isfinite(logits).all((1, 2))
        shape_valid = logits.shape == (len(rows), 4, 10)
        if not shape_valid:
            raise AssertionError("readout shape mismatch")
        next_digits = logits.argmax(-1)
        decimal_valid = ((next_digits >= 0) & (next_digits <= 9)).all(1)
        valid = finite & decimal_valid
        predicted = decode(next_digits)
        truth = trajectories[:, step_index]
        correct = valid & (predicted == truth)
        current_carry = carries[:, step_index - 1]
        for carry in range(5):
            eligible = aligned & (current_carry == carry)
            free_carry[str(carry)]["eligible"] += int(eligible.sum().item())
            free_carry[str(carry)]["correct"] += int((eligible & correct).sum().item())
        newly_failed = (~sticky) & (~correct)
        first_error[newly_failed] = step_index
        sticky |= ~correct
        invalid_nonfinite += int((~valid).sum().item())
        labels = digits(truth)
        free_steps.append({"step": step_index, "correct": int(correct.sum().item()),
                           "exact": float(correct.float().mean().item()),
                           "digit_ce": _ce(logits, labels, valid)})
        aligned &= correct
        predicted_digits = next_digits  # entire canonical hard reset; next step starts h0=0

    endpoint_values = decode(predicted_digits)
    endpoint_correct_mask = endpoint_values == trajectories[:, horizon]
    endpoint_correct = int(endpoint_correct_mask.sum().item())
    trajectory_correct = int((~sticky).sum().item())
    histogram = {"none": trajectory_correct} | {
        str(step): int((first_error == step).sum().item()) for step in range(1, horizon + 1)
    }

    teacher_steps = []
    teacher_invalid_nonfinite = 0
    teacher_carry: dict[str, dict[str, Any]] = {
        str(c): {"count": 0, "correct": 0, "ce_sums": [0.0] * 4} for c in range(5)
    }
    for step_index in range(1, horizon + 1):
        prior = trajectories[:, step_index - 1]
        logits = model.readout(model.step(model.encode_state(digits(prior)), context))
        finite = torch.isfinite(logits).all((1, 2))
        teacher_invalid_nonfinite += int((~finite).sum().item())
        next_digits = logits.argmax(-1)
        predicted = decode(next_digits)
        truth = trajectories[:, step_index]
        correct = finite & (predicted == truth)
        labels = digits(truth)
        teacher_steps.append({"step": step_index, "correct": int(correct.sum().item()),
                              "exact": float(correct.float().mean().item()),
                              "digit_ce": _ce(logits, labels, finite)})
        current_carry = carries[:, step_index - 1]
        for carry in range(5):
            mask = current_carry == carry
            count = int(mask.sum().item())
            if not count:
                continue
            bucket = teacher_carry[str(carry)]
            bucket["count"] += count
            bucket["correct"] += int((mask & correct).sum().item())
            ce_values = _ce(logits, labels, mask & finite)
            for place, value in enumerate(ce_values):
                if value is not None:
                    bucket["ce_sums"][place] += value * int((mask & finite).sum().item())
    for bucket in teacher_carry.values():
        count = bucket["count"]
        bucket["digit_ce"] = [value / count if count else None for value in bucket.pop("ce_sums")]
        bucket["exact"] = bucket["correct"] / count if count else None

    by_addend = {}
    for addend in sorted(set(row["a"] for row in rows)):
        mask = addends == addend
        by_addend[str(addend)] = {
            "count": int(mask.sum().item()),
            "endpoint_correct": int((mask & endpoint_correct_mask).sum().item()),
            "trajectory_correct": int((mask & ~sticky).sum().item()),
        }
    return {
        "T": horizon, "records": len(rows), "endpoint_correct": endpoint_correct,
        "trajectory_correct": trajectory_correct,
        "invalid_nonfinite_count": invalid_nonfinite + teacher_invalid_nonfinite,
        "free_running_invalid_nonfinite_count": invalid_nonfinite,
        "teacher_forced_invalid_nonfinite_count": teacher_invalid_nonfinite,
        "first_error_histogram": histogram, "free_running_per_step": free_steps,
        "teacher_forced_per_step": teacher_steps, "by_addend": by_addend,
        "teacher_forced_by_oracle_carry": teacher_carry,
        "free_running_by_oracle_carry_while_aligned": free_carry,
        "context_encode_calls": 1,
    }


def consecutive_certificate(results: Sequence[dict[str, Any]]) -> int:
    certified = 1
    for expected, result in zip(RUNGS, results):
        if result["T"] != expected:
            raise ValueError("rungs are incomplete or unordered")
        total = result["records"]
        teacher_exact = all(step["correct"] == total for step in result["teacher_forced_per_step"])
        if (result["endpoint_correct"] != total or result["trajectory_correct"] != total
                or result["invalid_nonfinite_count"] != 0 or not teacher_exact):
            break
        certified = expected
    return certified


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {f"p:{name}": value.detach().cpu().clone() for name, value in model.named_parameters()} | {
        f"b:{name}": value.detach().cpu().clone() for name, value in model.named_buffers()
    }


def _status(root: Path, path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short", "--", str(path.relative_to(root))], cwd=root, text=True
        ).strip() or "clean"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "unavailable"


def run_r3(root: str | Path = ".", data_root: str | Path = "data/generated/s1_addcarry_v1",
           metrics_path: str | Path = "metrics_s1_r3_track_a_seed77.json",
           provenance_path: str | Path = "artifacts/s1_r3_track_a_seed77_provenance.json") -> dict[str, Any]:
    root, data_root = Path(root), Path(data_root)
    checkpoint, frozen = verify_frozen_inputs(root, data_root)
    model = S1GRUTransducer(32, 64)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != 25930:
        raise RuntimeError("loaded model parameter count mismatch")
    if sum(module is model.step for module in model.modules()) != 1:
        raise RuntimeError("exactly one step module is required")
    if tuple(inspect.signature(model.step.forward).parameters) != ("state", "context"):
        raise RuntimeError("step signature mismatch")
    before = _snapshot(model)

    with JsonlParseMonitor() as monitor:
        populations = {horizon: load_depth_rows(data_root, horizon) for horizon in RUNGS}
    expected_parsed = [f"depth_t_{horizon}.jsonl" for horizon in RUNGS]
    if monitor.opened != expected_parsed:
        raise RuntimeError(f"unexpected parsed JSONL set: {monitor.opened}")
    population_audit = audit_depth_populations(populations)
    results = [evaluate_rung(model, populations[horizon], horizon, torch.device("cpu")) for horizon in RUNGS]
    repeated = [evaluate_rung(model, populations[horizon], horizon, torch.device("cpu")) for horizon in RUNGS]
    deterministic = results == repeated
    unchanged = all(torch.equal(value, _snapshot(model)[name]) for name, value in before.items())
    if not deterministic or not unchanged:
        raise RuntimeError("evaluation is nondeterministic or mutated frozen tensors")
    certified = consecutive_certificate(results)
    all_teacher_exact = all(all(step["correct"] == 512 for step in result["teacher_forced_per_step"])
                            for result in results)
    r3_pass = certified == 64 and all_teacher_exact
    unique_errors = sum(512 - step["correct"] for step in results[-1]["teacher_forced_per_step"])
    epsilon = unique_errors / 32768
    report = {
        "evaluation": "S1-R3 Track-A seen-context depth seed77", "rungs": results,
        "population_audit": population_audit, "largest_consecutive_perfect_rung": certified,
        "r3_pass_through_t64": r3_pass, "frozen_r2_t1_empirical_error": frozen["r2_t1_empirical_error"],
        "t64_unique_edge_empirical_one_step_error": epsilon,
        "rule_of_three_upper_bound_if_zero_errors": 3 / 32768 if unique_errors == 0 else None,
        "population_error_caveat": "Empirical zero does not establish zero population error.",
        "one_minus_epsilon_pow_t_noncertifying": {str(t): (1 - epsilon) ** t for t in RUNGS},
        "claim_scope": "Tied hard-state carry composition for seen addends {1,2,4,5,7,8} only.",
        "excluded_claims": ["held-out addends", "original G2", "latent recurrence", "wider width", "other operations"],
        "parsed_jsonl": monitor.opened, "model_visible_fields": ["canonical x0 digits", "canonical a digits"],
        "deterministic_repeat": deterministic, "parameters_buffers_unchanged": unchanged,
        "no_training": True, "optimizer_constructed": False, "device": "cpu", "dtype": "torch.float32",
    }
    metrics_path = Path(metrics_path)
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluator = Path(__file__).resolve()
    test_source = root.resolve() / "tests/test_s1_r3_depth.py"
    provenance = {
        "evaluator_sha256": sha256(evaluator), "evaluator_git_status": _status(root.resolve(), evaluator),
        "test_sha256": sha256(test_source), "test_git_status": _status(root.resolve(), test_source),
        "verified_input_hashes": frozen["verified_hashes"], "checkpoint_config": checkpoint["config"],
        "parameter_count": 25930, "final_epoch": 60, "optimizer_steps": 4740,
        "device": "cpu", "dtype": "torch.float32", "parsed_jsonl": monitor.opened,
        "model_visible_fields": report["model_visible_fields"], "no_training": True,
        "optimizer_constructed": False, "gate": {"r3_pass_through_t64": r3_pass,
                                                   "certified_through": certified},
        "metrics_sha256": sha256(metrics_path),
    }
    provenance_path = Path(provenance_path)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_r3(), indent=2))
