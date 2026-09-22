"""Frozen-weight strict Track-A depth evaluation for ``s0_perm_v1``.

This module is intentionally separate from the source that trained the frozen C2
checkpoint. It performs provenance verification before inference and never
constructs an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from benchmark.s0_research import C1PointerControl, decimal_digits, load_split


RUNGS = (1, 2, 4, 8, 16, 32, 64)
EXPECTED_CONFIG = {
    "seed": 75, "width": 64, "learning_rate": 3e-3, "weight_decay": 1e-4,
    "batch_size": 256, "max_steps": 1200, "eval_batch_size": 1024, "curve_every": 50,
}
EXPECTED_PARAMETERS = 33108
EXPECTED_FINAL_STEP = 1200


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8"))
    return digest.hexdigest()


def verify_frozen_inputs(repo_root: str | Path, data_root: str | Path,
                         checkpoint_path: str | Path, training_provenance_path: str | Path
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all frozen provenance before any model inference occurs."""
    repo_root, data_root = Path(repo_root), Path(data_root)
    checkpoint_path, training_provenance_path = Path(checkpoint_path), Path(training_provenance_path)
    provenance = json.loads(training_provenance_path.read_text(encoding="utf-8"))
    checks = {
        "checkpoint_sha256": sha256(checkpoint_path) == provenance["checkpoint_sha256"],
        "dataset_config_sha256": sha256(data_root / "dataset_config.json") == provenance["dataset_config_sha256"],
        "dataset_manifest_sha256": sha256(data_root / "artifact_manifest.json") == provenance["dataset_manifest_sha256"],
    }
    for relative, expected in provenance["source_sha256"].items():
        checks[f"source:{relative}"] = sha256(repo_root / relative) == expected
    on_disk_manifest = json.loads((data_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    checks["manifest_matches_training_provenance"] = on_disk_manifest == provenance["dataset_manifest"]
    for relative, expected in on_disk_manifest["files"].items():
        checks[f"artifact:{relative}"] = sha256(data_root / relative) == expected
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("frozen provenance mismatch before inference: " + ", ".join(failures))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata_checks = {
        "checkpoint_config": checkpoint["config"] == EXPECTED_CONFIG == provenance["config"],
        "checkpoint_parameter_count": checkpoint["parameter_count"] == EXPECTED_PARAMETERS == provenance["parameter_count"],
        "checkpoint_final_step": checkpoint["final_train_step"] == EXPECTED_FINAL_STEP == provenance["final_train_step"],
    }
    failures = [name for name, passed in metadata_checks.items() if not passed]
    if failures:
        raise RuntimeError("checkpoint metadata mismatch before inference: " + ", ".join(failures))
    checks.update(metadata_checks)
    return checkpoint, {"checks": checks, "training_provenance": provenance}


def _digits(values: torch.Tensor) -> torch.Tensor:
    return torch.stack((values.remainder(10), values.div(10, rounding_mode="floor")), dim=1)


def _tensor_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {f"parameter:{name}": value.detach().cpu().clone() for name, value in model.named_parameters()} | {
        f"buffer:{name}": value.detach().cpu().clone() for name, value in model.named_buffers()
    }


def _step_metrics(logits: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    labels = _digits(truth)
    predicted_digits = logits.argmax(dim=-1)
    predicted = predicted_digits[:, 0] + 10 * predicted_digits[:, 1]
    return {
        "correct": int((predicted == truth).sum().item()),
        "exact": float((predicted == truth).float().mean().item()),
        "digit_ce": [float(F.cross_entropy(logits[:, digit], labels[:, digit]).item()) for digit in range(2)],
    }


@torch.no_grad()
def evaluate_rung(model: nn.Module, rows: Sequence[dict[str, Any]], horizon: int,
                  device: torch.device) -> dict[str, Any]:
    """Evaluate strict free-running and teacher-forced lanes for one rung."""
    if len(rows) == 0 or any(row["T"] != horizon for row in rows):
        raise ValueError("rows must be nonempty and match the requested horizon")
    signature = inspect.signature(model.step.forward if isinstance(model.step, nn.Module) else model.step)
    if tuple(signature.parameters) != ("state", "context"):
        raise TypeError("step must accept exactly (state, context)")
    step_identity = id(model.step)
    tables = torch.tensor([row["perm"] for row in rows], dtype=torch.long, device=device)
    starts = torch.tensor([row["x0"] for row in rows], dtype=torch.long, device=device)
    trajectories = torch.tensor([row["trajectory"] for row in rows], dtype=torch.long, device=device)
    moduli = torch.tensor([row["M"] for row in rows], dtype=torch.long, device=device)
    context = model.encode_context(tables)  # once per record, batched

    predicted = starts.clone()
    sticky_failure = torch.zeros(len(rows), dtype=torch.bool, device=device)
    first_error = torch.zeros(len(rows), dtype=torch.long, device=device)
    invalid_count = 0
    free_steps: list[dict[str, Any]] = []
    for step_index in range(1, horizon + 1):
        if id(model.step) != step_identity:
            raise AssertionError("step identity changed during free-running evaluation")
        state = model.encode_state(_digits(predicted))
        logits = model.readout(model.step(state, context))
        predicted_digits = logits.argmax(dim=-1)
        predicted = predicted_digits[:, 0] + 10 * predicted_digits[:, 1]
        truth = trajectories[:, step_index]
        invalid = (predicted < 0) | (predicted >= moduli)
        error = predicted != truth
        newly_failed = (~sticky_failure) & (invalid | error)
        first_error[newly_failed] = step_index
        sticky_failure |= invalid | error
        invalid_count += int(invalid.sum().item())
        free_steps.append({"step": step_index, **_step_metrics(logits, truth)})

    endpoint_correct = int((predicted == trajectories[:, horizon]).sum().item())
    trajectory_correct = int((~sticky_failure).sum().item())
    histogram = {"none": trajectory_correct}
    histogram.update({str(step): int((first_error == step).sum().item()) for step in range(1, horizon + 1)})

    teacher_steps: list[dict[str, Any]] = []
    for step_index in range(1, horizon + 1):
        if id(model.step) != step_identity:
            raise AssertionError("step identity changed during teacher-forced evaluation")
        true_prior = trajectories[:, step_index - 1]
        logits = model.readout(model.step(model.encode_state(_digits(true_prior)), context))
        teacher_steps.append({"step": step_index, **_step_metrics(logits, trajectories[:, step_index])})

    return {
        "T": horizon, "records": len(rows), "endpoint_correct": endpoint_correct,
        "trajectory_correct": trajectory_correct, "invalid_decode_count": invalid_count,
        "first_error_step_histogram": histogram, "free_running_per_step": free_steps,
        "teacher_forced_per_step": teacher_steps,
    }


def consecutive_certificate(results: Sequence[dict[str, Any]]) -> int:
    certified = 0
    for expected, result in zip(RUNGS, results):
        if result["T"] != expected:
            raise ValueError("rungs must be complete and ordered")
        total = result["records"]
        if result["endpoint_correct"] != total or result["trajectory_correct"] != total:
            break
        certified = expected
    return certified


def run_depth(repo_root: str | Path, data_root: str | Path, checkpoint_path: str | Path,
              training_provenance_path: str | Path, metrics_path: str | Path,
              depth_provenance_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    checkpoint, verified = verify_frozen_inputs(repo_root, data_root, checkpoint_path, training_provenance_path)
    dev = torch.device(device)
    model = C1PointerControl(width=checkpoint["config"]["width"]).to(dev)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("constructed model parameter count mismatch")
    before = _tensor_snapshot(model)

    populations = {1: load_split(data_root, "test_context")[:512]}
    populations.update({horizon: load_split(data_root, f"depth_t_{horizon}") for horizon in RUNGS[1:]})
    results = [evaluate_rung(model, populations[horizon], horizon, dev) for horizon in RUNGS]
    repeated = [evaluate_rung(model, populations[horizon], horizon, dev) for horizon in RUNGS]
    deterministic = results == repeated
    unchanged = all(torch.equal(value, _tensor_snapshot(model)[name]) for name, value in before.items())
    if not deterministic or not unchanged:
        raise RuntimeError("frozen evaluation was nondeterministic or mutated model tensors")

    certified = consecutive_certificate(results)
    epsilon = 1.0 - results[0]["endpoint_correct"] / results[0]["records"]
    empirical_curve = {str(result["T"]): (1.0 - epsilon) ** result["T"] for result in results}
    report = {
        "stage": "S0", "version": "s0_perm_v1", "control": "C2 seed75 frozen Track A",
        "rungs": results, "certified_through_T": certified, "g3_pass_through_T64": certified == 64,
        "s1_unlocked": certified >= 16, "t1_empirical_epsilon": epsilon,
        "t1_rule_of_three_upper_bound": 3 / results[0]["records"],
        "independence_curve_one_minus_epsilon_pow_T": empirical_curve,
        "independence_curve_caveat": "Empirical epsilon=0 does not prove zero population error.",
        "ood_size_status": "G4 locked: frozen C2 M24 T1 exact was 11.33%; no M24 depth evaluated.",
        "deterministic_repeat": deterministic, "parameters_and_buffers_unchanged": unchanged,
        "no_training": True, "optimizer_constructed": False,
    }
    metrics_path = Path(metrics_path)
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    depth_source = Path(__file__).resolve()
    resolved_repo_root = Path(repo_root).resolve()
    depth_provenance = {
        "depth_evaluator_sha256": sha256(depth_source),
        "depth_evaluator_path": str(depth_source.relative_to(resolved_repo_root)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_source_sha256": verified["training_provenance"]["source_sha256"],
        "dataset_config_sha256": sha256(Path(data_root) / "dataset_config.json"),
        "dataset_manifest_sha256": sha256(Path(data_root) / "artifact_manifest.json"),
        "record_sha256": {f"T{horizon}": rows_sha256(populations[horizon]) for horizon in RUNGS},
        "source_file_sha256": {
            "T1:test_context_full_file": sha256(Path(data_root) / "test_context.jsonl")
        } | {f"T{horizon}:depth_t_{horizon}_full_file": sha256(Path(data_root) / f"depth_t_{horizon}.jsonl")
             for horizon in RUNGS[1:]},
        "record_counts": {str(horizon): len(populations[horizon]) for horizon in RUNGS},
        "device": str(dev), "dtype": str(next(model.parameters()).dtype),
        "checkpoint_config": checkpoint["config"], "parameter_count": EXPECTED_PARAMETERS,
        "final_train_step": EXPECTED_FINAL_STEP, "verification_checks": verified["checks"],
        "no_training": True, "optimizer_constructed": False,
        "metrics_path": str(metrics_path), "metrics_sha256": sha256(metrics_path),
    }
    Path(depth_provenance_path).write_text(json.dumps(depth_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen S0 C2 with strict Track A")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="data/generated/s0_perm_v1")
    parser.add_argument("--checkpoint", default="artifacts/s0_perm_v1_c2_seed75.pt")
    parser.add_argument("--training-provenance", default="artifacts/s0_perm_v1_c2_seed75_provenance.json")
    parser.add_argument("--metrics", default="metrics_s0_perm_v1_g3_track_a_seed75.json")
    parser.add_argument("--provenance", default="artifacts/s0_perm_v1_g3_track_a_seed75_provenance.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_depth(args.repo_root, args.data_root, args.checkpoint, args.training_provenance,
                       args.metrics, args.provenance, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
