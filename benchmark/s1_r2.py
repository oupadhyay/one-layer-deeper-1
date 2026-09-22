"""Independent seed-77 replication of the frozen S1 seen-context C1 protocol."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
from typing import Any
from unittest.mock import patch

import torch

from benchmark.s1_c1 import (
    CONFIG as C1_CONFIG,
    FORBIDDEN_PROJECTED_FIELDS,
    OneStepHarness,
    S1GRUTransducer,
    _predict,
    _snapshot,
    evaluate_rows,
    load_eval_rows,
    load_projected_train,
    train_fixed,
    verify_dataset,
)


R2_CONFIG = replace(C1_CONFIG, seed=77)


def config_diff() -> dict[str, dict[str, Any]]:
    before, after = asdict(C1_CONFIG), asdict(R2_CONFIG)
    return {key: {"c1": before[key], "r2": after[key]} for key in before if before[key] != after[key]}


class JsonlParseMonitor(AbstractContextManager["JsonlParseMonitor"]):
    """Instrument actual JSONL opens in a delimited parsing phase."""

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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _status(root: Path, path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short", "--", str(path.relative_to(root))], cwd=root, text=True
        ).strip() or "clean"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "unavailable"


def run_r2(data_root: str | Path = "data/generated/s1_addcarry_v1",
           metrics_path: str | Path = "metrics_s1_r2_gru_seed77.jsonl",
           checkpoint_path: str | Path = "artifacts/s1_r2_gru_seed77.pt",
           provenance_path: str | Path = "artifacts/s1_r2_gru_seed77_provenance.json",
           device: str = "cpu") -> dict[str, Any]:
    diff = config_diff()
    if diff != {"seed": {"c1": 76, "r2": 77}}:
        raise RuntimeError(f"R2 config differs from C1 beyond seed: {diff}")
    random.seed(R2_CONFIG.seed)
    torch.manual_seed(R2_CONFIG.seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(device)

    # Byte hashing is provenance verification, not JSONL parsing/inference.
    dataset_provenance = verify_dataset(data_root)
    hash_verified_files = sorted(dataset_provenance["verified_file_sha256"])
    application_log: list[str] = []
    with JsonlParseMonitor() as optimization_monitor:
        train_dataset = load_projected_train(data_root, application_log)
    if optimization_monitor.opened != ["train.jsonl"] or application_log != ["train.jsonl"]:
        raise RuntimeError("optimization parsed a file other than train.jsonl")
    if set(train_dataset[0]) != {"input_ids", "labels"} or set(train_dataset[0]) & FORBIDDEN_PROJECTED_FIELDS:
        raise RuntimeError("optimization projection leaked metadata")

    model = S1GRUTransducer(R2_CONFIG.embedding_dim, R2_CONFIG.hidden_dim).to(dev)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 25930:
        raise RuntimeError("frozen architecture parameter count changed")
    metrics_path = Path(metrics_path)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as output:
        steps, elapsed = train_fixed(model, train_dataset, R2_CONFIG, dev, output)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": asdict(R2_CONFIG),
                    "parameter_count": parameter_count, "final_epoch": R2_CONFIG.epochs,
                    "optimizer_steps": steps, "replication": "S1-R2"}, checkpoint_path)

        evaluation_application_log: list[str] = []
        with JsonlParseMonitor() as evaluation_monitor:
            train_rows = load_eval_rows(data_root, "train", evaluation_application_log)
            test_state_rows = load_eval_rows(data_root, "test_state", evaluation_application_log)
            test_context_rows = load_eval_rows(data_root, "test_context", evaluation_application_log)
        expected_evaluation = ["train.jsonl", "test_state.jsonl", "test_context.jsonl"]
        if evaluation_monitor.opened != expected_evaluation or evaluation_application_log != expected_evaluation:
            raise RuntimeError("post-freeze parsed-file set differs from contract")

        evaluations = {
            "train": evaluate_rows(model, train_rows, R2_CONFIG.eval_batch_size, dev),
            "test_state": evaluate_rows(model, test_state_rows, R2_CONFIG.eval_batch_size, dev),
            "test_context_non_gating": evaluate_rows(model, test_context_rows, R2_CONFIG.eval_batch_size, dev),
        }
        state_predictions = _predict(model, test_state_rows, R2_CONFIG.eval_batch_size, dev)
        context_predictions = _predict(model, test_context_rows, R2_CONFIG.eval_batch_size, dev)
        first_state_errors = sum(pred != row["labels"]
                                 for pred, row in zip(state_predictions[:5000], test_state_rows[:5000]))
        first_context_errors = sum(pred != row["labels"]
                                   for pred, row in zip(context_predictions[:5000], test_context_rows[:5000]))
        first_context = test_context_rows[:5000]
        mixture = {
            "per_a": {str(a): sum(row["a"] == a for row in first_context) for a in (3, 6, 9)},
            "per_carry": {str(c): sum(row["carry_length"] == c for row in first_context) for c in range(5)},
        }

        gates = {
            "train_20004_exact": evaluations["train"]["overall"]["exact_correct"] == 20004,
            "test_state_at_most_5_errors": 5004 - evaluations["test_state"]["overall"]["exact_correct"] <= 5,
            "test_state_c2_204_exact": evaluations["test_state"]["by_carry"]["2"]["exact_correct"] == 204,
            "test_state_c3_34_exact": evaluations["test_state"]["by_carry"]["3"]["exact_correct"] == 34,
            "test_state_c4_5_exact": evaluations["test_state"]["by_carry"]["4"]["exact_correct"] == 5,
            "config_only_seed_changed": diff == {"seed": {"c1": 76, "r2": 77}},
            "instrumented_access": optimization_monitor.opened == ["train.jsonl"],
        }
        before = _snapshot(model)
        sample = torch.tensor([row["input_ids"] for row in test_state_rows[:128]], device=dev)
        direct, harness, repeated = model(sample), OneStepHarness(model).run(sample), model(sample)
        purity = {
            "direct_harness_logits_bitwise_equal": torch.equal(direct, harness),
            "direct_harness_predictions_equal": torch.equal(direct.argmax(-1), harness.argmax(-1)),
            "repeated_logits_bitwise_equal": torch.equal(direct, repeated),
            "parameters_buffers_unchanged": all(torch.equal(value, _snapshot(model)[name])
                                                  for name, value in before.items()),
            "step_instances": sum(module is model.step for module in model.modules()),
            "step_signature": list(inspect.signature(model.step.forward).parameters),
        }
        gates["api_eval_purity"] = (
            purity["direct_harness_logits_bitwise_equal"]
            and purity["direct_harness_predictions_equal"]
            and purity["repeated_logits_bitwise_equal"]
            and purity["parameters_buffers_unchanged"]
            and purity["step_instances"] == 1
            and purity["step_signature"] == ["state", "context"]
        )
        gates["r2_pass"] = all(gates.values())

        root = Path(__file__).resolve().parents[1]
        c1_provenance = json.loads((root / "artifacts" / "s1_c1_gru_seed76_provenance.json").read_text(encoding="utf-8"))
        test_source = root / "tests" / "test_s1_r2.py"
        provenance = {
            "replication": "S1-R2 seen-context replication", "config": asdict(R2_CONFIG),
            "c1_config": asdict(C1_CONFIG), "exact_config_diff": diff,
            "architecture_source": "benchmark/s1_c1.py", "architecture_source_sha256": _sha256(root / "benchmark" / "s1_c1.py"),
            "architecture_source_matches_c1_provenance": _sha256(root / "benchmark" / "s1_c1.py") == c1_provenance["trainer_sha256"],
            "runner_sha256": _sha256(Path(__file__)), "runner_git_status": _status(root, Path(__file__)),
            "test_sha256": _sha256(test_source), "test_git_status": _status(root, test_source),
            "checkpoint_sha256": _sha256(checkpoint_path), "parameter_count": parameter_count,
            "final_epoch": R2_CONFIG.epochs, "optimizer_steps": steps,
            "dataset_generator_sha256": json.loads((Path(data_root) / "dataset_config.json").read_text(encoding="utf-8"))["generator_sha256"],
            "dataset_config_sha256": dataset_provenance["config_sha256"],
            "dataset_manifest_sha256": dataset_provenance["manifest_sha256"],
            "hash_verified_files": hash_verified_files,
            "optimization_parsed_jsonl": optimization_monitor.opened,
            "post_freeze_parsed_jsonl": evaluation_monitor.opened,
            "depth_jsonl_parsed": [], "device": str(dev), "dtype": str(next(model.parameters()).dtype),
            "outer_step_applications": 1, "no_scheduler": True,
            "test_context_diagnostic": evaluations["test_context_non_gating"],
        }
        provenance_path = Path(provenance_path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        final = {
            "kind": "final", "replication": "S1-R2 seen-context replication",
            "config": asdict(R2_CONFIG), "exact_config_diff": diff,
            "parameter_count": parameter_count, "optimizer_steps": steps,
            "elapsed_training_seconds": elapsed, "evaluations": evaluations,
            "first_5000_test_state_errors": first_state_errors,
            "first_5000_test_context_errors_non_gating": first_context_errors,
            "first_5000_test_context_mixture": mixture, "a1_c4_test_support": 0,
            "gates": gates, "purity": purity,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": provenance["checkpoint_sha256"],
            "provenance": str(provenance_path),
        }
        output.write(json.dumps(final, sort_keys=True) + "\n")
    return final


if __name__ == "__main__":
    print(json.dumps(run_r2(), indent=2))
