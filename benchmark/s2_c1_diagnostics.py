"""Evaluation-only supplemental reporting for the frozen S2 C1 seed-78 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from benchmark.s2_c1 import S2Transformer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    actual = {name: sha256(root / name) for name in manifest["files"]}
    if actual != manifest["files"]:
        raise RuntimeError("dataset artifact manifest mismatch")
    return actual


def read_rows(root: Path, name: str, parsed: list[str]) -> list[dict[str, Any]]:
    if name not in {"train", "test_state"}:
        raise ValueError("diagnostics may parse only train and test_state")
    path = root / f"{name}.jsonl"
    parsed.append(path.as_posix())
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def summarize(rows: list[dict[str, Any]], predictions: torch.Tensor,
              losses: torch.Tensor) -> dict[str, Any]:
    labels = torch.tensor([row["labels"] for row in rows])
    correct = predictions.eq(labels)
    exact = correct.all(dim=1)
    return {
        "count": len(rows),
        "exact_correct": int(exact.sum()),
        "exact_accuracy": float(exact.double().mean()),
        "per_place": [
            {
                "place": place,
                "ce": float(losses[:, place].double().mean()),
                "correct": int(correct[:, place].sum()),
                "accuracy": float(correct[:, place].double().mean()),
            }
            for place in range(4)
        ],
    }


def grouped(rows: list[dict[str, Any]], predictions: torch.Tensor, losses: torch.Tensor,
            key: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    result = {}
    for value in sorted({key(row) for row in rows}, key=str):
        indices = [index for index, row in enumerate(rows) if key(row) == value]
        result[str(value)] = summarize(
            [rows[index] for index in indices], predictions[indices], losses[indices]
        )
    return result


def threshold_rows(rows: list[dict[str, Any]], predictions: torch.Tensor,
                   losses: torch.Tensor, distance: int) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        if abs(2 * row["x0"] - row["N"]) != distance:
            continue
        prediction = predictions[index].tolist()
        output.append({
            "index": index, "x": row["x0"], "N": row["N"],
            "prediction": prediction, "target": row["labels"],
            "exact": prediction == row["labels"],
            "per_place_ce": [float(value) for value in losses[index].tolist()],
        })
    return output


@torch.no_grad()
def evaluate(model: S2Transformer, rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_parts = []
    for start in range(0, len(rows), 1024):
        ids = torch.tensor([row["input_ids"] for row in rows[start:start + 1024]])
        logits_parts.append(model(ids))
    logits = torch.cat(logits_parts)
    labels = torch.tensor([row["labels"] for row in rows])
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view(-1, 4)
    return logits, logits.argmax(dim=-1), losses


def run(data_root: Path, checkpoint: Path, original_metrics: Path,
        original_provenance: Path, output: Path, provenance_output: Path) -> dict[str, Any]:
    data_root, checkpoint = Path(data_root), Path(checkpoint)
    dataset_hashes = verify_dataset(data_root)
    parsed: list[str] = []
    train = read_rows(data_root, "train", parsed)
    test = read_rows(data_root, "test_state", parsed)

    model = S2Transformer().cpu().float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    train_logits, train_predictions, train_losses = evaluate(model, train)
    test_logits, test_predictions, test_losses = evaluate(model, test)
    repeated_logits, repeated_predictions, _ = evaluate(model, test)

    original_final = json.loads(original_metrics.read_text(encoding="utf-8").splitlines()[-1])
    train_summary = summarize(train, train_predictions, train_losses)
    test_summary = summarize(test, test_predictions, test_losses)
    original_agreement = {
        "train_exact_correct": train_summary["exact_correct"] == original_final["evaluations"]["train"]["exact_correct"],
        "test_exact_correct": test_summary["exact_correct"] == original_final["evaluations"]["test_state"]["exact_correct"],
        "test_error_predictions": all(
            test_predictions[item["index"]].tolist() == item["prediction"]
            for item in original_final["test_errors"]
        ),
    }
    if not all(original_agreement.values()) or test_summary["count"] - test_summary["exact_correct"] != 11:
        raise RuntimeError("frozen predictions disagree with original final metrics")

    def split_report(rows: list[dict[str, Any]], predictions: torch.Tensor,
                     losses: torch.Tensor) -> dict[str, Any]:
        return {
            "overall": summarize(rows, predictions, losses),
            "by_N": grouped(rows, predictions, losses, lambda row: row["N"]),
            "by_reduction": grouped(rows, predictions, losses, lambda row: row["reduction"][0]),
            "by_carry_count": grouped(rows, predictions, losses, lambda row: row["carry_counts"][0]),
            "by_carry_mask": grouped(rows, predictions, losses, lambda row: "".join(map(str, row["carry_masks"][0]))),
            "leading_zero_output": grouped(rows, predictions, losses, lambda row: row["labels"][3] == 0)["True"],
        }

    test_threshold = threshold_rows(test, test_predictions, test_losses, 1)
    train_threshold = threshold_rows(train, train_predictions, train_losses, 3)
    errors = [
        {"index": index, "x": row["x0"], "N": row["N"], "target": row["labels"],
         "prediction": test_predictions[index].tolist(),
         "per_place_ce": [float(value) for value in test_losses[index].tolist()]}
        for index, row in enumerate(test) if test_predictions[index].tolist() != row["labels"]
    ]
    report = {
        "kind": "supplemental_frozen_diagnostics", "control": "s2_c1_transformer_v1", "seed": 78,
        "train": split_report(train, train_predictions, train_losses),
        "test_state": split_report(test, test_predictions, test_losses),
        "threshold": {"test_abs_2x_minus_N_eq_1_gate": test_threshold,
                      "train_abs_2x_minus_N_eq_3_diagnostic": train_threshold},
        "singleton_predictions": [
            {"index": index, "N": row["N"], "x": row["x0"], "target": row["labels"],
             "prediction": train_predictions[index].tolist()}
            for index, row in enumerate(train) if row["x0"] in (0, row["N"] - 1)
        ],
        "test_errors": {"cap": 32, "total": len(errors), "rows": errors[:32]},
        "preserved_gate": {"test_errors": len(errors), "g1_pass": False, "no_selection": True},
        "original_final_agreement": original_agreement,
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    provenance = {
        "checkpoint_sha256": sha256(checkpoint),
        "training_runner_sha256": sha256(Path(__file__).with_name("s2_c1.py")),
        "original_metrics_sha256": sha256(original_metrics),
        "original_provenance_sha256": sha256(original_provenance),
        "diagnostics_source_sha256": sha256(Path(__file__)),
        "diagnostics_test_sha256": sha256(Path(__file__).parents[1] / "tests" / "test_s2_c1_diagnostics.py"),
        "dataset_hashes": dataset_hashes, "parsed_jsonl": parsed,
        "device": "cpu", "dtype": "torch.float32", "eval": True,
        "no_training": True, "no_optimizer": True, "weights_only": True, "strict_load": True,
        "deterministic_repeat": torch.equal(test_logits, repeated_logits) and torch.equal(test_predictions, repeated_predictions),
        "nonmutation": all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in before.items()),
        "supplemental_diagnostics_sha256": sha256(output),
    }
    provenance_output.write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/generated/s2_doublemod_seen_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/s2_c1_transformer_seed78.pt"))
    parser.add_argument("--original-metrics", type=Path, default=Path("metrics_s2_c1_transformer_seed78.jsonl"))
    parser.add_argument("--original-provenance", type=Path, default=Path("artifacts/s2_c1_transformer_seed78_provenance.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/s2_c1_transformer_seed78_diagnostics.json"))
    parser.add_argument("--provenance-output", type=Path, default=Path("artifacts/s2_c1_transformer_seed78_diagnostics_provenance.json"))
    args = parser.parse_args()
    result = run(args.data_root, args.checkpoint, args.original_metrics,
                 args.original_provenance, args.output, args.provenance_output)
    print(json.dumps({"test_errors": result["test_errors"]["total"],
                      "g1_pass": result["preserved_gate"]["g1_pass"]}))
