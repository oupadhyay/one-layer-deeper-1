"""Single declared S1 C1 one-step GRU transducer positive control."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class C1Config:
    seed: int = 76
    embedding_dim: int = 32
    hidden_dim: int = 64
    batch_size: int = 256
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    eval_batch_size: int = 2048
    curve_every_epochs: int = 5
    max_training_seconds: float = 1800.0


CONFIG = C1Config()
FORBIDDEN_PROJECTED_FIELDS = {
    "x0", "a", "y", "T", "split", "context_id", "trajectory",
    "carry_length", "carry_lengths", "seed",
}


class ProjectedDataset(Dataset[dict[str, torch.Tensor]]):
    """Stores only the two model/training fields after parsing."""

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape[1] != 10:
            raise ValueError("input_ids must have shape [N,10]")
        if labels.ndim != 2 or labels.shape[1] != 4:
            raise ValueError("labels must have shape [N,4]")
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


def load_projected_train(data_root: str | Path, opened_during_optimization: list[str]) -> ProjectedDataset:
    path = Path(data_root) / "train.jsonl"
    opened_during_optimization.append(path.name)
    inputs, labels = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            inputs.append(row["input_ids"])
            labels.append(row["labels"])
    return ProjectedDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long))


def load_eval_rows(data_root: str | Path, split: str, opened_after_freeze: list[str]) -> list[dict[str, Any]]:
    if split not in ("train", "test_state", "test_context"):
        raise ValueError("C1 evaluation is restricted to one-step splits")
    path = Path(data_root) / f"{split}.jsonl"
    opened_after_freeze.append(path.name)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class S1GRUTransducer(nn.Module):
    """Generic learned LSD-to-MSD sequence transducer with no programmed arithmetic."""

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, embedding_dim)
        self.cell = nn.GRUCell(2 * embedding_dim, hidden_dim)
        self.digit_head = nn.Linear(hidden_dim, 10)
        self.hidden_dim = hidden_dim
        self.step = _FourPlaceStep(self)

    def encode_state(self, canonical_digits: torch.Tensor) -> torch.Tensor:
        if canonical_digits.ndim != 2 or canonical_digits.shape[1] != 4:
            raise ValueError("encode_state expects [batch,4] canonical digits")
        return self.embedding(canonical_digits)

    def encode_context(self, canonical_addend_digits: torch.Tensor) -> torch.Tensor:
        if canonical_addend_digits.ndim != 2 or canonical_addend_digits.shape[1] != 4:
            raise ValueError("encode_context expects [batch,4] canonical digits")
        return self.embedding(canonical_addend_digits)

    def readout(self, states: torch.Tensor) -> torch.Tensor:
        return self.digit_head(states)

    def canonical_from_input_ids(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2 or ids.shape[1] != 10:
            raise ValueError("expected fixed [batch,10] input layout")
        if not torch.all(ids[:, 0] == 0) or not torch.all(ids[:, 5] == 1):
            raise ValueError("structural marker mismatch")
        state_tokens, addend_tokens = ids[:, 1:5], ids[:, 6:10]
        if not torch.all((state_tokens >= 2) & (state_tokens <= 11)):
            raise ValueError("invalid state digit token")
        if not torch.all((addend_tokens >= 2) & (addend_tokens <= 11)):
            raise ValueError("invalid addend digit token")
        return state_tokens - 2, addend_tokens - 2

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        state_digits, addend_digits = self.canonical_from_input_ids(ids)
        state = self.encode_state(state_digits)
        context = self.encode_context(addend_digits)
        return self.readout(self.step(state, context))


class _FourPlaceStep(nn.Module):
    """One outer task step containing four tied, local place updates."""

    def __init__(self, owner: S1GRUTransducer) -> None:
        super().__init__()
        # Avoid registering a cyclic child module while retaining shared identity.
        object.__setattr__(self, "_owner", owner)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        owner = object.__getattribute__(self, "_owner")
        hidden = state.new_zeros((state.shape[0], owner.hidden_dim))
        outputs = []
        for place in range(4):
            hidden = owner.cell(torch.cat((state[:, place], context[:, place]), dim=-1), hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class OneStepHarness:
    def __init__(self, model: S1GRUTransducer) -> None:
        signature = inspect.signature(model.step.forward)
        if tuple(signature.parameters) != ("state", "context"):
            raise TypeError("step must accept exactly (state, context)")
        self.model = model
        self.step_identity = id(model.step)

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        if id(self.model.step) != self.step_identity:
            raise AssertionError("step identity changed")
        state_digits, addend_digits = self.model.canonical_from_input_ids(input_ids)
        state = self.model.encode_state(state_digits)
        context = self.model.encode_context(addend_digits)
        return self.model.readout(self.model.step(state, context))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_status(repo_root: Path, path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short", "--", str(path.relative_to(repo_root))],
            cwd=repo_root, text=True,
        ).strip() or "clean"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "unavailable"


def verify_dataset(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {}
    for name, expected in manifest["files"].items():
        actual = _sha256(root / name)
        checks[name] = actual
        if actual != expected:
            raise RuntimeError(f"dataset manifest mismatch before training: {name}")
    return {
        "manifest": manifest, "verified_file_sha256": checks,
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(root / "dataset_config.json"),
    }


@torch.no_grad()
def _projected_metrics(model: nn.Module, dataset: ProjectedDataset, batch_size: int,
                       device: torch.device) -> dict[str, Any]:
    model.eval()
    ce = torch.zeros(4, dtype=torch.float64)
    correct = torch.zeros(4, dtype=torch.long)
    exact = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        logits = model(ids)
        predictions = logits.argmax(-1)
        exact += int((predictions == labels).all(1).sum().item())
        for place in range(4):
            ce[place] += F.cross_entropy(logits[:, place], labels[:, place], reduction="sum").double().cpu()
            correct[place] += (predictions[:, place] == labels[:, place]).sum().cpu()
    count = len(dataset)
    return {
        "count": count, "exact_correct": exact, "exact": exact / count,
        "place_ce": (ce / count).tolist(), "place_accuracy": (correct.double() / count).tolist(),
    }


def train_fixed(model: S1GRUTransducer, dataset: ProjectedDataset, config: C1Config,
                device: torch.device, metrics_output: Any) -> tuple[int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    started = time.perf_counter()
    steps = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch in loader:
            if time.perf_counter() - started >= config.max_training_seconds:
                raise TimeoutError("fixed C1 reached the 30-minute training cap")
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite C1 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.item()) * len(ids)
            examples += len(ids)
        if epoch == 1 or epoch % config.curve_every_epochs == 0 or epoch == config.epochs:
            train_metrics = _projected_metrics(model, dataset, config.eval_batch_size, device)
            metrics_output.write(json.dumps({
                "kind": "curve", "epoch": epoch, "optimizer_steps": steps,
                "mean_training_loss": loss_sum / examples,
                "train_exact": train_metrics["exact"],
                "elapsed_training_seconds": time.perf_counter() - started,
            }, sort_keys=True) + "\n")
            metrics_output.flush()
    return steps, time.perf_counter() - started


def _dataset_from_rows(rows: Sequence[dict[str, Any]]) -> ProjectedDataset:
    return ProjectedDataset(
        torch.tensor([row["input_ids"] for row in rows], dtype=torch.long),
        torch.tensor([row["labels"] for row in rows], dtype=torch.long),
    )


@torch.no_grad()
def evaluate_rows(model: S1GRUTransducer, rows: Sequence[dict[str, Any]], batch_size: int,
                  device: torch.device) -> dict[str, Any]:
    overall = _projected_metrics(model, _dataset_from_rows(rows), batch_size, device)

    def metrics(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _projected_metrics(model, _dataset_from_rows(selected), batch_size, device)

    by_addend = {str(a): metrics([row for row in rows if row["a"] == a])
                  for a in sorted({row["a"] for row in rows})}
    by_carry = {str(carry): metrics([row for row in rows if row["carry_length"] == carry])
                for carry in range(5) if any(row["carry_length"] == carry for row in rows)}
    by_addend_carry = {
        str(a): {str(carry): metrics(selected)
                 for carry in range(5)
                 if (selected := [row for row in rows if row["a"] == a and row["carry_length"] == carry])}
        for a in sorted({row["a"] for row in rows})
    }
    leading_zero = {
        "y_lt_1000": metrics([row for row in rows if row["y"] < 1000]),
        "y_lt_100": metrics([row for row in rows if row["y"] < 100]),
        "y_lt_10": metrics([row for row in rows if row["y"] < 10]),
    }
    return {"overall": overall, "by_addend": by_addend, "by_carry": by_carry,
            "by_addend_carry": by_addend_carry, "leading_zero_subsets": leading_zero}


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {f"p:{name}": value.detach().cpu().clone() for name, value in model.named_parameters()} | {
        f"b:{name}": value.detach().cpu().clone() for name, value in model.named_buffers()
    }


def run_c1(data_root: str | Path, metrics_path: str | Path, checkpoint_path: str | Path,
           provenance_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(device)
    dataset_provenance = verify_dataset(data_root)
    optimization_opens: list[str] = []
    train_dataset = load_projected_train(data_root, optimization_opens)
    if set(train_dataset[0]) != {"input_ids", "labels"} or set(train_dataset[0]) & FORBIDDEN_PROJECTED_FIELDS:
        raise RuntimeError("optimization projection leaked metadata")
    model = S1GRUTransducer(CONFIG.embedding_dim, CONFIG.hidden_dim).to(dev)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as output:
        steps, elapsed = train_fixed(model, train_dataset, CONFIG, dev, output)
        # Freeze final weights before any evaluation metadata is opened.
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": asdict(CONFIG),
                    "parameter_count": parameter_count, "final_epoch": CONFIG.epochs,
                    "optimizer_steps": steps}, checkpoint_path)

        evaluation_opens: list[str] = []
        train_rows = load_eval_rows(data_root, "train", evaluation_opens)
        test_state_rows = load_eval_rows(data_root, "test_state", evaluation_opens)
        test_context_rows = load_eval_rows(data_root, "test_context", evaluation_opens)
        evaluations = {
            "train": evaluate_rows(model, train_rows, CONFIG.eval_batch_size, dev),
            "test_state": evaluate_rows(model, test_state_rows, CONFIG.eval_batch_size, dev),
            "test_context": evaluate_rows(model, test_context_rows, CONFIG.eval_batch_size, dev),
        }
        test_state_predictions = _predict(model, test_state_rows, CONFIG.eval_batch_size, dev)
        test_context_predictions = _predict(model, test_context_rows, CONFIG.eval_batch_size, dev)
        first_5000_state_errors = sum(pred != row["labels"]
                                      for pred, row in zip(test_state_predictions[:5000], test_state_rows[:5000]))
        first_5000_context_errors = sum(pred != row["labels"]
                                        for pred, row in zip(test_context_predictions[:5000], test_context_rows[:5000]))
        first_context = test_context_rows[:5000]
        first_context_mixture = {
            "per_a": {str(a): sum(row["a"] == a for row in first_context) for a in (3, 6, 9)},
            "per_carry": {str(c): sum(row["carry_length"] == c for row in first_context) for c in range(5)},
        }

        gates = {
            "train_20004_exact": evaluations["train"]["overall"]["exact_correct"] == 20004,
            "test_state_at_most_5_errors": evaluations["test_state"]["overall"]["count"]
                                               - evaluations["test_state"]["overall"]["exact_correct"] <= 5,
            "test_state_c2_204_exact": evaluations["test_state"]["by_carry"]["2"]["exact_correct"] == 204,
            "test_state_c3_34_exact": evaluations["test_state"]["by_carry"]["3"]["exact_correct"] == 34,
            "test_state_c4_5_exact": evaluations["test_state"]["by_carry"]["4"]["exact_correct"] == 5,
            "anti_leakage": optimization_opens == ["train.jsonl"],
        }
        before = _snapshot(model)
        sample = torch.tensor([row["input_ids"] for row in test_state_rows[:128]], device=dev)
        direct = model(sample)
        harness_logits = OneStepHarness(model).run(sample)
        repeated = model(sample)
        purity = {
            "direct_harness_logits_bitwise_equal": torch.equal(direct, harness_logits),
            "repeated_logits_bitwise_equal": torch.equal(direct, repeated),
            "parameters_buffers_unchanged": all(torch.equal(value, _snapshot(model)[name]) for name, value in before.items()),
            "step_instances": sum(module is model.step for module in model.modules()),
            "step_signature": list(inspect.signature(model.step.forward).parameters),
        }
        gates["api_eval_purity"] = all((purity["direct_harness_logits_bitwise_equal"],
                                         purity["repeated_logits_bitwise_equal"],
                                         purity["parameters_buffers_unchanged"],
                                         purity["step_instances"] == 1,
                                         purity["step_signature"] == ["state", "context"]))
        gates["g1_pass"] = all(gates.values())

        repo_root = Path(__file__).resolve().parents[1]
        test_source = repo_root / "tests" / "test_s1_c1.py"
        provenance = {
            "control": "s1_c1_gru_transducer_v1", "config": asdict(CONFIG),
            "final_epoch": CONFIG.epochs, "optimizer_steps": steps,
            "parameter_count": parameter_count, "seed": CONFIG.seed,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "trainer_sha256": _sha256(Path(__file__)), "trainer_git_status": _git_status(repo_root, Path(__file__)),
            "test_sha256": _sha256(test_source), "test_git_status": _git_status(repo_root, test_source),
            "dataset_config_sha256": dataset_provenance["config_sha256"],
            "dataset_manifest_sha256": dataset_provenance["manifest_sha256"],
            "dataset_generator_sha256": json.loads((Path(data_root) / "dataset_config.json").read_text(encoding="utf-8"))["generator_sha256"],
            "dataset_verified_file_sha256": dataset_provenance["verified_file_sha256"],
            "optimization_files_opened": optimization_opens,
            "post_freeze_evaluation_files_opened": evaluation_opens,
            "no_depth_files_opened": not any("depth" in name for name in optimization_opens + evaluation_opens),
            "device": str(dev), "dtype": str(next(model.parameters()).dtype),
            "no_scheduler": True, "outer_step_applications": 1,
        }
        provenance_path = Path(provenance_path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        final = {
            "kind": "final", "control": "s1_c1_gru_transducer_v1", "config": asdict(CONFIG),
            "parameter_count": parameter_count, "optimizer_steps": steps,
            "elapsed_training_seconds": elapsed, "evaluations": evaluations,
            "first_5000_test_state_errors": first_5000_state_errors,
            "first_5000_test_context_errors": first_5000_context_errors,
            "first_5000_test_context_mixture": first_context_mixture,
            "a1_c4_test_support": 0, "gates": gates, "purity": purity,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": provenance["checkpoint_sha256"],
            "provenance": str(provenance_path),
        }
        output.write(json.dumps(final, sort_keys=True) + "\n")
    return final


@torch.no_grad()
def _predict(model: nn.Module, rows: Sequence[dict[str, Any]], batch_size: int,
             device: torch.device) -> list[list[int]]:
    result = []
    for start in range(0, len(rows), batch_size):
        ids = torch.tensor([row["input_ids"] for row in rows[start:start + batch_size]], device=device)
        result.extend(model(ids).argmax(-1).cpu().tolist())
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the single declared S1 C1/G1 control")
    parser.add_argument("--data-root", default="data/generated/s1_addcarry_v1")
    parser.add_argument("--metrics", default="metrics_s1_c1_gru_seed76.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/s1_c1_gru_seed76.pt")
    parser.add_argument("--provenance", default="artifacts/s1_c1_gru_seed76_provenance.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_c1(args.data_root, args.metrics, args.checkpoint, args.provenance, args.device), indent=2))


if __name__ == "__main__":
    cli()
