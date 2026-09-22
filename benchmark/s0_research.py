"""Isolated S0 Track-A harness and the declared C1/G1 positive control.

This module deliberately has no dependency on the public runner or submission API.
Model-visible arguments are only a canonical state and an immutable permutation
table; horizons and macrostep indices remain evaluator metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def decimal_digits(value: int) -> tuple[int, int]:
    """Return the canonical, fixed-width, LSD-first decimal representation."""
    if not 0 <= value <= 99:
        raise ValueError("S0 states must fit in two decimal digits")
    return value % 10, value // 10


def decode_digits(digits: Sequence[int]) -> int:
    if len(digits) != 2 or any(not 0 <= int(d) <= 9 for d in digits):
        raise ValueError("state must be exactly two decimal digits")
    return int(digits[0]) + 10 * int(digits[1])


def load_split(data_root: str | Path, split: str) -> list[dict[str, Any]]:
    """Load one generated split without involving a public dataset factory."""
    path = Path(data_root) / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PointerStep(nn.Module):
    """The single shared macrostep module used at every Track-A iteration."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width)

    def forward(self, state: torch.Tensor, context: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        keys, values = context
        scores = torch.einsum("bd,bmd->bm", self.query(state), keys) / math.sqrt(keys.shape[-1])
        return torch.einsum("bm,bmd->bd", scores.softmax(dim=-1), values)


class C1PointerControl(nn.Module):
    """Learned feedforward pointer attention; it contains no permutation indexing.

    Position keys, state queries, and table-value representations are learned.
    Consequently the answer is obtained by soft pointer attention over the entire
    encoded table, rather than by indexing/gathering with ``x``.
    """

    def __init__(self, width: int = 64, max_size: int = 24) -> None:
        super().__init__()
        self.width = width
        self.state_digit = nn.Embedding(10, width)
        self.state_encoder = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(max_size, width) / math.sqrt(width))
        self.context_digit = nn.Embedding(10, width)
        self.context_encoder = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.step = PointerStep(width)
        self.digit_heads = nn.ModuleList([nn.Linear(width, 10), nn.Linear(width, 10)])

    def encode_state(self, canonical_digits: torch.Tensor) -> torch.Tensor:
        if canonical_digits.ndim != 2 or canonical_digits.shape[1] != 2:
            raise ValueError("encode_state expects [batch, 2] LSD-first digits")
        return self.state_encoder(self.state_digit(canonical_digits).flatten(1))

    def encode_context(self, context_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context_table.ndim != 2 or context_table.shape[1] > self.position.shape[0]:
            raise ValueError("encode_context expects [batch, M], M <= max_size")
        digits = torch.stack((context_table.remainder(10), context_table.div(10, rounding_mode="floor")), -1)
        values = self.context_encoder(self.context_digit(digits).flatten(2))
        keys = self.position[: context_table.shape[1]].unsqueeze(0).expand(context_table.shape[0], -1, -1)
        return keys, values

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(state) for head in self.digit_heads], dim=1)

    def forward(self, state_digits: torch.Tensor, context_table: torch.Tensor) -> torch.Tensor:
        context = self.encode_context(context_table)
        state = self.encode_state(state_digits)
        return self.readout(self.step(state, context))


class TrackAEvaluator:
    """Strict evaluator that destroys latent/scratch state after every macrostep."""

    def __init__(self, model: nn.Module) -> None:
        required = ("encode_context", "encode_state", "step", "readout")
        if any(not hasattr(model, name) for name in required):
            raise TypeError(f"Track-A models must implement {required}")
        signature = inspect.signature(model.step.forward if isinstance(model.step, nn.Module) else model.step)
        if tuple(signature.parameters) != ("state", "context"):
            raise TypeError("step must accept exactly (state, context); T/index are forbidden")
        self.model = model
        self.step_identity = id(model.step)

    @torch.no_grad()
    def run(self, canonical_digits: torch.Tensor, context_table: torch.Tensor, horizon: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if horizon < 0:
            raise ValueError("horizon must be nonnegative")
        context = self.model.encode_context(context_table)  # exactly once
        state = self.model.encode_state(canonical_digits)
        scores: list[torch.Tensor] = []
        for _ in range(horizon):
            if id(self.model.step) != self.step_identity:
                raise AssertionError("step module identity changed across macrosteps")
            state = self.model.step(state, context)
            logits = self.model.readout(state)
            scores.append(logits)
            decoded = logits.argmax(dim=-1)
            # Canonical hard decode/re-encode is the sole state crossing the boundary.
            values = decoded[:, 0] + 10 * decoded[:, 1]
            canonical = torch.stack((values.remainder(10), values.div(10, rounding_mode="floor")), 1)
            state = self.model.encode_state(canonical)
        return (scores[-1].argmax(dim=-1) if scores else canonical_digits.clone()), scores


@dataclass(frozen=True)
class C1Config:
    seed: int = 74
    width: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_steps: int = 1200
    eval_batch_size: int = 1024
    curve_every: int = 50


@dataclass(frozen=True)
class C2Config:
    """Predeclared independent eligible-control replication."""

    seed: int = 75
    width: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_steps: int = 1200
    eval_batch_size: int = 1024
    curve_every: int = 50


FULL_CONFIG = C1Config()
SMOKE_CONFIG = C1Config(width=16, batch_size=8, max_steps=2, eval_batch_size=32, curve_every=1)
C2_FULL_CONFIG = C2Config()


def _tensors(rows: Sequence[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, ...]:
    states = torch.tensor([decimal_digits(r["x0"]) for r in rows], device=device)
    tables = torch.tensor([r["perm"] for r in rows], device=device)
    labels = torch.tensor([decimal_digits(r["y"]) for r in rows], device=device)
    return states, tables, labels


@torch.no_grad()
def _evaluate(model: nn.Module, rows: Sequence[dict[str, Any]], batch: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    correct = [0, 0]
    ce = [0.0, 0.0]
    exact = 0
    predictions: list[list[int]] = []
    for start in range(0, len(rows), batch):
        state, table, labels = _tensors(rows[start:start + batch], device)
        logits = model(state, table)
        pred = logits.argmax(-1)
        predictions.extend(pred.cpu().tolist())
        exact += (pred == labels).all(1).sum().item()
        for digit in range(2):
            correct[digit] += (pred[:, digit] == labels[:, digit]).sum().item()
            ce[digit] += F.cross_entropy(logits[:, digit], labels[:, digit], reduction="sum").item()
    n = len(rows)
    return {"exact": exact / n, "digit_ce": [v / n for v in ce],
            "digit_accuracy": [v / n for v in correct], "predictions": predictions}


def _without_predictions(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "predictions"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_c1_g1(data_root: str | Path, metrics_path: str | Path, *, smoke: bool = False,
              device: str = "cpu") -> dict[str, Any]:
    """Run the one declared C1 experiment (or its deterministic unit-test smoke)."""
    cfg = SMOKE_CONFIG if smoke else FULL_CONFIG
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(device)
    train = load_split(data_root, "train")
    test_state = load_split(data_root, "test_state")
    test_context = load_split(data_root, "test_context")
    model = C1PointerControl(cfg.width).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    started = time.perf_counter()
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as output:
        for step in range(1, cfg.max_steps + 1):
            indices = torch.randint(len(train), (cfg.batch_size,), generator=generator).tolist()
            state, table, labels = _tensors([train[i] for i in indices], dev)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(state, table)
            loss = sum(F.cross_entropy(logits[:, d], labels[:, d]) for d in range(2))
            loss.backward()
            optimizer.step()
            if step % cfg.curve_every == 0 or step == cfg.max_steps:
                output.write(json.dumps({"kind": "curve", "step": step, "loss": loss.item()}) + "\n")
        results = {name: _evaluate(model, rows, cfg.eval_batch_size, dev) for name, rows in
                   (("train", train), ("test_state", test_state), ("test_context", test_context))}
        errors = sum(results["test_state"]["predictions"][i] != list(decimal_digits(test_state[i]["y"]))
                     for i in range(min(5000, len(test_state))))
        counts: dict[int, int] = {}
        for row in test_state:
            counts[row["y"]] = counts.get(row["y"], 0) + 1
        baseline = max(counts.values()) / len(test_state)
        for result in results.values():
            result.pop("predictions")
        final = {"kind": "final", "config": asdict(cfg), "parameter_count": sum(p.numel() for p in model.parameters()),
                 "elapsed_seconds": time.perf_counter() - started, "metrics": results,
                 "first_5000_test_state_errors": errors, "marginal_baseline": baseline,
                 "g1_pass": len(test_state) >= 5000 and errors <= 5}
        output.write(json.dumps(final, sort_keys=True) + "\n")
    return final


def run_c2_g2(data_root: str | Path, metrics_path: str | Path, checkpoint_path: str | Path,
              provenance_path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Run and freeze the sole predeclared seed-75 C2/G2 replication.

    Only the canonical T1 train/test files are loaded. Depth metadata and files are
    not consumed by this function.
    """
    cfg = C2_FULL_CONFIG
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(device)
    data_root = Path(data_root)
    train = load_split(data_root, "train")
    test_state = load_split(data_root, "test_state")
    test_context = load_split(data_root, "test_context")
    ood_size = load_split(data_root, "ood_size")
    model = C1PointerControl(cfg.width).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    started = time.perf_counter()
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as output:
        for step in range(1, cfg.max_steps + 1):
            indices = torch.randint(len(train), (cfg.batch_size,), generator=generator).tolist()
            state, table, labels = _tensors([train[i] for i in indices], dev)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # Strict eligible API: encode once, apply exactly one shared step, read out.
            context = model.encode_context(table)
            latent = model.encode_state(state)
            logits = model.readout(model.step(latent, context))
            loss = sum(F.cross_entropy(logits[:, digit], labels[:, digit]) for digit in range(2))
            loss.backward()
            optimizer.step()
            if step % cfg.curve_every == 0 or step == cfg.max_steps:
                output.write(json.dumps({"kind": "curve", "step": step, "loss": loss.item()}) + "\n")

        split_rows = {"train": train, "test_state": test_state, "test_context": test_context,
                      "ood_size_t1_diagnostic": ood_size}
        raw = {name: _evaluate(model, rows, cfg.eval_batch_size, dev)
               for name, rows in split_rows.items()}
        first_5000_errors = {
            name: sum(raw[name]["predictions"][index] != list(decimal_digits(rows[index]["y"]))
                      for index in range(5000))
            for name, rows in (("test_state", test_state), ("test_context", test_context))
        }
        leading_zero: dict[str, Any] = {}
        for name, rows in split_rows.items():
            subset = [row for row in rows if row["y"] < 10]
            leading_zero[name] = {
                "count": len(subset),
                **_without_predictions(_evaluate(model, subset, cfg.eval_batch_size, dev)),
            }
        results = {name: _without_predictions(result) for name, result in raw.items()}
        g2_pass = (
            first_5000_errors["test_state"] <= 10
            and first_5000_errors["test_context"] <= 10
            and all(value < 0.01 for name in ("test_state", "test_context")
                    for value in results[name]["digit_ce"])
        )

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                    "parameter_count": parameter_count, "final_train_step": cfg.max_steps}, checkpoint_path)
        manifest_path = data_root / "artifact_manifest.json"
        source_paths = [Path(__file__), Path(__file__).parents[1] / "data" / "s0_perm_v1.py"]
        provenance = {
            "control": "C2",
            "config": asdict(cfg),
            "parameter_count": parameter_count,
            "final_train_step": cfg.max_steps,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "source_sha256": {str(path.relative_to(Path(__file__).parents[1])).replace("\\", "/"): _sha256(path)
                              for path in source_paths},
            "dataset_config_sha256": _sha256(data_root / "dataset_config.json"),
            "dataset_manifest_sha256": _sha256(manifest_path),
            "dataset_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        }
        provenance_path = Path(provenance_path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        final = {"kind": "final", "control": "C2", "config": asdict(cfg),
                 "parameter_count": parameter_count, "elapsed_seconds": time.perf_counter() - started,
                 "metrics": results, "leading_zero_metrics": leading_zero,
                 "first_5000_errors": first_5000_errors, "g2_pass": g2_pass,
                 "checkpoint": str(checkpoint_path), "checkpoint_sha256": provenance["checkpoint_sha256"],
                 "provenance": str(provenance_path)}
        output.write(json.dumps(final, sort_keys=True) + "\n")
    return final


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated declared S0 C1/G1 control")
    parser.add_argument("data_root")
    parser.add_argument("metrics_path")
    parser.add_argument("--smoke", action="store_true", help="two-step deterministic test config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--c2", action="store_true", help="run the predeclared seed-75 C2/G2 replication")
    parser.add_argument("--checkpoint", default="artifacts/s0_perm_v1_c2_seed75.pt")
    parser.add_argument("--provenance", default="artifacts/s0_perm_v1_c2_seed75_provenance.json")
    args = parser.parse_args()
    if args.c2:
        if args.smoke:
            parser.error("--smoke and --c2 cannot be combined")
        result = run_c2_g2(args.data_root, args.metrics_path, args.checkpoint, args.provenance,
                           device=args.device)
    else:
        result = run_c1_g1(args.data_root, args.metrics_path, smoke=args.smoke, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
