"""Authorized S1F-C1 seed-76 fixed numeric-context intervention."""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import time
from typing import Any
from unittest.mock import patch

import torch

from benchmark.s1_c1 import (CONFIG, FORBIDDEN_PROJECTED_FIELDS, OneStepHarness,
    S1GRUTransducer, _predict, _snapshot, evaluate_rows, load_eval_rows,
    load_projected_train, train_fixed, verify_dataset)


def fixed_numeric_table() -> torch.Tensor:
    rows = []
    for d in range(10):
        rows.append([d / 9, math.sin(2*math.pi*d/10), math.cos(2*math.pi*d/10),
                     math.sin(4*math.pi*d/10), math.cos(4*math.pi*d/10),
                     math.sin(6*math.pi*d/10), math.cos(6*math.pi*d/10), 1.0] + [0.0]*24)
    return torch.tensor(rows, dtype=torch.float32)


class S1FixedNumericContextTransducer(S1GRUTransducer):
    """C1 model whose sole intervention is independent fixed context lookup."""
    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        if embedding_dim != 32:
            raise ValueError("S1F requires embedding_dim=32")
        super().__init__(embedding_dim, hidden_dim)
        self.register_buffer("fixed_context_table", fixed_numeric_table(), persistent=True)

    def encode_context(self, canonical_addend_digits: torch.Tensor) -> torch.Tensor:
        if canonical_addend_digits.ndim != 2 or canonical_addend_digits.shape[1] != 4:
            raise ValueError("encode_context expects [batch,4] canonical digits")
        return self.fixed_context_table[canonical_addend_digits]


class JsonlParseMonitor(AbstractContextManager["JsonlParseMonitor"]):
    def __init__(self) -> None:
        self.opened: list[str] = []
    def __enter__(self) -> "JsonlParseMonitor":
        original, monitor = Path.open, self
        def monitored(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path.suffix == ".jsonl": monitor.opened.append(path.name)
            return original(path, *args, **kwargs)
        self.patcher = patch.object(Path, "open", monitored); self.patcher.start(); return self
    def __exit__(self, *args: Any) -> None: self.patcher.stop()


def sha(path: str | Path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def table_sha() -> str: return hashlib.sha256(fixed_numeric_table().numpy().tobytes()).hexdigest()


def _initialization_check() -> bool:
    random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    parent = S1GRUTransducer(CONFIG.embedding_dim, CONFIG.hidden_dim)
    random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    model = S1FixedNumericContextTransducer(CONFIG.embedding_dim, CONFIG.hidden_dim)
    p, q = dict(parent.named_parameters()), dict(model.named_parameters())
    return p.keys() == q.keys() and all(torch.equal(p[n], q[n]) for n in p)


def run(data_root: str | Path, metrics_path: str | Path, checkpoint_path: str | Path,
        provenance_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    if not _initialization_check(): raise RuntimeError("parent/S1F initialization mismatch")
    random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed); torch.use_deterministic_algorithms(True)
    dev, root = torch.device(device), Path(__file__).resolve().parents[1]
    dataset_provenance = verify_dataset(data_root)  # byte hashing only
    manual: list[str] = []
    with JsonlParseMonitor() as optimization:
        train = load_projected_train(data_root, manual)
    if optimization.opened != ["train.jsonl"] or manual != optimization.opened: raise RuntimeError("parse contract")
    if set(train[0]) != {"input_ids", "labels"} or set(train[0]) & FORBIDDEN_PROJECTED_FIELDS: raise RuntimeError("projection leak")
    # Third identical reset: construction used for the one authorized optimization.
    random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
    model = S1FixedNumericContextTransducer(CONFIG.embedding_dim, CONFIG.hidden_dim).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 25930 or model.fixed_context_table.numel() != 320: raise RuntimeError("architecture count")
    metrics_path, checkpoint_path, provenance_path = map(Path, (metrics_path, checkpoint_path, provenance_path))
    metrics_path.parent.mkdir(parents=True, exist_ok=True); checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as out:
        steps, elapsed = train_fixed(model, train, CONFIG, dev, out)
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        torch.save({"state_dict": model.state_dict(), "config": asdict(CONFIG), "parameter_count": trainable,
                    "fixed_buffer_values": 320, "final_epoch": 60, "optimizer_steps": steps,
                    "experiment": "S1F-C1"}, checkpoint_path)
        opened: list[str] = []
        with JsonlParseMonitor() as evaluation:
            rows = {s: load_eval_rows(data_root, s, opened) for s in ("train", "test_state", "test_context")}
        expected = ["train.jsonl", "test_state.jsonl", "test_context.jsonl"]
        if evaluation.opened != expected or opened != expected: raise RuntimeError("evaluation parse contract")
        ev = {s: evaluate_rows(model, r, CONFIG.eval_batch_size, dev) for s, r in rows.items()}
        preds = {s: _predict(model, r, CONFIG.eval_batch_size, dev) for s, r in rows.items()}
        errors = {s: len(rows[s])-ev[s]["overall"]["exact_correct"] for s in rows}
        first5000 = len(rows["test_context"][:5000])-sum(p == r["labels"] for p,r in zip(preds["test_context"][:5000], rows["test_context"][:5000]))
        context_a_errors = {str(a): ev["test_context"]["by_addend"][str(a)]["count"]-ev["test_context"]["by_addend"][str(a)]["exact_correct"] for a in (3,6,9)}
        gates = {"train_20004_exact": errors["train"] == 0, "test_state_at_most_5_errors": errors["test_state"] <= 5,
          "test_state_c2_204_exact": ev["test_state"]["by_carry"]["2"]["exact_correct"] == 204,
          "test_state_c3_34_exact": ev["test_state"]["by_carry"]["3"]["exact_correct"] == 34,
          "test_state_c4_5_exact": ev["test_state"]["by_carry"]["4"]["exact_correct"] == 5,
          "test_context_at_most_30_errors": errors["test_context"] <= 30,
          "context_each_a_at_most_10_errors": all(x <= 10 for x in context_a_errors.values()),
          "context_c2_1620_exact": ev["test_context"]["by_carry"]["2"]["exact_correct"] == 1620,
          "context_c3_162_exact": ev["test_context"]["by_carry"]["3"]["exact_correct"] == 162,
          "context_c4_18_exact": ev["test_context"]["by_carry"]["4"]["exact_correct"] == 18,
          "first_5000_context_at_most_5_errors": first5000 <= 5, "instrumented_access": optimization.opened == ["train.jsonl"]}
        before = _snapshot(model); sample = torch.tensor([r["input_ids"] for r in rows["test_context"][:128]], device=dev)
        direct, harness, repeat = model(sample), OneStepHarness(model).run(sample), model(sample)
        purity = {"direct_harness_logits_bitwise_equal": torch.equal(direct,harness), "direct_harness_predictions_equal": torch.equal(direct.argmax(-1),harness.argmax(-1)),
                  "repeated_logits_bitwise_equal": torch.equal(direct,repeat), "state_unchanged": all(torch.equal(v,_snapshot(model)[n]) for n,v in before.items())}
        gates["implementation"] = all(purity.values()) and _initialization_check() and trainable == 25930
        gates["s1f_c1_pass"] = all(gates.values())
        test_path = root/"tests"/"test_s1f_c1.py"
        provenance = {"experiment":"S1F-C1 seed76 fixed numeric context", "config":asdict(CONFIG), "config_equal_c1":True,
          "intervention":"encode_context only: independent lookup in fixed 10x32 float32 table", "table_values":fixed_numeric_table().tolist(),
          "table_raw_float32_sha256":table_sha(), "trainable_parameter_count":trainable, "fixed_buffer_value_count":320,
          "initialization_named_trainable_bitwise_equal":_initialization_check(), "parent_source_sha256":sha(root/"benchmark"/"s1_c1.py"),
          "runner_sha256":sha(__file__), "test_sha256":sha(test_path), "dataset_manifest_sha256":dataset_provenance["manifest_sha256"],
          "dataset_config_sha256":dataset_provenance["config_sha256"], "dataset_verified_file_sha256":dataset_provenance["verified_file_sha256"],
          "optimization_parsed_jsonl":optimization.opened, "post_freeze_parsed_jsonl":evaluation.opened, "depth_jsonl_parsed":[],
          "checkpoint_sha256":sha(checkpoint_path), "final_epoch":60, "optimizer_steps":steps, "device":str(dev), "dtype":"torch.float32",
          "no_scheduler":True, "no_training_selection_retry":True, "training_curve_in_metrics":True, "purity":purity}
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
        final = {"kind":"final","experiment":"S1F-C1","config":asdict(CONFIG),"optimizer_steps":steps,"elapsed_training_seconds":elapsed,
                 "evaluations":ev,"errors":errors,"context_a_errors":context_a_errors,"first_5000_test_context_errors":first5000,
                 "gates":gates,"purity":purity,"checkpoint_sha256":provenance["checkpoint_sha256"],"provenance":str(provenance_path)}
        out.write(json.dumps(final,sort_keys=True)+"\n")
    return final


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/generated/s1_addcarry_v1"); p.add_argument("--metrics",default="metrics_s1f_c1_seed76.jsonl"); p.add_argument("--checkpoint",default="artifacts/s1f_c1_seed76.pt"); p.add_argument("--provenance",default="artifacts/s1f_c1_seed76_provenance.json"); p.add_argument("--device",default="cpu"); a=p.parse_args()
    print(json.dumps(run(a.data_root,a.metrics,a.checkpoint,a.provenance,a.device),indent=2))
