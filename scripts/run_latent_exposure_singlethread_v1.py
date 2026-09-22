"""Deterministic, single-thread latent exposure protocol (version 1)."""

from __future__ import annotations

import os

# These must precede torch's import (and therefore creation of either BLAS pool).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

# This block must precede every project/model import and all tensor operations.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
assert torch.are_deterministic_algorithms_enabled()

import argparse
import hashlib
import json
import platform
import random
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements  # noqa: E402
from benchmark.manifest import load_manifest  # noqa: E402
from benchmark.runner import _evaluate, _loss_and_accuracy  # noqa: E402
from benchmark.validation import assert_state_versions_unchanged, capture_state_versions  # noqa: E402
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders  # noqa: E402
import competition_loop2_latent as candidate  # noqa: E402

MANIFEST = ROOT / "benchmark/manifests/local_competition_c1_e1_cpu.json"
SOURCE = ROOT / "competition_loop2_latent.py"
DATASET_EXPECTED = "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123"
ARTIFACTS = ROOT / "artifacts"
RESULT = ARTIFACTS / "latent_exposure_singlethread_v1_results.json"
FINAL = ARTIFACTS / "latent_exposure_singlethread_v1_final.pt"
MILESTONES = (200, 400, 800, 1600)
MODEL_SEED, LOADER_SEED = 74, 45


def thread_settings() -> dict[str, Any]:
    settings = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    assert settings == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "torch_num_threads": 1, "torch_num_interop_threads": 1,
                        "deterministic_algorithms": True}
    return settings


def sha256(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    """Unambiguous recursive encoding; independent of torch.save containers."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
        return (b"tensor:" + str(tensor.dtype).encode() + b":" +
                json.dumps(list(tensor.shape), separators=(",", ":")).encode() +
                b":" + struct.pack(">Q", len(raw)) + raw)
    if value is None:
        return b"none"
    if isinstance(value, bool):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"float64:" + struct.pack(">d", value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"str:" + struct.pack(">Q", len(raw)) + raw
    if isinstance(value, bytes):
        return b"bytes:" + struct.pack(">Q", len(value)) + value
    if isinstance(value, np.generic):
        return b"numpy-scalar:" + str(value.dtype).encode() + b":" + value.tobytes()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (b"numpy-array:" + str(array.dtype).encode() + b":" +
                json.dumps(list(array.shape), separators=(",", ":")).encode() + b":" + array.tobytes())
    if isinstance(value, dict):
        pairs = [(_canonical_bytes(k), _canonical_bytes(v)) for k, v in value.items()]
        pairs.sort(key=lambda pair: pair[0])
        return b"mapping:" + b"".join(struct.pack(">Q", len(k)) + k +
                                      struct.pack(">Q", len(v)) + v for k, v in pairs)
    if isinstance(value, (list, tuple)):
        tag = b"list:" if isinstance(value, list) else b"tuple:"
        return tag + b"".join(struct.pack(">Q", len(item)) + item
                              for item in map(_canonical_bytes, value))
    raise TypeError(f"unsupported canonical hash type: {type(value)!r}")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def seed_all(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8 emits no BOM; replace is atomic on the same filesystem.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def capacity_preflight(spec: ModelSpec) -> dict[str, Any]:
    """The frozen mixed T1/T2/T3 d32 AdamW(3e-3) capacity check."""
    x = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8],
                      [2, 10, 9, 10, 3, 15, 16, 4, 9],
                      [2, 10, 9, 10, 3, 11, 9, 4, 10]])
    labels = torch.tensor([[9, 9, 12], [8, 10, 14], [9, 16, 7]])
    positions = torch.tensor([[6, 7, 8]] * 3)
    batch_indices = torch.arange(3)[:, None]
    seed_all()
    model = candidate.Model(spec, d_model=32, heads=4).float().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for update in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        selected = model(x)[0][batch_indices, positions]
        loss = F.cross_entropy(selected.flatten(0, 1), labels.flatten())
        loss.backward()
        optimizer.step()
        if selected.argmax(-1).eq(labels).all():
            break
    train_correct = int(selected.argmax(-1).eq(labels).all(1).sum())
    versions = capture_state_versions(model)
    model.eval()
    with torch.no_grad():
        evaluated = model(x)[0][batch_indices, positions]
    assert_state_versions_unchanged(model, versions)
    result = {"rows": 3, "true_t": [1, 2, 3], "d_model": 32,
              "optimizer": "AdamW(lr=3e-3)", "maximum_updates": 100,
              "updates_run": update, "train_correct": train_correct,
              "same_weight_eval_correct": int(evaluated.argmax(-1).eq(labels).all(1).sum()),
              "final_cross_entropy": float(loss.detach()), "state_unchanged": True}
    if result["train_correct"] != 3 or result["same_weight_eval_correct"] != 3:
        raise RuntimeError(f"capacity preflight failed: {result}")
    return result


def setup():
    settings = thread_settings()
    manifest = load_manifest(MANIFEST)
    if (manifest.data.seed, manifest.data.batch_size, manifest.data.data_root) != (
            LOADER_SEED, 64, DATASET_EXPECTED):
        raise RuntimeError("frozen E1 manifest/data contract drift")
    spec = ModelSpec(infer_vocab_size(manifest.data), infer_max_seq_len(manifest.data),
                     manifest.model_state.maximum_elements)
    preflight = capacity_preflight(spec)
    # Official state is always fresh and is created only after preflight.
    loaders = make_dataloaders(manifest.data, device=torch.device("cpu"))
    seed_all(MODEL_SEED)
    model = candidate.build_model(spec).float().train()
    bundle = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
    loaders["train"].generator.manual_seed(LOADER_SEED)  # immediately before iterator
    epoch_start = loaders["train"].generator.get_state().clone()
    iterator = iter(loaders["train"])
    return manifest, loaders, model, bundle, iterator, epoch_start, preflight, settings


def batch_digest(batch: dict[str, torch.Tensor]) -> str:
    target_key = "targets" if "targets" in batch else "labels"
    return canonical_digest({"input_ids": batch["input_ids"], "targets": batch[target_key]})


def train_update(model, bundle, batch, manifest):
    bundle.optimizer.zero_grad(set_to_none=True)
    loss, exact, count, labels = _loss_and_accuracy(model, batch, manifest, torch.device("cpu"))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = max(group["lr"] for group in bundle.optimizer.param_groups)  # before step
    bundle.optimizer.step()
    if bundle.scheduler is not None:
        bundle.scheduler.step()
    return float(loss.detach()), exact, count, labels, float(grad), lr


def probe(output: Path) -> dict[str, Any]:
    manifest, loaders, model, bundle, iterator, epoch_start, preflight, settings = setup()
    losses, batches = [], []
    epoch = consumed = examples = 0
    batches_per_epoch = len(loaders["train"])
    for _update in range(1, 21):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            consumed = 0
            epoch_start = loaders["train"].generator.get_state().clone()
            iterator = iter(loaders["train"])
            batch = next(iterator)
        batches.append(batch_digest(batch))
        loss, _, count, _, _, _ = train_update(model, bundle, batch, manifest)
        losses.append(loss)
        consumed += 1
        examples += count
    payload = {
        "protocol": "latent_exposure_singlethread_v1", "settings": settings,
        "preflight": preflight, "losses": losses, "ordered_batch_digests": batches,
        "model_state_digest": canonical_digest(model.state_dict()),
        "optimizer_state_digest": canonical_digest(bundle.optimizer.state_dict()),
        "scheduler_state_digest": canonical_digest(None if bundle.scheduler is None else bundle.scheduler.state_dict()),
        "loader_generator_digest": canonical_digest(loaders["train"].generator.get_state()),
        "position": {"update": 20, "examples": examples, "epoch_index": epoch,
                     "batches_consumed_in_current_epoch": consumed,
                     "batches_per_epoch": batches_per_epoch,
                     "epoch_start_generator_digest": canonical_digest(epoch_start)},
    }
    artifact = {"comparison_payload": payload}
    write_json(output, artifact)
    return artifact


def subset_loader(loader, indices) -> DataLoader:
    return DataLoader(Subset(loader.dataset, indices), batch_size=128, shuffle=False,
                      drop_last=False, collate_fn=loader.collate_fn, num_workers=0)


def evaluate(model, loader, manifest) -> dict[str, Any]:
    versions = capture_state_versions(model)
    was_training = model.training
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    started = time.perf_counter()
    try:
        metrics = _evaluate(model, loader, manifest, torch.device("cpu"),
                            deadline=float("inf"), budget_seconds=float("inf"))
        assert_state_versions_unchanged(model, versions)
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(torch_rng)
        model.train()
    metrics.update(model_state_unchanged=True, rng_state_restored=True,
                   entered_from_training=was_training,
                   diagnostic_seconds=time.perf_counter() - started)
    return metrics


def checkpoint(model, bundle, loader, update, examples, epoch, consumed,
               batches_per_epoch, epoch_start) -> dict[str, Any]:
    return {
        "protocol": "latent_exposure_singlethread_v1",
        "model": model.state_dict(), "optimizer": bundle.optimizer.state_dict(),
        "scheduler": None if bundle.scheduler is None else bundle.scheduler.state_dict(),
        "train_loader_generator_state": loader.generator.get_state(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state()},
        "update": update, "examples": examples,
        "iterator_position": {
            "epoch_index": epoch, "batches_consumed_in_current_epoch": consumed,
            "batches_per_epoch": batches_per_epoch,
            "epoch_start_generator_state": epoch_start,
            "resume_instructions": (
                "Restore model/optimizer/scheduler and all RNG states. Set the train-loader generator "
                "to epoch_start_generator_state, create the epoch iterator, skip exactly "
                "batches_consumed_in_current_epoch batches, verify the resulting generator state equals "
                "train_loader_generator_state, then continue at update+1 without rebuilding any state."
            ),
        },
    }


def full_run() -> dict[str, Any]:
    source_before, manifest_before = sha256(SOURCE), sha256(MANIFEST)
    manifest, loaders, model, bundle, iterator, epoch_start, preflight, settings = setup()
    dataset_path = ROOT / manifest.data.data_root
    dataset_before = sha256(dataset_path)
    train_records, test_records = loaders["train"].dataset.records, loaders["test"].dataset.records
    train_ids = {f"T{t}": [i for i, row in enumerate(train_records) if row["time_steps"] == t]
                 for t in (1, 2, 3)}
    test_ids = {f"T{t}": [i for i, row in enumerate(test_records) if row["time_steps"] == t]
                for t in (1, 2, 3)}
    if {k: len(v) for k, v in train_ids.items()} != {"T1": 200, "T2": 200, "T3": 200}:
        raise RuntimeError("train partitions drifted")
    if {k: len(v) for k, v in test_ids.items()} != {"T1": 50, "T2": 50, "T3": 50}:
        raise RuntimeError("test partitions drifted")
    old = ARTIFACTS / "competition_loop2_latent_exposure_update1600.pt"
    result: dict[str, Any] = {
        "status": "running", "protocol": "latent_exposure_singlethread_v1",
        "contract": {"fresh_update0": True, "probe_or_old_checkpoint_loaded": False,
                     "continuous_model_optimizer_scheduler_iterator": True,
                     "updates": 1600, "batch_size": 64, "loss": "full cross entropy",
                     "gradient_clip": 1.0, "optimizer": "C1B AdamW/warmup20",
                     "model_seed": MODEL_SEED, "loader_seed": LOADER_SEED,
                     "no_resets_early_stop_or_checkpoint_selection": True},
        "determinism": settings, "preflight": preflight,
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform()},
        "hashes": {"source": source_before, "harness": sha256(Path(__file__)),
                   "manifest": manifest_before, "dataset": dataset_before,
                   "milestone_states": {},
                   "historical_8thread_checkpoint": sha256(old) if old.is_file() else None},
        "partitions": {"full_train": len(train_records), "test": len(test_records),
                       "train_by_t": {k: len(v) for k, v in train_ids.items()},
                       "test_by_t": {k: len(v) for k, v in test_ids.items()}},
        "model_state_elements": count_model_state_elements(model),
        "milestones": {}, "training": {}, "diagnostics": {"elapsed_seconds": 0.0},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    epoch = consumed = examples = 0
    batches_per_epoch = len(loaders["train"])
    training_seconds = 0.0
    lr_counts: Counter[str] = Counter()
    lr_representatives: dict[str, float] = {}
    final_batch = {}
    for update in range(1, 1601):
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            consumed = 0
            epoch_start = loaders["train"].generator.get_state().clone()
            iterator = iter(loaders["train"])
            batch = next(iterator)
        loss, exact, count, labels, grad, lr = train_update(model, bundle, batch, manifest)
        training_seconds += time.perf_counter() - started
        consumed += 1
        examples += count
        key = format(lr, ".17g")
        lr_counts[key] += 1
        if update in (1, 19, 20, 21, *MILESTONES):
            lr_representatives[str(update)] = lr
        final_batch = {"cross_entropy": loss, "exact_accuracy": exact, "example_count": count,
                       "label_count": labels, "clipped_gradient_norm": grad, "lr_used": lr}
        if update not in MILESTONES:
            continue
        diagnostic_start = time.perf_counter()
        evaluation = {
            "full_train": evaluate(model, subset_loader(loaders["train"], range(len(train_records))), manifest),
            "train_by_t": {k: evaluate(model, subset_loader(loaders["train"], ids), manifest)
                           for k, ids in train_ids.items()},
        }
        diagnostic_seconds = time.perf_counter() - diagnostic_start
        state_path = ARTIFACTS / f"latent_exposure_singlethread_v1_update{update}.pt"
        torch.save(checkpoint(model, bundle, loaders["train"], update, examples, epoch, consumed,
                              batches_per_epoch, epoch_start), state_path)
        result["hashes"]["milestone_states"][str(update)] = sha256(state_path)
        result["milestones"][str(update)] = {
            "update": update, "examples": examples, "training_seconds": training_seconds,
            "diagnostic_seconds": diagnostic_seconds, "evaluation": evaluation,
            "checkpoint": str(state_path.relative_to(ROOT)), "final_batch": final_batch,
            "model_state_digest": canonical_digest(model.state_dict()),
        }
        result["training"] = {"elapsed_seconds": training_seconds, "examples": examples,
                              "lr_counts": dict(lr_counts), "lr_representatives": lr_representatives}
        result["diagnostics"]["elapsed_seconds"] += diagnostic_seconds
        write_json(RESULT, result)
    diagnostic_start = time.perf_counter()
    final_eval = {
        "test": evaluate(model, loaders["test"], manifest),
        "test_by_t": {k: evaluate(model, subset_loader(loaders["test"], ids), manifest)
                      for k, ids in test_ids.items()},
        "seen_N_T1": evaluate(model, loaders["depth_t_1"], manifest),
        "ood_T6": evaluate(model, loaders["ood"], manifest),
    }
    result["diagnostics"]["elapsed_seconds"] += time.perf_counter() - diagnostic_start
    torch.save(model.state_dict(), FINAL)
    result["hashes"]["final_weights"] = sha256(FINAL)
    result["final_evaluation"] = final_eval
    exposure = result["milestones"]["1600"]["evaluation"]["full_train"]["correct_examples"] >= 60
    seen = final_eval["seen_N_T1"]["correct_examples"] >= 4
    alternate = (final_eval["test"]["correct_examples"] > 9 and
                 final_eval["test"]["loss"] < 1.8830257671)
    result["gates"] = {
        "exact_rule": "exposure full train>=60/600; competition exposure AND (seen>=4/38 OR (test>9/150 AND CE<1.8830257671))",
        "exposure_pass": exposure, "seen_pass": seen, "test_ce_pass": alternate,
        "competition_pass": exposure and (seen or alternate),
    }
    after = {"source": sha256(SOURCE), "manifest": sha256(MANIFEST), "dataset": sha256(dataset_path)}
    result["no_mutation"] = {key: after[key] == result["hashes"][key]
                             for key in ("source", "manifest", "dataset")}
    if not all(result["no_mutation"].values()):
        raise RuntimeError(f"frozen inputs mutated: {result['no_mutation']}")
    result["status"] = "complete"
    write_json(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", metavar="OUTPUT_JSON", type=Path,
                       help="run exactly 20 updates and write one comparison JSON (no checkpoint)")
    group.add_argument("--run", action="store_true",
                       help="run a fresh continuous 1,600-update exposure protocol")
    args = parser.parse_args()
    if args.probe is not None:
        probe(args.probe)
        print(json.dumps({"status": "probe_complete", "output": str(args.probe)}))
    else:
        result = full_run()
        print(json.dumps({"status": result["status"], "result": str(RESULT),
                          "final_weights": str(FINAL)}, indent=2))


if __name__ == "__main__":
    main()
