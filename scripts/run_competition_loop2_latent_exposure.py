"""Guarded 1,600-update continuous-latent exposure-curve diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements  # noqa: E402
from benchmark.manifest import load_manifest  # noqa: E402
from benchmark.runner import _evaluate, _loss_and_accuracy  # noqa: E402
from benchmark.validation import (  # noqa: E402
    assert_state_versions_unchanged,
    capture_state_versions,
)
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders  # noqa: E402
import competition_loop2_latent as candidate  # noqa: E402
from scripts.run_competition_loop2_latent import capacity_preflight  # noqa: E402

MANIFEST = ROOT / "benchmark/manifests/local_competition_c1_e1_cpu.json"
SOURCE = ROOT / "competition_loop2_latent.py"
PRIOR = ROOT / "artifacts/competition_loop2_latent_final.pt"
OUTPUT = ROOT / "artifacts/competition_loop2_latent_exposure_results.json"
ABORT = ROOT / "artifacts/competition_loop2_latent_exposure_abort.json"
FINAL = ROOT / "artifacts/competition_loop2_latent_exposure_final.pt"
MILESTONES = (200, 400, 800, 1600)
MODEL_SEED, LOADER_SEED = 74, 45


def sha256(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def seed_all(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_loader(loader, indices) -> DataLoader:
    return DataLoader(
        Subset(loader.dataset, indices), batch_size=128, shuffle=False,
        drop_last=False, collate_fn=loader.collate_fn, num_workers=0,
    )


def evaluate(model, loader, manifest) -> dict:
    versions = capture_state_versions(model)
    was_training = model.training
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    started = time.perf_counter()
    try:
        metrics = _evaluate(model, loader, manifest, torch.device("cpu"),
                            deadline=float("inf"), budget_seconds=float("inf"))
        assert_state_versions_unchanged(model, versions)
    finally:
        # Diagnostics must not perturb the continuing stochastic training stream.
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        model.train()
    metrics["model_state_unchanged"] = True
    metrics["rng_state_restored"] = True
    metrics["entered_from_training"] = was_training
    metrics["diagnostic_seconds"] = time.perf_counter() - started
    return metrics


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def checkpoint_payload(model, bundle, update, examples, loader, epoch, consumed,
                       batches_per_epoch, epoch_start_state) -> dict:
    return {
        "model": model.state_dict(), "optimizer": bundle.optimizer.state_dict(),
        "scheduler": None if bundle.scheduler is None else bundle.scheduler.state_dict(),
        "update": update, "examples": examples,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state()},
        "train_loader_generator_state": loader.generator.get_state(),
        "iterator_position": {
            "epoch_index": epoch, "batches_consumed_in_current_epoch": consumed,
            "batches_per_epoch": batches_per_epoch,
            "epoch_start_generator_state": epoch_start_state,
            "resume_instructions": (
                "Reset the loader generator to epoch_start_generator_state, recreate the epoch "
                "iterator, skip batches_consumed_in_current_epoch batches, then require its "
                "current generator state to equal train_loader_generator_state before training."
            ),
        },
    }


def verify_prior(model) -> tuple[bool, dict]:
    expected = torch.load(PRIOR, map_location="cpu", weights_only=True)
    actual = model.state_dict()
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    unequal = []
    for key in sorted(set(actual) & set(expected)):
        if not torch.equal(actual[key], expected[key]):
            unequal.append({"key": key, "current_sha256": tensor_hash(actual[key]),
                            "prior_sha256": tensor_hash(expected[key])})
    proof = {"same_keys": not missing and not extra, "missing_keys": missing,
             "extra_keys": extra, "mismatched_tensors": unequal,
             "all_tensors_bitwise_equal": not unequal}
    return not missing and not extra and not unequal, proof


def one_update_hash() -> dict:
    """Reproduce update 1 in a standalone process to test CPU determinism."""
    manifest = load_manifest(MANIFEST)
    spec = ModelSpec(infer_vocab_size(manifest.data), infer_max_seq_len(manifest.data),
                     manifest.model_state.maximum_elements)
    capacity_preflight(spec, False)
    loaders = make_dataloaders(manifest.data, device=torch.device("cpu"))
    seed_all()
    model = candidate.build_model(spec).float().train()
    bundle = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
    loaders["train"].generator.manual_seed(LOADER_SEED)
    batch = next(iter(loaders["train"]))
    bundle.optimizer.zero_grad(set_to_none=True)
    loss, _, _, _ = _loss_and_accuracy(model, batch, manifest, torch.device("cpu"))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    bundle.optimizer.step()
    if bundle.scheduler is not None:
        bundle.scheduler.step()
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {"state_sha256": digest.hexdigest(), "loss": float(loss.detach()),
            "threads": torch.get_num_threads()}


def run() -> dict:
    if not PRIOR.is_file():
        raise FileNotFoundError(f"required frozen prior checkpoint absent: {PRIOR}")
    manifest = load_manifest(MANIFEST)
    if manifest.data.seed != LOADER_SEED or manifest.data.batch_size != 64:
        raise RuntimeError("frozen manifest loader contract drift")
    prior_hash = sha256(PRIOR)
    spec = ModelSpec(infer_vocab_size(manifest.data), infer_max_seq_len(manifest.data),
                     manifest.model_state.maximum_elements)
    # Reproduce the original process, including its capacity preflight before
    # official loader creation. The original harness reseeded afterward, but
    # this call can initialize CPU kernels and thread-pool execution paths.
    preflight = capacity_preflight(spec, False)
    if preflight["train_correct"] != 3 or preflight["same_weight_eval_correct"] != 3:
        raise RuntimeError("frozen capacity preflight did not reproduce")
    loaders = make_dataloaders(manifest.data, device=torch.device("cpu"))
    seed_all()
    model = candidate.build_model(spec).float().train()
    bundle = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
    # This reset is intentionally the last operation before constructing the iterator.
    loaders["train"].generator.manual_seed(LOADER_SEED)
    epoch, consumed, batches_per_epoch = 0, 0, len(loaders["train"])
    epoch_start_state = loaders["train"].generator.get_state().clone()
    iterator = iter(loaders["train"])

    train_records = loaders["train"].dataset.records
    train_ids = {f"T{t}": [i for i, row in enumerate(train_records)
                             if row["time_steps"] == t] for t in (1, 2, 3)}
    test_records = loaders["test"].dataset.records
    test_ids = {f"T{t}": [i for i, row in enumerate(test_records)
                            if row["time_steps"] == t] for t in (1, 2, 3)}
    if {k: len(v) for k, v in train_ids.items()} != {f"T{t}": 200 for t in (1, 2, 3)}:
        raise RuntimeError("train partition drift")
    if {k: len(v) for k, v in test_ids.items()} != {f"T{t}": 50 for t in (1, 2, 3)}:
        raise RuntimeError("test partition drift")

    previous_abort = None
    if ABORT.is_file():
        previous_abort = {"artifact": str(ABORT.relative_to(ROOT)), "sha256": sha256(ABORT),
                          "reason": "update-200 bitwise mismatch when original capacity preflight was omitted"}
    result = {
        "status": "running", "attempt_history": ([previous_abort] if previous_abort else []) + [
            {"fresh_replay_from_update": 0,
             "reproduction_change": "include original capacity preflight before official loader creation"}
        ],
        "protocol": {"diagnostic_only_not_competition_budget_result": True, "device": "cpu",
                     "dtype": "float32", "model_seed": MODEL_SEED, "loader_seed": LOADER_SEED,
                     "batch_size": 64, "updates": 1600, "loss": "full cross entropy",
                     "gradient_clip": 1.0, "optimizer": "C1B AdamW/warmup20",
                     "continuous_objects": ["model", "optimizer", "scheduler", "train_iterator"],
                     "no_milestone_resets": True, "no_early_stopping": True,
                     "checkpoint_selection": "none"},
        "hashes": {"source": sha256(SOURCE), "harness": sha256(Path(__file__)),
                   "manifest": sha256(MANIFEST), "dataset": sha256(ROOT / manifest.data.data_root),
                   "prior_checkpoint": prior_hash, "milestone_states": {}},
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "torch": torch.__version__, "platform": platform.platform(),
                        "torch_num_threads": torch.get_num_threads()},
        "reproduction_proof": {"fresh_replay": True,
                               "loader_reset_immediately_before_first_iterator": True,
                               "original_capacity_preflight_replayed": preflight,
                               "model_state_elements": count_model_state_elements(model)},
        "partitions": {"train_by_t": {k: len(v) for k, v in train_ids.items()},
                       "test_by_t": {k: len(v) for k, v in test_ids.items()},
                       "full_train": len(train_records), "test": len(test_records)},
        "milestones": {}, "training": {"elapsed_seconds": 0.0, "examples": 0,
                                          "lr_representatives": {}, "lr_counts": {}},
        "diagnostics": {"elapsed_seconds": 0.0}, "no_mutation": {},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    examples, training_elapsed = 0, 0.0
    lr_counts: Counter[str] = Counter()
    lr_representatives = {}
    final_loss = final_exact = final_grad = None
    for update in range(1, 1601):
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            consumed = 0
            epoch_start_state = loaders["train"].generator.get_state().clone()
            iterator = iter(loaders["train"])
            batch = next(iterator)
        bundle.optimizer.zero_grad(set_to_none=True)
        loss, final_exact, count, labels = _loss_and_accuracy(
            model, batch, manifest, torch.device("cpu"))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        final_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = max(group["lr"] for group in bundle.optimizer.param_groups)
        lr_key = format(lr, ".17g")
        lr_counts[lr_key] += 1
        if update in (1, 19, 20, 21, *MILESTONES):
            lr_representatives[str(update)] = lr
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
        training_elapsed += time.perf_counter() - started
        consumed += 1
        examples += count
        final_loss = float(loss.detach())

        if update not in MILESTONES:
            continue
        prior_proof = None
        if update == 200:
            matches, prior_proof = verify_prior(model)
            result["reproduction_proof"]["update_200_prior_comparison"] = prior_proof
            if not matches:
                result["status"] = "aborted_before_update_201"
                result["abort"] = prior_proof
                result["training"].update(elapsed_seconds=training_elapsed, examples=examples,
                                          lr_representatives=lr_representatives,
                                          lr_counts=dict(lr_counts))
                write_json(ABORT, result)
                write_json(OUTPUT, result)
                raise RuntimeError(f"update-200 replay mismatch; evidence: {ABORT}")

        diagnostic_started = time.perf_counter()
        train_eval = {
            "full_train": evaluate(model, subset_loader(loaders["train"], range(len(train_records))), manifest),
            "train_by_t": {key: evaluate(model, subset_loader(loaders["train"], ids), manifest)
                           for key, ids in train_ids.items()},
        }
        diagnostic_elapsed = time.perf_counter() - diagnostic_started
        state_path = ROOT / f"artifacts/competition_loop2_latent_exposure_update{update}.pt"
        torch.save(checkpoint_payload(model, bundle, update, examples, loaders["train"], epoch,
                                      consumed, batches_per_epoch, epoch_start_state), state_path)
        result["hashes"]["milestone_states"][str(update)] = sha256(state_path)
        result["milestones"][str(update)] = {
            "update": update, "examples": examples, "elapsed_training_seconds": training_elapsed,
            "diagnostic_seconds": diagnostic_elapsed, "evaluation": train_eval,
            "state_path": str(state_path.relative_to(ROOT)), "lr_used": lr,
            "final_batch": {"cross_entropy": final_loss, "exact_accuracy": final_exact,
                            "example_count": count, "label_count": labels,
                            "clipped_gradient_norm": float(final_grad)},
        }
        result["training"].update(elapsed_seconds=training_elapsed, examples=examples,
                                  lr_representatives=lr_representatives,
                                  lr_counts=dict(lr_counts))
        result["diagnostics"]["elapsed_seconds"] += diagnostic_elapsed
        result["no_mutation"][str(update)] = all(
            metrics["model_state_unchanged"] for metrics in
            [train_eval["full_train"], *train_eval["train_by_t"].values()])
        write_json(OUTPUT, result)  # durable evidence before any later work

    diagnostic_started = time.perf_counter()
    final_eval = {
        "test": evaluate(model, loaders["test"], manifest),
        "test_by_t": {key: evaluate(model, subset_loader(loaders["test"], ids), manifest)
                      for key, ids in test_ids.items()},
        "seen_N_T1": evaluate(model, loaders["depth_t_1"], manifest),
        "ood_T6": evaluate(model, loaders["ood"], manifest),
    }
    final_diagnostic = time.perf_counter() - diagnostic_started
    torch.save(model.state_dict(), FINAL)
    if sha256(PRIOR) != prior_hash:
        raise RuntimeError("frozen prior checkpoint changed during run")
    result["hashes"]["final_weights"] = sha256(FINAL)
    result["final_evaluation"] = final_eval
    result["diagnostics"]["elapsed_seconds"] += final_diagnostic
    exposure = result["milestones"]["1600"]["evaluation"]["full_train"]["correct_examples"] >= 60
    seen = final_eval["seen_N_T1"]["correct_examples"] >= 4
    test = final_eval["test"]
    alternate = test["correct_examples"] > 9 and test["loss"] < 1.8830257671
    result["gate"] = {
        "predeclared": "exposure: train>=60/600; competition: exposure AND (seen T1>=4/38 OR (test>9/150 AND CE<1.8830257671))",
        "exposure_pass": exposure, "seen_t1_pass": seen, "test_ce_pass": alternate,
        "competition_pass": exposure and (seen or alternate),
    }
    result["status"] = "complete"
    result["attempt_history"][-1].update(completed_update=1600, status="complete")
    write_json(OUTPUT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="execute the guarded 1,600-update diagnostic")
    group.add_argument("--one-update-hash", action="store_true",
                       help="print a standalone update-1 determinism hash; write no artifact")
    args = parser.parse_args()
    if args.one_update_hash:
        print(json.dumps(one_update_hash(), sort_keys=True))
        return
    result = run()
    print(json.dumps({"status": result["status"], "output": str(OUTPUT),
                      "final_weights": str(FINAL)}, indent=2))


if __name__ == "__main__":
    main()
