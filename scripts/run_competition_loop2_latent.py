"""Guarded fixed screen for the Loop-2 continuous latent candidate."""

from __future__ import annotations

import argparse, hashlib, json, platform, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements  # noqa: E402
from benchmark.manifest import load_manifest  # noqa: E402
from benchmark.runner import _evaluate, _loss_and_accuracy  # noqa: E402
from benchmark.validation import capture_state_versions, assert_state_versions_unchanged  # noqa: E402
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders  # noqa: E402
import competition_loop2_latent as candidate  # noqa: E402

MANIFEST = ROOT / "benchmark/manifests/local_competition_c1_e1_cpu.json"
OUTPUT = ROOT / "artifacts/competition_loop2_latent_results.json"
CHECKPOINT = ROOT / "artifacts/competition_loop2_latent_final.pt"
MODEL_SEED, LOADER_SEED = 74, 45


def seed_all(seed=MODEL_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256(path):
    if path.is_file(): return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode()); digest.update(child.read_bytes())
    return digest.hexdigest()


def subset_loader(loader, indices, batch_size=128):
    return DataLoader(Subset(loader.dataset, indices), batch_size=batch_size, shuffle=False,
                      drop_last=False, collate_fn=loader.collate_fn, num_workers=0)


def train(model, loader, manifest, updates):
    bundle = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
    iterator, examples, lrs, start = iter(loader), 0, [], time.perf_counter()
    final = None
    for _ in range(updates):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        bundle.optimizer.zero_grad(set_to_none=True)
        final, exact, count, labels = _loss_and_accuracy(model, batch, manifest, torch.device("cpu"))
        if not torch.isfinite(final): raise FloatingPointError("non-finite training loss")
        final.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # LR used by this update must be captured before optimizer.step.
        lrs.append(max(group["lr"] for group in bundle.optimizer.param_groups))
        bundle.optimizer.step()
        if bundle.scheduler: bundle.scheduler.step()
        examples += count
    elapsed = time.perf_counter() - start
    return {"updates": updates, "examples": examples, "elapsed_seconds": elapsed,
            "updates_per_second": updates / elapsed, "examples_per_second": examples / elapsed,
            "lr_used_by_update": lrs, "update_1_lr": lrs[0],
            "below_peak_lr_updates": sum(lr < 8e-4 for lr in lrs),
            "peak_lr_updates": sum(lr == 8e-4 for lr in lrs), "peak_lr": max(lrs),
            "final_batch": {"cross_entropy": float(final.detach()), "exact_accuracy": exact,
                            "correct_examples": int(round(exact * count)), "example_count": count,
                            "label_count": labels, "clipped_gradient_norm": float(grad_norm)}}


def evaluate(model, loader, manifest):
    versions = capture_state_versions(model); model.eval()
    metrics = _evaluate(
        model,
        loader,
        manifest,
        torch.device("cpu"),
        deadline=float("inf"),
        budget_seconds=float("inf"),
    )
    assert_state_versions_unchanged(model, versions); metrics["state_unchanged"] = True
    return metrics


def capacity_preflight(spec, smoke):
    x = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8],
                      [2, 10, 9, 10, 3, 15, 16, 4, 9],
                      [2, 10, 9, 10, 3, 11, 9, 4, 10]])
    labels = torch.tensor([[9, 9, 12], [8, 10, 14], [9, 16, 7]])
    positions = torch.tensor([[6, 7, 8]] * 3); batch = torch.arange(3)[:, None]
    seed_all(); model = candidate.Model(spec, d_model=32, heads=4).float().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3); maximum = 1 if smoke else 100
    updates = 0
    for updates in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True); selected = model(x)[0][batch, positions]
        loss = F.cross_entropy(selected.flatten(0, 1), labels.flatten()); loss.backward(); optimizer.step()
        if selected.argmax(-1).eq(labels).all(): break
    train_correct = int(selected.argmax(-1).eq(labels).all(1).sum())
    versions = capture_state_versions(model); model.eval()
    with torch.no_grad(): evaluated = model(x)[0][batch, positions]
    assert_state_versions_unchanged(model, versions)
    return {"rows": 3, "true_t": [1, 2, 3], "d_model": 32, "optimizer": "AdamW(lr=3e-3)",
            "maximum_updates": 100, "updates_run": updates, "train_correct": train_correct,
            "same_weight_eval_correct": int(evaluated.argmax(-1).eq(labels).all(1).sum()),
            "final_cross_entropy": float(loss.detach()), "state_unchanged": True}


def run(smoke=False):
    manifest = load_manifest(MANIFEST)
    if manifest.data.seed != LOADER_SEED: raise ValueError("loader seed drift")
    spec = ModelSpec(infer_vocab_size(manifest.data), infer_max_seq_len(manifest.data),
                     manifest.model_state.maximum_elements)
    preflight = capacity_preflight(spec, smoke)
    if smoke: return {"smoke": "passed", "preflight": preflight}
    # The scored run always receives fresh loader, model, and optimizer state.
    loaders = make_dataloaders(manifest.data, device=torch.device("cpu"))
    seed_all(); model = candidate.build_model(spec).float().train()
    loaders["train"].generator.manual_seed(LOADER_SEED)
    training = train(model, loaders["train"], manifest, 200)
    train_all = subset_loader(loaders["train"], range(len(loaders["train"].dataset)))
    records = loaders["test"].dataset.records
    test_ids = {f"T{t}": [i for i, row in enumerate(records) if row["time_steps"] == t] for t in (1, 2, 3)}
    evaluation = {"full_train": evaluate(model, train_all, manifest),
                  "test": evaluate(model, loaders["test"], manifest),
                  "test_by_t": {k: evaluate(model, subset_loader(loaders["test"], v), manifest)
                                for k, v in test_ids.items()},
                  "ood_T6": evaluate(model, loaders["ood"], manifest),
                  "OOD_N_T1": evaluate(model, loaders["depth_ood_n_t_1"], manifest)}
    evaluation["seen_N"] = {f"T{t}": evaluate(model, loaders[f"depth_t_{t}"], manifest)
                             for t in (1, 2, 4, 8) if f"depth_t_{t}" in loaders}
    train_correct = evaluation["full_train"]["correct_examples"]
    test = evaluation["test"]; seen_t1 = evaluation["seen_N"]["T1"]["correct_examples"]
    signal = train_correct >= 60 and (seen_t1 >= 4 or
             (test["correct_examples"] > 9 and test["loss"] < 1.8830257671))
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    result = {
        "protocol": {"manifest": str(MANIFEST.relative_to(ROOT)), "model_seed": MODEL_SEED,
                     "loader_seed": LOADER_SEED, "fresh_exact_manifest_loaders": True,
                     "train_generator_reset_immediately_before_training": True, "device": "cpu",
                     "dtype": "float32", "batch_size": 64, "updates": 200, "grad_clip": 1,
                     "optimizer_identity_reused": candidate.build_optimizer is __import__("competition_submission_c1b").SUBMISSION.build_optimizer,
                     "lr_accounting": "captured before optimizer.step; update1=4e-5, updates1-19 below peak, 20-200 peak",
                     "checkpoint_policy": "final only; one candidate", "evaluation_nonmutation": True},
        "hashes": {"candidate": sha256(ROOT / "competition_loop2_latent.py"),
                   "runner": sha256(Path(__file__)), "manifest": sha256(MANIFEST),
                   "dataset": sha256(ROOT / manifest.data.data_root), "checkpoint": sha256(CHECKPOINT)},
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
        "state_elements": count_model_state_elements(model), "preflight": preflight,
        "training": training, "evaluation": evaluation,
        "partition_size_proof": {k: len(v) for k, v in test_ids.items()},
        "gate": {"predeclared": "train>=60/600 AND (seen T1>=4/38 OR (test>9/150 AND CE<1.8830257671))",
                 "full_train_pass": train_correct >= 60, "seen_or_test_pass": seen_t1 >= 4 or
                 (test["correct_examples"] > 9 and test["loss"] < 1.8830257671), "signal": signal}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="execute exact 200-update screen")
    group.add_argument("--smoke", action="store_true", help="one-step smoke; write no files")
    args = parser.parse_args(); result = run(smoke=args.smoke)
    print(json.dumps(result if args.smoke else {"output": str(OUTPUT), "checkpoint": str(CHECKPOINT)}, indent=2))


if __name__ == "__main__": main()
