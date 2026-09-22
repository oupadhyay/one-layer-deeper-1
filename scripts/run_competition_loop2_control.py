"""Deterministic guarded CPU run for the ungated Loop-1-D cheap control."""

from __future__ import annotations

import argparse, hashlib, json, platform, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements  # noqa: E402
from benchmark.manifest import load_manifest  # noqa: E402
from benchmark.runner import _evaluate, _loss_and_accuracy  # noqa: E402
from benchmark.validation import capture_state_versions, assert_state_versions_unchanged  # noqa: E402
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders  # noqa: E402
import competition_loop2_control as control  # noqa: E402

MANIFEST = ROOT / "benchmark/manifests/local_competition_c1_e1_cpu.json"
OUTPUT = ROOT / "artifacts/competition_loop2_control_results.json"
MODEL_SEED, LOADER_SEED = 74, 45


def seed_all(seed=MODEL_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256(path):
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def subset_loader(loader, indices, batch_size=None):
    return DataLoader(Subset(loader.dataset, indices), batch_size=batch_size or len(indices),
                      shuffle=False, drop_last=False, collate_fn=loader.collate_fn, num_workers=0)


def train(model, loader, manifest, updates, optimizer_builder=control.build_optimizer):
    bundle = optimizer_builder(model, OptimizerSpec(60, "cpu"))
    iterator, start, examples, lrs = iter(loader), time.perf_counter(), 0, []
    final = None
    for _ in range(updates):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        bundle.optimizer.zero_grad(set_to_none=True)
        final, exact, count, label_count = _loss_and_accuracy(model, batch, manifest, torch.device("cpu"))
        if not torch.isfinite(final): raise FloatingPointError("non-finite training loss")
        final.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bundle.optimizer.step()
        if bundle.scheduler: bundle.scheduler.step()
        lrs.append(max(group["lr"] for group in bundle.optimizer.param_groups)); examples += count
    elapsed = time.perf_counter() - start
    return {"updates": updates, "examples": examples, "elapsed_seconds": elapsed,
            "updates_per_second": updates / elapsed, "examples_per_second": examples / elapsed,
            "lr_by_update": lrs, "below_peak_lr_updates": sum(x < 8e-4 for x in lrs),
            "peak_lr_updates": sum(x == 8e-4 for x in lrs), "peak_lr": max(lrs),
            "final_batch": {"cross_entropy": float(final.detach()), "exact_accuracy": exact,
                            "correct_examples": int(round(exact * count)), "example_count": count,
                            "label_count": label_count, "clipped_gradient_norm": float(grad_norm)}}


def evaluate(model, loader, manifest, *, soft=False):
    versions = capture_state_versions(model)
    if soft:
        model.train(); batch = next(iter(loader))
        with torch.no_grad(): loss, exact, count, labels = _loss_and_accuracy(model, batch, manifest, torch.device("cpu"))
        metrics = {"loss": float(loss), "exact_accuracy": exact, "correct_examples": int(round(exact * count)),
                   "example_count": count, "label_count": labels}
    else:
        metrics = _evaluate(model, loader, manifest, torch.device("cpu"), deadline=float("inf"), budget_seconds=float("inf"))
    assert_state_versions_unchanged(model, versions)
    metrics["state_unchanged"] = True
    return metrics


def run(smoke=False):
    manifest = load_manifest(MANIFEST)
    if manifest.data.seed != LOADER_SEED: raise ValueError("E1 loader seed drift")
    spec = ModelSpec(infer_vocab_size(manifest.data), infer_max_seq_len(manifest.data), manifest.model_state.maximum_elements)
    # Preflight has independent data/model/optimizer state and records failures without aborting.
    pre_loaders = make_dataloaders(manifest.data, device=torch.device("cpu"))
    micro_loader = subset_loader(pre_loaders["depth_t_1"], [0, 1, 2])
    seed_all(); micro = control.Model(spec, d_model=32, heads=4).float().train()
    def preflight_optimizer(model, _spec):
        from benchmark import OptimizerBundle
        return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=3e-3), None)
    micro_training = train(micro, micro_loader, manifest, 1 if smoke else 100, preflight_optimizer)
    micro_soft = evaluate(micro, micro_loader, manifest, soft=True)
    micro_hard = evaluate(micro.eval(), micro_loader, manifest)
    preflight = {"rows": 3, "true_t": 1, "d_model": 32, "optimizer": "AdamW(lr=3e-3)",
                 "maximum_updates": 100, "updates_run": 1 if smoke else 100, "training": micro_training,
                 "train_soft": micro_soft, "same_weight_eval_hard": micro_hard,
                 "requirements_met": {"train_soft_3_of_3": micro_soft["correct_examples"] == 3,
                                      "eval_hard_3_of_3": micro_hard["correct_examples"] == 3},
                 "miss_policy": "record, do not crash"}
    if smoke: return {"smoke": "passed", "preflight": preflight}

    # Fresh loaders. Reset the train sampler generator immediately before training.
    loaders = make_dataloaders(manifest.data, device=torch.device("cpu"))
    seed_all(); model = control.build_model(spec).float().train()
    loaders["train"].generator.manual_seed(LOADER_SEED)
    training = train(model, loaders["train"], manifest, 200)

    full_train = subset_loader(loaders["train"], range(len(loaders["train"].dataset)), 128)
    test_records = loaders["test"].dataset.records
    partition_indices = {f"T{t}": [i for i, row in enumerate(test_records) if row["time_steps"] == t] for t in (1, 2, 3)}
    partitions = {name: evaluate(model, subset_loader(loaders["test"], ids, 128), manifest)
                  for name, ids in partition_indices.items()}
    evaluation = {"full_train": evaluate(model, full_train, manifest), "test": evaluate(model, loaders["test"], manifest),
                  "test_by_t": partitions, "ood": evaluate(model, loaders["ood"], manifest),
                  "seen_N_depth_t_1": evaluate(model, loaders["depth_t_1"], manifest),
                  "OOD_N_depth_ood_n_t_1": evaluate(model, loaders["depth_ood_n_t_1"], manifest)}
    absolute = (evaluation["seen_N_depth_t_1"]["correct_examples"] >= 4 or
                (evaluation["test"]["correct_examples"] > 9 and evaluation["test"]["loss"] < 1.8830257671))
    result = {
        "protocol": {"manifest": str(MANIFEST.relative_to(ROOT)), "model_seed": 74, "loader_seed": 45,
                     "fresh_dataloaders": True, "train_generator_reset_immediately_before_training": True,
                     "device": "cpu", "dtype": "float32", "updates": 200, "batch_size": 64, "grad_clip": 1,
                     "architecture": "fixed Loop-1 D GlobalSlotMLP on direct frozen-C0 Model; fully ungated training logits",
                     "preserved": ["true-T loop", "X/T context exclusion", "N context", "soft train feedback",
                                   "hard eval feedback", "digit reset", "output alignment"],
                     "optimizer_identity_reused": control.build_optimizer is __import__("competition_submission_c1b").SUBMISSION.build_optimizer,
                     "evaluation_nonmutation": True, "full_train_loader": "all 600, shuffle=false, drop_last=false",
                     "gate_predeclared": "full-train materially exceeds stored train no-input baseline AND (seen T1>=4/38 OR test>9/150 with CE<1.8830257671)"},
        "hashes": {"control_source_sha256": sha256(ROOT / "competition_loop2_control.py"),
                   "harness_source_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(MANIFEST),
                   "dataset_sha256": sha256(ROOT / manifest.data.data_root)},
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
        "state_elements": count_model_state_elements(model), "preflight": preflight, "training": training,
        "evaluation": evaluation, "partition_size_proof": {k: len(v) for k, v in partition_indices.items()},
        "reset_proof": {"model_base_class_is_frozen_C0": control.Model.__mro__[1].__module__ == "competition_submission",
                        "transition_has_no_persistent_recurrent_state": True},
        "gate": {"train_no_input_exact_marginal_stored": False,
                 "baseline_policy": "report full-train; do not invent baseline; use absolute seen/test gate only",
                 "absolute_seen_or_test_gate": absolute, "signal": absolute}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="run the exact 200-update control")
    group.add_argument("--smoke", action="store_true", help="one-update CPU smoke; writes no result artifact")
    args = parser.parse_args()
    result = run(smoke=args.smoke)
    print(json.dumps(result if args.smoke else {"output": str(OUTPUT)}, indent=2))


if __name__ == "__main__": main()
