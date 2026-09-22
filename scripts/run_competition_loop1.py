"""Deterministic, fixed-update CPU screen for the five Loop-1 candidates."""

from __future__ import annotations

import argparse, hashlib, json, random, sys, time, traceback
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
import competition_loop1_candidates as candidates  # noqa: E402

MANIFEST = ROOT / "benchmark/manifests/local_competition_c1_e1_cpu.json"
OUTPUT = ROOT / "artifacts/competition_loop1_results.json"


def seed_all(seed=74):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256(path):
    h = hashlib.sha256()
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for p in files:
        h.update(p.relative_to(path).as_posix().encode())
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def batch_loader(loader, indices):
    return DataLoader(Subset(loader.dataset, indices), batch_size=len(indices), shuffle=False,
                      collate_fn=loader.collate_fn, num_workers=0)


def train_fixed(model, loader, manifest, updates):
    bundle = candidates.build_optimizer(model, OptimizerSpec(60, "cpu"))
    iterator, start, peak_lr, examples = iter(loader), time.perf_counter(), 0.0, 0
    final = None
    for _ in range(updates):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        bundle.optimizer.zero_grad(set_to_none=True)
        final, exact, count, _ = _loss_and_accuracy(model, batch, manifest, torch.device("cpu"))
        if not torch.isfinite(final): raise FloatingPointError("non-finite training loss")
        final.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bundle.optimizer.step()
        if bundle.scheduler: bundle.scheduler.step()
        peak_lr = max(peak_lr, *(g["lr"] for g in bundle.optimizer.param_groups)); examples += count
    elapsed = time.perf_counter() - start
    return {"loss": float(final.detach()), "exact": exact, "updates": updates, "examples": examples,
            "elapsed_seconds": elapsed, "updates_per_second": updates / elapsed,
            "examples_per_second": examples / elapsed, "below_peak_lr_updates": min(updates, 19),
            "peak_lr_updates": max(0, updates - 19), "peak_lr": peak_lr}


def evaluate(model, loader, manifest):
    versions = capture_state_versions(model)
    metrics = _evaluate(model, loader, manifest, torch.device("cpu"), deadline=float("inf"), budget_seconds=float("inf"))
    assert_state_versions_unchanged(model, versions)
    metrics["deterministic_nonmutation"] = True
    return metrics


def evaluate_soft(model, loader, manifest):
    """Measure the three-row probe while retaining training (soft reset) behavior."""
    model.train(); versions = capture_state_versions(model)
    batch = next(iter(loader))
    with torch.no_grad():
        loss, exact, count, loss_count = _loss_and_accuracy(
            model, batch, manifest, torch.device("cpu")
        )
    assert_state_versions_unchanged(model, versions)
    return {"loss": float(loss.detach()), "exact_accuracy": exact,
            "correct_examples": int(round(exact * count)), "example_count": count,
            "label_count": loss_count, "deterministic_nonmutation": True}


def run_candidate(name, spec, loaders, manifest):
    # The official depth_t_1 split supplies fixed, true labels; exactly three rows.
    source = loaders["depth_t_1"]
    micro_loader = batch_loader(source, [0, 1, 2])
    seed_all(); micro = candidates.build_model(spec, name, 32, 4)
    micro_train = train_fixed(micro, micro_loader, manifest, 100)
    micro_soft = evaluate_soft(micro, micro_loader, manifest)
    micro_hard = evaluate(micro.eval(), micro_loader, manifest)

    seed_all(); model = candidates.BUILDERS[name](spec).float()
    state = count_model_state_elements(model)
    training = train_fixed(model.train(), loaders["train"], manifest, 200)
    wanted = ("test", "ood", "depth_t_1", "depth_ood_n_t_1")
    evaluation = {split: evaluate(model, loaders[split], manifest) for split in wanted}
    micro_exact = micro_hard["correct_examples"] == micro_hard["example_count"]
    seen_t1 = evaluation["depth_t_1"]
    test = evaluation["test"]
    architecture_signal = (
        seen_t1["correct_examples"] >= 4
        or (test["correct_examples"] > 9 and test["loss"] < 1.8830257671)
    )
    gates = {"finite": bool(np.isfinite(training["loss"])), "under_state_budget": state <= spec.maximum_model_state_elements,
             "micro_hard_exact_3_of_3": micro_exact, "eval_nonmutation": all(x["deterministic_nonmutation"] for x in evaluation.values()),
             "architecture_signal": architecture_signal}
    mechanically_valid = all(value for key, value in gates.items() if key != "architecture_signal")
    status = "promising" if mechanically_valid and architecture_signal else "rejected"
    return {"status": status, "rejection_gates": gates,
            "state_elements": state, "micro_overfit": {"training": micro_train, "train_soft": micro_soft, "eval_hard": micro_hard},
            "screen_training": training, "evaluation": evaluation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the full 5 x 200-update screen")
    args = parser.parse_args()
    if not args.run:
        parser.error("full screen is guarded; pass --run explicitly")
    manifest = load_manifest(MANIFEST); seed_all()
    data_config = manifest.data
    loaders = make_dataloaders(data_config, device=torch.device("cpu"))
    spec = ModelSpec(infer_vocab_size(data_config), infer_max_seq_len(data_config), manifest.model_state.maximum_elements)
    result = {"protocol": {"model_seed": 74, "data_loader_seed": data_config.seed, "batch_size": 64, "updates": 200, "device": "cpu", "dtype": "float32", "grad_clip": 1,
                           "optimizer_identity": "competition_submission_c1b.SUBMISSION.build_optimizer",
                           "evaluated_splits": ["test", "ood", "depth_t_1", "depth_ood_n_t_1"],
                           "screen_rule": "promising only if mechanically valid and seen-N T1 >=4/38, or test >9/150 with CE <1.8830257671"},
              "hashes": {"candidate_source_sha256": sha256(ROOT / "competition_loop1_candidates.py"),
                         "harness_source_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(MANIFEST),
                         "dataset_sha256": sha256(ROOT / manifest.data.data_root)}, "candidates": {}}
    for name in candidates.TRANSITIONS:
        print(f"candidate={name} status=running", flush=True)
        try: result["candidates"][name] = run_candidate(name, spec, loaders, manifest)
        except Exception as exc:
            result["candidates"][name] = {"status": "failed", "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        print(f"candidate={name} status={result['candidates'][name]['status']}", flush=True)
    print(OUTPUT)


if __name__ == "__main__": main()
