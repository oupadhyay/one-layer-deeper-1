"""Deterministic single-thread x2 latent-inner-refinement protocol (version 1)."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Execute an exact, mechanically renamed copy of the validated canonical protocol.
# Keeping this transformation explicit makes future canonical hash/probe fixes apply here too.
canonical_path = ROOT / "scripts/run_latent_exposure_singlethread_v1.py"
source = canonical_path.read_text(encoding="utf-8")
body = source[source.index("import argparse"):source.rindex("if __name__ == \"__main__\":")]
body = body.replace("import competition_loop2_latent as candidate", "import competition_loop3_latent_x2 as candidate")
body = body.replace('ROOT / "competition_loop2_latent.py"', 'ROOT / "competition_loop3_latent_x2.py"')
body = body.replace("latent_exposure_singlethread_v1", "latent_inner_refinement_x2_singlethread_v1")
body = body.replace("competition_loop2_latent_exposure_update1600.pt", "competition_loop2_latent_exposure_update1600.pt")
namespace = globals()
exec(compile(body, str(canonical_path), "exec"), namespace)

_canonical_probe = probe
_canonical_full_run = full_run
PARENT_SECONDS_PER_UPDATE = 673.5659546004026 / 1600
PROBE1 = ARTIFACTS / "latent_inner_refinement_x2_singlethread_v1_probe1.json"
PROBE2 = ARTIFACTS / "latent_inner_refinement_x2_singlethread_v1_probe2.json"


def probe(output: Path):
    manifest, loaders, model, bundle, iterator, epoch_start, preflight, settings = setup()
    losses, batches = [], []
    epoch = consumed = examples = 0
    batches_per_epoch = len(loaders["train"])
    started = time.perf_counter()
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
    seconds_per_update = (time.perf_counter() - started) / 20
    artifact = {"comparison_payload": {
        "protocol": "latent_inner_refinement_x2_singlethread_v1", "settings": settings,
        "preflight": preflight, "losses": losses, "ordered_batch_digests": batches,
        "model_state_digest": canonical_digest(model.state_dict()),
        "optimizer_state_digest": canonical_digest(bundle.optimizer.state_dict()),
        "scheduler_state_digest": canonical_digest(
            None if bundle.scheduler is None else bundle.scheduler.state_dict()),
        "loader_generator_digest": canonical_digest(loaders["train"].generator.get_state()),
        "position": {"update": 20, "examples": examples, "epoch_index": epoch,
                     "batches_consumed_in_current_epoch": consumed,
                     "batches_per_epoch": batches_per_epoch,
                     "epoch_start_generator_digest": canonical_digest(epoch_start)},
    }}
    artifact["timing"] = {
        "candidate_training_seconds_per_update": seconds_per_update,
        "parent_frozen_seconds_per_update": PARENT_SECONDS_PER_UPDATE,
        "ratio": seconds_per_update / PARENT_SECONDS_PER_UPDATE,
    }
    write_json(output, artifact)
    return artifact


def _authorize_full_run():
    # Authorization I/O is deliberately after the mandatory per-process capacity preflight.
    manifest = load_manifest(MANIFEST)
    preflight_spec = ModelSpec(infer_vocab_size(manifest.data), infer_max_seq_len(manifest.data),
                               manifest.model_state.maximum_elements)
    capacity_preflight(preflight_spec)
    if not PROBE1.is_file() or not PROBE2.is_file():
        raise RuntimeError(f"full run requires stored probes {PROBE1.name} and {PROBE2.name}")
    first = json.loads(PROBE1.read_text(encoding="utf-8"))
    second = json.loads(PROBE2.read_text(encoding="utf-8"))
    if first.get("comparison_payload") != second.get("comparison_payload"):
        raise RuntimeError("probe comparison_payload mismatch (timing is intentionally excluded)")
    ratio = max(first["timing"]["ratio"], second["timing"]["ratio"])
    if ratio > 3:
        raise RuntimeError(f"resource ratio {ratio:.3f} exceeds hard limit 3.0")
    if ratio > 2.5:
        raise RuntimeError(f"resource ratio {ratio:.3f} requires investigation before full run")


def full_run():
    _authorize_full_run()
    result = _canonical_full_run()
    train_t1 = result["milestones"]["1600"]["evaluation"]["train_by_t"]["T1"]
    correct, ce = train_t1["correct_examples"], train_t1["loss"]
    if correct >= 20 or ce <= 1.60: verdict = "PASS"
    elif correct <= 10 and ce >= 1.75: verdict = "REJECT"
    else: verdict = "INCONCLUSIVE"
    result["gates"]["mechanistic"] = {"verdict": verdict, "train_t1_correct": correct,
                                          "train_t1_cross_entropy": ce}
    result["equal_update_parent_context"] = {
        "source": "frozen parent result milestones", "selection_or_stopping_used": False,
        "note": "Context only; candidate was trained continuously for exactly 1600 updates."
    }
    write_json(RESULT, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", metavar="OUTPUT_JSON", type=Path)
    group.add_argument("--run", action="store_true")
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
