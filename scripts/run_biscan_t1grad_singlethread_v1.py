"""Deterministic single-thread canonical BiScan T1-gradient protocol (version 1)."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
import json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the reviewed checkpoint/iterator/digest implementation. Both executed
# dependencies are recorded in every final result; this wrapper contains all
# protocol differences rather than silently relying on parent defaults.
parent_path = ROOT / "scripts/run_latent_exposure_singlethread_v1.py"
source = parent_path.read_text(encoding="utf-8")
body = source[source.index("import argparse"):source.rindex('if __name__ == "__main__":')]
body = body.replace("import competition_loop2_latent as candidate",
                    "import competition_loop6_biscan_t1grad as candidate")
body = body.replace('ROOT / "competition_loop2_latent.py"',
                    'ROOT / "competition_loop6_biscan_t1grad.py"')
body = body.replace("latent_exposure_singlethread_v1", "biscan_t1grad_singlethread_v1")
exec(compile(body, str(parent_path), "exec"), globals())

PARENT_SECONDS_PER_UPDATE = 707.5528632008 / 1600
PROBE1 = ARTIFACTS / "biscan_t1grad_singlethread_v1_probe1.json"
PROBE2 = ARTIFACTS / "biscan_t1grad_singlethread_v1_probe2.json"


def capacity_preflight(spec):
    """Three true-T1 rows must memorize under the authorized cheap optimizer."""
    x = torch.tensor([[2,10,9,10,3,8,12,4,8], [2,10,9,10,3,15,16,4,8],
                      [2,10,9,10,3,11,9,4,8]])
    labels = torch.tensor([[9,9,12], [8,13,16], [8,11,16]])
    positions, bi = torch.tensor([[6,7,8]]*3), torch.arange(3)[:,None]
    seed_all(); model=candidate.build_model(spec).float().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-3)
    for update in range(1,101):
        optimizer.zero_grad(set_to_none=True); selected=model(x)[0][bi,positions]
        loss=F.cross_entropy(selected.flatten(0,1),labels.flatten()); loss.backward(); optimizer.step()
        if selected.argmax(-1).eq(labels).all(): break
    train_correct=int(selected.argmax(-1).eq(labels).all(1).sum()); model.eval()
    with torch.no_grad(): evaluated=model(x)[0][bi,positions]
    result={"rows":3,"true_t":[1,1,1],"optimizer":"AdamW(lr=3e-3)",
            "maximum_updates":100,"updates_run":update,"train_correct":train_correct,
            "same_weight_eval_correct":int(evaluated.argmax(-1).eq(labels).all(1).sum())}
    if result["train_correct"] != 3 or result["same_weight_eval_correct"] != 3:
        raise RuntimeError(f"capacity preflight failed: {result}")
    return result


_base_train_update=train_update
_probe_horizons=None
_probe_scan_counts=None
def train_update(model,bundle,batch,manifest):
    if _probe_horizons is not None:
        values=model.debug_execution(batch["input_ids"],batch.get("attention_mask"))["parsed_steps"]
        _probe_horizons.update(map(int,values.tolist()))
        _probe_scan_counts["active_row_macrosteps"] += int(values.sum())
        _probe_scan_counts["vectorized_scan_calls"] += int(values.max())
        _probe_scan_counts["down_cell_calls"] += int(values.max()) * model.max_seq_len
        _probe_scan_counts["up_cell_calls"] += int(values.max()) * model.max_seq_len
    return _base_train_update(model,bundle,batch,manifest)

def probe(output:Path):
    global _probe_horizons, _probe_scan_counts
    _probe_horizons=Counter()
    _probe_scan_counts=Counter()
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
    payload = {
        "protocol": "biscan_t1grad_singlethread_v1", "settings": settings,
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
    payload["batch_horizon_histogram"]={f"T{k}":v for k,v in sorted(_probe_horizons.items())}
    payload["scan_call_counts"]=dict(_probe_scan_counts)
    payload["backward_gate"]={"T1":1.0,"T_not_1":0.01,"forward_values_ungated":True}
    artifact = {"comparison_payload": payload,
                "timing": {"candidate_training_seconds_per_update": seconds_per_update,
                           "d112_parent_frozen_seconds_per_update": PARENT_SECONDS_PER_UPDATE,
                           "ratio": seconds_per_update / PARENT_SECONDS_PER_UPDATE}}
    _probe_horizons=None; _probe_scan_counts=None; write_json(output,artifact); return artifact


def _authorize_full_run():
    if not PROBE1.is_file() or not PROBE2.is_file():
        raise RuntimeError("full run requires both canonical deterministic probe artifacts")
    first = json.loads(PROBE1.read_text(encoding="utf-8"))
    second = json.loads(PROBE2.read_text(encoding="utf-8"))
    if first.get("comparison_payload") != second.get("comparison_payload"):
        raise RuntimeError("probe comparison_payload mismatch")
    ratio = max(first["timing"]["ratio"], second["timing"]["ratio"])
    if ratio > 3:
        raise RuntimeError(f"resource ratio {ratio:.3f} rejects the implementation")
    if ratio > 2.5:
        raise RuntimeError(f"resource ratio {ratio:.3f} requires review")


_base_setup=setup
_last_runtime=None
def setup():
    global _last_runtime
    result=_base_setup(); _last_runtime=(result[1],result[2],result[0]); return result

_base_full_run=full_run
def full_run():
    _authorize_full_run()
    result=_base_full_run()
    loaders,model,manifest=_last_runtime
    result["final_evaluation"]["depth_ood_n_t_1"]=evaluate(model,loaders["depth_ood_n_t_1"],manifest)
    t1=result["milestones"]["1600"]["evaluation"]["train_by_t"]["T1"]
    correct,ce=t1["correct_examples"],t1["loss"]
    verdict="PASS" if correct>=60 or ce<=1.10 else ("REJECT" if correct<=25 and ce>=1.45 else "INCONCLUSIVE")
    final=result["final_evaluation"]
    competition=dict(result["gates"])
    result["gates"]={
      "competition":competition,
      "one_step_learnability":{"verdict":verdict,"train_t1_correct":correct,"train_t1_cross_entropy":ce},
      "scan_transfer":{"pass":final["test_by_t"]["T1"]["correct_examples"]>=4 and
        final["seen_N_T1"]["correct_examples"]>=4 and final["test_by_t"]["T1"]["loss"]<1.8830257671},
      "ood_n_t1_supporting":{"correct":final["depth_ood_n_t_1"]["correct_examples"],"target":8,"total":512},
      "promotion_pass":competition["competition_pass"] and
        final["test_by_t"]["T1"]["correct_examples"]>=4 and
        final["seen_N_T1"]["correct_examples"]>=4 and
        final["test_by_t"]["T1"]["loss"]<1.8830257671}
    result["provenance"]={"dynamic_dependencies":{
      str(parent_path.relative_to(ROOT)):sha256(parent_path),
      "competition_loop6_biscan_t1grad.py":sha256(SOURCE)}}
    write_json(RESULT,result); return result


if __name__ == "__main__": main()
