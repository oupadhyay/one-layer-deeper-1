"""Deterministic bounded preflight for the factorized pair reducer."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
import argparse, json, math, random, sys, time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

# Safe source adaptation keeps this runner independent of import-time runner side effects.
parent_path=ROOT/"scripts/run_biscan_e5_diversity_singlethread_v1.py"
source=parent_path.read_text(encoding="utf-8")
body=source[source.index("import json, sys, time"):source.rindex('if __name__=="__main__":')]
body=body.replace("competition_loop6_biscan_t1grad","competition_loop11_factorized_pair_reducer")
body=body.replace("biscan_e5_diversity_singlethread_v1","factorized_pair_reducer_singlethread_v1")
exec(compile(body,str(parent_path),"exec"),globals())

FROZEN_BISCAN_COST=.034
PROBE1=ARTIFACTS/"factorized_pair_reducer_singlethread_v1_probe1.json"
PROBE2=ARTIFACTS/"factorized_pair_reducer_singlethread_v1_probe2.json"
_widths=Counter(); _pairs=_reducers=_resets=0

_actual_train_update=_base_train_update
def train_update(model,bundle,batch,manifest):
    global _pairs,_reducers,_resets
    if _horizons is not None:
        debug=model.debug_execution(batch["input_ids"],batch.get("attention_mask"))
        _horizons.update(map(int,debug["parsed_steps"].tolist()))
        _widths.update(map(int,debug["widths"].tolist()))
        _counts["active_macrosteps"]+=debug["active_macrosteps"]
        _pairs+=debug["pair_token_count"]; _reducers+=debug["reducer_calls"]; _resets+=debug["scratch_resets"]
    return _actual_train_update(model,bundle,batch,manifest)

_inherited_probe=probe
def probe(path:Path):
    global _pairs,_reducers,_resets
    _widths.clear(); _pairs=_reducers=_resets=0
    artifact=_inherited_probe(path); payload=artifact["comparison_payload"]
    payload["protocol"]="factorized_pair_reducer_singlethread_v1"
    payload["width_histogram"]={str(k):v for k,v in sorted(_widths.items())}
    payload["pair_token_count"]=_pairs; payload["reducer_call_count"]=_reducers
    payload["scratch_reset_count"]=_resets; payload["training_objective"]="plain endpoint cross entropy"
    payload["backward_gate"]={"enabled":False,"horizon_scale":1.0}
    artifact["timing"]["e1_biscan_frozen_seconds_per_update"]=FROZEN_BISCAN_COST
    artifact["timing"]["ratio"]=artifact["timing"]["candidate_training_seconds_per_update"]/FROZEN_BISCAN_COST
    write_json(path,artifact); return artifact

TRAINABILITY_JSON=ARTIFACTS/"factorized_pair_reducer_singlethread_v1_trainability400.json"
TRAINABILITY_PT=ARTIFACTS/"factorized_pair_reducer_singlethread_v1_trainability400.pt"

def aggregate_train(model, dataset, collate_fn):
    """Evaluate collated training batches without touching training iterator or RNG state."""
    versions=capture_state_versions(model); was_training=model.training
    py_rng,np_rng,torch_rng=random.getstate(),np.random.get_state(),torch.get_rng_state()
    totals={t:{"loss_sum":0.,"tokens":0,"correct":0,"examples":0} for t in (1,2,3)}
    places=[{"loss_sum":0.,"tokens":0} for _ in range(4)]; all_loss=all_tokens=all_correct=all_examples=0
    unique_predictions=set()
    loader=DataLoader(dataset,batch_size=128,shuffle=False,drop_last=False,
                      collate_fn=collate_fn,num_workers=0)
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                ids=batch["input_ids"]; mask=batch.get("attention_mask")
                targets=batch["targets"] if "targets" in batch else batch["labels"]
                positions=batch.get("target_positions")
                logits,_=model(ids,attention_mask=mask)
                if positions is None:
                    token_logits,token_targets=logits[:,:-1].float(),targets[:,1:]
                else:
                    bi=torch.arange(ids.shape[0])[:,None]
                    token_logits=logits[bi,positions.clamp_min(0)].float(); token_targets=targets
                valid=token_targets.ne(-100); losses=F.cross_entropy(
                    token_logits[valid],token_targets[valid],reduction="none")
                loss_matrix=torch.zeros_like(token_targets,dtype=torch.float32)
                loss_matrix[valid]=losses
                predictions=token_logits.argmax(-1)
                row_exact=((predictions==token_targets)|~valid).all(1)
                _,_,_,steps=model.parse(ids,ids.ne(0) if mask is None else mask.bool())
                for t in (1,2,3):
                    rows=steps.eq(t); rv=valid&rows[:,None]
                    totals[t]["loss_sum"]+=float(loss_matrix[rv].sum())
                    totals[t]["tokens"]+=int(rv.sum()); totals[t]["correct"]+=int(row_exact[rows].sum())
                    totals[t]["examples"]+=int(rows.sum())
                for row in range(ids.shape[0]):
                    vals=predictions[row,valid[row]].tolist()
                    code=0
                    for value in vals: code=code*model.vocab_size+int(value)
                    unique_predictions.add((len(vals),code))
                    indices=valid[row].nonzero().flatten().tolist()
                    for place,index in enumerate(reversed(indices)):
                        if place<4:
                            places[place]["loss_sum"]+=float(loss_matrix[row,index]); places[place]["tokens"]+=1
                all_loss+=float(losses.sum()); all_tokens+=int(valid.sum())
                all_correct+=int(row_exact.sum()); all_examples+=ids.shape[0]
        assert_state_versions_unchanged(model,versions)
    finally:
        random.setstate(py_rng); np.random.set_state(np_rng); torch.set_rng_state(torch_rng)
        model.train(was_training)
    metric=lambda x:{"loss":x["loss_sum"]/x["tokens"],"exact_accuracy":x["correct"]/x["examples"],
                     "correct_examples":x["correct"],"example_count":x["examples"],"token_count":x["tokens"]}
    return ({"full_train":{"loss":all_loss/all_tokens,"exact_accuracy":all_correct/all_examples,
                            "correct_examples":all_correct,"example_count":all_examples,"token_count":all_tokens},
             "train_by_t":{f"T{t}":metric(totals[t]) for t in (1,2,3)},
             "output_place_token_ce":{name:{"loss":p["loss_sum"]/p["tokens"],"token_count":p["tokens"]}
                                      for name,p in zip(("LSD","tens","hundreds","thousands"),places)},
             "unique_predicted_sequences":len(unique_predictions),"all_row_modal_collapse":len(unique_predictions)==1,
             "model_state_unchanged":True,"rng_state_restored":True})

def trainability():
    _authorize_full_run()
    source_before,manifest_before=sha256(SOURCE),sha256(MANIFEST)
    manifest,loaders,model,bundle,iterator,epoch_start,preflight,settings=setup()
    dataset_path=ROOT/manifest.data.data_root; dataset_before=sha256(dataset_path)
    epoch=consumed=examples=0; batches_per_epoch=len(loaders["train"]); final_batch={}; started=time.perf_counter()
    for update in range(1,401):
        try: batch=next(iterator)
        except StopIteration:
            epoch+=1; consumed=0; epoch_start=loaders["train"].generator.get_state().clone()
            iterator=iter(loaders["train"]); batch=next(iterator)
        loss,exact,count,labels,grad,lr=train_update(model,bundle,batch,manifest)
        consumed+=1; examples+=count
        final_batch={"cross_entropy":loss,"exact_accuracy":exact,"example_count":count,
                     "label_count":labels,"clipped_gradient_norm":grad,"lr_used":lr}
    training_seconds=time.perf_counter()-started
    aggregate=aggregate_train(model,loaders["train"].dataset,loaders["train"].collate_fn)
    state=checkpoint(model,bundle,loaders["train"],400,examples,epoch,consumed,batches_per_epoch,epoch_start)
    state["protocol"]="factorized_pair_reducer_singlethread_v1_trainability400"
    torch.save(state,TRAINABILITY_PT); checkpoint_hash=sha256(TRAINABILITY_PT)
    t1=aggregate["train_by_t"]["T1"]; place=aggregate["output_place_token_ce"]
    clauses={"t1_learnability":t1["loss"]<=1.90 or t1["correct_examples"]>=80,
             "lsd_ce":place["LSD"]["loss"]<2.25,"tens_ce":place["tens"]["loss"]<2.25,
             "finite_metrics":all(math.isfinite(x) for x in [aggregate["full_train"]["loss"],t1["loss"],
                 place["LSD"]["loss"],place["tens"]["loss"]]),
             "no_all_row_modal_collapse":not aggregate["all_row_modal_collapse"]}
    probes={p.name:sha256(p) for p in (PROBE1,PROBE2)}
    result={"status":"complete","protocol":"factorized_pair_reducer_singlethread_v1_trainability400",
      "contract":{"fresh_update0":True,"updates":400,"batch_size":64,"model_seed":74,"loader_seed":45,
        "official_e5_train_loader":True,"official_one_forward_one_backward_train_update":True,
        "loss":"plain endpoint cross entropy","gradient_clip":1.0,"probe_or_checkpoint_loaded":False,
        "no_reset_selection_continuation_or_held_out_evaluation":True},
      "correction_history":[],"determinism":settings,"preflight":preflight,
      "model_state_elements":count_model_state_elements(model),"training_elapsed_seconds":training_seconds,
      "final_batch":final_batch,"aggregate":aggregate,
      "position":{"update":400,"examples":examples,"epoch_index":epoch,
        "batches_consumed_in_current_epoch":consumed,"batches_per_epoch":batches_per_epoch,
        "epoch_start_generator_digest":canonical_digest(epoch_start)},
      "digests":{"model":canonical_digest(model.state_dict()),"optimizer":canonical_digest(bundle.optimizer.state_dict()),
        "scheduler":canonical_digest(bundle.scheduler.state_dict()),
        "loader_generator":canonical_digest(loaders["train"].generator.get_state())},
      "hashes":{"candidate":source_before,"harness":sha256(Path(__file__)),"manifest":manifest_before,
        "dataset":dataset_before,"probes":probes,"checkpoint":checkpoint_hash},
      "promotion_gate":{"rule":"T1 CE <=1.90 OR exact >=80/1600; LSD and tens CE <2.25; finite; no all-row modal collapse",
        "clauses":clauses,"pass":all(clauses.values()),"failed_clauses":[k for k,v in clauses.items() if not v]}}
    after=(sha256(SOURCE),sha256(MANIFEST),sha256(dataset_path))
    if after!=(source_before,manifest_before,dataset_before): raise RuntimeError("frozen input mutation")
    write_json(TRAINABILITY_JSON,result); return result

def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe",metavar="OUTPUT_JSON",type=Path); group.add_argument("--run",action="store_true")
    group.add_argument("--trainability",action="store_true")
    args=parser.parse_args()
    if args.probe is not None: probe(args.probe)
    elif args.run: full_run()
    else: trainability()

if __name__=="__main__": main()
