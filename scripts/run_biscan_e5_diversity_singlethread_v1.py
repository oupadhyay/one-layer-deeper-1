"""Deterministic fixed-update canonical BiScan E5 data-diversity screen."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
import json, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

parent_path=ROOT/"scripts/run_latent_exposure_singlethread_v1.py"
source=parent_path.read_text(encoding="utf-8")
body=source[source.index("import argparse"):source.rindex('if __name__ == "__main__":')]
body=body.replace("import competition_loop2_latent as candidate","import competition_loop6_biscan_t1grad as candidate")
body=body.replace('ROOT / "competition_loop2_latent.py"','ROOT / "competition_loop6_biscan_t1grad.py"')
body=body.replace('ROOT / "benchmark/manifests/local_competition_c1_e1_cpu.json"','ROOT / "benchmark/manifests/local_biscan_e5_diversity_cpu.json"')
body=body.replace('data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123','data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123')
body=body.replace("latent_exposure_singlethread_v1","biscan_e5_diversity_singlethread_v1")
body=body.replace('{"T1": 200, "T2": 200, "T3": 200}','{"T1": 1600, "T2": 1600, "T3": 1600}')
body=body.replace('{"T1": 50, "T2": 50, "T3": 50}','{"T1": 400, "T2": 400, "T3": 400}')
exec(compile(body,str(parent_path),"exec"),globals())

FROZEN_BISCAN_COST=687.6041304974351/1600
PROBE1=ARTIFACTS/"biscan_e5_diversity_singlethread_v1_probe1.json"
PROBE2=ARTIFACTS/"biscan_e5_diversity_singlethread_v1_probe2.json"

def capacity_preflight(spec):
    x=torch.tensor([[2,10,9,10,3,8,12,4,8],[2,10,9,10,3,15,16,4,8],[2,10,9,10,3,11,9,4,8]])
    labels=torch.tensor([[9,9,12],[8,13,16],[8,11,16]]); positions=torch.tensor([[6,7,8]]*3); bi=torch.arange(3)[:,None]
    seed_all(); model=candidate.build_model(spec).float().train(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-3)
    for update in range(1,101):
        optimizer.zero_grad(set_to_none=True); selected=model(x)[0][bi,positions]
        F.cross_entropy(selected.flatten(0,1),labels.flatten()).backward(); optimizer.step()
        if selected.argmax(-1).eq(labels).all(): break
    train_correct=int(selected.argmax(-1).eq(labels).all(1).sum()); model.eval()
    with torch.no_grad(): evaluated=model(x)[0][bi,positions]
    result={"rows":3,"true_t":[1,1,1],"optimizer":"AdamW(lr=3e-3)","maximum_updates":100,"updates_run":update,"train_correct":train_correct,"same_weight_eval_correct":int(evaluated.argmax(-1).eq(labels).all(1).sum())}
    if result["train_correct"]!=3 or result["same_weight_eval_correct"]!=3: raise RuntimeError(f"capacity preflight failed: {result}")
    seed_all(); return result

_base_train_update=train_update; _horizons=None; _counts=None
def train_update(model,bundle,batch,manifest):
    if _horizons is not None:
        values=model.debug_execution(batch["input_ids"],batch.get("attention_mask"))["parsed_steps"]
        _horizons.update(map(int,values.tolist())); maximum=int(values.max())
        _counts["active_row_macrosteps"]+=int(values.sum()); _counts["vectorized_scan_calls"]+=maximum
        _counts["down_cell_calls"]+=maximum*model.max_seq_len; _counts["up_cell_calls"]+=maximum*model.max_seq_len
    return _base_train_update(model,bundle,batch,manifest)

def probe(output:Path):
    global _horizons,_counts
    _horizons=Counter(); _counts=Counter()
    manifest,loaders,model,bundle,iterator,epoch_start,preflight,settings=setup()
    losses,batches=[],[]; epoch=consumed=examples=0; batches_per_epoch=len(loaders["train"]); started=time.perf_counter()
    for _update in range(1,21):
        try: batch=next(iterator)
        except StopIteration:
            epoch+=1; consumed=0; epoch_start=loaders["train"].generator.get_state().clone(); iterator=iter(loaders["train"]); batch=next(iterator)
        batches.append(batch_digest(batch)); loss,_,count,_,_,_=train_update(model,bundle,batch,manifest)
        losses.append(loss); consumed+=1; examples+=count
    seconds=(time.perf_counter()-started)/20
    payload={"protocol":"biscan_e5_diversity_singlethread_v1","settings":settings,"preflight":preflight,"losses":losses,"ordered_batch_digests":batches,
             "model_state_digest":canonical_digest(model.state_dict()),"optimizer_state_digest":canonical_digest(bundle.optimizer.state_dict()),
             "scheduler_state_digest":canonical_digest(None if bundle.scheduler is None else bundle.scheduler.state_dict()),"loader_generator_digest":canonical_digest(loaders["train"].generator.get_state()),
             "position":{"update":20,"examples":examples,"epoch_index":epoch,"batches_consumed_in_current_epoch":consumed,"batches_per_epoch":batches_per_epoch,"epoch_start_generator_digest":canonical_digest(epoch_start)},
             "batch_horizon_histogram":{f"T{k}":v for k,v in sorted(_horizons.items())},"scan_call_counts":dict(_counts),"backward_gate":{"T1":1.0,"T_not_1":0.01,"forward_values_ungated":True}}
    artifact={"comparison_payload":payload,"timing":{"candidate_training_seconds_per_update":seconds,"e1_biscan_frozen_seconds_per_update":FROZEN_BISCAN_COST,"ratio":seconds/FROZEN_BISCAN_COST}}
    _horizons=None; _counts=None; write_json(output,artifact); return artifact

def _authorize_full_run():
    if not PROBE1.is_file() or not PROBE2.is_file(): raise RuntimeError("full run requires two canonical probes")
    a=json.loads(PROBE1.read_text()); b=json.loads(PROBE2.read_text())
    if a["comparison_payload"]!=b["comparison_payload"]: raise RuntimeError("probe comparison_payload mismatch")
    ratio=max(a["timing"]["ratio"],b["timing"]["ratio"])
    if ratio>3: raise RuntimeError("resource ratio rejects experiment")
    if ratio>2.5: raise RuntimeError("resource ratio requires review")

_base_setup=setup; _last_runtime=None
def setup():
    global _last_runtime
    result=_base_setup(); _last_runtime=(result[1],result[2],result[0]); return result

_inherited_full=full_run
def full_run():
    _authorize_full_run(); result=_inherited_full(); loaders,model,manifest=_last_runtime
    result["final_evaluation"]["depth_ood_n_t_1"]=evaluate(model,loaders["depth_ood_n_t_1"],manifest)
    final=result["final_evaluation"]; train=result["milestones"]["1600"]["evaluation"]
    t1=train["train_by_t"]["T1"]; exposure=train["full_train"]["correct_examples"]>=480
    learn=t1["correct_examples"]>=320 or t1["loss"]<=1.5
    test=final["test_by_t"]["T1"]; seen=final["seen_N_T1"]; oodn=final["depth_ood_n_t_1"]
    transfer=(test["correct_examples"]>=40 and test["loss"]<1.8 and seen["correct_examples"]>=50 and seen["loss"]<1.8 and oodn["correct_examples"]>=16)
    result["gates"]={"exposure":{"pass":exposure,"rule":"full train >=480/4800"},"one_step_learnability":{"pass":learn,"rule":"train T1 >=320/1600 or CE <=1.5","correct":t1["correct_examples"],"cross_entropy":t1["loss"]},"variable_n_transfer":{"pass":transfer,"rule":"test T1 >=40/400 and CE <1.8; seen-N T1 >=50/512 and CE <1.8; OOD-N T1 >=16/512","test_t1":test,"seen_n_t1":seen,"ood_n_t1":oodn},"depth_t6_supporting":{"correct":final["ood_T6"]["correct_examples"],"total":600,"target":30},"promotion_pass":exposure and learn and transfer}
    result.setdefault("provenance",{})["dynamic_dependencies"]={str(parent_path.relative_to(ROOT)):sha256(parent_path),"competition_loop6_biscan_t1grad.py":sha256(SOURCE),"benchmark/manifests/local_biscan_e5_diversity_cpu.json":sha256(MANIFEST)}
    write_json(RESULT,result); return result

if __name__=="__main__": main()
