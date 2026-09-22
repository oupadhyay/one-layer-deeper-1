"""Deterministic bounded compressed axial cellular E5 protocol."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

# Reuse the final audited train-only screen while pinning this protocol's substitutions.
parent_path=ROOT/"scripts/run_factorized_pair_reducer_d96k3_singlethread_v1.py"
source=parent_path.read_text(encoding="utf-8")
body=source[source.index("import argparse, json, math"):source.rindex('if __name__=="__main__":')]
body=body.replace("competition_loop12_factorized_pair_reducer_d96k3","competition_loop13_compressed_axial_cellular")
body=body.replace("factorized_pair_reducer_d96k3_singlethread_v1","compressed_axial_cellular_singlethread_v1")
body=body.replace("_actual_train_update","_parent_actual_train_update")
body=body.replace("_inherited_probe","_parent_inherited_probe")
body=body.replace("_,_,_,steps=model.parse", "_,_,steps=model.parse")
body=body.replace("The initial probes entered the predeclared 2.5-to-3.0 review band. A trainability-only review authorized 400 updates after the projected runtime fit the bounded CPU budget. The 1,600-update path remained blocked.",
                  "The first trainability attempt completed its updates but the inherited four-value parse assumption stopped aggregation; no artifact was saved. The harness was corrected for the candidate's three-value parser and a fresh update0-to-400 screen was executed. The 1,600-update path remained blocked.")
exec(compile(body,str(parent_path),"exec"),globals())

FROZEN_BISCAN_COST=.034
PROBE1=ARTIFACTS/"compressed_axial_cellular_singlethread_v1_probe1.json"
PROBE2=ARTIFACTS/"compressed_axial_cellular_singlethread_v1_probe2.json"
RESOURCE=ARTIFACTS/"compressed_axial_cellular_singlethread_v1_resource.json"
TRAINABILITY_JSON=ARTIFACTS/"compressed_axial_cellular_singlethread_v1_trainability400.json"
TRAINABILITY_PT=ARTIFACTS/"compressed_axial_cellular_singlethread_v1_trainability400.pt"
_widths=Counter(); _axes=Counter(); _resets=0

_actual_train_update=_base_train_update
def train_update(model,bundle,batch,manifest):
    global _resets
    if _horizons is not None:
        debug=model.debug_execution(batch["input_ids"],batch.get("attention_mask"))
        _horizons.update(map(int,debug["parsed_steps"].tolist()))
        _widths.update(map(int,debug["widths"].tolist()))
        _counts["active_macrosteps"]+=debug["active_macrosteps"]
        _resets+=debug["workspace_resets"]
        _axes["H"]+=debug["horizontal_cell_calls"]; _axes["V"]+=debug["vertical_cell_calls"]
    return _actual_train_update(model,bundle,batch,manifest)

_inherited_probe=probe
def probe(path:Path):
    global _resets
    _widths.clear(); _axes.clear(); _resets=0
    artifact=_inherited_probe(path); payload=artifact["comparison_payload"]
    payload["protocol"]="compressed_axial_cellular_singlethread_v1"
    payload["width_histogram"]={str(k):v for k,v in sorted(_widths.items())}
    payload["workspace_reset_count"]=_resets
    payload["axis_cell_calls"]={k:_axes[k] for k in ("H","V")}
    payload["microtick_order"]=["H","V","H","V"]
    payload["backward_gate"]={"T1":1.0,"T2":.01,"T3":.01}
    artifact["timing"]["e5_biscan_frozen_seconds_per_update"]=FROZEN_BISCAN_COST
    artifact["timing"]["ratio"]=artifact["timing"]["candidate_training_seconds_per_update"]/FROZEN_BISCAN_COST
    write_json(path,artifact); return artifact

def authorize_trainability_after_review():
    if not PROBE1.is_file() or not PROBE2.is_file(): raise RuntimeError("trainability requires two probes")
    first=json.loads(PROBE1.read_text()); second=json.loads(PROBE2.read_text())
    if first["comparison_payload"]!=second["comparison_payload"]: raise RuntimeError("probe payload mismatch")
    ratio=max(first["timing"]["ratio"],second["timing"]["ratio"])
    decision="authorized" if ratio<=2.5 else ("review-stop" if ratio<=3.0 else "reject")
    record={"decision":decision,"maximum_ratio":ratio,"limit":2.5,
            "full_1600_update_run_authorized":False}
    if ratio>2.5:
        write_json(RESOURCE,record); raise RuntimeError("resource gate blocks trainability")
    return record

def _authorize_full_run():
    return authorize_trainability_after_review()

if __name__=="__main__": main()
