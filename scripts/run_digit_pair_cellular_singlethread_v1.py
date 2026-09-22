"""Deterministic fixed-update digit-pair cellular E5 protocol."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

# Execute the complete E5 checkpoint/evaluation harness after pinned protocol substitutions.
parent_path=ROOT/"scripts/run_biscan_e5_diversity_singlethread_v1.py"
source=parent_path.read_text(encoding="utf-8")
body=source[source.index("import json, sys, time"):source.rindex('if __name__=="__main__":')]
body=body.replace("competition_loop6_biscan_t1grad","competition_loop10_digit_pair_cellular")
body=body.replace("biscan_e5_diversity_singlethread_v1","digit_pair_cellular_singlethread_v1")
exec(compile(body,str(parent_path),"exec"),globals())

FROZEN_BISCAN_COST=.034
_widths=Counter()

def train_update(model,bundle,batch,manifest):
    debug=model.debug_execution(batch["input_ids"],batch.get("attention_mask"))
    if _horizons is not None:
        steps, widths=debug["parsed_steps"],debug["widths"]
        _horizons.update(map(int,steps.tolist())); _widths.update(map(int,widths.tolist()))
        _counts["workspace_initialization_count"]+=int(steps.max())
        _counts["vectorized_cell_calls"]+=int(steps.max())*2*int(widths.max())
        _counts["active_row_cell_calls"]+=int((2*widths*steps).sum())
    return _base_train_update(model,bundle,batch,manifest)

_inherited_probe=probe
def probe(path:Path):
    _widths.clear()
    artifact=_inherited_probe(path)
    artifact["comparison_payload"]["protocol"]="digit_pair_cellular_singlethread_v1"
    artifact["comparison_payload"]["width_histogram"]={str(k):v for k,v in sorted(_widths.items())}
    write_json(path,artifact)
    return artifact

if __name__=="__main__": main()
