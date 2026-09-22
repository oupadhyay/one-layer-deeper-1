"""Deterministic single-thread reversible-SSM T1-gradient protocol v1."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

# Execute the reviewed loop-7 wrapper after mechanical, pinned architecture substitutions.
loop7_harness_path=ROOT/"scripts/run_relative_place_attention_t1grad_singlethread_v1.py"
source=loop7_harness_path.read_text(encoding="utf-8")
body=source[source.index("import json, sys, time"):source.rindex('if __name__=="__main__":')]
body=body.replace("competition_loop7_relative_place_attention_t1grad","competition_loop8_reversible_ssm_t1grad")
body=body.replace("relative_place_attention_t1grad_singlethread_v1","reversible_ssm_t1grad_singlethread_v1")
body=body.replace("vectorized_block_calls","vectorized_transition_calls")
body=body.replace("attention_block_calls","F_mixer_calls")
body=body.replace(' _counts["readout_calls"]+=maximum',' _counts["G_mixer_calls"]+=maximum; _counts["endpoint_readout_calls"]+=1')
exec(compile(body,str(loop7_harness_path),"exec"),globals())

_parent_probe=probe
def probe(output:Path):
    artifact=_parent_probe(output)
    artifact["comparison_payload"]["coupling_scales"]={"F":0.1,"G":0.1,"bounded_by":"tanh"}
    write_json(output,artifact)
    return artifact

_parent_full=full_run
def full_run():
    result=_parent_full()
    dependencies=result.setdefault("provenance",{})["dynamic_dependencies"]
    dependencies[str(loop7_harness_path.relative_to(ROOT))]=sha256(loop7_harness_path)
    dependencies["scripts/run_latent_exposure_singlethread_v1.py"]=sha256(ROOT/"scripts/run_latent_exposure_singlethread_v1.py")
    write_json(RESULT,result)
    return result

if __name__=="__main__": main()
