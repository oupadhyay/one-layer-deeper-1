"""Deterministic single-thread d64 capacity-regularization protocol (version 1)."""

from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Inherit the validated current protocol mechanically and pin every dynamically
# executed source in the emitted result.
parent_path = ROOT / "scripts/run_latent_x2_t1grad_singlethread_v1.py"
source = parent_path.read_text(encoding="utf-8")
body = source[source.index("import json"):source.rindex('if __name__ == "__main__":')]
body = body.replace("import competition_loop4_latent_x2_t1grad as candidate",
                    "import competition_loop5_latent_x2_t1grad_d64 as candidate")
body = body.replace('ROOT / "competition_loop4_latent_x2_t1grad.py"',
                    'ROOT / "competition_loop5_latent_x2_t1grad_d64.py"')
body = body.replace("latent_x2_t1grad_singlethread_v1",
                    "latent_x2_t1grad_d64_singlethread_v1")
exec(compile(body, str(parent_path), "exec"), globals())


_base_full_run_d64 = full_run
def full_run():
    result = _base_full_run_d64()
    result["capacity_regularization_test"] = {
        "predeclared": True,
        "only_test_after_x2_t1gate_interference_pass": True,
        "change": "default d_model 112 -> 64",
        "automatic_follow_up": False,
    }
    result["gates"]["capacity_regularization"] = {
        "verdict": "MEASURED_BY_COMPETITION_AND_INTERFERENCE_GATES",
        "automatic_follow_up": False,
    }
    dependencies = result.setdefault("provenance", {}).setdefault("dynamic_dependencies", {})
    dependencies[str(parent_path.relative_to(ROOT))] = sha256(parent_path)
    write_json(RESULT, result)
    return result


if __name__ == "__main__": main()
