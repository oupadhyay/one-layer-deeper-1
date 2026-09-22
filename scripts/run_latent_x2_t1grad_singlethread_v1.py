"""Deterministic single-thread latent x2 T1-gradient-gate protocol (version 1)."""

from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Mechanically inherit the reviewed x2 protocol, while pinning every dynamically
# executed source in provenance via SOURCE plus this wrapper's explicit hash below.
parent_path = ROOT / "scripts/run_latent_inner_refinement_x2_singlethread_v1.py"
source = parent_path.read_text(encoding="utf-8")
body = source[source.index("import argparse"):source.rindex('if __name__ == "__main__":')]
body = body.replace("import competition_loop3_latent_x2 as candidate",
                    "import competition_loop4_latent_x2_t1grad as candidate")
body = body.replace('ROOT / "competition_loop3_latent_x2.py"',
                    'ROOT / "competition_loop4_latent_x2_t1grad.py"')
body = body.replace("latent_inner_refinement_x2_singlethread_v1", "latent_x2_t1grad_singlethread_v1")
exec(compile(body, str(parent_path), "exec"), globals())


def capacity_preflight(spec):
    x = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8], [2, 10, 9, 10, 3, 15, 16, 4, 8],
                      [2, 10, 9, 10, 3, 11, 9, 4, 8]])
    labels = torch.tensor([[9, 9, 12], [8, 13, 16], [8, 11, 16]])
    positions, bi = torch.tensor([[6, 7, 8]] * 3), torch.arange(3)[:, None]
    seed_all(); model = candidate.Model(spec, d_model=32).float().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for update in range(1, 101):
        optimizer.zero_grad(set_to_none=True); selected = model(x)[0][bi, positions]
        loss = F.cross_entropy(selected.flatten(0, 1), labels.flatten()); loss.backward(); optimizer.step()
        if selected.argmax(-1).eq(labels).all(): break
    model.eval()
    with torch.no_grad(): evaluated = model(x)[0][bi, positions]
    result = {"rows": 3, "true_t": [1, 1, 1], "inputs": [15, 89, 42],
              "residues_mod_323": [225, 169, 149], "d_model": 32,
              "optimizer": "AdamW(lr=3e-3)", "maximum_updates": 100, "updates_run": update,
              "train_correct": int(selected.argmax(-1).eq(labels).all(1).sum()),
              "same_weight_eval_correct": int(evaluated.argmax(-1).eq(labels).all(1).sum())}
    if result["train_correct"] != 3 or result["same_weight_eval_correct"] != 3:
        raise RuntimeError(f"capacity preflight failed: {result}")
    return result


_base_train_update = train_update
_probe_horizons = None
def train_update(model, bundle, batch, manifest):
    if _probe_horizons is not None:
        horizons = model.debug_execution(batch["input_ids"], batch.get("attention_mask"))["parsed_steps"]
        _probe_horizons.update(map(int, horizons.tolist()))
    return _base_train_update(model, bundle, batch, manifest)


_base_probe = probe
def probe(output: Path):
    global _probe_horizons
    _probe_horizons = Counter()
    artifact = _base_probe(output)
    payload = artifact["comparison_payload"]
    payload["batch_horizon_histogram"] = {f"T{k}": v for k, v in sorted(_probe_horizons.items())}
    payload["backward_gate"] = {"T1": 1.0, "T_not_1": 0.01, "forward_values_ungated": True}
    _probe_horizons = None
    write_json(output, artifact)
    return artifact


_base_full_run = full_run
def full_run():
    result = _base_full_run()
    t1 = result["milestones"]["1600"]["evaluation"]["train_by_t"]["T1"]
    correct, ce = t1["correct_examples"], t1["loss"]
    verdict = "PASS" if correct >= 60 or ce <= 1.10 else ("REJECT" if correct <= 25 and ce >= 1.45 else "INCONCLUSIVE")
    result["gates"]["interference"] = {"verdict": verdict, "train_t1_correct": correct,
                                         "train_t1_cross_entropy": ce, "automatic_follow_up": False}
    result["backward_gate_disclosure"] = {"T1": 1.0, "T_not_1": 0.01,
                                           "reported_forward_ce": "ordinary unweighted"}
    result.setdefault("provenance", {})["dynamic_dependencies"] = {
        str(parent_path.relative_to(ROOT)): sha256(parent_path),
        "scripts/run_latent_exposure_singlethread_v1.py": sha256(ROOT / "scripts/run_latent_exposure_singlethread_v1.py")}
    write_json(RESULT, result); return result


if __name__ == "__main__": main()
