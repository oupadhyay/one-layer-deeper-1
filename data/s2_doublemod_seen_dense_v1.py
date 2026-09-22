"""Deterministic S2-D1 dense complement dataset and independent G0 audit."""
from __future__ import annotations

import argparse, hashlib, json, math, random, shutil, tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from data.s2_doublemod_seen_v1 import (EXPECTED_MODULI, RUNGS, SEED, carry_mask,
                                        digits, input_ids, transition)

STAGE = "s2_doublemod_seen_dense_v1"
EXPECTED_DENSE_COUNTS = [3413, 3667, 3905, 4157, 4405, 4661, 4937, 5171]
EXPECTED_TOTAL = 34316
PARENT_FILES = {
    "test_state.jsonl": "4accb941940d5ff9d513b3d887c20a589e9c6c348b9833db64129b01fc65fcb2",
    "depth_t_2.jsonl": "e8d54f79c3e51657d89138268fccc09950cdb6a43090c37fe5a08c81dac87435",
    "depth_t_4.jsonl": "8bed5b1df217a26035c9888bbdb634d24aed056b049b6178d03053ee0f25b951",
    "depth_t_8.jsonl": "fd407bef75e85a2f7a48ddb8afc2b3d36b32b7ba9b55aac4ffbdb051d75a2f54",
    "depth_t_16.jsonl": "66acfdc5612b7e1320c964f4ca197fb70fb744dbacbb8555c62d288b3e3fed79",
    "depth_t_32.jsonl": "de8e9757832c8f5c23d6207f62f1307f56e9353b337b96e688887112402711ad",
    "depth_t_64.jsonl": "1bec3d8b712620a91072a6e52e39f88f0476bf03c0120ca701a20b81f9718d97",
}
PARENT_PINS = {
    "generator": "acfd1e6bbc47e3b079d8e09b0a684ecd39cf730ef3c87b9a521eb10c9e5c5529",
    "dataset_config.json": "ec87f59ae1786cb32e8462a07a22c0f6f1be3c1221d3d0a70daac8b280007304",
    "artifact_manifest.json": "47c4ec7c5cc5887cc39cf0ff86dbc9fe25a7040b31f5640f2c1aaaad57423568",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def dense_record(n: int, x: int) -> dict[str, Any]:
    y = transition(x, n)
    mask = carry_mask(x)
    return {"stage": STAGE, "split": "dense_train", "N": n, "x0": x, "T": 1,
            "y": y, "trajectory": [x, y], "reduction": [int(2*x >= n)],
            "carry_masks": [mask], "carry_counts": [sum(mask)],
            "input_ids": input_ids(x, n), "labels": digits(y), "seed": SEED}

def _parent_default() -> Path:
    return Path(__file__).resolve().parent / "generated" / "s2_doublemod_seen_v1"

def verify_parent(parent: Path) -> None:
    source = Path(__file__).with_name("s2_doublemod_seen_v1.py")
    actual = {"generator": sha256(source), **{n: sha256(parent/n) for n in
              ("dataset_config.json", "artifact_manifest.json")}}
    if actual != PARENT_PINS:
        raise RuntimeError(f"parent provenance drift: {actual}")
    for name, expected in PARENT_FILES.items():
        if sha256(parent/name) != expected: raise RuntimeError(f"parent {name} drift")

def generate(output_dir: str | Path, parent_dir: str | Path | None = None) -> dict[str, Any]:
    out, parent = Path(output_dir), Path(parent_dir) if parent_dir else _parent_default()
    verify_parent(parent); out.mkdir(parents=True, exist_ok=True)
    test = rows(parent/"test_state.jsonl")
    depth = rows(parent/"depth_t_64.jsonl")
    excluded = {n: ({r["x0"] for r in test if r["N"] == n} |
                    {x for r in depth if r["N"] == n for x in r["trajectory"][:-1]})
                for n in EXPECTED_MODULI}
    train = [dense_record(n, x) for n in EXPECTED_MODULI for x in range(n) if x not in excluded[n]]
    random.Random(SEED).shuffle(train)
    if [sum(r["N"] == n for r in train) for n in EXPECTED_MODULI] != EXPECTED_DENSE_COUNTS:
        raise RuntimeError("dense count drift")
    (out/"train.jsonl").write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"))+"\n" for r in train), encoding="utf-8", newline="\n")
    for name in PARENT_FILES: shutil.copyfile(parent/name, out/name)
    config = {"stage": STAGE, "control": "S2-D1 dense_density_control seed78", "seed": SEED,
              "operation": "y=(2*x)%N", "moduli": EXPECTED_MODULI, "width": 4,
              "decimal_order": "LSD-first", "input_layout": "[STATE,x[4],MODULUS,N[4]]",
              "model_visible_fields": ["x0", "N"], "dense_train_split": "dense_train",
              "dense_policy": "every (N,x) absent from frozen test_state and frozen T64 source-edge partition; no sampling or balancing",
              "split_counts": {"train": len(train), "test_state": len(test), **{f"depth_t_{t}": 512 for t in RUNGS}},
              "per_modulus_dense_counts": dict(zip(map(str, EXPECTED_MODULI), EXPECTED_DENSE_COUNTS)),
              "parent_root": "data/generated/s2_doublemod_seen_v1", "parent_pins": PARENT_PINS,
              "copied_file_sha256": PARENT_FILES, "generator_path": "data/s2_doublemod_seen_dense_v1.py",
              "generator_sha256": sha256(Path(__file__).resolve())}
    (out/"dataset_config.json").write_text(json.dumps(config, sort_keys=True, indent=2)+"\n", encoding="utf-8", newline="\n")
    files = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != "artifact_manifest.json"}
    manifest = {"algorithm": "sha256", "excluded": ["artifact_manifest.json"], "files": files}
    (out/"artifact_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n", encoding="utf-8", newline="\n")
    return config

def audit(root_dir: str | Path, parent_dir: str | Path | None = None, regenerate: bool = True) -> dict[str, Any]:
    root, parent = Path(root_dir), Path(parent_dir) if parent_dir else _parent_default()
    verify_parent(parent)
    cfg = json.loads((root/"dataset_config.json").read_text(encoding="utf-8"))
    man = json.loads((root/"artifact_manifest.json").read_text(encoding="utf-8"))
    assert man["excluded"] == ["artifact_manifest.json"] and "artifact_manifest.json" not in man["files"]
    assert all(sha256(root/name) == value for name, value in man["files"].items())
    assert cfg["generator_sha256"] == sha256(Path(__file__).resolve())
    assert all((root/name).read_bytes() == (parent/name).read_bytes() for name in PARENT_FILES)
    train, test, depth = rows(root/"train.jsonl"), rows(root/"test_state.jsonl"), rows(root/"depth_t_64.jsonl")
    dense = {(r["N"], r["x0"]) for r in train}; frozen = {(r["N"], r["x0"]) for r in test}
    all_states = {(n, x) for n in EXPECTED_MODULI for x in range(n)}
    edges = {(r["N"], x) for r in depth for x in r["trajectory"][:-1]}
    assert len(train) == EXPECTED_TOTAL and len(dense) == EXPECTED_TOTAL
    assert not dense & frozen and not dense & edges and not frozen & edges
    assert dense | frozen | edges == all_states and len(all_states) == 71084
    assert [sum(r["N"] == n for r in train) for n in EXPECTED_MODULI] == EXPECTED_DENSE_COUNTS
    assert all(r["stage"] == STAGE and r["split"] == "dense_train" for r in train)
    for r in train:
        assert r["y"] == (2*r["x0"]) % r["N"] and r["labels"] == digits(r["y"]) and r["input_ids"] == input_ids(r["x0"], r["N"])
    test_targets = {(r["N"], r["y"]) for r in test if abs(2*r["x0"]-r["N"]) == 1}
    train_targets = {(r["N"], r["y"]) for r in train}
    assert len(test_targets) == 16 and not test_targets & train_targets
    assert all((n, x) in frozen for n in EXPECTED_MODULI for x in range(n) if abs(2*x-n) == 1)
    assert all((n, x) in dense for n in EXPECTED_MODULI for x in range(n) if abs(2*x-n) == 3)
    assert all((n, 0) in dense and (n, n-1) in dense for n in EXPECTED_MODULI)
    regen = True
    if regenerate:
        with tempfile.TemporaryDirectory() as td:
            generate(td, parent)
            regen = {p.name for p in root.iterdir()} == {p.name for p in Path(td).iterdir()} and all((root/p.name).read_bytes() == p.read_bytes() for p in Path(td).iterdir())
        assert regen
    distribution = {str(n): {"count": sum(r["N"] == n for r in train),
        "reduction": dict(Counter(r["reduction"][0] for r in train if r["N"] == n)),
        "leading_zero_targets": sum(r["N"] == n and r["labels"][3] == 0 for r in train)} for n in EXPECTED_MODULI}
    return {"passed": True, "stage": STAGE, "split_counts": cfg["split_counts"],
            "per_modulus_dense_counts": EXPECTED_DENSE_COUNTS, "partition_states": len(all_states),
            "unique_t64_source_edges": len(edges), "threshold_test_targets_absent_from_same_N_train_targets": True,
            "copied_files_byte_identical": True, "canonical_visibility": ["x0", "N"],
            "manifest_self_excluding": True, "byte_identical_regeneration": regen,
            "distribution": distribution, "artifact_hashes": man["files"], "parent_pins": PARENT_PINS}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("generate", "audit")); parser.add_argument("output")
    args = parser.parse_args(); print(json.dumps(generate(args.output) if args.action == "generate" else audit(args.output), indent=2, sort_keys=True))
