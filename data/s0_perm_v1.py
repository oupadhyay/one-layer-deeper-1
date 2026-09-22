"""Deterministic, isolated S0 permutation dataset and G0 audits.

Model input layout is ``STATE, ones(x0), tens(x0), TABLE,`` followed by the
LSD-first two-digit encoding of ``perm[0] ... perm[M-1]``.  Token ids 0 and 1
are STATE and TABLE; ids 2--11 are decimal digits 0--9.  Labels are exactly the
two LSD-first digit tokens for the endpoint.  In particular, T, context ids,
step numbers, and trajectories are never model-visible.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Iterator


STAGE = "S0"
VERSION = "s0_perm_v1"
DEFAULT_SEED = 20260729
TOKEN_MAP = {"STATE": 0, "TABLE": 1, **{str(i): i + 2 for i in range(10)}}
DEPTHS = (2, 4, 8, 16, 32, 64)
EXPECTED_COUNTS = {
    "train": 26_000, "test_state": 6_000, "test_context": 5_120,
    "ood_size": 4_800, **{f"depth_t_{t}": 512 for t in DEPTHS},
}
ARTIFACT_MANIFEST = "artifact_manifest.json"


def _digits(value: int) -> list[int]:
    return [TOKEN_MAP[str(value % 10)], TOKEN_MAP[str(value // 10)]]


def _tokens(x0: int, perm: list[int]) -> list[int]:
    return [TOKEN_MAP["STATE"], *_digits(x0), TOKEN_MAP["TABLE"],
            *(token for value in perm for token in _digits(value))]


def _trajectory(perm: list[int], x0: int, steps: int) -> list[int]:
    values = [x0]
    for _ in range(steps):
        values.append(perm[values[-1]])
    return values


def _record(split: str, context_id: str, perm: list[int], x0: int,
            steps: int, seed: int) -> dict[str, Any]:
    trajectory = _trajectory(perm, x0, steps)
    return {
        "stage": STAGE, "version": VERSION, "split": split,
        "context_id": context_id, "M": len(perm), "perm": perm, "x0": x0,
        "T": steps, "y": trajectory[-1], "trajectory": trajectory,
        "input_ids": _tokens(x0, perm), "labels": _digits(trajectory[-1]),
        "seed": seed,
    }


def _fresh_perm(rng: random.Random, size: int, seen: set[tuple[int, ...]]) -> list[int]:
    while True:
        perm = list(range(size))
        rng.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            return perm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _generator_git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(Path(__file__).resolve())],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "clean"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generate_s0_perm_v1(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Generate the complete normative S0 dataset and return its configuration."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    handles: dict[str, Any] = {}

    def emit(record: dict[str, Any]) -> None:
        split = record["split"]
        if split not in handles:
            handles[split] = (root / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        handles[split].write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    try:
        # Rotating held-out domains and codomains make both state marginals exact.
        for index in range(2_000):
            held_x = [(3 * index + j) % 16 for j in range(3)]
            held_y = [(3 * index + j + 7) % 16 for j in range(3)]
            perm = [-1] * 16
            ys = held_y[:]
            rng.shuffle(ys)
            for x, y in zip(held_x, ys):
                perm[x] = y
            rest_x = [x for x in range(16) if x not in held_x]
            rest_y = [y for y in range(16) if y not in held_y]
            rng.shuffle(rest_y)
            for x, y in zip(rest_x, rest_y):
                perm[x] = y
            key = tuple(perm)
            if key in seen:  # astronomically unlikely; preserve exact balancing
                raise RuntimeError("duplicate balanced training permutation")
            seen.add(key)
            cid = f"train_{index:04d}"
            for x in rest_x:
                emit(_record("train", cid, perm, x, 1, seed))
            for x in held_x:
                emit(_record("test_state", cid, perm, x, 1, seed))

        for index in range(320):
            perm = _fresh_perm(rng, 16, seen)
            for x in range(16):
                emit(_record("test_context", f"context_{index:04d}", perm, x, 1, seed))
        for index in range(200):
            perm = _fresh_perm(rng, 24, seen)
            for x in range(24):
                emit(_record("ood_size", f"ood24_{index:04d}", perm, x, 1, seed))

        # Exact x/y quotas provide strict marginal certification at every rung.
        for steps in DEPTHS:
            y_counts: Counter[int] = Counter()
            short = 0
            for index in range(512):
                x = index % 16
                while True:
                    perm = _fresh_perm(rng, 16, seen)
                    trajectory = _trajectory(perm, x, steps)
                    y = trajectory[-1]
                    is_short = perm[x] == x or perm[perm[x]] == x
                    if y_counts[y] < 32 and (not is_short or short < 25):
                        break
                    seen.remove(tuple(perm))
                y_counts[y] += 1
                short += int(is_short)
                emit(_record(f"depth_t_{steps}", f"depth{steps}_{index:04d}",
                             perm, x, steps, seed))
    finally:
        for handle in handles.values():
            handle.close()

    generator_path = Path(__file__).resolve()
    config = {
        "stage": STAGE, "version": VERSION, "generator_seed": seed,
        "generator_sha256": _sha256(generator_path),
        "generator_path": "data/s0_perm_v1.py",
        "repo_head": _repo_head(),
        "generator_git_status": _generator_git_status(),
        "state_width": 2, "label_width": 2,
        "decimal_order": "LSD-first", "explicit_leading_zeroes": True,
        "token_map": TOKEN_MAP, "input_layout": "STATE x0[2] TABLE perm[0:M][2]",
        "model_visible_fields": ["x0", "perm"], "split_counts": EXPECTED_COUNTS,
        "train_contexts": 2_000, "test_contexts": 320,
        "ood_size_contexts": 200, "depth_records_per_rung": 512,
        "c6_expected_exact": {"M16": 1 / 16, "M24": 1 / 24},
    }
    (root / "dataset_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    artifact_names = ["dataset_config.json", *(f"{split}.jsonl" for split in EXPECTED_COUNTS)]
    artifact_manifest = {
        "algorithm": "sha256",
        "excluded": [ARTIFACT_MANIFEST],
        "files": {name: _sha256(root / name) for name in sorted(artifact_names)},
    }
    (root / ARTIFACT_MANIFEST).write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return config


def _rows(root: Path) -> Iterator[dict[str, Any]]:
    for split in EXPECTED_COUNTS:
        with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def audit_s0_perm_v1(root: str | Path) -> dict[str, Any]:
    """Independently validate G0 invariants; raise AssertionError on any failure."""
    root = Path(root)
    config = json.loads((root / "dataset_config.json").read_text(encoding="utf-8"))
    generator_path = Path(__file__).resolve()
    assert config["generator_sha256"] == _sha256(generator_path)
    assert config["generator_path"] == "data/s0_perm_v1.py"
    assert config["repo_head"] and config["generator_git_status"]
    artifact_manifest = json.loads((root / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
    assert artifact_manifest["algorithm"] == "sha256"
    assert artifact_manifest["excluded"] == [ARTIFACT_MANIFEST]
    for name, expected_hash in artifact_manifest["files"].items():
        assert _sha256(root / name) == expected_hash
    counts: Counter[str] = Counter()
    contexts: dict[str, set[str]] = {}
    edges: dict[str, set[tuple[str, int, int]]] = {"train": set(), "test_state": set()}
    marginals: dict[str, dict[str, Counter[int]]] = {}
    perms: dict[str, tuple[int, ...]] = {}
    short: Counter[str] = Counter()
    depth_edges: set[tuple[str, int, int]] = set()
    for row in _rows(root):
        assert set(row) == {"stage", "version", "split", "context_id", "M", "perm",
                            "x0", "T", "y", "trajectory", "input_ids", "labels", "seed"}
        assert row["stage"] == STAGE and row["version"] == VERSION
        assert row["seed"] == config["generator_seed"]
        split = row["split"]
        counts[split] += 1
        contexts.setdefault(split, set()).add(row["context_id"])
        assert sorted(row["perm"]) == list(range(row["M"]))
        trajectory = _trajectory(row["perm"], row["x0"], row["T"])
        assert row["trajectory"] == trajectory and row["y"] == trajectory[-1]
        assert row["labels"] == _digits(row["y"]) and len(row["labels"]) == 2
        assert row["input_ids"] == _tokens(row["x0"], row["perm"])
        assert len(row["input_ids"]) == 4 + 2 * row["M"]
        key = tuple(row["perm"])
        old = perms.setdefault(row["context_id"], key)
        assert old == key
        if split in edges:
            edges[split].add((row["context_id"], row["x0"], row["y"]))
        if split != "train":
            bucket = marginals.setdefault(split, {"x0": Counter(), "y": Counter()})
            bucket["x0"][row["x0"]] += 1
            bucket["y"][row["y"]] += 1
        if split.startswith("depth_t_"):
            depth_edges.update(
                (row["context_id"], source, target)
                for source, target in zip(trajectory, trajectory[1:])
            )
            short[split] += int(row["perm"][row["x0"]] == row["x0"] or
                                row["perm"][row["perm"][row["x0"]]] == row["x0"])
    assert dict(counts) == EXPECTED_COUNTS
    assert edges["train"].isdisjoint(edges["test_state"])
    train_ids = contexts["train"]
    assert contexts["test_state"] == train_ids
    fresh_splits = [split for split in contexts if split not in ("train", "test_state")]
    for offset, split in enumerate(fresh_splits):
        assert contexts[split].isdisjoint(train_ids)
        for other in fresh_splits[offset + 1:]:
            assert contexts[split].isdisjoint(contexts[other])
    assert len(perms) == len(set(perms.values()))
    modal: dict[str, dict[str, float]] = {}
    entropy_bits: dict[str, dict[str, float]] = {}
    for split, fields in marginals.items():
        size = 24 if split == "ood_size" else 16
        modal[split] = {}
        entropy_bits[split] = {}
        for field, frequency in fields.items():
            value = max(frequency.values()) / counts[split]
            assert value < 1.5 / size
            modal[split][field] = value
            probabilities = [count / counts[split] for count in frequency.values()]
            entropy = -sum(probability * math.log2(probability) for probability in probabilities)
            assert math.isclose(entropy, math.log2(size), abs_tol=1e-12)
            entropy_bits[split][field] = entropy
    for split, value in short.items():
        assert value / counts[split] <= .05
    assert config["token_map"] == TOKEN_MAP and config["split_counts"] == EXPECTED_COUNTS
    assert config["state_width"] == config["label_width"] == 2
    assert config["generator_seed"] is not None
    depth_training_edge_overlap = depth_edges & edges["train"]
    assert not depth_training_edge_overlap
    c6_expected_exact = config.get(
        "c6_expected_exact", {"M16": 1 / 16, "M24": 1 / 24}
    )
    if config["generator_seed"] == DEFAULT_SEED:
        with tempfile.TemporaryDirectory(prefix="s0-perm-v1-audit-") as regenerated:
            regenerated_root = Path(regenerated)
            generate_s0_perm_v1(regenerated_root, seed=DEFAULT_SEED)
            expected_names = {*artifact_manifest["files"], ARTIFACT_MANIFEST}
            assert {path.name for path in regenerated_root.iterdir()} == expected_names
            for name in expected_names:
                assert (root / name).read_bytes() == (regenerated_root / name).read_bytes()
    return {
        "passed": True,
        "oracle_accuracy": 1.0,
        "split_counts": dict(counts),
        "modal_frequencies": modal,
        "entropy_bits": entropy_bits,
        "c6_expected_exact": c6_expected_exact,
        "depth_training_edge_overlap_fraction": 0.0,
        "generator_sha256": config["generator_sha256"],
        "repo_head": config["repo_head"],
        "generator_git_status": config["generator_git_status"],
        "artifact_hashes_verified": len(artifact_manifest["files"]),
        "byte_identical_regeneration": config["generator_seed"] == DEFAULT_SEED,
        "short_cycle_counts": dict(short),
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate or audit isolated s0_perm_v1 data")
    parser.add_argument("mode", choices=("generate", "audit"))
    parser.add_argument("root")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate_s0_perm_v1(args.root, args.seed) if args.mode == "generate" else audit_s0_perm_v1(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
