"""Generate and audit the isolated S1 fixed-width decimal add-with-carry stage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Sequence


DEFAULT_SEED = 20260729
MODULUS = 10_000
WIDTH = 4
A_TRAIN = (1, 2, 4, 5, 7, 8)
A_TEST = (3, 6, 9)
RUNGS = (2, 4, 8, 16, 32, 64)
TRAIN_PER_A = 3334
TEST_STATE_PER_A = 834
DEPTH_ROOTS_PER_A = {1: 86, 2: 86, 4: 85, 5: 85, 7: 85, 8: 85}
EXPECTED_COUNTS = {
    "train": TRAIN_PER_A * len(A_TRAIN),
    "test_state": TEST_STATE_PER_A * len(A_TRAIN),
    "test_context": MODULUS * len(A_TEST),
    **{f"depth_t_{horizon}": 512 for horizon in RUNGS},
}


def digits(value: int) -> list[int]:
    if not 0 <= value < MODULUS:
        raise ValueError("state must be in [0, 10000)")
    return [(value // (10 ** place)) % 10 for place in range(WIDTH)]


def decode(values: Sequence[int]) -> int:
    if len(values) != WIDTH or any(not 0 <= int(value) <= 9 for value in values):
        raise ValueError("canonical state requires four decimal digits")
    return sum(int(value) * 10 ** place for place, value in enumerate(values))


def transition(x: int, a: int) -> int:
    return (x + a) % MODULUS


def carry_length(x: int, a: int) -> int:
    """Number of decimal places through which the carry propagates."""
    carry = 1 if x % 10 + a >= 10 else 0
    if not carry:
        return 0
    length = 1
    for place in range(1, WIDTH):
        if (x // (10 ** place)) % 10 != 9:
            return length
        length += 1
    return WIDTH


def _audit_carry_length(x: int, a: int) -> int:
    """Independent audit implementation, deliberately separate from generation."""
    if (x % 10) + a < 10:
        return 0
    if (x // 10) % 10 < 9:
        return 1
    if (x // 100) % 10 < 9:
        return 2
    if (x // 1000) % 10 < 9:
        return 3
    return 4


def input_ids(x: int, a: int) -> list[int]:
    return [0, *[digit + 2 for digit in digits(x)], 1, *[digit + 2 for digit in digits(a)]]


def _record(split: str, a: int, x: int, horizon: int, seed: int) -> dict[str, Any]:
    trajectory = [x]
    carries: list[int] = []
    for _ in range(horizon):
        carries.append(carry_length(trajectory[-1], a))
        trajectory.append(transition(trajectory[-1], a))
    return {
        "stage": "S1", "version": "s1_addcarry_v1", "split": split,
        "context_id": f"addend_{a}", "a": a, "x0": x, "T": horizon,
        "y": trajectory[-1], "trajectory": trajectory,
        "carry_length": carries[0], "carry_lengths": carries,
        "input_ids": input_ids(x, a), "labels": digits(trajectory[-1]), "seed": seed,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_status(root: Path, source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short", "--", str(source.relative_to(root))], cwd=root, text=True
        ).strip() or "clean"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "unavailable"


def _select_depth_roots(rng: random.Random) -> list[tuple[int, int]]:
    # Preserve two rare examples per feasible stratum/addend for one-step splits.
    reserved: set[tuple[int, int]] = set()
    for a in A_TRAIN:
        for stratum in (3, 4):
            candidates = [x for x in range(MODULUS) if carry_length(x, a) == stratum]
            reserved.update((a, x) for x in candidates[: min(2, len(candidates))])

    roots: list[tuple[int, int]] = []
    for a in A_TRAIN:
        candidates = list(range(MODULUS))
        rng.shuffle(candidates)
        used: set[int] = set()
        selected: list[int] = []
        # Seed long-carry exposure where enough rare support remains after reservation.
        preferred = [x for x in candidates if any(carry_length((x + step * a) % MODULUS, a) >= 3
                                                  for step in range(64))]
        candidates = preferred[:1] + [x for x in candidates if x not in set(preferred[:1])]
        for root in candidates:
            states = [(root + step * a) % MODULUS for step in range(65)]
            edges = states[:-1]
            if len(set(states)) != 65 or any(edge in used or (a, edge) in reserved for edge in edges):
                continue
            selected.append(root)
            used.update(edges)
            if len(selected) == DEPTH_ROOTS_PER_A[a]:
                break
        if len(selected) != DEPTH_ROOTS_PER_A[a]:
            raise RuntimeError(f"could not allocate unique depth segments for a={a}")
        roots.extend((a, root) for root in selected)
    rng.shuffle(roots)
    return roots


def _allocate_one_step(a: int, excluded: set[int], train_count: int, test_count: int,
                       rng: random.Random) -> tuple[list[int], list[int]]:
    groups = {stratum: [x for x in range(MODULUS) if x not in excluded and carry_length(x, a) == stratum]
              for stratum in range(5)}
    for values in groups.values():
        rng.shuffle(values)
    train: list[int] = []
    test: list[int] = []
    # Explicitly place every supported stratum in both splits before random fill.
    for stratum, values in groups.items():
        if values:
            train.append(values.pop())
        if values:
            test.append(values.pop())
    pool = [value for values in groups.values() for value in values]
    rng.shuffle(pool)
    train_needed = train_count - len(train)
    test_needed = test_count - len(test)
    train.extend(pool[:train_needed])
    test.extend(pool[train_needed:train_needed + test_needed])
    if len(train) != train_count or len(test) != test_count or set(train) & set(test):
        raise RuntimeError(f"one-step allocation failed for a={a}")
    return train, test


def _interleave(records: list[dict[str, Any]], rng: random.Random) -> None:
    rng.shuffle(records)


def generate_s1_addcarry_v1(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    roots = _select_depth_roots(rng)
    depth_edges: dict[int, set[int]] = defaultdict(set)
    for a, root in roots:
        depth_edges[a].update((root + step * a) % MODULUS for step in range(64))

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in A_TRAIN:
        train_x, test_x = _allocate_one_step(a, depth_edges[a], TRAIN_PER_A, TEST_STATE_PER_A, rng)
        splits["train"].extend(_record("train", a, x, 1, seed) for x in train_x)
        splits["test_state"].extend(_record("test_state", a, x, 1, seed) for x in test_x)
    for a in A_TEST:
        splits["test_context"].extend(_record("test_context", a, x, 1, seed) for x in range(MODULUS))
    for horizon in RUNGS:
        name = f"depth_t_{horizon}"
        splits[name] = [_record(name, a, root, horizon, seed) for a, root in roots]
    for records in splits.values():
        _interleave(records, rng)

    carry_counts = {
        name: {str(a): {str(c): sum(row["a"] == a and row["carry_length"] == c for row in records)
                        for c in range(5)} for a in sorted({row["a"] for row in records})}
        for name, records in splits.items() if name in ("train", "test_state", "test_context")
    }
    carry_support = {
        str(a): {str(c): sum(carry_length(x, a) == c for x in range(MODULUS)) for c in range(5)}
        for a in (*A_TRAIN, *A_TEST)
    }
    source = Path(__file__).resolve()
    repo_root = source.parents[1]
    config = {
        "stage": "S1", "version": "s1_addcarry_v1", "generator_seed": seed,
        "generator_path": "data/s1_addcarry_v1.py", "generator_sha256": _sha256(source),
        "repo_head": _repo_head(repo_root), "generator_git_status": _source_status(repo_root, source),
        "operation": "F_a(x)=(x+a) mod 10000", "modulus": MODULUS, "width": WIDTH,
        "decimal_order": "LSD-first", "explicit_leading_zeroes": True,
        "a_train": list(A_TRAIN), "a_test": list(A_TEST), "nested_depth_roots": True,
        "depth_rungs": list(RUNGS), "depth_root_count": 512,
        "split_counts": {name: len(records) for name, records in splits.items()},
        "carry_counts": carry_counts, "carry_support": carry_support,
        "rare_carry_allocation_policy": (
            "Reserve up to two c3/c4 edges per seen addend before depth selection, then place one in "
            "train and one in test-state when support permits. The sole a=1,c=4 edge is assigned to train."
        ),
        "model_visible_fields": ["x0", "a"],
        "input_layout": "[STATE, x0[4], ADDEND, a[4]]", "label_width": WIDTH,
        "token_map": {"STATE": 0, "ADDEND": 1, **{str(digit): digit + 2 for digit in range(10)}},
    }
    for name, records in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    (output / "dataset_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    files = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.name != "artifact_manifest.json"}
    manifest = {"algorithm": "sha256", "excluded": ["artifact_manifest.json"], "files": files}
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return config


def _load(root: Path, split: str) -> list[dict[str, Any]]:
    with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _position_stats(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, float]]:
    result = []
    for place in range(WIDTH):
        counts = Counter(digits(row[field])[place] for row in rows)
        total = len(rows)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        result.append({"entropy_bits": entropy, "modal_frequency": max(counts.values()) / total})
    return result


def _conditional_c6(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for a in sorted({row["a"] for row in rows}):
        selected = [row for row in rows if row["a"] == a]
        modal = [Counter(row["labels"][place] for row in selected).most_common(1)[0][0] for place in range(WIDTH)]
        result[str(a)] = sum(row["labels"] == modal for row in selected) / len(selected)
    return result


def audit_s1_addcarry_v1(root_dir: str | Path, verify_regeneration: bool = True) -> dict[str, Any]:
    root = Path(root_dir)
    config = json.loads((root / "dataset_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    source = Path(__file__).resolve()
    if config["generator_sha256"] != _sha256(source):
        raise AssertionError("executing generator source hash differs from config")
    for name, expected in manifest["files"].items():
        if _sha256(root / name) != expected:
            raise AssertionError(f"artifact hash mismatch: {name}")
    splits = {name: _load(root, name) for name in EXPECTED_COUNTS}
    if {name: len(rows) for name, rows in splits.items()} != EXPECTED_COUNTS:
        raise AssertionError("split counts differ from frozen contract")

    oracle_rows = 0
    for name, rows in splits.items():
        for row in rows:
            expected = [row["x0"]]
            carries = []
            for _ in range(row["T"]):
                carries.append(_audit_carry_length(expected[-1], row["a"]))
                expected.append((expected[-1] + row["a"]) % MODULUS)
            if row["trajectory"] != expected or row["y"] != expected[-1] or row["carry_lengths"] != carries:
                raise AssertionError(f"oracle mismatch in {name}")
            if row["labels"] != digits(row["y"]) or len(row["labels"]) != WIDTH:
                raise AssertionError("noncanonical or ignored answer slot")
            if row["input_ids"] != input_ids(row["x0"], row["a"]):
                raise AssertionError("model-visible tensor reconstruction mismatch")
            oracle_rows += 1

    train_edges = {(row["a"], row["x0"]) for row in splits["train"]}
    test_edges = {(row["a"], row["x0"]) for row in splits["test_state"]}
    depth_edges = {(row["a"], x) for row in splits["depth_t_64"] for x in row["trajectory"][:-1]}
    if len(depth_edges) != 512 * 64:
        raise AssertionError("depth directed edges are not unique")
    if train_edges & test_edges or depth_edges & (train_edges | test_edges):
        raise AssertionError("directed edge quarantine failed")
    if {row["a"] for row in splits["test_context"]} & set(A_TRAIN):
        raise AssertionError("held-out contexts overlap seen addends")
    for row in splits["depth_t_64"]:
        if len(set(row["trajectory"])) != 65:
            raise AssertionError("depth trajectory repeats within 64 steps")
    full_by_root = {(row["a"], row["x0"]): row for row in splits["depth_t_64"]}
    for horizon in RUNGS[:-1]:
        for row in splits[f"depth_t_{horizon}"]:
            full = full_by_root[(row["a"], row["x0"])]
            if row["trajectory"] != full["trajectory"][:horizon + 1]:
                raise AssertionError("nested rung is not a T64 prefix")

    actual_carries = {
        name: {str(a): {str(c): sum(row["a"] == a and row["carry_length"] == c for row in splits[name])
                        for c in range(5)} for a in sorted({row["a"] for row in splits[name]})}
        for name in ("train", "test_state", "test_context")
    }
    if actual_carries != config["carry_counts"]:
        raise AssertionError("carry counts differ from config")
    for a in A_TRAIN:
        for c in range(5):
            support = config["carry_support"][str(a)][str(c)]
            if support >= 2 and (actual_carries["train"][str(a)][str(c)] == 0
                                 or actual_carries["test_state"][str(a)][str(c)] == 0):
                raise AssertionError(f"supported carry stratum missing from one-step splits: a={a}, c={c}")

    byte_identical = True
    if verify_regeneration:
        with tempfile.TemporaryDirectory() as temporary:
            generate_s1_addcarry_v1(temporary, config["generator_seed"])
            generated = Path(temporary)
            names = sorted(path.name for path in root.iterdir())
            byte_identical = names == sorted(path.name for path in generated.iterdir()) and all(
                (root / name).read_bytes() == (generated / name).read_bytes() for name in names
            )
            if not byte_identical:
                raise AssertionError("same-seed regeneration is not byte-identical")

    entropy = {name: {field: _position_stats(rows, field) for field in ("x0", "y")}
               for name, rows in splits.items()}
    depth_carry_per_step = {
        name: [{str(c): sum(row["carry_lengths"][step] == c for row in rows) for c in range(5)}
               for step in range(rows[0]["T"])]
        for name, rows in splits.items() if name.startswith("depth_t_")
    }
    first_5000 = splits["test_context"][:5000]
    first_5000_composition = {
        "per_a": {str(a): sum(row["a"] == a for row in first_5000) for a in A_TEST},
        "per_carry": {str(c): sum(row["carry_length"] == c for row in first_5000) for c in range(5)},
    }
    if any(count == 0 for count in first_5000_composition["per_a"].values()):
        raise AssertionError("first 5000 test-context rows do not interleave all contexts")
    return {
        "passed": True, "oracle_accuracy": 1.0, "oracle_rows": oracle_rows,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "carry_counts": actual_carries, "depth_carry_per_step": depth_carry_per_step,
        "position_stats": entropy,
        "conditional_on_a_c6_exact": {name: _conditional_c6(rows) for name, rows in splits.items()},
        "train_test_edge_overlap": 0, "depth_one_step_edge_overlap": 0,
        "unique_depth_edges": len(depth_edges), "first_5000_test_context_composition": first_5000_composition,
        "nested_depth_rungs": True, "trajectory_repeats_within_64": 0,
        "model_visible_fields": ["x0", "a"], "byte_identical_regeneration": byte_identical,
        "generator_sha256": config["generator_sha256"], "repo_head": config["repo_head"],
        "generator_git_status": config["generator_git_status"],
        "artifact_hashes_verified": len(manifest["files"]),
    }


class StrictAPIContract:
    """G0-only API validator; it performs no learned computation."""

    def __init__(self, model: Any) -> None:
        signature = inspect.signature(model.step.forward)
        if tuple(signature.parameters) != ("state", "context"):
            raise TypeError("step must accept only (state, context)")
        self.model = model
        self.step_identity = id(model.step)

    def canonical_reset(self, predicted_digits: Sequence[int]) -> Any:
        if id(self.model.step) != self.step_identity:
            raise AssertionError("step identity changed")
        return self.model.encode_state(digits(decode(predicted_digits)))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate or audit s1_addcarry_v1")
    parser.add_argument("action", choices=("generate", "audit"))
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = (generate_s1_addcarry_v1(args.output, args.seed) if args.action == "generate"
              else audit_s1_addcarry_v1(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
