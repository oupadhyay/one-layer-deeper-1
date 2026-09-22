"""Build static dashboard metadata from frozen result and provenance files."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "research_portal"
RUNGS = (1, 2, 4, 8, 16, 32, 64)


def load_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def linked_path(record: dict, label: str) -> Path | None:
    target = record.get("links", {}).get(label)
    return (PORTAL / target).resolve() if target else None


def fraction(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", value)
    if not match or int(match.group(2)) == 0:
        return None
    return int(match.group(1)), int(match.group(2))


def metric_rungs(record: dict, prefix: str) -> list[dict | None]:
    result = []
    for rung in RUNGS:
        parsed = fraction(record.get("metrics", {}).get(f"{prefix}T{rung}", {}).get("exact"))
        result.append(
            {"timeSteps": rung, "correct": parsed[0], "total": parsed[1]}
            if parsed else None
        )
    return result


def official_seed(result: dict) -> dict:
    seeds = (result.get("result") or {}).get("seeds", [])
    return seeds[0] if seeds else {}


def official_rungs(seed: dict, key: str) -> list[dict | None]:
    raw = seed.get("depth_profile", {}).get(key, [])
    indexed = {item.get("time_steps"): item for item in raw}
    return [
        {
            "timeSteps": rung,
            "correct": indexed[rung]["correct_examples"],
            "total": indexed[rung]["example_count"],
        }
        if rung in indexed else None
        for rung in RUNGS
    ]


def provenance_rungs(provenance: dict, key: str, fallback_total: int) -> list[dict | None]:
    outcome = provenance.get("outcome")
    values = outcome.get(key) if isinstance(outcome, dict) else None
    if not isinstance(values, list):
        return [None] * len(RUNGS)
    return [
        {"timeSteps": rung, "correct": int(values[index]), "total": fallback_total}
        if index < len(values) and values[index] is not None else None
        for index, rung in enumerate(RUNGS)
    ]


def merge_rungs(*candidates: list[dict | None]) -> list[dict]:
    merged = [
        next((candidate[index] for candidate in candidates if candidate[index]), None)
        for index in range(len(RUNGS))
    ]
    return [rung for rung in merged if rung is not None]


def nested_value(data: dict, paths: tuple[tuple[str, ...], ...]) -> object:
    for path in paths:
        value: object = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value not in (None, "", {}):
                return value
    return None


def package_path(provenance: dict) -> str | None:
    value = nested_value(
        provenance,
        (
            ("package", "path"),
            ("package", "source"),
            ("source", "path"),
            ("package_source",),
        ),
    )
    return str(value).replace("\\", "/") if value else None


def architecture_note(record: dict, provenance: dict) -> str:
    value = nested_value(
        provenance,
        (("architecture",), ("mechanism",), ("hypothesis",), ("configuration",)),
    )
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    return f"{record['group']} — {record['protocol']}"


def inferred_tier(record: dict, result: dict) -> str | None:
    tier = result.get("tier")
    if tier in {"easy", "medium", "hard"}:
        return tier
    identifier = record["id"]
    for candidate in ("easy", "medium", "hard"):
        if f"remote_{candidate}" in identifier:
            return candidate
    return None


def main() -> None:
    source = (PORTAL / "experiments.js").read_text(encoding="utf-8")
    match = re.fullmatch(r".*?const EXPERIMENTS = (\[.*\]);\s*", source, re.S)
    if not match:
        raise RuntimeError("experiments registry is not a JSON-compatible assignment")
    records = json.loads(match.group(1))
    details = {}
    for record in records:
        result = load_json(linked_path(record, "result"))
        provenance = load_json(linked_path(record, "provenance"))
        created_at = result.get("created_at") or result.get("run_created_at")
        tier = inferred_tier(record, result)
        if not created_at or not tier:
            continue
        seed = official_seed(result)
        t1_total = fraction(record.get("metrics", {}).get("seenT1", {}).get("exact"))
        ood_t1_total = fraction(record.get("metrics", {}).get("oodNT1", {}).get("exact"))
        seen = merge_rungs(
            official_rungs(seed, "rungs"),
            provenance_rungs(provenance, "seen_n_rungs", t1_total[1] if t1_total else 512),
            metric_rungs(record, "seen"),
        )
        ood = merge_rungs(
            official_rungs(seed, "ood_n_rungs"),
            provenance_rungs(provenance, "ood_n_rungs", ood_t1_total[1] if ood_t1_total else 512),
            metric_rungs(record, "oodN"),
        )
        details[record["id"]] = {
            "createdAt": created_at,
            "tier": tier,
            "dataset": result.get("dataset_id") or result.get("dataset") or result.get("suite"),
            "submissionId": result.get("id") or result.get("submission_id") or provenance.get("submission_id"),
            "runId": result.get("run_id") or provenance.get("run_id"),
            "packagePath": package_path(provenance),
            "architecture": architecture_note(record, provenance),
            "seenRungs": seen,
            "oodRungs": ood,
        }
    output = "const RUN_DETAILS = " + json.dumps(details, separators=(",", ":"), sort_keys=True) + ";\n"
    (PORTAL / "run_details.js").write_text(output, encoding="utf-8")
    print(f"wrote {len(details)} dated remote run details")


if __name__ == "__main__":
    main()
