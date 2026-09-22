"""Submit the validated scale batch sequentially under the remote active-job gate."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "conditional_scale_remote_batch_live.json"
MEDIUM = [
    ("conditional_medium_focal05_lr9_v1", "a2c10fbb-60c6-44a2-b1e3-9d54b311bcee"),
    ("conditional_medium_plain_lr9_v1", None),
    ("conditional_medium_focal10_lr9_v1", None),
    ("conditional_medium_focal05_lr6_v1", None),
    ("conditional_medium_focal05_batch128_v1", None),
    ("conditional_medium_focal05_d96_v1", None),
    ("conditional_medium_focal05_lr10_v1", None),
    ("conditional_medium_focal05_warm32_v1", None),
    ("conditional_medium_focal05_batch512_v1", None),
    ("conditional_medium_focal05_d128_v1", None),
]
HARD = ("conditional_dynamic_hard_v1", None)
TERMINAL = {"succeeded", "failed", "cancelled"}


def command(*arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "client.cli", *arguments],
        cwd=ROOT,
        capture_output=True,
    )
    output = result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace")
    if result.returncode:
        raise RuntimeError(output)
    return output


def jobs() -> list[dict]:
    output = command("jobs", "--all", "--json")
    return json.loads(output[output.find("["):])


def write_log(records: list[dict]) -> None:
    LOG.write_text(json.dumps({"attempts": records}, indent=2), encoding="utf-8")


def wait_for(identifier: str, records: list[dict]) -> None:
    while True:
        job = next(item for item in jobs() if item["id"] == identifier)
        records[-1]["status"] = job["status"]
        records[-1]["run_id"] = job.get("run_id")
        records[-1]["finished_at"] = job.get("finished_at")
        write_log(records)
        if job["status"] in TERMINAL:
            return
        time.sleep(30)


def submit(name: str, tier: str, records: list[dict]) -> str:
    path = ROOT / "submissions" / name / "submission.py"
    arguments = ["submit", "--tier", tier]
    if tier == "medium":
        arguments.extend(("--dataset", "m3"))
    output = command(*arguments, str(path))
    match = re.search(r"queued:\s+([0-9a-f-]{36})", output)
    if not match:
        raise RuntimeError(f"accepted submission ID was not found:\n{output}")
    identifier = match.group(1)
    records.append({"name": name, "tier": tier, "submission_id": identifier, "status": "queued"})
    write_log(records)
    return identifier


def main() -> None:
    records: list[dict] = []
    for index, (name, existing) in enumerate(MEDIUM):
        if existing:
            records.append({"name": name, "tier": "medium", "submission_id": existing, "status": "accepted"})
            identifier = existing
            write_log(records)
        else:
            identifier = submit(name, "medium", records)
        wait_for(identifier, records)
        if records[-1]["status"] == "failed":
            records[-1]["note"] = "The exact configuration failed; the next independent package remains valid."
            write_log(records)
    name, _ = HARD
    identifier = submit(name, "hard", records)
    wait_for(identifier, records)


if __name__ == "__main__":
    main()
