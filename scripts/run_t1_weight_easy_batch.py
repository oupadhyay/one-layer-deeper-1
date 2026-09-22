"""Submit the six validated T1-weight Easy packages sequentially."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "t1_weight_easy_batch_live.json"
NAMES = (
    "conditional_t1_weight2_easy_v1",
    "conditional_t1_weight4_easy_v1",
    "conditional_t1_weight8_easy_v1",
    "conditional_t1_weight16_easy_v1",
    "conditional_t1_weight4_plain_easy_v1",
    "conditional_t1_weight4_d64_easy_v1",
)


def main() -> None:
    attempts = []
    for name in NAMES:
        path = ROOT / "submissions" / name / "submission.py"
        result = subprocess.run(
            [sys.executable, "-m", "client.cli", "submit", "--tier", "easy",
             "--dataset", "e5", "--wait", str(path)],
            cwd=ROOT,
            capture_output=True,
        )
        output = result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace")
        match = re.search(r"(?:queued|submission)[: ]+\s*([0-9a-f-]{36})", output)
        attempts.append({
            "name": name,
            "submission_id": match.group(1) if match else None,
            "return_code": result.returncode,
            "output": output,
        })
        LOG.write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(output)


if __name__ == "__main__":
    main()
