from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.s1_c1 import CONFIG as C1_CONFIG, S1GRUTransducer, load_eval_rows, load_projected_train
from benchmark.s1_r2 import JsonlParseMonitor, R2_CONFIG, config_diff


class S1R2Tests(unittest.TestCase):
    def test_config_diff_is_seed_only(self) -> None:
        self.assertEqual(config_diff(), {"seed": {"c1": 76, "r2": 77}})
        c1, r2 = asdict(C1_CONFIG), asdict(R2_CONFIG)
        r2["seed"] = c1["seed"]
        self.assertEqual(c1, r2)

    def test_replication_imports_exact_frozen_architecture(self) -> None:
        model = S1GRUTransducer(R2_CONFIG.embedding_dim, R2_CONFIG.hidden_dim)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 25930)
        self.assertEqual(model.__class__, S1GRUTransducer)

    def test_actual_optimization_jsonl_open_is_instrumented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"input_ids": [0, 2, 2, 2, 2, 1, 3, 2, 2, 2], "labels": [1, 0, 0, 0]}
            (root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "test_state.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            manual: list[str] = []
            with JsonlParseMonitor() as monitor:
                load_projected_train(root, manual)
            self.assertEqual(monitor.opened, ["train.jsonl"])
            self.assertEqual(manual, ["train.jsonl"])

            # Prove interception is independent of the application's manual log.
            with JsonlParseMonitor() as monitor:
                load_eval_rows(root, "test_state", [])
            self.assertEqual(monitor.opened, ["test_state.jsonl"])

    def test_depth_split_is_rejected_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with JsonlParseMonitor() as monitor:
                with self.assertRaises(ValueError):
                    load_eval_rows(directory, "depth_t_2", [])
            self.assertEqual(monitor.opened, [])


if __name__ == "__main__":
    unittest.main()
