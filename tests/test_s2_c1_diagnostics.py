import unittest
from pathlib import Path

from benchmark.s2_c1_diagnostics import sha256, verify_dataset


class S2C1DiagnosticsTests(unittest.TestCase):
    def test_repaired_dataset_hashes_and_frozen_inputs(self):
        root = Path("data/generated/s2_doublemod_seen_v1")
        hashes = verify_dataset(root)
        self.assertEqual(hashes["train.jsonl"], "cb78b65e281484b54b5c957af4149943857e389fbf926339499c0113de984f25")
        self.assertEqual(hashes["test_state.jsonl"], "4accb941940d5ff9d513b3d887c20a589e9c6c348b9833db64129b01fc65fcb2")
        self.assertEqual(sha256(Path("artifacts/s2_c1_transformer_seed78.pt")),
                         "78114f47b9516862cca5ae6191dad06318d3a81d1b52aab3d27f2b0d33c18ab7")

    def test_supplement_has_expected_frozen_counts(self):
        import json
        report = json.loads(Path("artifacts/s2_c1_transformer_seed78_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["train"]["overall"]["exact_correct"], 16000)
        self.assertEqual(report["test_state"]["overall"]["exact_correct"], 3989)
        self.assertEqual(report["test_errors"]["total"], 11)
        self.assertEqual(len(report["threshold"]["test_abs_2x_minus_N_eq_1_gate"]), 16)
        self.assertEqual(len(report["threshold"]["train_abs_2x_minus_N_eq_3_diagnostic"]), 16)
        self.assertFalse(report["preserved_gate"]["g1_pass"])


if __name__ == "__main__":
    unittest.main()
