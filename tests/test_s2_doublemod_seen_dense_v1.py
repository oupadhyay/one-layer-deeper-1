import json, tempfile, unittest
from pathlib import Path

from data.s2_doublemod_seen_dense_v1 import (EXPECTED_DENSE_COUNTS, PARENT_FILES,
    audit, generate, sha256)

ROOT=Path("data/generated/s2_doublemod_seen_dense_v1")
PARENT=Path("data/generated/s2_doublemod_seen_v1")

class DenseDatasetTests(unittest.TestCase):
    def test_g0_partition_thresholds_and_visibility(self):
        result=audit(ROOT,regenerate=False)
        self.assertTrue(result["passed"] and result["manifest_self_excluding"])
        self.assertEqual(result["partition_states"],71084)
        self.assertEqual(result["per_modulus_dense_counts"],EXPECTED_DENSE_COUNTS)
        self.assertEqual(result["canonical_visibility"],["x0","N"])
        self.assertTrue(result["threshold_test_targets_absent_from_same_N_train_targets"])
    def test_copies_hashes_and_parent_metadata(self):
        for name,h in PARENT_FILES.items():
            self.assertEqual(sha256(ROOT/name),h); self.assertEqual((ROOT/name).read_bytes(),(PARENT/name).read_bytes())
        copied=json.loads((ROOT/"test_state.jsonl").read_text().splitlines()[0])
        dense=json.loads((ROOT/"train.jsonl").read_text().splitlines()[0])
        self.assertEqual((copied["stage"],copied["split"]),("s2_doublemod_seen_v1","test_state"))
        self.assertEqual((dense["stage"],dense["split"]),("s2_doublemod_seen_dense_v1","dense_train"))
    def test_manifest_no_bom_permutation_and_regeneration(self):
        self.assertFalse((ROOT/"artifact_manifest.json").read_bytes().startswith(b"\xef\xbb\xbf"))
        man=json.loads((ROOT/"artifact_manifest.json").read_text()); self.assertNotIn("artifact_manifest.json",man["files"])
        with tempfile.TemporaryDirectory() as td:
            generate(td,PARENT)
            for p in Path(td).iterdir(): self.assertEqual(p.read_bytes(),(ROOT/p.name).read_bytes())

if __name__=="__main__": unittest.main()
