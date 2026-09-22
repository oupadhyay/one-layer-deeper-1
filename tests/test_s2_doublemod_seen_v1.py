from pathlib import Path
import inspect
import json
import unittest

from data.s2_doublemod_seen_v1 import (EXPECTED_COUNTS, EXPECTED_MODULI,
    StrictAPIContract, audit, carry_mask, derive_moduli, digits, input_ids)

ROOT=Path(__file__).parents[1]/"data"/"generated"/"s2_doublemod_seen_v1"

class S2Tests(unittest.TestCase):
    def test_encoding_moduli_carry_and_layout(self):
        self.assertEqual(derive_moduli(),EXPECTED_MODULI)
        self.assertEqual(digits(7),[7,0,0,0])
        self.assertEqual(input_ids(7,8009),[0,9,2,2,2,1,11,2,2,10])
        self.assertEqual(carry_mask(9999),[1,1,1,1])
    def test_normative_audit(self):
        r=audit(ROOT)
        self.assertTrue(r["passed"] and r["byte_identical_regeneration"])
        self.assertEqual(r["split_counts"],EXPECTED_COUNTS)
        self.assertEqual(r["unique_depth_edges"],32768)
    def test_per_step_leading_zero_uses_intermediate_target(self):
        rows=[json.loads(line) for line in (ROOT/"depth_t_2.jsonl").read_text(encoding="utf-8").splitlines()]
        n=EXPECTED_MODULI[0]
        selected=[row for row in rows if row["N"]==n]
        intermediate=sum(digits(row["trajectory"][1])[3]==0 for row in selected)/len(selected)
        endpoint=sum(row["labels"][3]==0 for row in selected)/len(selected)
        self.assertNotEqual(intermediate,endpoint)
        report=audit(ROOT,regenerate=False)
        self.assertEqual(report["per_modulus_step_diagnostics"]["depth_t_2"][str(n)][0]
                         ["leading_zero_output_frequency"],intermediate)
    def test_strict_g0(self):
        class Step:
            def forward(self,state,context): return state
        class Harness:
            step=Step()
            def encode_state(self,x): return tuple(x)
            def encode_context(self,n): return tuple(n)
        c=StrictAPIContract(Harness())
        self.assertEqual(tuple(inspect.signature(c.model.step.forward).parameters),("state","context"))
        self.assertEqual(c.canonical_reset([3,2,1,0]),(3,2,1,0))
        class BadStep:
            def forward(self,state,context,T): return state
        Harness.step=BadStep()
        with self.assertRaises(TypeError): StrictAPIContract(Harness())

if __name__=="__main__": unittest.main()
