from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from data.s1_addcarry_v1 import (
    A_TEST, A_TRAIN, EXPECTED_COUNTS, StrictAPIContract, audit_s1_addcarry_v1,
    carry_length, decode, digits, input_ids,
)


ROOT = Path(__file__).parents[1] / "data" / "generated" / "s1_addcarry_v1"


class S1AddCarryV1Tests(unittest.TestCase):
    def test_fixed_width_alignment_carries_and_visible_layout(self) -> None:
        self.assertEqual(digits(7), [7, 0, 0, 0])
        self.assertEqual(decode([7, 0, 0, 0]), 7)
        self.assertEqual(input_ids(9998, 7), [0, 10, 11, 11, 11, 1, 9, 2, 2, 2])
        self.assertEqual([carry_length(x, 1) for x in (8, 9, 99, 999, 9999)], [0, 1, 2, 3, 4])

    def test_normative_g0_audit(self) -> None:
        report = audit_s1_addcarry_v1(ROOT)
        self.assertTrue(report["passed"])
        self.assertEqual(report["split_counts"], EXPECTED_COUNTS)
        self.assertEqual(report["train_test_edge_overlap"], 0)
        self.assertEqual(report["depth_one_step_edge_overlap"], 0)
        self.assertTrue(report["byte_identical_regeneration"])

    def test_context_sets_and_rare_carries_are_explicit(self) -> None:
        report = audit_s1_addcarry_v1(ROOT, verify_regeneration=False)
        self.assertFalse(set(A_TRAIN) & set(A_TEST))
        for split in ("train", "test_state"):
            self.assertGreater(sum(report["carry_counts"][split][str(a)][str(c)]
                                   for a in A_TRAIN for c in (3, 4)), 0)

    def test_strict_api_hides_t_and_canonically_resets(self) -> None:
        class Step:
            def forward(self, state, context):
                return state

        class Model:
            step = Step()

            def encode_state(self, state_digits):
                return tuple(state_digits)

            def encode_context(self, addend_digits):
                return tuple(addend_digits)

        contract = StrictAPIContract(Model())
        self.assertEqual(tuple(inspect.signature(contract.model.step.forward).parameters), ("state", "context"))
        self.assertEqual(contract.canonical_reset([3, 2, 1, 0]), (3, 2, 1, 0))


if __name__ == "__main__":
    unittest.main()
