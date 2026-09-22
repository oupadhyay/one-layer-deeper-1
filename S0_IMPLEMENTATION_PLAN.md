# `s0_perm_v1` Implementation and Experiment Plan

## Scope and isolation

Implement an internal research benchmark only. Do not modify the public competition
runner, submission API, E5 datasets, existing metrics, or remote evaluation flow.
The deliverable covers G0 and one declared C1/G1 experiment; C2 and every depth
experiment remain locked.

## Dataset contract

- Add `data/s0_perm_v1.py` with deterministic JSONL generation and auditing.
- Canonical state: two fixed-width LSD-first decimal digits with explicit zeroes.
- Context: a complete permutation table over `M=16` (or `M=24` for OOD size).
- Model-visible tokens contain state and table contents only. `T`, outer-step index,
  context IDs, and trajectories remain evaluator metadata.
- Generate the normative train, test-state, test-context, OOD-size, and depth-rung
  splits with the counts and seeds from the Oracle specification.
- Store complete trajectories and enough metadata to independently recompute every
  directed transition edge.

## G0 audit suite

Implement tests and a reusable audit function covering:

1. Oracle endpoint and complete-trajectory recomputation.
2. Exactly two supervised output digits, including leading zeroes, with no ignored
   answer labels.
3. Train/test-state edge disjointness and fresh context IDs for context, size, and
   depth splits.
4. Endpoint/state marginal entropy and maximum modal frequency below `1.5 / M` on
   certification splits, with the marginal baseline recorded.
5. Proof that T and step indices are absent from model-visible tensors.
6. At most 5% of depth starts on fixed points or two-cycles.
7. Byte-identical regeneration for the same seed.
8. Internal Track A harness tests for one shared step module and canonical
   decode/re-encode scratch reset.

## Internal evaluator API

Add `benchmark/s0_research.py`, separate from the public benchmark runner, with:

```text
context = encode_context(context_table)
state = encode_state(canonical_state_digits)
repeat T times:
    state = step(state, context)
logits = readout(state)
```

The strict Track A evaluator hard-decodes and scores every macrostep, then calls
`encode_state` on the decoded two-digit state before the next step. It verifies one
step-module identity and never passes T or an outer-step index to model code.

## C1 positive control and G1 experiment

- Implement one unrestricted learned pointer-attention T1 control with separate
  state and context encoders.
- Train only on the 26,000 T1 training edges.
- Use one predeclared optimizer/configuration and a bounded local horizon; no sweep.
- Evaluate on all test-state and test-context records, while applying G1 to the
  first deterministic 5,000 test-state examples as specified.
- Record train curves, exact accuracy, per-digit CE/accuracy, modal baseline, timing,
  parameter count, and the exact number of G1 errors.
- G1 passes only at `>=99.9%` exact on 5,000 held-out test-state examples (at most
  five errors). If it fails, stop; do not run C2 or depth.

## Verification and artifacts

- Run focused S0 tests, the existing contract tests, dataset audit CLI, and
  `git diff --check`.
- Preserve generated dataset files and local metrics under distinct S0 names.
- Do not commit, push, submit remotely, or overwrite prior artifacts.
- Report G0 and G1 evidence to the `Generalized recursion diagnosis` Oracle only
  after local verification and the single C1 experiment complete.
