# Hard Run Preparation

## Decision

Do not submit either current candidate remotely.

The Medium reflection-memory result does not scale to sparse 64-bit prompts,
and its width-safe implementation uses explicit decimal carry/borrow logic.
That implementation is retained only as a local upper-bound diagnostic at
`submissions/diagnostic_explicit_reflection_hard_v1/submission.py`.

The remotely legal candidate is
`submissions/learned_pairwise_canonical_hard_v2/submission.py`. It performs all
cross-position routing and state updates with learned modules, uses hard
straight-through recurrent feedback, canonical leading-zero supervision, and
T1/T2/T4 endpoint weights 1.0/0.1/0.005.

## Local results

### Explicit-reflection diagnostic

- M6, 60 seconds: T1 121/140, T2 121/140, T4-T64 124/140.
- Sparse variable-N 64-bit proxy, 60 seconds: test 0/900, OOD 0/1000.
- Conclusion: the strong Medium result depends on dense small-N orbit coverage.

### Learned pairwise candidate

- M6, 60 seconds: T1 0/140, T2 1/140, standard test 1/390.
- Sparse variable-N 64-bit proxy, 60 seconds: test 0/900, OOD 0/1000.
- Sparse variable-N 64-bit proxy, 600 seconds and 4,800 updates: test 0/900,
  OOD 0/1000; final train loss 4.1451 and evaluation losses 2.4709/2.4537.
- Conclusion: the model learns token marginals slowly but not exact arithmetic
  or a closed recurrence.

## Validation

Both sources compile and pass the syntactic submission source-policy checker.
Only the learned pairwise candidate satisfies the stated semantic requirement
that arithmetic carry/borrow behavior be learned rather than programmed.

Hard evaluation is private, so the generated 64-bit dataset is a proxy rather
than a claim to reproduce the hidden evaluator. Its purpose is to reject
architectures that rely on exhaustive small-modulus coverage.

## Deadline submission

At the user's explicit 9:00 PM PT fallback deadline, the explicit-reflection
diagnostic was submitted as the strongest locally measured candidate despite
the semantic-policy concern and its zero-exact 64-bit proxy result.

- Submission: `d4750ec5-5446-4e3f-80bc-e6842c5e00c7`
- Tier/dataset: Hard / h1
- Initial status: running
- Remaining daily Hard attempts after acceptance: 0
