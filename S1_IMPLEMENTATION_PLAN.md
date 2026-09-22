# S1 add-with-carry implementation plan

1. Generate fixed-width, LSD-first decimal records for `F_a(x)=(x+a) mod 10000`, keeping state and addend physically separate and all evaluator metadata model-invisible.
2. Reserve rare carry-length edges, then select 512 deterministic nested depth roots whose 64-step directed edges are unique per addend and quarantined from one-step splits.
3. Allocate exactly 20,004 train and 5,004 test-state edges from the remaining seen-addend support, with every feasible per-addend carry stratum represented; generate the exhaustive 30,000 held-out-addend context split.
4. Emit source/config provenance and a self-excluding SHA-256 artifact manifest on the first generation.
5. Audit oracle endpoints and trajectories, canonical labels, split/edge isolation, carry counts, nested trajectories, cycle freedom, entropy/modal floors, model-visible reconstruction, and byte-identical regeneration.
6. Test only the generic strict API contract and canonical reset semantics. Do not construct or train a learned model and do not evaluate depth predictions.

Definition of done: normative artifacts generated, G0 audit green, focused and existing S0 tests green, and all hashes/results preserved for review.
