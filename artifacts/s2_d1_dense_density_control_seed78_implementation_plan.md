# S2-D1 dense density control seed78 implementation plan

1. Pin and validate the repaired S2 parent generator, config, manifest, copied test split, and six copied depth splits.
2. Generate an isolated `s2_doublemod_seen_dense_v1` root. Construct dense train as the exact complement of frozen test-state and frozen T64 source edges for each modulus, deterministically permute it, and byte-copy frozen evaluation/depth artifacts.
3. Independently audit the 71,084-state partition, trajectories, thresholds, distributions, canonical model visibility, manifest exclusion, provenance pins, and byte-identical regeneration; emit a UTF-8 no-BOM G0 report.
4. Add an isolated 5,040-step CPU runner which imports (never duplicates) `benchmark.s2_c1.S2Transformer`, proves seed-78 initialization equality, restricts JSONL access by phase, and emits checkpoint/metrics/provenance only when explicitly run.
5. Add and execute focused generation/audit and non-training runner tests plus bytecode compilation. Do not invoke the D1 runner.
