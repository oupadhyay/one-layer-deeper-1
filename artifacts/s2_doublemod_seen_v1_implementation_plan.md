# S2 double-mod seen-context implementation plan (frozen before generation)

* Stage/version is `s2_doublemod_seen_v1`; all randomness uses a private
  `random.Random(20260730)` instance. No existing source or artifact is edited.
* Independently scan each width-250 bin of `[8000,10000)` and choose its least
  odd prime with `ord_N(2) >= 512`; abort unless the resulting ordered list is
  `[8009,8263,8501,8753,9001,9257,9533,9767]`.
* Reserve all four boundary states first: `|2x-N|=3` goes to train and
  `|2x-N|=1` goes to test-state. Reserve `x=0` and `x=N-1` for train. Thus
  `x=N-1` is the explicit train/reduction singleton policy and test has no zero.
* Select depth before filling one-step splits. For each modulus, shuffle
  nonzero, nonreserved roots deterministically. Greedily accept a root iff its
  65-state T64 orbit has no repeated state, none of its 64 directed source
  edges is boundary/reserved/already used, until 64 roots are accepted.
* From remaining states, independently shuffle reduction and nonreduction
  candidates per modulus. Force the reserved singleton/boundary assignments,
  then fill train to 1000/class and test-state to 250/class. Splits and depth
  are edge-disjoint; records are deterministically shuffled to interleave N.
* Every record has canonical visible input `[STATE,x[4],MODULUS,N[4]]`, with
  LSD-first decimal digits and explicit zero labels. Metadata (trajectory,
  reduction and decimal carry diagnostics) is never required to reconstruct
  model input. Carry bit `p` is the outgoing carry after doubling decimal
  position `p`, including position 3 (thousands overflow).
* The six depth rungs use the exact same 512 roots and exact trajectory
  prefixes. Generation writes LF UTF-8 JSON/JSONL, a config containing source
  SHA/path/status and HEAD, then a self-excluding SHA-256 manifest. Audit uses
  separate primality/order/transition/carry implementations and same-seed
  temporary regeneration. The external G0 report records the config hash,
  avoiding config/manifest self-reference.
* G0 is contract validation only: fixed-width canonical `encode_state`,
  separate `encode_context`, one stable `step.forward(state, context)`, and
  canonical reset. It creates no model, optimizer, or inference workload.
