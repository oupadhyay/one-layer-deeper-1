# One Layer Deeper Findings

These findings came from repeated modular-squaring experiments. They are evidence, not universal laws.

## Evaluation Lessons

- Certification required every example at a rung to be correct. Near-perfect averages did not certify a rung.
- Seen depth, OOD depth, and OOD condition were separate capabilities.
- Standard test and endpoint OOD accuracy frequently disagreed with depth-rung accuracy.
- Per-token accuracy and error position were useful for distinguishing global failure from one-digit carry or boundary failures.
- Selection by aggregate score repeatedly favored models that did not improve the desired certified rung.

## Easy E6 Case Study

The public E6 dataset used fixed `N=247`, training depths 1–4, 588 training examples, and 32 Seen examples per depth rung.

- The >50% T1 gate required at least 17/32 exact examples.
- The strongest historical Seen T1 result was 5/32 from a 40,970-parameter contextual bidirectional-GRU transducer.
- A same-source replication later scored only 1/32, demonstrating substantial seed variance.
- A 15-run LR, warmup, weight-decay, and batch screen did not beat 5/32.
- Another 20 interaction variants and 11 structured-loss variants remained at or below 3/32.
- Smaller exhaustive datasets E7–E10 did not automatically make the same architecture succeed; their Seen T1 counts were zero in transfer tests.

The E6 withheld T1 split revealed a tempting symmetry:

- 29/32 target residues already appeared among T1 training outputs.
- 22/32 withheld operands had an `N-x` partner in T1 training.

Neither a learned centered even-coordinate model nor same-label contrastive learning converted that statistical opportunity into T1 generalization. Knowing an invariance exists is not enough; the architecture must learn the transformation relating equivalent inputs.

## Interpolation Can Be Misleading

A centered commutative-algebra model reached:

- Final train loss about 0.011.
- 42/148 standard test exact.
- 42/60 endpoint OOD exact.
- Only 1/32 Seen depth-T1 exact.

This was strong evidence of interpolation and memorization without learning the held-out recurrence. Use this pattern as a warning: standard generalization across sampled prompts is not equivalent to learning the recursive operator.

## Medium Evidence

On public M6:

- `N=1517` was fixed.
- Seen depth rungs had 140 examples each.
- A strong conditional transducer produced Seen rung counts `[3, 4, 5, 3, 2, 2, 4]`.
- It reached 190/390 standard test and 350/500 endpoint OOD.
- It certified no depth rung.

This again separated sampled-prompt interpolation from exact transition learning.

Spatial-gating and axial variants produced the most encouraging Medium hill-climbing signal during earlier campaigns, but gains remained slow and did not establish Hard transfer. Their useful lesson was architectural: mixing across digit positions mattered more than adding a generic global MLP, and normalization/axial choices materially affected optimization.

## Hard Evidence

The best observed Hard family remained extremely sparse:

- A D56 quadratic/polynomial-style run produced Seen counts `[0, 1, 1, 0, 0, 0, 0]` and OOD-condition counts `[0, 1, 0, 0, 0, 0, 0]`.
- Mean exact accuracy was roughly 0.0733%.
- Another dynamic Hard candidate produced one exact example only at T16, not T1.

Sparse isolated hits at deeper rungs did not imply a learned recurrence. They can arise from collisions, fixed points, or attractors. Require repeatable T1 gains before interpreting deep isolated hits as progress.

## Architecture Families Tested

Broad experiments included:

- Contextual bidirectional-GRU transducers.
- Spatial-gating units and axial variants.
- Neural-GPU and cellular workspaces.
- Bidirectional scans and state-space models.
- Relative-position attention and direct Transformers.
- DeltaNet/fast-weight models.
- Tensor-product and bilinear models.
- Factorized digit-pair reducers.
- Autoregressive digit decoders.
- Persistent-edge and global-query models.
- Direct, projected, factorized, and literal polynomial recurrences.
- Learned centered symmetry and Fourier-coordinate models.
- Commutative latent-algebra recurrences.
- Same-output supervised contrastive objectives.
- Focal, hard-example, smoothing, T1 weighting, and optimizer sweeps.

These historical families did not achieve reliable exact depth certification. The later canonical-quotient experiments below exposed the missing mechanism; do not merely rename or widen one of the failed families.

## Canonical-Quotient Breakthrough

Local AMD experiments on fixed-modulus Easy datasets established that withheld-operand failure was primarily a state-identification problem.

- A learned centered/radix RBF model fit T1 training almost perfectly but scored 0/32 held-out T1. Approximate canonicalization let the decoder distinguish symmetry partners and memorize them separately.
- The exact reflection orbit key `abs(2*x - N)` forced `x` and `N-x` to share one learned categorical transition. E6 immediately reached 22/32 T1, exactly the number of held-out operands with a reflected T1 training partner.
- Masking absent high-order digit slots before categorical feedback raised the same model to 18/32 T2 and 12/32 T4. Unmasked inactive digits can silently corrupt an otherwise correct recurrence.
- Full modular square-root equivalence covered 29/32 E6 T1 operands. An N-conditioned integer-period resonance bank reached exactly 29/32, confirming that factor/CRT quotient state was the relevant representation.
- The three uncovered E6 operands formed one entire unseen square-root orbit, proving that quotient sharing alone cannot infer a transition for an orbit absent from direct supervision.
- On E9 and E10, every held-out T1 operand had a square-root-equivalent T1 training representative. The resonance model reached 14/14 and 30/30 respectively and certified Seen T1 on both datasets after 10,000 local updates.
- The certified E9 checkpoint reached 10/14 T2; E10 reached 24/30 T2. Categorical recurrence composed substantially beyond its directly optimized T1 transition, but incomplete transition coverage still caused deeper decay.
- M6 provided a width-transfer test at `N=1517=37*41`: all 140 held-out T1 operands had a square-root-equivalent T1 training representative. The same 108,585-state-element resonance recurrence reached 86/140 T1, 27/140 T2, and 7/140 T4 after 40,000 local AMD updates in 560 seconds. It did not certify T1, but improved the historical M6 rung scale from single digits to 61.4% T1.
- M6 unseen-modulus transfer remained absent at 1/256 OOD-N T1 and endpoint OOD was 9/500. Fixed candidate periods condition successfully on a compatible Seen modulus but are not a learned factorization mechanism.

Interpretation:

1. Exact quotient-state sharing can convert memorization into withheld-operand transfer.
2. Approximate learned coordinates are not interchangeable with exact discrete canonicalization.
3. A tied categorical transition can compose once its state lies on a closed residue manifold.
4. Certification still requires every evaluated orbit to be represented or the arithmetic transition itself to be learned.
5. Fixed small-period banks do not scale to large hidden factors and performed poorly on unseen moduli. Treat them as mechanism probes, not complete Medium/Hard solutions.
6. Increasing quotient-class count increases fitting time and capacity pressure: Easy reached its coverage ceiling near 10,000 updates, while M6 still underfit its 100% coverage ceiling after 40,000 updates.

Rule caution: exact orbit indices and sharply factor-selective period banks may fall near the competition's task-specific-solver boundary. They are valid local scientific probes; obtain an organizer ruling before using such a representation in a hosted submission.

## Failure Patterns

### Soft expected-digit feedback

Repeatedly decoding to a distribution, taking expected digits, and re-encoding caused information loss and off-manifold drift. It can work as a control but is a weak default for long composition.

### Literal polynomial updates

Quadratic updates such as `a⊙z² + b⊙z + c` lacked cross-digit communication when applied independently. Adding spatial mixing improved capacity but did not establish exact recurrence or Hard transfer.

### Global latent collapse

Pooling the entire operand into one vector made optimization easy but discarded place-specific structure. Such models could fit ordinary tests and still fail every certification rung.

### Pairwise capacity without learned algorithm

Pairwise and tensor-product workspaces preserved interactions, yet unconstrained pair MLPs still memorized. Interaction capacity is necessary but not sufficient.

### Structured losses

Focal losses, hard-token emphasis, label smoothing, and T1 weighting changed losses and token behavior but did not overcome representational failure.

### Scaling and resource pressure

Several larger fast-weight, TPR, and SGU runs aborted or received too few useful updates. Always compare completed updates and throughput before attributing failure to the architecture.

### Joint deep training before closure

Adding T2-T4 endpoint loss from initialization destabilized the factor-resonance transition. Random early categorical states made deep gradients meaningless and overwhelmed T1 learning. Even low-weight T2 fine-tuning damaged shared transition weights. First establish T1 closure, preserve that checkpoint unchanged, and isolate any deeper adaptation from the learned one-step map.

## Campaign Discipline

- Use one cheap dataset as a gate, but confirm on a second dataset before transfer.
- Predeclare a stopping rule for each family.
- Preserve the best candidate unchanged before experimental edits.
- Replicate outliers; single-seed gains were not reliable.
- Spend Medium and Hard quota only on hypotheses supported by Easy rung gains.
- Record failures and resource aborts in the result portal; absence of a score is still evidence about feasibility.

## Canonical-State Closure Breakthrough

A localized learned table over exact factor-quotient classes isolated a decisive recurrence bug on fixed-modulus Medium datasets:

- A global quotient decoder reached M6 T1 `140/140` but only T2 `45/140`.
- The visible T1 answer was correct while omitted high-order output slots were unsupervised.
- Those arbitrary hidden digits were fed back as part of the next integer, so the model's internal state was not the same canonical value that the evaluator observed.
- A localized quotient-class table left unobserved high slots at canonical zero and immediately reached `140/140` at every Seen rung T1-T64 and 100% ordinary Seen/OOD on M6 after a 60-second local run.
- The same probe reached 100% Seen/OOD and certified T64 on M7, M8, and M10. M9 reached 97.6% Seen; its seven T1 misses exactly matched seven held-out quotient classes absent from T1 training.

Diagnostic rule: if T1 is exact but T2 is not, inspect every latent component fed into the next tick, including slots excluded from the visible loss. Equality of visible answers does not imply equality of recurrent state.

Localized capacity also matters. A categorical quotient basis with a global MLP reached M6 `91/140` T1 after 600 seconds; direct localized addressing reached the quotient-coverage ceiling in 60 seconds. Wider global decoding was slower and worse.

Replacing the zero-initialized table with random initialization and a generic confidence gate preserved M6's 100% Seen/OOD and T64 certification in 60 seconds. Thus canonical feedback does not depend on prohibited hard-coded weights. The exact factorization and quotient hash remain local mechanism probes, not established legal hosted candidates under rules forbidding hard-coded algorithms and task-specific solvers. Automated source validation accepts them but cannot provide an organizer ruling. Translate the finding into learned routing over digit representations before submission.

## Endpoint-Depth Identifiability

Before claiming that endpoint supervision identifies a one-step transition, compute the gcd of supervised depths:

- M1/M2 depths `{4,8,16}` identify at most a fourth iterate.
- M3 depth `{2}` and M5 depths `{2,4,8}` identify at most a second iterate.
- M4 depth `{8}` identifies at most an eighth iterate.
- M6-M10 depths `{1,2,4}` directly identify the one-step transition on covered states.

When the gcd is greater than one, tied unrolling and endpoint gradients do not uniquely select the intended compositional root. Multiple transitions can have the same supervised iterate. Treat high ordinary endpoint accuracy as insufficient evidence of correct recurrence, and do not require T1 recovery from such a dataset unless an independent legal inductive bias supplies the missing identification.

## Exact Quotient-Oracle Ceiling

A local-only diagnostic with exact factor discovery, residue squaring, and CRT reconstruction reached the full ceiling on every Medium dataset M1-M10 after learning only a randomly initialized decimal renderer:

- Every Seen T1 and T2 rung was exact.
- Every OOD-N T1 and T2 rung was exact.
- Seen and OOD-N profiles certified through T64 on all ten datasets.
- This included variable-modulus M3-M5 and moduli up to seven decimal digits / 22 bits.

This establishes that canonical CRT quotient recurrence is sufficient across the entire public Medium suite. The remaining research problem is not representational sufficiency; it is learning factor/residue routing and CRT-like recombination from permitted endpoint supervision.

The oracle is intentionally not a competition candidate: exact factor discovery and CRT arithmetic are task-specific solver logic. Use it only as an upper-bound diagnostic and ablate one oracle component at a time when designing a legal learned analogue.

## Final Hard-Transfer Retrospective

The late reflection-memory campaign demonstrated why promotion rules must include
a sparse target-width gate:

- Dual-hash reflection memory with canonical zero feedback reached an aggregate
  1,813 exact T1+T2 cases across fixed-modulus M6-M10 screens.
- A small T4 endpoint weight of 0.005 improved this aggregate, confirming that
  shallow graph expansion can improve closure without overwhelming T1.
- The same mechanism scored 0/900 test and 0/1000 OOD on a sparse variable-N
  64-bit proxy. Its Medium success came from dense small-N equivalence coverage,
  not a modulus-functional squaring transition.
- A legal learned pairwise candidate with straight-through feedback, canonical
  zero supervision, and the same endpoint weights scored 0/900 and 0/1000 after
  600 local seconds and 4,800 updates. Pairwise capacity and falling loss did not
  produce exact arithmetic.
- The hosted explicit-reflection Hard run scored approximately 0.01%, certified
  no Seen or OOD-N rung, and underperformed the earlier 0.05% personal best.
- The submission retained `max_steps=20000`, stopped after about 1,515 of 3,600
  available training seconds, and wasted the remaining budget. Its chance-level
  losses indicate that removing the cap was necessary preflight hygiene but not
  sufficient to fix the architecture.

Campaign lesson: define the hidden-tier objective before hill-climbing public
accuracy. A candidate should not advance toward Hard until it has non-chance
exact T1 on fresh states at target width, positive evidence across multiple
condition identities, adequate target-width throughput, and a clear semantic
legality rationale. Validate these independently; a source checker cannot prove
semantic legality, and a dense fixed-condition test cannot prove sparse transfer.
