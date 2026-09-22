---
name: architecting-recursive-models
description: Designs, evaluates, and iterates learned architectures for recursive functions, algorithmic state transitions, depth generalization, and exact-match certification. Use when building recurrent neural systems that must apply one learned transition repeatedly and generalize to unseen depths, operands, widths, or problem sizes.
---

# Architecting Recursive Models

Design recursive models around a learned transition that remains meaningful when composed many more times than it was trained. Optimize for exact recurrence generalization, not ordinary interpolation.

## Start With the Contract

Before proposing an architecture, establish:

1. The mathematical state, condition, and transition:
   \[
   z_{t+1}=F_\theta(z_t,c)
   \]
2. Which inputs are fixed, Seen, OOD, or withheld.
3. Training depths and evaluation depths.
4. Whether output width or state width changes.
5. The exact certification criterion per rung.
6. Runtime, update, parameter, and submission constraints.
7. Forbidden solver behavior, hard-coded operations, data augmentation, or intermediate supervision.

Do not infer success from aggregate test accuracy when the target is certified depth generalization.

## Align the Campaign Before Experimenting

Resolve these questions with the user at the start. If the answers are available
from manifests, rules, APIs, or local hardware inspection, verify them directly
instead of asking the user to investigate:

1. Is the actual objective public-tier accuracy or a legal transition that
   transfers to the widest hidden tier, even if its public score starts lower?
2. What exact result promotes a candidate between tiers? Define counts, datasets,
   seeds, widths, and runtimes rather than saying "looks promising."
3. Which programmed operations are semantically forbidden? Automated source
   validation is not a legality ruling; ask the organizer before building around
   a borderline symmetry, canonicalizer, carry, reduction, or quotient.
4. What local accelerators, runtime, datasets, and experiment volume are
   available? Establish the cheapest faithful local gate immediately.
5. What evidence from successful systems can the user share, and which details
   are observation versus speculation?
6. What are the quota reset timezone, final deadline, risk tolerance, and minimum
   time reserved for a full preflight and hosted evaluation?

Write a one-sentence campaign contract. A strong default is: optimize only a
legal learned recurrence; promote from Easy after replicated withheld T1 gains;
promote toward Hard only after positive exact T1 on a sparse, target-width proxy.
Treat ordinary Seen accuracy, token accuracy, and loss as diagnostics.

## Use a Rung-First Scorecard

Rank candidates in this order unless the task specifies otherwise:

1. Exact Seen count at the target rung, especially T1.
2. Exact OOD count at that rung.
3. Certified maximum Seen and OOD depth.
4. Exact counts across all depth rungs.
5. Per-digit or per-token accuracy.
6. Error position and error type.
7. Standard test accuracy, endpoint OOD accuracy, and loss.

Treat token accuracy and error position as diagnostics, not substitutes for exact sequence accuracy. A model can have excellent token accuracy while never certifying one example.

## Diagnose the Failure Axis

Separate four capabilities:

| Axis | Diagnostic |
|---|---|
| Representation | Can one state encode the full operand and condition? |
| One-step transition | Does Seen T1 improve on withheld inputs? |
| Closure | Is the predicted next state valid input to the same transition? |
| Stability | Does composition preserve correctness at larger T? |

Interpret common patterns:

- High train and standard-test accuracy but weak T1: interpolation or memorization, not recurrence learning.
- Good T1 but collapse by T2/T4: transition is not closed or feedback causes distribution drift.
- Good Seen but poor OOD condition: condition encoding is lookup-like rather than functional.
- Good token accuracy but low exact accuracy: a localized carry, boundary, or high-order digit failure.
- Flat nonzero counts at every depth: default-output collisions or attractor collapse.
- Falling train loss without rung gains: stop optimizer tuning and change representation.

## Architecture Requirements

### Preserve a persistent state

Prefer:

```text
input tokens ──▶ encoder ──▶ z0
condition    ──▶ encoder ──▶ c
                       ┌─────────────┐
z0 ──────────────────▶│ shared Fθ   │──▶ z1 ──▶ shared Fθ ──▶ ...
                       └─────────────┘
final zT ──▶ decoder ──▶ output tokens
```

Avoid decoding to token probabilities and re-encoding expected token values after every tick unless categorical recurrence is itself the hypothesis. That bottleneck introduces irreversible information loss and train/evaluation mismatch.

### Make the transition closed

The output of `Fθ` must live in the same coordinate system and scale as its input. Use tied transition weights across all macrosteps. Normalization can help, but verify it does not erase magnitude needed by the algorithm.

### Provide cross-position communication

Independent per-digit cells cannot represent multiplication, carry, reduction, or global routing. Add one of:

- Bidirectional scans.
- Axial or local spatial mixing with enough propagation steps.
- Pairwise or tensor-product workspaces.
- State-space scans in both directions.
- Global summaries plus position-preserving local state.

Global pooling alone usually loses place-specific information. Local convolution alone may not propagate far enough within the time budget.

### Condition every transition

For recurrences parameterized by an input such as a modulus, inject the condition into every step, not only the initial state or decoder. Prefer feature-wise modulation, low-rank factors, or conditioned gates over a single pooled additive bias.

### Preserve interactions

Recursive arithmetic commonly needs multiplicative interactions. Useful generic learned operators include:

- Factorized bilinear maps.
- Commutative products.
- Tensor-product representations.
- Quadratic residual branches.
- Spatial gating and axial mixing.
- Fast-weight or state-space updates.

A quadratic term alone is not an algorithm. It still needs positional structure, condition dependence, closure, and a usable optimization path.

### Keep discrete feedback an explicit choice

Compare:

- Persistent latent recurrence.
- Soft categorical feedback.
- Straight-through hard categorical feedback.

Soft feedback is differentiable but drifts off the training manifold. Hard feedback preserves valid symbols but blocks gradients without an estimator. Straight-through feedback can help but may optimize a biased transition. Treat these as architecture ablations, not implementation details.

## Training Strategy

1. Train the same shared transition across all supplied depths.
2. Balance sequence losses so longer outputs do not dominate merely by token count.
3. Use rung weighting only after establishing that the architecture can learn the rung.
4. Prefer endpoint supervision first; add auxiliary losses only when they preserve the task boundary.
5. Do not manufacture intermediate traces or transformed operands when prohibited.
6. Monitor completed updates, throughput, and resource failures. A larger model that receives fewer useful updates can regress.
7. Replicate surprising gains before tuning around them.

Auxiliary objectives must answer a diagnosed problem. Input reconstruction prevents encoder collapse but does not prove recurrence learning. Same-label contrastive learning encourages grouping but does not teach the transformation that produces the equivalence. Structured token losses cannot rescue a missing representation.

## Experimental Workflow

### 1. Select a cheap gate

Choose the smallest dataset whose withheld split tests the desired capability. Compute the exact count required for the milestone before submitting anything.

Also build the cheapest target-scale proxy early. For arithmetic benchmarks it
should separate:

- Fresh states under condition identities present in training.
- Entirely unseen condition identities.
- Target operand width and state width.
- T1 from T2 and deeper composition.

Do not use exhaustive small-condition performance as evidence for a sparse wide
target. Compute equivalence-class or lookup coverage before training; if that
coverage vanishes at target scale, reject the mechanism as a transfer candidate.

### 2. Establish controls

Record:

- Best historical architecture.
- Best current exact rung count.
- Same architecture under a fresh seed.
- Simplest plausible baseline.

### 3. Run architecture families, not random variants

For each family, test:

1. Full hypothesis.
2. One ablation that removes the claimed mechanism.
3. One compute-matched control.

Stop a family when the full model does not materially improve the target rung. Do not exhaust quota on optimizer neighborhoods after the representation hypothesis is falsified.

### 4. Rank by the milestone

Select strictly by target-rung exact count. Use loss, standard accuracy, token accuracy, and deeper totals only as tie-breakers or diagnostics.

### 5. Replicate and transfer

- Replicate a substantial gate improvement before scaling.
- Transfer to a second Easy condition or size.
- Run Medium only after repeatable Easy evidence.
- Use occasional Hard probes to test transfer, not to tune blindly.

Promotion toward the hidden tier requires all of:

- Non-chance exact T1 on fresh states at target width.
- Positive evidence on more than one condition identity.
- A learned rather than lookup-keyed condition representation.
- Throughput sufficient to receive a useful number of full-width updates.

If any requirement fails, keep the candidate as a mechanism probe and do not
spend hidden-tier quota on it.

### 6. Maintain an evidence ledger

For every run capture:

- Architecture and exact difference from parent.
- Dataset, seed, quota tier, update count, and runtime outcome.
- Seen and OOD counts for every rung.
- Certified max T.
- Standard and OOD exact accuracy and loss.
- Token accuracy and error-position profile when available.
- Hypothesis verdict: supported, contradicted, or unresolved.

## Scaling Decisions

Scale only when all are true:

- The mechanism improves exact rung counts, not just average loss.
- The result replicates.
- The model is not update-starved.
- The state remains stable under additional compositions.
- The architecture supports wider inputs without changing its learned semantics.

Do not assume Easy gains transfer to Medium or Hard. Increasing width changes routing and carry distance; increasing depth compounds transition error. Treat width generalization, condition generalization, and depth generalization as separate tests.

## Full-Budget Preflight

Before consuming scarce quota:

1. Re-read the exported submission object and verify that `max_steps`, batch
   sizes, schedulers, and evaluator limits permit the full runtime budget.
2. Run target-width forward and backward smoke tests with the longest permitted
   sequence and depth.
3. Measure updates per second at target width; account for operators such as
   pairwise workspaces whose cost changes from \(O(W)\) to \(O(W^2)\).
4. Run the faithful sparse-width proxy for enough time to observe an exact-rung
   learning curve, not merely falling token loss.
5. Repeat semantic legality review separately from syntactic source validation.
6. Reserve enough wall-clock time for upload, queueing, the complete hosted run,
   status retrieval, and one contingency. Do not begin final local sweeps inside
   that reserve.

Never assume more training will rescue a model whose exact target-width count is
flat at zero and whose evaluation loss remains at chance. Conversely, do not cap
an otherwise viable run early by accidentally retaining an exploratory
`max_steps` value.

## Legality and Scientific Integrity

When a benchmark requires learned random weights:

- Keep parsing separate from solving.
- Do not encode factors, lookup tables, residue catalogs, carries, reductions, or task-specific routes.
- Do not reconstruct operands into scalars and apply the target arithmetic outside learned modules when programmed arithmetic is forbidden.
- Do not create transformed examples, hidden labels, or intermediate traces inside the submission.
- Use generic learned operators whose parameters are randomly initialized and trained by the evaluator.
- State uncertainty about the rules before spending quota on a borderline design.

Read [competition findings](reference/competition-findings.md) for concrete evidence and [architecture patterns](reference/architecture-patterns.md) for implementation templates and ablations.
