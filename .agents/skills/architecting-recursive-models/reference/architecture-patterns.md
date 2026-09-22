# Recursive Architecture Patterns

Use these as hypothesis templates, not guaranteed solutions.

## Persistent Conditioned State

```python
condition = condition_encoder(n_tokens)
state = state_encoder(x_tokens)
for tick in range(max_steps):
    proposed = transition(state, condition)
    state = where(active_at(tick), proposed, state)
logits = decoder(state, condition)
```

Key ablation: decode and re-encode categorical state after every tick. If persistent state wins at T2+, closure was the relevant mechanism.

## Conditioned Bilinear Transition

```python
left = U(state)
right = V(state)
gate = 1 + tanh(C(condition))
proposed = norm(O(left * right * gate) + B(condition))
```

Useful for multiplicative recurrences. Compare against a parameter-matched GELU MLP. If neither improves T1, bilinear capacity is not the missing mechanism.

## Spatially Mixed Recurrent Grid

```python
workspace = initialize(position_tokens, condition_tokens)
for microstep in range(propagation_steps):
    local = depthwise_conv(norm(workspace))
    global_context = axial_scan(workspace)
    workspace = gated_update(workspace, local, global_context)
```

Set propagation steps from required communication distance, then verify runtime. Ablate the axial/global path to measure whether long-range communication matters.

## Pairwise Workspace

```python
pair[i, j] = init(x_i, x_j, n_i, n_j, relative_position(i, j))
pair = shared_pair_cell(pair)
digit_state[i] = reduce_j(pair[i, j])
```

Use low-rank or shared reductions to avoid width-specific tables. Ablate pair interactions while preserving parameter count.

## Categorical Recurrence

```python
soft = logits.softmax(-1)
hard = one_hot(soft.argmax(-1))
feedback = hard - soft.detach() + soft  # training only
```

Compare soft, hard, and straight-through variants. Inspect whether errors appear immediately at T1 or compound only after feedback.

Mask inactive digit slots to the canonical zero token before feedback. An unconstrained high-order slot may be absent from the loss but still corrupt the next recurrent state.

## Canonical Quotient State

When the transition has an exact input symmetry, force equivalent operands to share state rather than asking a loss to discover the equivalence approximately.

Reflection-orbit mechanism probe:

```python
orbit = abs(2 * x_integer - n_integer)
logits = learned_transition_table[orbit]
```

Periodic factor-coordinate probe:

```python
period = arange(candidate_min, candidate_max)
n_phase = 2 * pi * n / period
resonance = exp(-sharpness * (1 - cos(n_phase)))
x_even = cos(2 * pi * x / period)
state = concat(resonance, resonance * x_even)
logits = learned_transition(state)
```

The first enforces `x ~ N-x`. The second can identify the larger CRT square-root quotient when periods resonate with factors of `N`. In both cases, output digits remain learned and must be categorically reapplied at each macrostep.

Use quotient coverage as a pre-training ceiling:

1. Count held-out operands whose equivalence class has a T1 training representative.
2. Predict the maximum transfer available from state sharing alone.
3. Compare measured T1 to that ceiling.
4. If measured T1 reaches the ceiling but does not certify, the remaining problem is arithmetic transition learning, not optimization.

These are strong local mechanism probes. Exact task symmetries or factor-selective features may require an organizer ruling before competition submission.

## Useful Minimal Ablations

Choose one based on the claimed mechanism:

- Remove condition injection after initialization.
- Remove cross-position communication.
- Replace bilinear product with matched MLP.
- Replace persistent latent state with categorical re-encoding.
- Untie transition weights across training depths as a negative control.
- Remove normalization or move it before/after the transition.
- Reduce propagation microsteps while preserving width.

Do not make several changes in one ablation.

## Metric Interpretation Matrix

| T1 | Deep T | Token accuracy | Likely conclusion |
|---|---|---|---|
| Low | Low | Low | Representation or optimization failure |
| Low | Low | High | Localized digit/boundary failure or memorization |
| High | Low | High | Feedback, closure, or stability failure |
| High | High Seen | Low OOD condition | Condition lookup or poor size generalization |
| Sparse at arbitrary T | Sparse | Variable | Collision, fixed point, or attractor; replicate |

## Experiment Card

```text
Hypothesis:
Parent architecture:
Single mechanism changed:
Expected metric movement:
Cheap gate and threshold:
Falsification rule:
Seen rung counts:
OOD rung counts:
Certified max T:
Token/error-position diagnostics:
Completed updates and runtime:
Verdict:
Next action:
```

Require the expected metric movement and falsification rule before running the experiment.
