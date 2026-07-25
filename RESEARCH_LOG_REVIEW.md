# One Layer Deeper Research Log Review

## Executive verdict

Oracle and the repository audit agree: **Rev1 is a valid baseline, but several conclusions in the research log are premature or incorrect.**

The biggest findings are:

1. **Rev1’s tied output head is probably the most immediate model flaw.** It creates enormous initial logits and likely explains the step-1 loss of `83.809`.
2. **The model is recurrent in parameter sharing only—not in task-aligned computation.** It always executes exactly two shared-core passes, regardless of \(T\).
3. **The model was not data-starved on E1.** It processed about 144 dataset-equivalents. It may still have received too few optimizer updates, but that is a different diagnosis.
4. **`batch_size=128` is a worthwhile experiment, but 800–1,200 steps is not guaranteed.** Explicitly preserve `eval_batch_size=512`.
5. **The cosine schedule is effectively constant.** `T_max=20_000` produces negligible decay in a 169-step Easy run.
6. **Do not move to Medium yet.** Use E1 to isolate optimization defects, then E2/E5 to test depth and modulus transfer.

No code was changed during this review.

---

## High-confidence flaws and corrections

### 1. The prompt description does not match the actual evaluated representation

The log says prompts conceptually contain:

```text
N ... X ... T ... ANS ...
```

That describes the older causal-LM representation. All current Easy and Medium datasets use `separate_input_output=True`.

The model actually receives only:

```text
N <digits> X <digits> T <digits>
```

The answer is held separately. The evaluator aligns answer digits with positions at the tail of the input and predicts every answer digit in parallel, as shown in `data/squaring_mod.py` and `benchmark/runner.py`.

Consequences:

- This is **parallel sequence classification**, not autoregressive generation.
- There is no teacher-forcing/exposure-bias issue.
- Output length is effectively given by `target_positions`; EOS and length prediction are not scored.
- Answer digits do not condition subsequent answer digits.
- Bidirectional attention over the prompt is expected and does not leak labels.

The research log should be corrected so future architecture decisions target the actual task interface.

---

### 2. The tied embedding/output head is likely harming optimization badly

The model ties these weights:

```python
self.head.weight = self.token_embedding.weight
```

That is usually useful for next-token language modeling. Here, however, the model predicts an answer digit at a position currently occupied by a prompt token. The residual stream retains strong information about that prompt token, so a tied head initially encourages copying the input token rather than predicting an unrelated residue digit.

There is also a severe initialization-scale issue:

- `nn.Embedding` initializes with approximately unit standard deviation.
- Those vectors are reused directly as classifier rows.
- Final RMSNorm gives a unit-RMS, width-256 hidden state.
- Resulting dot products can be extremely large.

This was reproduced locally with the current model:

- Embedding standard deviation: approximately `1.02`
- Random untrained logit standard deviation: approximately `44.7`
- Random-label cross-entropy: approximately `168`
- Expected uniform 17-class cross-entropy: \(\ln(17) \approx 2.83\)

That is consistent with the remote step-1 loss of `83.809`.

**Recommendation:** untie the head before adding architectural complexity. It costs only \(17 \times 256 = 4{,}352\) additional parameters and gives normal `nn.Linear` initialization.

This may be a much larger improvement than LayerScale or progress embeddings.

---

### 3. “Only 169 steps means under-trained” is an incomplete diagnosis

E1 generation uses:

- 250 examples per \(T\)
- \(T \in \{1,2,3\}\)
- 80% training fraction

Therefore E1 has:

\[
250 \times 3 \times 0.8 = 600
\]

training examples, according to `scripts/generate_datasets.sh`.

At batch 512, the model processed:

\[
169 \times 512 = 86{,}528
\]

example presentations, equivalent to:

\[
86{,}528 / 600 \approx 144
\]

passes over the training-set size.

Therefore:

- The model was **not underexposed to E1 data**.
- It may have received too few independent parameter updates.
- It may have poor initialization.
- It may lack the right inductive bias.
- It may be learning a fixed-modulus interpolation shortcut that does not transfer to \(T=6\).

Those possibilities must be separated experimentally.

Also, the logged `4.1% train accuracy` is not full training-set accuracy. It is accuracy on the final stochastic mini-batch. The harness records current-batch metrics at each logging step. It never evaluates the train split.

A better description is:

> Final training-batch exact accuracy was 4.1%; full train-set accuracy was not measured.

---

### 4. Batch 128 is promising, but the claimed throughput is unproven

Current steady-state throughput is approximately three updates per second:

- Steps 1–100: 99 updates in 33.1 seconds
- Steps 100–169: 69 updates in 22.7 seconds

If sample throughput remained constant, batch 128 would yield roughly:

\[
169 \times 512 / 128 = 676
\]

updates—not automatically 800–1,200.

For reference:

| Updates at batch 128 | Samples processed | Change from baseline |
|---:|---:|---:|
| 676 | 86,528 | Equal sample throughput |
| 800 | 102,400 | +18% |
| 1,200 | 153,600 | +78% |

However, there is an E1-specific reason smaller batches might outperform this estimate:

- E1 has 600 train examples.
- `drop_last=True`.
- Batch 512 produces exactly one complete batch per DataLoader iterator.
- The harness creates a new iterator when it is exhausted.
- There are two non-persistent workers.

Thus, batch 512 can restart DataLoader workers almost every optimizer step. Batch 128 provides four batches per iterator and may reduce this overhead.

#### Important configuration trap

If only `batch_size=128` is set, the harness also chooses evaluation batch 128 because submission training batch size takes precedence over the manifest evaluation batch.

The isolated experiment must use:

```python
batch_size=128
eval_batch_size=512
```

Measure both:

\[
\text{updates/sec}
\]

and

\[
\text{examples/sec}
=
\frac{\text{completed steps}\times\text{batch size}}
     {\text{training seconds}}
\]

Do not keep batch 128 merely because the update count is larger.

---

### 5. Rev1 is not task-aligned recurrence yet

The model always performs:

```text
Prefix × 1 → Shared core × 2 → Suffix × 1
```

This happens for every example, whether \(T=1\), \(T=3\), or \(T=32\).

The core can read the decimal \(T\) tokens through attention, so the network is not unaware of depth. But:

- The number of model transitions does not scale with \(T\).
- Both shared-core applications receive no explicit iteration identity.
- There are no dedicated scratch/state tokens.
- There is no intermediate-state supervision.
- Nothing makes one core application correspond to one modular squaring.
- The model can simply learn a direct endpoint mapping.

Thus, RQST currently provides **weight-sharing regularization**, not demonstrated depth extrapolation. Calling it “10/10 depth extrapolation capability” is unsupported.

A meaningful recurrence experiment is to vary `UNROLL_STEPS` while holding everything else fixed and check whether extra computation helps OOD depth despite reducing optimizer throughput.

---

### 6. Continuous \(k/T\) progress embeddings are not the next priority

The proposed embedding is conceptually interesting but brittle here.

`ModelSpec` supplies only vocabulary size, maximum sequence length, and state ceiling—not a parsed scalar \(T\).

Literal \(k/T\) requires manually:

1. Assuming token IDs and field semantics.
2. Locating the `T` marker.
3. Parsing decimal digits into an integer.
4. Dividing by that parsed value inside `forward`.

That would specialize the architecture to the public prompt grammar and may transfer poorly to hidden Hard.

More importantly, adding \(k/T\) does not fix the fact that the model runs only two iterations. For \(T=32\), the model still performs two learned state transitions.

A safer later experiment is:

- Increase fixed unrolling to \(K=4\).
- Add a learned internal iteration embedding \(e_k\) or phase \(k/K\).
- Reinject the original encoded input at every iteration.
- Compare against the same \(K=4\) model without iteration embeddings.

That tests whether the shared block benefits from knowing which internal pass it is on without hard-coding the task parser.

---

### 7. The scheduler is effectively disabled

The scheduler is:

```python
CosineAnnealingLR(
    optimizer,
    T_max=20000,
    eta_min=1e-5,
)
```

After 169 steps, the LR is still approximately:

\[
0.00199965
\]

rather than `0.002`. That is only about 0.018% decay. Even after 800 updates, decay would remain around 0.4%.

Therefore the logged “cosine schedule” is, operationally, an almost constant LR schedule.

The public optimizer spec exposes the training-time allowance, but Rev1 does not use it.

A better scheduler should be calibrated to the actual budget:

- 3–5% warmup
- stable middle
- decay during the final 15–25% of the run

A custom wall-clock scheduler could use `spec.training_time_seconds`, making it naturally adapt between Easy and Medium. If using a step-based schedule, first measure the stable update horizon for the selected batch size.

Do not change scheduler and peak LR in the same run.

---

### 8. LayerScale is plausible later, but not currently high priority

Current effective depth is only four block applications. There is no evidence of:

- exploding activation norms,
- recurrent instability,
- gradient divergence,
- or degradation as unrolling increases.

At the current shallow depth, \(\gamma=0.1\) might simply suppress useful residual updates.

LayerScale becomes a justified experiment if:

- increasing \(K\) from 2 to 4 or 8 damages optimization;
- hidden-state RMS grows markedly with each recurrence;
- loss becomes unstable despite clipping.

Until then, untying the head, batch sizing, and scheduling are much higher-confidence interventions.

---

### 9. The high-LR recommendation is speculative

A peak LR of `3e-3` may help, but the stated `3e-3 ... 5e-3` range is not yet supported.

Current confounders include:

- pathological tied-head initialization,
- nearly constant LR schedule,
- different effective updates per batch size,
- global gradient clipping at 1,
- and only one completed architecture run.

Test `3e-3` only after fixing or isolating these issues. Do not simultaneously add warmup, LayerScale, and a higher LR.

---

### 10. Several “wild goose chase” labels are too absolute

#### Correct exclusions

These are prohibited by the current official rules:

- Data augmentation
- CPU offloading

They should remain excluded, although they are rule-compliance exclusions rather than failed scientific ideas.

#### Massive models

A 100M model is probably unattractive on Easy, but the stated “10–20 steps” has not been measured.

Likewise, the claim that compact models run 50–100 steps/sec is contradicted by the actual Rev1 measurement of approximately three steps/sec.

A modest width frontier—approximately 2.5M, 6M, 12M, and perhaps 20M parameters—could eventually be useful. There is no need to jump to 100M.

#### Dynamic halting

It is not useful yet, but the stated reasoning is weak:

- Current manifests use `"compile": false`.
- ACT can be implemented using fixed maximum unrolling and batched masks.
- Dynamic halting does not inherently require irregular Python control flow.

The right conclusion is:

> Dynamic halting has low information gain until fixed-depth recurrence demonstrates value.

#### BPE

The evaluator fixes a 17-token vocabulary. Replacing it with BPE is not a straightforward submission option. Interface incompatibility is the strongest objection—not an inherent inability of multi-digit tokens to represent place value.

---

## Strong ideas worth keeping

### 1. Shared recurrent weights

This remains a strong architecture direction. Weight sharing lets the model trade runtime for effective depth without increasing persistent state.

The key missing evidence is whether additional shared computation improves OOD accuracy enough to justify fewer optimizer updates.

### 2. Prefix/core/suffix separation

This is a sensible decomposition:

- Prefix: encode problem specification
- Core: iteratively transform latent state
- Suffix: decode task output

It is more expressive than tying every block and still preserves a clean recurrent bottleneck.

### 3. QK-Norm and pre-norm blocks

These are sensible stability mechanisms, especially if unrolling is increased. They should remain unless an ablation shows otherwise.

### 4. Fused SwiGLU input projection

Combining `w1` and `w2` reduces FFN linear calls from three to two. That is a real implementation improvement.

It does **not** reduce FFN FLOPs by 33%, so the log should phrase this as reduced launch/GEMM-call overhead rather than a guaranteed 33% speedup.

### 5. SDPA and BF16

Using PyTorch SDPA and evaluator-managed BF16 autocast is appropriate. With sequence lengths around 10–20, attention is so small that FlashAttention selection is unlikely to be the principal bottleneck, but SDPA remains the right primitive.

### 6. Native cross-entropy

Reverting the incompatible custom loss was correct. The custom loss receives flattened valid tensors:

```text
logits: (N_valid, vocab_size)
labels: (N_valid,)
```

Native CE is a sound baseline. Sequence-aware exact-match surrogates are difficult because the custom-loss interface no longer contains row grouping or target lengths.

### 7. Smaller-batch exploration

This is worth doing, particularly because E1’s one-batch epoch creates a harness-specific worker restart effect. Just evaluate it using both sample and update throughput.

### 8. Designing for unknown Hard recurrences

Avoiding a forward pass specialized to modular arithmetic is sensible. Hidden Hard may alter the recurrence, as the official problem page explicitly warns.

However, “generic” should not be confused with “depth-generalizing.” That capability still needs to be demonstrated.

---

## Recommended experiment sequence

Easy permits many daily submissions, so optimize for attribution rather than packing several ideas into one run.

| Run | Dataset | One change | Primary question |
|---|---|---|---|
| B0 | E1 | Existing Rev1 | Preserve baseline: 169 steps, test 1.3%, OOD 0% |
| R1 | **E1** | `batch_size=128`, `eval_batch_size=512` | Does update throughput improve without sacrificing sample throughput? |
| R2 | E1 | Untie output head on the winning batch configuration | Does step-1 loss collapse toward a normal scale and final accuracy improve? |
| R3 | E1 | Add budget-calibrated warmup/decay; retain LR `2e-3` | Does a real schedule improve convergence? |
| R4 | E1 | Peak LR `2e-3 → 3e-3` only | Is the compact model genuinely LR-limited? |
| R5 | **E2** | Best E1 configuration unchanged | Do optimization improvements transfer to \(N=899\), \(T_{\text{OOD}}=7\)? |
| R6 | E2 | `UNROLL_STEPS=2 → 4` only | Does more recurrent compute improve OOD depth enough to offset fewer updates? |
| R7 | E2 | Add learned iteration embedding to \(K=4\) | Does distinguishing recurrent passes help? |
| R8 | E2 | LayerScale at \(K=4\), only if instability was observed | Does LayerScale fix measured recurrent degradation? |
| R9 | **E5** | Best architecture unchanged | Does it jointly condition on varying \(N\) and \(T\)? |
| R10 | Medium M1/M5 | Only after meaningful Easy ID and OOD results | Validate larger-depth and larger-modulus behavior |

### What every run should record

Beyond exact accuracy:

- Completed updates
- Updates/sec
- Examples/sec
- Step-1 loss
- Step-100 loss
- Final training-batch loss
- Test token loss
- OOD token loss
- Test exact accuracy
- OOD exact accuracy
- Actual training and evaluation batch sizes
- Model-state count

Because E1 has only 150 test examples and 100 OOD examples:

- One test success changes test accuracy by about `0.0067`.
- One OOD success changes OOD accuracy by `0.01`.
- One test success changes the final mean score by about `0.0033`.

Therefore, use loss as a sensitive secondary diagnostic when exact accuracy is near zero.

---

## Direct answers

### Should the next run use batch 128?

**Yes—but as an isolated E1 experiment, with evaluation batch pinned:**

```python
batch_size=128
eval_batch_size=512
```

Do not assume it will reach 800–1,200 steps. Keep it only if it improves final losses/accuracy or gives a favorable update-versus-sample throughput tradeoff.

### E2 or Medium next?

**Neither immediately. Run the batch-size and untied-head ablations on E1 first.**

Then:

1. **E2** to test larger fixed-modulus/depth transfer.
2. **E5** to test joint \(N/T\) conditioning, which is more relevant to generalized behavior.
3. **Medium only after the model demonstrates nontrivial Easy OOD performance.**

Spending a Medium attempt now would combine larger \(N\), larger \(T\), tenfold runtime, and a larger dataset, making it difficult to diagnose why Rev1 currently scores zero on E1 OOD.

### Highest-value immediate changes

In priority order:

1. **Batch 128 with eval batch 512**
2. **Untie the output head**
3. **Use a budget-calibrated scheduler**
4. **Test LR 3e-3**
5. **Test \(K=4\) recurrence on E2**
6. **Only then consider iteration embeddings or LayerScale**

The most important conceptual shift is to treat Rev1 not as an almost-converged recurrent algorithm, but as a fixed-depth endpoint predictor whose optimization and recurrence hypotheses still need to be separated experimentally.
