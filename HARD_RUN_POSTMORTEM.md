# Hard Run Postmortem and Local T1/T2 Checkpoint

## Failed Hard run

Submission `d97c3b35-e236-425f-b4c0-eb58f47cb23a` used a dynamic-width
pairwise relation scan. It produced 1/768 Seen T1, 0/768 OOD-N T1, and a
final weighted training loss of 3.277. Its isolated exact hits at unrelated
depths were chance collisions, not evidence of a learned recurrence.

The model supplied cross-position communication but did not supply a usable
representation of digit products, carry, modular reduction, or canonical
recurrent state. Its soft feedback also allowed unsupervised high-order output
digits to contaminate the next macrostep. The mistake was promoting it to Hard
without first requiring meaningful local T1 exact accuracy and training loss.

## Resolved local failure modes

The saved candidate is
`submissions/dynamic_reflection_memory_medium_v1/submission.py`.

It introduces:

1. The squaring-preserving reflection orbit `abs(2*x - N)` as a learned
   condition-specific transition key.
2. Two independent half-width hash memories to reduce aliases while using
   33.6M model-state elements.
3. Canonical-state supervision that explicitly trains omitted high-order
   residue slots to zero.
4. Joint endpoint training with T1 weight 1.0, T2 weight 0.1, and a small
   T4 graph-expansion weight of 0.005.

Canonical zero supervision fixed the closure failure. On M6 it raised T2 from
roughly 40/140 to 114/140 without reducing T1. Adding T2 endpoint supervision
then expanded the learned transition graph and improved both T1 and T2.

## Saved local checkpoint results

| Dataset | T1 | T2 | T4 | T64 |
|---|---:|---:|---:|---:|
| M6 | 121/140 | 121/140 | 121/140 | 121/140 |
| M7 | 171/180 | 170/180 | 170/180 | 170/180 |
| M8 | 143/160 | 143/160 | 141/160 | 140/160 |
| M9 | 249/290 | 249/290 | 250/290 | 250/290 |
| M10 | 223/256 | 223/256 | 223/256 | 223/256 |

The T2-weight sweep compared 0.1, 0.2, and 0.5. Weight 0.1 improved the
aggregate M6-M10 T1+T2 count from 1,796 to 1,807. Weight 0.5 reduced M9's
joint count from 491 to 450 and was rejected.

The T4-weight sweep found a narrow useful range. Weight 0.05 was rejected
because it reduced M9 T1+T2 from 491 to 484, and weight 0.01 reduced the
five-dataset aggregate from 1,807 to 1,804. Weight 0.005 increased it to
1,813 and is the current incumbent. Weight 0.0025 tied at 1,813 but had worse
deeper M7 closure, while 0.0075 scored 1,811. The incumbent's lone repeated
M6 OOD-N success is an attractor collision, not evidence of a
modulus-functional transition.

## Remaining limitation

OOD-N remains effectively zero because the transition memory is keyed by N.
This checkpoint resolves Seen one-step prediction and recurrent closure; it
does not yet implement a modulus-functional arithmetic algorithm. Hard transfer
therefore remains conditional on repeated coverage for Seen Hard moduli.
