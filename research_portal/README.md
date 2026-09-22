# Research portal

This is a dependency-free static view of frozen One Layer Deeper attempts. Open `index.html` directly (a `file://` URL works), or from the repository root run:

```text
python -m http.server 8765
```

Then visit <http://localhost:8765/research_portal/>. Serving the repository root also makes the experiment artifact links available.

## Joint T1 transfer metric

The joint T1 score is the lower of each T1 accuracy divided by its public reference: seen-N `4/512` and OOD-N `2/512`. A score above `1.0` means that the same run strictly beats both references. The portal also reports balanced accuracy as the lower raw accuracy. A breakthrough pass requires at least 10% exact accuracy on both profiles. The outcome filter groups detailed verdict text into a small set of success, partial-success, failed-gate, aborted/invalid, and diagnostic categories.

## Checklist for adding an experiment

1. Append one registry record **after freeze**.
2. Use **frozen metrics only**.
3. Include result/provenance links and the decision/limitation.
4. Run `python -m unittest tests.test_research_portal`.
5. Do not rewrite old records without a provenance correction.
