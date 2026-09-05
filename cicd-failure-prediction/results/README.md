# results/ — which file is authoritative for what

Every number reported in the thesis comes from a file in this directory. Where
two files could answer the same question, this page states which one is
correct. That ambiguity is what this layout exists to prevent.

Regenerate everything below from `data/raw/` with:

```bash
.venv/bin/python -m src.run_phase2_5              # prepare + splits
.venv/bin/python -m src.run_corrected_evaluation  # all reported metrics + figures
```

## Authoritative

| File | What it is authoritative for |
|---|---|
| `grouped_cv.json` | **The reported results.** Commit-grouped five-fold cross-validation for the three classifiers and the four ablation configurations. Ranking metrics are pooled over all out-of-fold predictions; decision metrics are per-fold mean and standard deviation. |
| `thesis_tables.json` | Every table in Chapter 7, assembled from the above. Transcribe tables from here, not from prose. |
| `split_integrity.json` | Proof that the splits are what they claim: commit overlap between every pair of partitions, temporal invariants, per-partition class balance. |
| `metric_attribution_ladder.json` | Decomposition of the previously reported F1 of 0.5924 into leakage, threshold selection, and fold composition. |
| `threshold_selection_gap.json` | Residual optimism that selecting a threshold on evaluation data would still buy. |
| `chronological_evaluation.json` | Secondary evaluation on the per-repository chronological partition. |
| `business_impact.json` | Business estimate for the selected configuration. |
| `business_impact_all_configurations.json` | The same estimate for every configuration. |
| `business_impact_sensitivity.json` | Net saving against the assumed false-alarm cost, and the break-even cost per configuration. |
| `oof_predictions.npz` | One out-of-fold probability per row per model, plus ground truth. The source for figures 12, 13, 20, 21 and 22. |
| `phase2_5_summary.json` | Feature set, stoplist, and split sizes. |
| `phase3_summary.json` | Pipeline architecture validation. |
| `eda_report.txt` | Descriptive statistics of the raw dataset. |

## `single_fold_reference/` — not the reported numbers

Output of `src/run_phase4.py`, which evaluates on a single held-out fold and
persists the shipped model artefacts. Single-fold estimates on this dataset
vary by approximately 20 points of failure-class F1 depending on which fold is
drawn, because repository identity dominates the model and no grouped splitter
balances repository composition across folds. These files are retained because
Phase 4 is what trains and saves `models/`, and because the contrast with
`grouped_cv.json` is itself reported in Section 7.3.1. **Do not quote them.**

## `superseded/` — retained as the record, not as results

Produced under the evaluation protocol that Section 7.4.5 shows to be unsound:
a row-level split that shared commits between training and test, and a decision
threshold selected on the test set it was then reported on. They are kept
because the thesis quantifies the difference between that protocol and the
current one, and deleting the evidence would make that claim unverifiable.

| File | Why it is superseded |
|---|---|
| `business_impact_optimized.json` | Assumed a 30 per cent failure rate against the 11 per cent observed and charged nothing for false alarms, so the estimate tracked recall alone. This is the file reporting that the tuned model saved *less* than the untuned one. |
| `threshold_optimization.json` | Threshold sweep conducted on the test set and reported on the same test set. |
| `phase2_summary.json` | Output of the superseded Phase 2 preparation module. |
