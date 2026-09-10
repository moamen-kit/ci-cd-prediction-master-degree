# Review Findings

> **STATUS, 2026-09-05.** This document is the record of what was known on
> 2026-08-29 and is deliberately left unedited below this banner. It is no
> longer a description of the current code.
>
> F-1, F-2, F-3, F-4, F-5, F-10, F-11 and F-12 are **resolved**. F-6, F-7, F-8
> and F-9 remain **open and disclosed**. Current status, with the measurements
> that replaced the ones below, is in `CLAUDE.md`. The corrected headline is
> failure F1 **0.4216 ± 0.0638** under commit-grouped cross-validation, not the
> 0.533 anticipated here — the difference is that the threshold is now selected
> on a validation fold rather than on the test set, which cost a further 0.126.
>
> Two defects not in this review were found later: `src/run_phase2.py` had never
> been runnable, and Appendix B instructed the reader to run two scripts that do
> not exist.

Independent review of the CI/CD failure-prediction project, 2026-08-29.
Every claim below was verified against the code and data, not inferred from prose.
Findings are referenced elsewhere as **F-1** … **F-12**.

## Reproducibility: confirmed

Re-running `prepare_dataset()` → `build_xgboost_pipeline()` from
`data/raw/github_actions_real.csv` reproduces **ROC-AUC 0.884, PR-AUC 0.587,
F1@0.5 0.322, best F1 0.592 @ threshold 0.06** — identical to the reported
figures. The Appendix B bit-for-bit claim holds. Code quality is sound: type
hints throughout, module-level logging, clean phase separation, no silent
failures. The thesis also honestly reports that the hybrid hypothesis was
refuted, which is the correct call given the ablation.

The findings below concern the validity of the numbers, not their reproducibility.

---

## Critical

### F-1 — Duplicate-commit leakage across the stratified split

`data/preparation` splits **rows** (workflow runs), but every modelled feature is
**commit-level**. With 9,772 rows over 2,835 unique commits (3.45 runs/commit,
max 179), `train_test_split(stratify=...)` scatters near-duplicates across both
sides. They differ only in `workflow_name` / `branch` / `event`.

| Split | Test rows whose exact commit fingerprint also appears in train |
|---|---|
| Stratified (**primary**) | **1,726 / 1,955 = 88.3%** |
| Chronological | 115 / 1,955 = 5.9% |

Measured cost, via `GroupShuffleSplit(test_size=0.2, random_state=42)` on
`commit_sha` — same pipeline, same seed, grouped test failure rate 11.6%:

| Metric | Reported (stratified) | Commit-grouped | Δ |
|---|---|---|---|
| XGB best F1 | 0.592 | **0.533** | −5.9 pp |
| XGB PR-AUC | 0.587 | 0.540 | −4.7 pp |
| XGB ROC-AUC | 0.884 | 0.834 | −5.0 pp |
| RF best F1 | 0.508 | 0.457 | −5.1 pp |

**Fix:** `GroupShuffleSplit` / `StratifiedGroupKFold` keyed on `commit_sha`.
Requires retaining `commit_sha` through preparation as a grouping key (it is
currently dropped in `LEAKAGE_OR_REDUNDANT`).

**Impact:** ~6 pp of the headline 0.5924 is memorization. 0.53 remains defensible;
the surrounding narrative survives. The thesis discusses identity leakage and
post-execution leakage at length but never this.

### F-2 — The chronological split is not chronological

`chronological_split()` in `src/data_preparation.py` sorts by `commit_date` (when
the commit was authored) rather than `created_at` (when the run executed — the
actual prediction event).

- Test window spans **11 hours**: 2026-05-28 19:27 → 2026-05-29 06:06, against
  6 months of training data.
- **1,872 / 1,955 test rows (95.8%)** have `created_at` earlier than the training
  set's maximum `created_at`. In run-time terms the model trains on the future
  and tests on the past.
- The 6.65% test failure rate vs 12.05% train is an artifact of that 11-hour
  window, not temporal drift — but Figure 7's caption reads it as drift.
- Thesis **TC-001** asserts `min(test.commit_date) >= max(train.commit_date)` and
  records "Test min = 2025-12-08; train max = 2025-12-08". Wrong column, and the
  cited dates do not match the data (actual boundary: 2026-05-28 19:26).
  Equal min and max is a degenerate pass regardless.

**Fix:** sort by `created_at`, cut on a calendar boundary, assign whole commits
to one side of the cut.

### F-3 — Decision threshold selected on the test set, then reported on it

`src/run_phase5.py` sweeps 0.05–0.95 against `test_stratified.csv` and reports
F1 = 0.5924 at the argmax of that same set. There is no validation fold or CV.
The headline is optimistically biased by construction and the "+27 pp from one
hyperparameter" is partly a selection effect. The chronological transfer check
(62.07%) is a partial defense but lands on the split broken by F-2.

**Fix:** carve a validation split from train, select the threshold there, report
on the untouched test set.

---

## Substantive

### F-4 — Business-impact figures are unsupported

`compute_business_metrics()` in `src/train_evaluate.py`:

- hardcodes `failure_rate = 0.30` against the observed **0.11**;
- uses a 5-min → 0.5-min triage model instead of the $0.064 compute +
  $18.75 context-switch model `phase4.md` specified;
- **never subtracts false-alarm cost**, despite the spec requiring $2.50 each —
  at precision 0.566 that is ~143 unpriced false alarms/day;
- carries `routing_reduction_per_failure_seconds`, a leftover key from the
  archived synthetic-data project.

Self-contradiction in the artifacts: `business_impact.json` (RF, recall 0.696)
= **$428,854/yr**; `business_impact_optimized.json` (XGB, recall 0.621)
= **$382,802/yr**. The optimized model saves *less*, because savings scale with
recall alone. Phase 5 predicted the opposite. The thesis quotes $383k.

### F-5 — The ablation omits the configuration that tests the thesis

`run_ablation_study()` runs `hybrid_full`, `text_only`, `structured_only` —
`categorical_only` was never implemented. `phase4.md` named it explicitly:
*"If Categorical-only is competitive: that means repository identity drives
predictions, NOT commit content."*

Existing results make it the load-bearing experiment:

| Config | failure F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| structured_only | **0.379** | 0.588 | 0.877 |
| hybrid_full | 0.322 | 0.587 | 0.884 |
| text_only | 0.027 | 0.307 | 0.750 |
| categorical_only | *not run* | — | — |

Structured-only beats the full hybrid on F1. Combined with the 0.0%-38.3% failure-rate
spread across repositories and `elastic/elasticsearch` at 0.0% over 600 rows
(see DATA_PROFILE.md), `categorical_only` is likely to be competitive.

### F-6 — Spec/code drift in the feature set

| Feature | Spec (Phase 2.5) | Code | Consequence |
|---|---|---|---|
| `is_many_files` | `files_changed > 10` | `> median` | Built identically to `is_large_commit`; r = 0.595 — reintroduces the redundancy Phase 2.5 existed to remove |
| `is_off_hours_commit` | `hour < 8 or >= 18` | `< 6 or >= 18` | Minor |
| `branch` bucketing | top-20 | top-15 | Caption claims "21 each"; actual is 21 and 16 |
| `commit_message` | listed in `LEAKAGE_OR_REDUNDANT` | retained in `cicd_prepared.csv` | Harmless (not a feature) but doc and code disagree |

### F-7 — Preprocessing fit on train + test

The `total_changes` / `files_changed` medians, the top-K bucket vocabularies, and
the 693-token stoplist are all computed over all 9,772 rows in `prepare_dataset()`
before any split. Target-independent, so mild, but it belongs inside the pipeline
where `fit` sees train only.

### F-8 — Temporal features are computed in UTC

`is_off_hours_commit` and `is_weekend_commit` derive from raw UTC timestamps
across 18 globally distributed projects. 18:00 UTC is mid-afternoon in California
and midnight in Tokyo. As defined these two features are close to noise.

### F-9 — `files_changed` is censored at 300

The GitHub commits API truncates `files[]` at 300 entries. 110 rows (51 commits)
report exactly 300 for what may be far more. `log_files_changed` and
`avg_lines_per_file` inherit the ceiling. Not currently disclosed anywhere.

---

## Documentation and housekeeping

### F-10 — Leaked credential

A live GitHub PAT appears in plaintext in two places:
`phase0.md:13`, and `.claude/settings.local.json` lines 9 and 21.
**Revoke** at github.com/settings/tokens and redact both files before sharing
this project anywhere.

### F-11 — Figure and caption drift

- `fig_18` (stratified vs chronological) and `fig_19` (business impact) were
  never generated; the PR-curve figure from `phase4.md` is also absent.
- `figures/captions.md` contains **two** "Figure 10" entries.
- Figure 5's caption states that `run_attempt` and `run_duration_sec`
  "contribute distinct predictive signal" — the exact features Phase 2.5 dropped
  as post-execution leakage.
- Figure numbering diverges from the phase specs (spec `fig_15` = metrics
  comparison; actual `fig_15` = ablation).
- Figure 7's caption interprets the 11-hour test window as temporal drift (F-2).

### F-12 — Ambiguous artifacts

- Two competing "best" models: `models/best_model.joblib` = Random Forest
  (Phase 4, F1 0.475 at threshold 0.5) and `models/best_optimized.joblib` =
  XGBoost (Phase 5, F1 0.592 at threshold 0.06). Defensible as "best at default"
  vs "best tuned", but nothing in the filenames says so.
- `data/processed/train.csv` and `test.csv` are stale Phase 2 orphans superseded
  by the `_stratified` / `_chronological` pairs. Delete them so nothing trains on
  the wrong file.
- No README or reproduction script at `cicd-failure-prediction/` root, though
  thesis Appendix B promises reproduction instructions.
- `~$*.docx` Word lock files and duplicate thesis copies at repository root.

---

## Recommended order of work

F-1, F-2 and F-3 push the same direction and share one fix: **re-split by commit
group, cut chronologically on `created_at`, and hold out a validation fold for
threshold selection.** Expected honest headline: **F1 0.50–0.53** rather than
0.5924. Weaker, but defensible — and the thesis narrative ("structured features
dominate; threshold calibration matters more than feature engineering") is
unaffected. It is the number that moves, not the story.

Then add `categorical_only` (F-5) to explain *why* structured features dominate,
and rebuild the business model (F-4) with the observed 11% rate and a priced
false-alarm term.

Any change to a metric must propagate to four places: `results/*.json`, the
figure, its entry in `figures/captions.md`, and
`final-version-doc/files/4_Thesis_Source_Markdown.md`.
