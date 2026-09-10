# Handoff — where this project stands

**Last updated:** 2026-09-05 · **Commit at handoff:** see `git log -1`
**Purpose:** let a session on another device pick up without re-deriving anything.

Read this first, then [CLAUDE.md](CLAUDE.md) for conventions and
[REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) for the detailed evidence behind every
defect referenced as F-*n* below.

---

## Get running (any device)

```bash
git clone git@github.com:Moamenmuh13/ci-cd-prediction-master-degree.git
cd ci-cd-prediction-master-degree/cicd-failure-prediction
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import pandas, sklearn, xgboost; print('ok')"
```

`.venv/` is the only thing deliberately not in the repo. Everything else —
raw data, derived splits, trained models, figures, thesis documents — is tracked,
so nothing needs regenerating to start reading results.

Sanity check that you match the recorded baseline:

```bash
.venv/bin/python -m src.run_phase5     # reuses saved models, no retraining
# expect XGBoost: optimal threshold 0.06, failure F1 0.5924
```

---

## State of the work

### Done and verified

- **Pipeline reproduces exactly.** Re-running preparation → XGBoost from
  `data/raw/github_actions_real.csv` yields ROC-AUC 0.884, PR-AUC 0.587,
  F1@0.5 0.322, best F1 0.592 @ threshold 0.06 — matching every reported figure.
- **Full independent review complete.** 12 findings, F-1 … F-12, all measured
  against code and data rather than inferred. See REVIEW_FINDINGS.md.
- **Dataset fully profiled.** See DATA_PROFILE.md — substitutes for loading the
  5 MB CSV.
- **Repository published** with README, CLAUDE.md, REVIEW_FINDINGS.md,
  DATA_PROFILE.md, and this file.
- **Leaked GitHub PAT redacted** from `phase0.md`; it was never committed.

### Not started

Everything in the work queue below. **No code fixes have been applied yet.**
The pipeline on `main` is exactly as it was when the thesis was written.

---

## Work queue

Ordered by dependency. Items 1–3 share one fix and should land together.

### 1. Commit-grouped splitting — F-1 · blocks 2, 3, and every number downstream

`src/data_preparation.py`

`train_test_split(stratify=...)` splits *rows* (workflow runs), but every feature
is *commit-level*. 9,772 rows cover only 2,835 commits, so 88.3% of the stratified
test set shares a commit with train.

- Stop dropping `commit_sha` in `LEAKAGE_OR_REDUNDANT` (line ~102); carry it
  through preparation as a grouping key, excluded from `ALL_FEATURE_COLUMNS`.
- Replace `stratified_split()` (line 383) with `StratifiedGroupKFold` or
  `GroupShuffleSplit(test_size=0.2, random_state=42)` keyed on `commit_sha`.
- Assert zero `commit_sha` overlap between train and test as a hard check.

**Measured expectation:** XGB best F1 0.592 → **0.533**, PR-AUC 0.587 → 0.540,
ROC-AUC 0.884 → 0.834. RF best F1 0.508 → 0.457.

### 2. Fix the chronological split — F-2

`src/data_preparation.py:414`

`sort_col = "commit_date"` sorts by when the commit was *authored*, not when the
run *executed*. Result: an 11-hour test window, and 95.8% of test rows ran before
train's last run.

- Sort by `created_at`.
- Cut on a calendar boundary, not a row index, so the test period is a real span.
- Assign whole commits to one side of the cut.
- Update thesis **TC-001**, which currently validates `commit_date` and cites
  dates (2025-12-08) that do not match the data (actual boundary 2026-05-28).

### 3. Threshold selection on a validation fold — F-3

`src/run_phase5.py:338`

The threshold sweep runs against `test_stratified.csv` and the argmax is reported
on that same set — the headline 0.5924 is optimistically biased.

- Carve a validation split from train; select the threshold there.
- Report on the untouched test set.

**After 1–3 the honest headline should land near F1 0.50–0.53.** The thesis
narrative — structured features dominate, threshold calibration matters more than
feature engineering — is unaffected. Only the number moves.

### 4. Add the `categorical_only` ablation — F-5

`src/train_evaluate.py:242`

`run_ablation_study()` runs `hybrid_full`, `text_only`, `structured_only` — the
one configuration `phase4.md` named as the falsification test was never
implemented. Add a `build_categorical_only_preprocessor()` alongside the existing
two at line 272 and include it in the loop.

Likely competitive, given structured-only (F1 0.379) already beats the full
hybrid (0.322), the 0.0%-38.3% failure-rate spread across repositories, and
`elastic/elasticsearch` sitting at 0.0% failures over 600 rows. If it *is*
competitive, that is the honest explanation for why the hybrid claim failed and
belongs in the discussion chapter.

### 5. Rebuild the business model — F-4

`src/train_evaluate.py:300`

- `failure_rate = 0.30` at line 314 → use the observed **0.11**.
- Price false alarms; at precision 0.566 that is ~143/day currently unpriced.
- Adopt the `phase4.md` cost model ($0.064 compute + $18.75 context-switch)
  or document a different one explicitly.
- Remove `routing_reduction_per_failure_seconds`, a leftover key from the
  archived synthetic project.

Note the existing contradiction to resolve: the "optimized" model reports
**lower** savings ($382,802) than the unoptimized one ($428,854), because savings
scale with recall alone. The thesis quotes $383k.

### 6. Smaller corrections — F-6 … F-9, F-11, F-12

- `is_many_files` uses the median (spec: `> 10`), making it near-duplicate of
  `is_large_commit`, r = 0.595 — `src/data_preparation.py:298`
- `is_off_hours_commit` uses `< 6` (spec: `< 8`) — line 301
- `branch` bucketed top-15 while captions claim 21 — line 123
- Medians, bucket vocabularies and the stoplist are fit on train + test (F-7)
- Temporal flags computed in UTC across globally distributed projects (F-8)
- `files_changed` censored at 300 by the GitHub API, undisclosed (F-9)
- `fig_18`, `fig_19` and the PR-curve figure never generated; `captions.md` has
  two "Figure 10" entries; Figure 5's caption credits dropped leakage features
  with "distinct predictive signal" (F-11)
- Two competing "best" models with no naming distinction; stale
  `data/processed/train.csv` / `test.csv` Phase 2 orphans (F-12)

---

## Decisions already made — do not re-litigate

- `failure` is the positive class; labels stay as strings, XGBoost is wrapped in
  `LabelEncoderForBinary`.
- `run_duration_sec`, `run_attempt`, `is_retry`, `status`, `updated_at` are
  post-execution and stay out of every feature set. The strictly pre-execution
  framing is the thesis's central claim.
- Seed is 42 everywhere.
- The hybrid hypothesis was refuted by the ablation, and the thesis reports that
  honestly. That finding stands — the work now is explaining *why*, not rescuing
  the claim.
- Class imbalance is handled by `class_weight` / `scale_pos_weight`, not
  resampling.

## Propagation rule

Any changed metric must reach four places, or the repo becomes internally
inconsistent — it already has in a few spots:

1. `results/*.json`
2. the figure that plots it
3. its entry in `figures/captions.md`
4. `../final-version-doc/files/4_Thesis_Source_Markdown.md`

State which of the four you updated.

---

## Open questions for Moamen

1. **What is the goal now?** The delivery package targets a 2026-06-07 defense
   that has passed. Corrected re-submission, a paper, or documentation only?
   This determines whether the work queue is worth running at all.
2. **Is the reduced headline acceptable?** Fixing F-1 through F-3 moves the
   result from 0.5924 to roughly 0.50–0.53. Honest and defensible, but lower.
3. **Revoke the PAT.** The token was redacted from `phase0.md` and never
   committed, but it lived in plaintext on disk for months and is still inside
   `CI-CD-prediction.zip`. Revoke at github.com/settings/tokens.
4. **`final-version-doc/` vanished from disk** during the 2026-09-05 session and
   was restored from `CI-CD-prediction.zip`. Was that deletion intentional?
