# CI/CD Build-Failure Prediction

Predicting whether a GitHub Actions workflow run will fail, using only
information available **before** the run starts.

MSc Software Engineering thesis project. This directory is the complete
source package: code, raw data, processed data, trained models, figures, and
every reported metric.

---

## What the project does

Every time a developer saves a change to a software project, an automated system
rebuilds the project and runs its tests. This takes minutes to hours and costs
real compute. About 11 per cent of these runs fail.

This project predicts that failure at the moment the change is committed, using
only what is knowable at that point: which project, which workflow, which branch,
what triggered it, how large the change is, and what the commit message says.
Anything known only after the run — its duration, its retry count — is excluded
by construction, because a prediction that requires the run to have happened
cannot save the cost of the run.

---

## Headline result

Commit-grouped five-fold cross-validation, decision threshold selected on a
validation fold and never on the test data:

| Configuration | Failure F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.4311 ± 0.0633 | 0.4092 | 0.8276 |
| Random Forest | 0.4240 ± 0.0812 | 0.3909 | 0.8008 |
| XGBoost *(selected on PR-AUC)* | 0.4216 ± 0.0638 | 0.4803 | 0.8240 |
| **Categorical only** | **0.4808 ± 0.0767** | **0.5467** | **0.8649** |

Two things to read from this table:

1. **The three classifiers are not separable.** They differ by less than 0.01 of
   mean F1 while the standard deviation across folds exceeds 0.06.
2. **A model given nothing but categorical identity — repository, workflow,
   branch, trigger — beats the full hybrid model on every metric.** The system is
   substantially a project-level risk estimator rather than a commit-level one.
   This refutes the hypothesis the project began with, and it is reported as the
   central finding.

An earlier version of this work reported a failure-class F1 of 0.5924. That
figure was produced under an evaluation protocol since found to be unsound, and
`results/metric_attribution_ladder.json` decomposes the difference:

| Step | Protocol | F1 | Change |
|---|---|---|---|
| A | Row-level split, threshold selected on the test set | 0.5924 | — |
| B | Commits kept together across the split | 0.5326 | −0.060 |
| C | Threshold selected on a validation fold | 0.4065 | −0.126 |
| D | Five-fold cross-validation | 0.4216 | +0.015 |

---

## Dataset

| | |
|---|---|
| Source | GitHub Actions REST API (`/actions/runs` joined to `/commits/{sha}`) |
| Rows | 9,772 workflow runs |
| Distinct commits | 2,835 (mean 3.45 runs per commit, max 179) |
| Repositories | 18 large public projects across six languages |
| Collection cap | 600 most recent runs per repository |
| Filter | runs that completed with `success` or `failure` |
| Class balance | 8,700 success (89.03%) / 1,072 failure (10.97%) |
| Time span | 2025-11-25 → 2026-05-29 (run start times) |

`data/raw/github_actions_real.csv` is the collected ground truth and is never
modified. A plain-language description of every column, including why each is
used or excluded, is in `../defence/Dataset_Guide.xlsx`.

Three disclosed limitations of the data: `files_changed` is censored at 300 by
the GitHub API (110 rows); `author_association` is 100 per cent empty because it
was read from an endpoint that does not return it; and the two temporal features
are computed in UTC across globally distributed projects, which makes them close
to noise as defined.

---

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Versions matter. The reported figures were produced with **scikit-learn 1.8.0**
and **xgboost 3.2.0**, both pinned in `requirements.txt`. Other versions of those
two packages are not guaranteed to reproduce the numbers exactly.

Always invoke the interpreter as `.venv/bin/python`, and run modules from this
directory (`.venv/bin/python -m src.<module>`).

---

## Reproducing every reported number

```bash
./reproduce.sh
```

One command, about three minutes, from `data/raw/` to every metric and figure in
Chapter 7. It requires no network access, does not write to `data/raw/`, and
re-verifies split integrity as it runs before printing the headline table and
the attribution ladder.

Equivalent individual steps:

```bash
.venv/bin/python -m src.run_phase2_5              # prepare data, build and verify splits
.venv/bin/python -m src.run_corrected_evaluation  # evaluation, business model, figures
```

Optional, not required for any reported number:

```bash
export GITHUB_TOKEN="..."                          # re-collect from the API (~2 hours)
.venv/bin/python -m src.collect_github_data
.venv/bin/python -m src.run_phase2                 # exploratory figures
.venv/bin/python -m src.run_phase3                 # architecture validation
.venv/bin/python -m src.run_phase4                 # train and persist models/
```

---

## Layout

```
src/
  collect_github_data.py       GitHub API -> data/raw/github_actions_real.csv
  data_preparation.py          cleaning, feature engineering, text cleaning, ALL splits
  hybrid_pipeline.py           four-branch ColumnTransformer + LR / RF / XGBoost
  cross_validation.py          commit-grouped CV protocol and threshold selection
  train_evaluate.py            metrics, ablation, business cost model
  threshold_optimization.py    threshold sweep
  visualization.py             ThesisPlotter, 300 DPI figures
  run_corrected_evaluation.py  PRIMARY orchestrator: every reported metric and figure
  run_phase{2,2_5,3,4,5}.py    phase orchestrators

data/raw/          immutable collected dataset
data/processed/    prepared dataset + the three split families
results/           every reported metric as JSON  (see results/README.md)
figures/           22 figures at 300 DPI + captions.md
models/            trained pipelines + metadata sidecars
reproduce.sh       one-command reproduction
```

### Which split is which

| Family | Files | Status |
|---|---|---|
| Grouped | `{train,val,test}_grouped.csv` | **Primary.** `StratifiedGroupKFold` on `commit_sha`. |
| Chronological | `{train,val,test}_chronological.csv` | Secondary. Each repository cut at its own quantile of `created_at`. |
| Stratified | `{train,test}_stratified.csv` | **Do not evaluate on this.** Retained only to demonstrate the leakage it exhibits: 913 commits shared between train and test. |

### Which model is which

| File | What it is |
|---|---|
| `best_default_threshold_rf.joblib` | Best model at the default 0.5 threshold |
| `best_tuned_threshold_xgb.joblib` | Best model at its tuned threshold |

---

## Methodology notes

**Splits are grouped on `commit_sha`.** Every row is a workflow run, but every
feature describes a commit, and one commit can trigger up to 179 runs. Splitting
rows scatters near-duplicates across both sides. `results/split_integrity.json`
records that no commit appears in more than one partition.

**The decision threshold is never selected on evaluation data.** It is chosen on
an inner validation split of each fold's own training rows.
`results/threshold_selection_gap.json` measures the residual optimism that
selecting on evaluation data would still buy: between 0.006 and 0.021 of F1.

**Results are cross-validated, not taken from one split.** Repository identity
dominates the model, and no grouped splitter balances repository composition
across folds, so single-fold estimates on this dataset vary by roughly 20 points
of F1 depending on the fold drawn.

**Post-execution features are excluded everywhere.** `run_duration_sec`,
`run_attempt`, `is_retry`, `status` and `updated_at` never enter any feature set.

---

## Reading the results

`results/README.md` states which file is authoritative for which number.

Two subdirectories are deliberately **not** results:

- `results/single_fold_reference/` — Phase 4's single-fold output. Retained
  because Phase 4 trains and persists `models/`.
- `results/superseded/` — artifacts from the earlier, unsound protocol. Retained
  because the thesis quantifies the difference between the two protocols, and
  deleting the evidence would make that claim unverifiable.

---

## Project documents

| File | Purpose |
|---|---|
| `CLAUDE.md` | Working rules, current state, defect status |
| `DATA_PROFILE.md` | Dataset statistics and split integrity |
| `REVIEW_FINDINGS.md` | Independent review of 2026-08-29, with a status banner |
| `HANDOFF.md` | Work queue as of 2026-09-05 |
| `../defence/DEFENCE_BRIEF.md` | Presentation speaking aid |
| `../defence/Dataset_Guide.xlsx` | Committee-readable dataset description |

Seed is 42 throughout. `data/raw/` is never modified.
