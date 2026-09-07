# Predicting CI/CD Pipeline Build Failures Using Machine Learning

MSc Software Engineering thesis project — Moamen Mohamed Aly Hussein.
Cairo University, Faculty of Graduate Studies for Statistical Research.

Predicts GitHub Actions workflow failure from **pre-execution** commit features
(commit metadata + the commit message), using a four-branch hybrid pipeline:
numerical scaling, one-hot categoricals, binary flags, and TF-IDF text, fused via
`ColumnTransformer` and consumed by Logistic Regression / Random Forest / XGBoost.

## Dataset

9,772 real workflow runs from 18 active open-source repositories, collected from
the GitHub Actions API. 89.03% success / 10.97% failure, drawn from **2,835
distinct commits** — so 3.45 runs per commit, which is why every split is
grouped on the commit hash. **`failure` is the positive class.**

Full profile: [`cicd-failure-prediction/DATA_PROFILE.md`](cicd-failure-prediction/DATA_PROFILE.md).
Plain-language description for non-specialists: [`defence/Dataset_Guide.xlsx`](defence/Dataset_Guide.xlsx).

## Reported results

Commit-grouped five-fold cross-validation, decision threshold selected on a
validation fold and never on the test data:

| Configuration | Failure F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.4311 ± 0.0633 | 0.4092 | 0.8276 |
| Random Forest | 0.4240 ± 0.0812 | 0.3909 | 0.8008 |
| XGBoost *(selected on PR-AUC)* | 0.4216 ± 0.0638 | 0.4803 | 0.8240 |
| **Categorical only** | **0.4808 ± 0.0767** | **0.5467** | **0.8649** |

Two things to read from that table:

1. **The three classifiers are not separable.** They differ by less than 0.01 of
   mean F1 while the standard deviation across folds exceeds 0.06.
2. **A model given nothing but categorical identity — repository, workflow,
   branch, trigger — beats the full hybrid on every metric.** The hybrid
   hypothesis is refuted, and the system is best described as a project-level
   risk estimator rather than a commit-level one.

Business estimate: **$243,670**/year net of false alarms, reported alongside the
break-even false-alarm cost of **$20.41** at which it reaches zero.

### An earlier version of this work reported F1 = 0.5924

That figure came from an evaluation protocol since found to be unsound. Each
defect was corrected and its effect measured independently:

| Step | Protocol | Failure F1 | Change |
|---|---|---|---|
| A | Row-level split, threshold selected on the test set | 0.5924 | — |
| B | Commit-grouped split, threshold still selected on test | 0.5326 | −0.060 |
| C | Commit-grouped split, threshold selected on validation | 0.4065 | −0.126 |
| D | Commit-grouped five-fold cross-validation | 0.4216 | +0.015 |

Selecting the threshold on the evaluation data cost more than twice what the
duplicate-commit leakage did. Full account:
[`final-version-doc/files/6_Corrections_Since_Submission.md`](final-version-doc/files/6_Corrections_Since_Submission.md),
thesis Sections 7.3.1 and 7.4.5, and
[`cicd-failure-prediction/results/metric_attribution_ladder.json`](cicd-failure-prediction/results/metric_attribution_ladder.json).

[`cicd-failure-prediction/REVIEW_FINDINGS.md`](cicd-failure-prediction/REVIEW_FINDINGS.md)
is the original review that surfaced the first of these, retained unedited as a
record; its status banner says which findings are now closed.

## Reproducing

```bash
cd cicd-failure-prediction
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
./reproduce.sh
```

One command, about three minutes, from `data/raw/` to every reported number and
figure. It re-verifies split integrity as it runs and prints the headline table
and the attribution ladder.
[`cicd-failure-prediction/results/README.md`](cicd-failure-prediction/results/README.md)
states which results file is authoritative for what.

Seed is fixed at 42 throughout. `scikit-learn==1.8.0` and `xgboost==3.2.0` are
pinned because they determine the reported figures.

## Layout

```
cicd-failure-prediction/     the pipeline (see its CLAUDE.md for conventions)
  src/                       collection -> preparation -> training -> evaluation
    cross_validation.py        the commit-grouped CV protocol
    run_corrected_evaluation.py  produces every reported metric and figure
  data/raw/                  immutable source CSV (tracked)
  results/                   every reported metric as JSON (see its README.md)
  figures/                   22 figures at 300 DPI + captions.md
  models/                    trained pipelines + metadata sidecars
  reproduce.sh               one-command reproduction
final-version-doc/files/     thesis (.docx/.pdf), deck (.pptx/.pdf), Q&A guide,
                             corrections note, markdown source
defence/                     DEFENCE_BRIEF.md, Dataset_Guide.xlsx
tools/                       markdown -> .docx and -> .pdf generators
dataset/                     reference papers and auxiliary datasets
phase0.md … phase5.md        original phase specifications (intent, not a spec of the code)
```

## License / use

Academic work submitted for an MSc degree. The `dataset/` directory contains
third-party published papers included for reference only.
