# CI/CD Build-Failure Prediction — MSc thesis project

Predict GitHub Actions workflow failure from **pre-execution** commit features.
Binary target `conclusion`; **`failure` is the positive class**. Labels are
strings (`"success"` / `"failure"`), not 0/1 — XGBoost is wrapped in
`LabelEncoderForBinary` (`src/hybrid_pipeline.py`) to keep the fit/predict
contract uniform across the three estimators.

**Start with [HANDOFF.md](HANDOFF.md)** — current state, the open work queue,
and decisions already settled. Then [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) and
[DATA_PROFILE.md](DATA_PROFILE.md) before changing anything. They encode verified facts that are expensive to
re-derive and easy to get wrong.

## Environment

- Python: `.venv/bin/python` — **never** bare `python` (no pandas on the system
  interpreter). The venv has pandas, scikit-learn 1.8.0, xgboost 3.2.0.
- Run modules from this directory: `.venv/bin/python -m src.run_corrected_evaluation`
- Or just `./reproduce.sh` for everything (~3 min).
- `random_state = 42` everywhere. Never modify `data/raw/`.

## Layout

```
src/
  collect_github_data.py   GitHub API → data/raw/github_actions_real.csv
  data_preparation.py      cleaning, feature engineering, text cleaning, splits
  hybrid_pipeline.py       4-branch ColumnTransformer + LR / RF / XGB
  train_evaluate.py        metrics, ablation, business model, best-model selection
  threshold_optimization.py  threshold sweep
  eda.py visualization.py  ThesisPlotter, 300 DPI figures
  cross_validation.py      commit-grouped CV protocol + threshold selection
  run_corrected_evaluation.py  PRIMARY: every reported metric and figure
  run_phase{2,2_5,3,4,5}.py  orchestrators (phase 4 = single fold, see below)
data/raw/        immutable source CSV
data/processed/  cicd_prepared.csv (retains commit_sha as GROUP_KEY)
                 + {train,val,test}_grouped.csv        PRIMARY
                 + {train,val,test}_chronological.csv  secondary
                 + {train,test}_stratified.csv         leakage demonstration only
results/         every reported metric as JSON; see results/README.md for
                 which file is authoritative
figures/         22 PNGs + captions.md
models/          *.joblib + metadata sidecars
```

Phase specs live at the repository root (`phase0.md` … `phase5.md`). They are
**intent**, not a description of the code — several have drifted (F-6).

## Ground truth — verified, do not recompute unless asked

9,772 runs · 18 repositories · 89.03 / 10.97 success/failure · **2,835 unique
commits** · `created_at` spans 2025-11-25 → 2026-05-29.

**Reported headline (current).** Commit-grouped 5-fold CV, threshold selected on
an inner validation fold, ranking metrics pooled over out-of-fold predictions:

| configuration | failure F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.4311 ± 0.0633 | 0.4092 | 0.8276 |
| Random Forest | 0.4240 ± 0.0812 | 0.3909 | 0.8008 |
| XGBoost (selected on PR-AUC) | 0.4216 ± 0.0638 | 0.4803 | 0.8240 |
| **categorical_only** | **0.4808 ± 0.0767** | **0.5467** | **0.8649** |

The three main models are **not separable** — they differ by less than 0.01
while the cross-fold standard deviation exceeds 0.06. Do not describe any of
them as "the winner" on F1.

`categorical_only` beating everything is the project's central finding: the
system is substantially a project-level risk estimator, not a commit-level one.

**The old headline of 0.5924 is superseded.** It is retained only inside the
attribution narrative (`results/metric_attribution_ladder.json`): −0.060 from
duplicate-commit leakage, −0.126 from threshold selection on the test set,
+0.015 from fold averaging.

**Regenerate everything with `./reproduce.sh`** (~3 min from `data/raw/`).

## Defect status — do not re-litigate

Full original detail in REVIEW_FINDINGS.md, which keeps the record of what was
known when. Current status:

**Resolved.**
- **F-1 duplicate-commit leakage** → `grouped_split()` uses `StratifiedGroupKFold`
  on `commit_sha`. Verified zero overlap in `results/split_integrity.json`.
  **Any new split must still group on `commit_sha`.**
- **F-2 chronological split** → rewritten to cut per repository on `created_at`.
  A *global* cut cannot work here: the 600-run-per-repo cap gives per-repo
  coverage from 0.4 to 182 days, so any global 80/20 cut yields a ~9 h window
  over 11 of 18 repos. Per-repo invariant holds for all 18.
- **F-3 threshold on test** → selected on an inner validation fold. Residual gap
  measured in `results/threshold_selection_gap.json` (+0.006 to +0.021).
- **F-4 business model** → rebuilt to the phase4.md cost model with a priced
  false-alarm term; one implementation, not three. Break-even reported.
- **F-5 categorical_only** → implemented, and it wins.
- **F-11 caption drift** → `_append_caption` is now an upsert keyed on filename,
  which was the mechanism behind the duplicate Figure 10. Figures 5 and 7
  captions corrected.
- **F-12 ambiguous artifacts** → `results/README.md` states what is
  authoritative; single-fold output in `results/single_fold_reference/`,
  pre-correction artifacts in `results/superseded/`.

**Open, disclosed, not fixed.**
- **F-6 spec drift**: `is_many_files` uses the median (spec `> 10`),
  `is_off_hours` uses `< 6` (spec `< 8`). Left as-is; changing feature
  definitions after the freeze would invalidate the reported numbers.
- **F-7** medians, bucket vocabularies and the stoplist are fit on train+test.
  Target-independent, so mild. Belongs inside the pipeline. Disclosed in the
  thesis and in DEFENCE_BRIEF.md.
- **F-8** `is_off_hours_commit` / `is_weekend_commit` use UTC across globally
  distributed projects — near-noise as defined. Disclosed.
- **F-9** `files_changed` censored at 300 by the API (110 rows). Disclosed.

**Found later, not in the original review.**
- `src/run_phase2.py` had never been runnable: it imported
  `ENGINEERED_FEATURE_COLUMNS` and `split_dataset`, neither of which existed in
  `data_preparation.py` even in the initial commit. Appendix B's reproduction
  sequence could not have completed. Both symbols restored.
- Appendix B referenced `src/run_phase0.py` and `src/run_phase1.py`, which do
  not exist. Corrected; `./reproduce.sh` is now the entry point.
- Reference [1] (Patel 2019) could not be verified against the cited venue.
  Chapter 3 positions the whole contribution against it. **Moamen must check it
  against the source PDF.**

## Working rules

- **Quote metrics from `results/*.json`, never from thesis prose.** The two have
  already diverged.
- A changed number must propagate to four places: `results/*.json`, the figure,
  its entry in `figures/captions.md`, and
  `../final-version-doc/files/4_Thesis_Source_Markdown.md`. State which of the
  four you updated.
- **Do not retrain without asking.** A full Phase 4 run overwrites `models/`.
- **Never quote `results/single_fold_reference/` or `results/superseded/`.**
  Single-fold estimates on this data vary by ~20 points of F1 depending on the
  fold drawn, because repository identity dominates and no grouped splitter
  balances repository composition.
- **Flag any metric that improves without a mechanism.** This project has already
  been burned once by leakage; a jump with no causal story is a bug until proven
  otherwise.
- Report failures with the actual output. Never present a number as clean when it
  came from the stratified split without saying so.
- Do not add `run_duration_sec`, `run_attempt`, `is_retry`, `status` or
  `updated_at` to any feature set — they are post-execution and the strictly
  pre-execution framing is the thesis's central claim.

## Thesis prose conventions

Academic register: no contractions, no bullet lists inside chapter body text,
IEEE-numbered citations, figures referenced as "Figure 7.4" in text. Edit
`4_Thesis_Source_Markdown.md`; the `.docx` is a derived artifact.

## Do not commit or upload

`.venv/`, `*.joblib`, any CSV over 1 MB, `~$*.docx` lock files, and
~~`phase0.md` line 13~~ — the PAT is redacted and was revoked by the owner.
The literal string survives in git history at commit `c5851d0`; revocation, not
redaction, is what makes it inert.
