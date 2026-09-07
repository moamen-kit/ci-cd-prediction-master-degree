# Corrections Since the Previous Submission

**Thesis:** Predicting CI/CD Pipeline Build Failures Using Machine Learning Techniques
**Student:** Moamen Mohamed Aly Hussein (ID: 202401681)
**Prepared:** 2026-09-05

This note records what changed between the previously submitted version of this
thesis and the current one, and why. It is provided so that the committee can
see the difference stated plainly rather than discover it.

---

## Summary

An internal review of the evaluation protocol identified two defects that
inflated the reported performance. Both have been corrected, the affected
numbers have been regenerated throughout, and the magnitude of each defect has
been measured and is now reported in the thesis as Section 7.4.5.

The headline failure-class F1 falls from **0.5924** to **0.4216**. The model,
its hyperparameters, the dataset and the random seed are unchanged. What changed
is the method of measurement.

---

## Defect 1 — Duplicate commits shared across the split

Each row of the dataset is one workflow run, but every feature the model uses
describes a *commit*. The dataset contains 9,772 runs drawn from only 2,835
distinct commits, an average of 3.45 runs per commit and a maximum of 179.

The original evaluation split rows at random, which placed near-identical
observations of the same commit on both sides of the partition. Measured on that
split, **1,726 of the 1,955 test rows (88.3 per cent) shared a commit with the
training set.** The model was therefore rewarded partly for recognising commits
it had already been shown.

**Correction.** Every partition is now produced by a splitter that assigns all
runs of a given commit to exactly one side. The absence of any shared commit is
verified programmatically and recorded in `results/split_integrity.json`.

**Cost of the defect: 0.060 of failure-class F1.**

---

## Defect 2 — Decision threshold selected on the test set

The original evaluation chose the decision threshold by maximising failure-class
F1 over the test set, and then reported the resulting F1 as the performance of
the model on that same test set. A quantity selected to maximise a value cannot
also serve as an unbiased estimate of that value.

**Correction.** The threshold is now selected on a validation partition carved
from the training data under the same grouping constraint, and applied unchanged
to the test partition. The residual optimism that the original procedure would
still have purchased is measured and reported in
`results/threshold_selection_gap.json`.

**Cost of the defect: 0.126 of failure-class F1.**

This was the larger of the two, by a factor of approximately two. It is also the
less conspicuous: the threshold sweep is a legitimate technique, and only its
placement relative to the evaluation data was wrong.

---

## Consequential change — cross-validation replaces a single split

Repository identity is the strongest single predictor available to the model,
and no grouped splitter balances the composition of repositories across
partitions. Single-partition estimates of failure-class F1 on this dataset were
found to vary across a range of approximately 20 percentage points depending on
which partition was drawn.

Primary results are therefore now obtained by five-fold commit-grouped
cross-validation. Each of the 9,772 runs receives exactly one prediction from a
model trained without any run of that run's commit.

---

## Attribution

Each correction is applied independently, with the classifier, hyperparameters
and seed held constant.

| Step | Evaluation protocol | Failure F1 | Change |
|---|---|---|---|
| A | Row-level stratified split, threshold selected on the test set | 0.5924 | — |
| B | Commit-grouped split, threshold still selected on the test set | 0.5326 | −0.060 |
| C | Commit-grouped split, threshold selected on a validation partition | 0.4065 | −0.126 |
| D | Commit-grouped five-fold cross-validation | 0.4216 | +0.015 |

Reproducible via `./reproduce.sh`; recorded in
`results/metric_attribution_ladder.json`.

---

## Other corrections made at the same time

**The chronological evaluation was not chronological.** It sorted by the commit
authoring date rather than the run execution date, which produced a test
partition spanning approximately eleven hours in which 95.8 per cent of runs had
in fact executed *before* the last run in the training set. It is now cut per
repository on execution time, and for all eighteen repositories the earliest
test execution occurs no earlier than the latest training execution.

**The business model did not implement its own specification.** It assumed a 30
per cent failure rate against the 11 per cent observed, and charged nothing for
false alarms, which made the estimated saving a function of recall alone — and
is why the earlier version reported that the tuned model saved *less* than the
untuned one. Rebuilt against the specified cost model, the estimate falls from
$382,802 to $243,670 per year, and is now reported alongside the break-even
false-alarm cost at which it reaches zero.

**An ablation specified in the project plan had never been run.** The
categorical-only configuration, which receives nothing but repository, workflow
name, branch and trigger, outperforms the full hybrid model on every metric
(F1 0.4808 against 0.4216). This refutes the project's original hypothesis more
comprehensively than previously reported, and is now stated as the central
empirical finding: the system is substantially a project-level risk estimator
rather than a commit-level one.

**Two reproduction defects.** `src/run_phase2.py` imported two symbols that do
not exist and therefore could never run, and Appendix B instructed the reader to
execute two scripts absent from the repository. Both are fixed, and the whole
pipeline now reproduces from raw data with a single command in about three
minutes.

---

## What did not change

The dataset, the feature engineering, the model architecture, the
hyperparameters and the random seed are all unchanged. No result improved as a
consequence of these corrections. The narrative of the thesis — that structured
features dominate, that the hybrid hypothesis is refuted, and that decision-rule
calibration matters — is unaffected and in each case is now supported by
stronger evidence.

---

## Position taken

The corrected figures are lower. The author judges the disclosure to be of
greater value than the figures it replaces: an evaluation protocol that cannot
be defended does not become defensible by producing a larger number. Both
defects are of a kind documented across the wider literature — Kapoor and
Narayanan [21] find leakage of this class in 294 papers across seventeen fields
— and both leave every individual step of an analysis looking correct while
inflating its result.
