# Data Profile — `data/raw/github_actions_real.csv`

Machine-generated summary so Claude can reason about the dataset without loading
the 5 MB CSV. Regenerate after any change to data collection.

**Verified:** 2026-08-29 · **Rows:** 9,772 · **Columns:** 20

## Collection

Collected via `src/collect_github_data.py` from the GitHub Actions API
(`/actions/runs` joined to `/commits/{sha}`), capped at 600 runs per repository.
Kept only `status == "completed"` and `conclusion in {success, failure}`.

- `created_at` (workflow run start): **2025-11-25 → 2026-05-29** (6 months)
- `commit_date` (commit authored): **2015-05-18 → 2026-05-29** (11 years)

Those two ranges differ because re-runs and merge-base commits attach old commits
to recent runs. **`created_at` is the prediction event; `commit_date` is not.**

## Target

`conclusion` — string labels, **`failure` is the positive class**.

| Class | Count | Share |
|---|---|---|
| success | 8,700 | 89.03% |
| failure | 1,072 | 10.97% |

`scale_pos_weight = 8.12` (LR/RF use `class_weight='balanced'`).

## Grain: rows are workflow runs, not commits

| | |
|---|---|
| Rows | 9,772 |
| Unique `commit_sha` | **2,835** |
| Unique `commit_message` | 2,281 |
| Runs per commit | mean 3.45 · median 2 · max 179 |
| Commits with >1 run | 1,658 |
| Commits with **mixed** outcomes | 432 (3,641 rows) |

Every modelled feature is commit-level, so multiple rows per commit are
near-duplicates differing only in `workflow_name` / `branch` / `event`.
**Any split must group on `commit_sha`.** See [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) F-1.

## Cardinality

| Column | Unique | Handling |
|---|---|---|
| `repository` | 18 | one-hot as-is |
| `workflow_name` | 623 | bucketed to top-20 + `__other__` = 21 |
| `branch` | 1,396 | bucketed to top-15 + `__other__` = 16 |
| `event` | 14 | one-hot as-is |
| `commit_author` | 557 | dropped after `is_bot_author`; feeds the stoplist |
| `commit_sha` | 2,835 | dropped |

Fused matrix after preprocessing: 5 numerical + 66 one-hot + 6 binary + ~1,814–3,000 TF-IDF.

## Failure rate by repository

| Repository | n | Failure % |
|---|---|---|
| prisma/prisma | 600 | 38.3 |
| rust-lang/rust | 600 | 25.7 |
| kubernetes/kubernetes | 115 | 20.0 |
| pytorch/pytorch | 600 | 18.8 |
| facebook/react | 600 | 14.8 |
| vercel/next.js | 600 | 13.3 |
| sveltejs/svelte | 427 | 12.6 |
| huggingface/transformers | 594 | 9.9 |
| pandas-dev/pandas | 357 | 9.8 |
| python/cpython | 600 | 7.0 |
| nodejs/node | 600 | 5.8 |
| expressjs/express | 600 | 5.8 |
| microsoft/vscode | 600 | 5.5 |
| nestjs/nest | 600 | 5.0 |
| tensorflow/tensorflow | 600 | 3.7 |
| scikit-learn/scikit-learn | 479 | 3.5 |
| ruby/ruby | 600 | 3.5 |
| **elastic/elasticsearch** | 600 | **0.0** |

A 38× spread across repositories. `elastic/elasticsearch` contributes 600 rows
(6.1% of the dataset) with **zero** failures — one-hot `repository` makes it a
perfect success predictor, which inflates the categorical branch. Note this
whenever discussing whether repository identity dominates the model.

## Failure rate by trigger event (n ≥ 50)

| Event | n | Failure % |
|---|---|---|
| schedule | 667 | 23.2 |
| pull_request | 4,260 | 13.3 |
| dynamic | 1,039 | 11.1 |
| push | 1,745 | 8.6 |
| workflow_run | 980 | 6.4 |
| issue_comment | 181 | 5.5 |
| pull_request_target | 652 | 1.7 |
| issues | 61 | 0.0 |
| status | 132 | 0.0 |

Tail below 50 rows: `workflow_dispatch` 27 · `merge_group` 14 ·
`repository_dispatch` 12 · `branch_protection_rule` 1 · `release` 1.

## Numerical columns (raw, pre-log)

| | lines_added | lines_deleted | total_changes | files_changed | run_duration_sec |
|---|---|---|---|---|---|
| median | 23 | 7 | 35 | 2 | 115 |
| 75% | 102 | 36 | 147 | 7 | 807 |
| mean | 1,142 | 696 | 1,838 | 14.2 | 7,013 |
| max | 4,266,967 | 148,252 | 4,266,967 | **300** | 1,913,001 |

Extreme right skew — hence the `log1p` family. 31 rows exceed 100k total changes
(vendored/generated blobs).

**`files_changed` is censored at 300.** The GitHub commits API truncates `files[]`
at 300 entries, so 110 rows (51 commits) report exactly 300 for what may be far
more. `log_files_changed` and `avg_lines_per_file` inherit that ceiling.

`run_duration_sec` is **post-execution** and excluded from all feature sets.

## Commit messages

Length in chars: median 116 · mean 294 · min 5 · max 1,000.
1,021 rows (10.4%) sit at the 1,000-char collection cap → `was_truncated`.
After `clean_commit_message_for_nlp` + the 693-token stoplist: mean 31.6 tokens;
83 rows (0.8%) clean to the empty string.

## Missing values

Only two columns: `commit_author` 82 nulls (→ `__unknown__`) and
`author_association` **9,772 nulls (100%, dropped)** — the collector read it from
the commits endpoint, which does not return that field at commit level.

## Splits on disk

| File | Rows | success/failure | Status |
|---|---|---|---|
| `train_grouped.csv` | 6,254 | 89.03 / 10.97 | **PRIMARY** |
| `val_grouped.csv` | 1,565 | 89.01 / 10.99 | threshold selection only |
| `test_grouped.csv` | 1,953 | 89.04 / 10.96 | **PRIMARY** |
| `train_chronological.csv` | 6,014 | 88.79 / 11.21 | secondary |
| `val_chronological.csv` | 1,197 | 88.64 / 11.36 | secondary |
| `test_chronological.csv` | 1,636 | 90.59 / 9.41 | secondary |
| `train_stratified.csv` | 7,817 | 89.02 / 10.98 | leakage demonstration only |
| `test_stratified.csv` | 1,955 | 89.05 / 10.95 | leakage demonstration only |

The grouped split is produced by `StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=42)` on `commit_sha`. Grouping alone is not enough: plain
`GroupShuffleSplit` let the failure rate drift to 9.71 / 14.72 / 11.64 per cent
across the three folds, which would confound threshold transfer.

The chronological split cuts **each repository at its own quantiles** of
`created_at`, assigns whole commits to one side, and discards straddlers
(925 rows). A single global cut is not viable on this dataset: the 600-run cap
gives per-repository coverage from 0.4 days (`ruby/ruby`) to 182 days
(`expressjs/express`), and 88.7 per cent of rows fall in May 2026, so any global
80/20 cut yields a ~9 hour test window covering 11 of 18 repositories.

**Integrity, verified in `results/split_integrity.json`:**

- grouped and chronological: **zero** commits shared between any pair of folds
- chronological: `min(test.created_at) >= max(train.created_at)` holds for
  **all 18 of 18** repositories; pooled test window spans 57 days
- stratified: **913 commits shared** between train and test — this is the defect
  it is retained to demonstrate, not a split to evaluate on

Primary results are not read from a single fold. They come from commit-grouped
5-fold cross-validation (`src/cross_validation.py`), because single-fold
estimates on this dataset vary by roughly 20 points of failure-class F1
depending on which fold is drawn.
