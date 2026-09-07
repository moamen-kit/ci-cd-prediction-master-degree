"""Commit-grouped cross-validation — the primary evaluation protocol.

A single held-out fold is not a safe basis for a headline number on this
dataset. Repository identity is the strongest available feature (the
``categorical_only`` ablation outranks the full hybrid model), and no grouped
splitter balances the repository mix, so the composition of any one fold moves
the result. Measured on the same data and the same pipeline:

======================================  ==========  ===========
fold drawn by                           XGB best F1  ROC-AUC
======================================  ==========  ===========
``GroupShuffleSplit(random_state=42)``       0.533        0.834
``StratifiedGroupKFold`` fold 1              0.560        0.873
======================================  ==========  ===========

The gap is fold composition, not modelling: the ``StratifiedGroupKFold`` fold
over-represents both ``prisma/prisma`` (38.3 per cent failure) and
``elastic/elasticsearch`` (0.0 per cent failure) by roughly 43 per cent
relative to their share of the corpus, which makes repository identity more
discriminative on that fold than on the population.

Cross-validation removes the choice. Every row is predicted exactly once, by a
model that never saw any run of that row's commit:

* **Ranking metrics** (ROC-AUC, PR-AUC) are computed once over the pooled
  out-of-fold probabilities, so they use all 9,772 rows.
* **Decision metrics** (precision, recall, F1) are computed per fold and
  reported as mean and standard deviation, because they depend on a threshold
  and the spread is itself a result worth reporting.
* **The threshold is never chosen on evaluation data.** Within each fold, an
  inner validation set is carved from that fold's training rows — grouped on
  ``commit_sha`` like everything else — the threshold is selected there, and
  only then applied to the held-out fold.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .data_preparation import GROUP_KEY, TARGET
from .hybrid_pipeline import prepare_features_targets
from .threshold_optimization import find_optimal_threshold
from .utils import get_logger

_LOGGER = get_logger(__name__)

POSITIVE_LABEL = "failure"


def _positive_proba(pipeline: Any, x: pd.DataFrame) -> np.ndarray:
    classes = list(pipeline.classes_)
    return pipeline.predict_proba(x)[:, classes.index(POSITIVE_LABEL)]


def _binarise(y: Any) -> np.ndarray:
    return (np.asarray(y).astype(str) == POSITIVE_LABEL).astype(int)


def _decision_metrics(y_true_bin: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision_failure": float(precision_score(y_true_bin, pred, zero_division=0)),
        "recall_failure": float(recall_score(y_true_bin, pred, zero_division=0)),
        "f1_failure": float(f1_score(y_true_bin, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_bin, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "per_fold": [round(float(v), 4) for v in arr],
    }


def run_grouped_cv(
    df: pd.DataFrame,
    pipeline_factories: dict[str, Callable[[], Any]],
    n_splits: int = 5,
    random_state: int = 42,
    inner_val_splits: int = 5,
) -> dict[str, Any]:
    """Run commit-grouped CV with per-fold, validation-selected thresholds.

    ``pipeline_factories`` maps a display name to a zero-argument callable
    returning a fresh, unfitted pipeline — a factory rather than an instance so
    that every fold trains a genuinely independent model.
    """
    groups = df[GROUP_KEY].astype(str)
    y_all = df[TARGET].astype(str)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(outer.split(df, y_all, groups))

    results: dict[str, Any] = {}

    for name, factory in pipeline_factories.items():
        _LOGGER.info("=" * 70)
        _LOGGER.info("Cross-validating %s over %d commit-grouped folds", name, n_splits)
        _LOGGER.info("=" * 70)

        oof_proba = np.full(len(df), np.nan)
        per_fold: list[dict[str, Any]] = []
        fit_seconds = 0.0

        for fold_id, (train_pool_idx, test_idx) in enumerate(folds, start=1):
            train_pool = df.iloc[train_pool_idx]

            # Inner grouped split: threshold is selected here, never on test.
            inner = StratifiedGroupKFold(
                n_splits=inner_val_splits, shuffle=True, random_state=random_state
            )
            inner_train_idx, inner_val_idx = next(
                inner.split(
                    train_pool,
                    train_pool[TARGET].astype(str),
                    train_pool[GROUP_KEY].astype(str),
                )
            )
            fit_df = train_pool.iloc[inner_train_idx]
            val_df = train_pool.iloc[inner_val_idx]
            test_df = df.iloc[test_idx]

            x_fit, y_fit = prepare_features_targets(fit_df)
            x_val, y_val = prepare_features_targets(val_df)
            x_test, y_test = prepare_features_targets(test_df)

            pipeline = factory()
            t0 = time.perf_counter()
            pipeline.fit(x_fit, y_fit)
            fit_seconds += time.perf_counter() - t0

            val_proba = _positive_proba(pipeline, x_val)
            chosen = find_optimal_threshold(_binarise(y_val), val_proba, "f1")
            threshold = float(chosen["optimal_threshold"])

            test_proba = _positive_proba(pipeline, x_test)
            oof_proba[test_idx] = test_proba
            y_test_bin = _binarise(y_test)

            fold_metrics = _decision_metrics(y_test_bin, test_proba, threshold)
            fold_metrics.update(
                {
                    "fold": fold_id,
                    "n_fit": int(len(fit_df)),
                    "n_val": int(len(val_df)),
                    "n_test": int(len(test_df)),
                    "test_failure_rate_pct": round(float(y_test_bin.mean() * 100.0), 3),
                    "roc_auc": float(roc_auc_score(y_test_bin, test_proba)),
                    "pr_auc": float(average_precision_score(y_test_bin, test_proba)),
                    "f1_at_default_0.5": float(
                        f1_score(y_test_bin, (test_proba >= 0.5).astype(int), zero_division=0)
                    ),
                    "commit_overlap_fit_test": int(
                        len(set(fit_df[GROUP_KEY].astype(str)) & set(test_df[GROUP_KEY].astype(str)))
                    ),
                    "commit_overlap_val_test": int(
                        len(set(val_df[GROUP_KEY].astype(str)) & set(test_df[GROUP_KEY].astype(str)))
                    ),
                }
            )
            per_fold.append(fold_metrics)
            _LOGGER.info(
                "  fold %d: thr=%.2f F1=%.4f PR-AUC=%.4f ROC-AUC=%.4f (overlap %d/%d)",
                fold_id, threshold, fold_metrics["f1_failure"],
                fold_metrics["pr_auc"], fold_metrics["roc_auc"],
                fold_metrics["commit_overlap_fit_test"],
                fold_metrics["commit_overlap_val_test"],
            )

        assert not np.isnan(oof_proba).any(), "every row must receive one out-of-fold prediction"
        y_all_bin = _binarise(y_all)

        results[name] = {
            "n_splits": n_splits,
            "fit_time_sec_total": round(fit_seconds, 3),
            "pooled_out_of_fold": {
                "n_rows": int(len(df)),
                "roc_auc": round(float(roc_auc_score(y_all_bin, oof_proba)), 4),
                "pr_auc": round(float(average_precision_score(y_all_bin, oof_proba)), 4),
            },
            "threshold": _mean_std([f["threshold"] for f in per_fold]),
            "f1_failure": _mean_std([f["f1_failure"] for f in per_fold]),
            "precision_failure": _mean_std([f["precision_failure"] for f in per_fold]),
            "recall_failure": _mean_std([f["recall_failure"] for f in per_fold]),
            "balanced_accuracy": _mean_std([f["balanced_accuracy"] for f in per_fold]),
            "roc_auc_per_fold": _mean_std([f["roc_auc"] for f in per_fold]),
            "pr_auc_per_fold": _mean_std([f["pr_auc"] for f in per_fold]),
            "f1_at_default_0.5": _mean_std([f["f1_at_default_0.5"] for f in per_fold]),
            "folds": per_fold,
            "_oof_proba": oof_proba,
        }

    return results


def cv_summary_table(cv_results: dict[str, Any]) -> str:
    """Render the CV result as a fixed-width table for the phase log."""
    header = (
        f"{'model':<22} {'F1 (mean±sd)':>18} {'precision':>16} {'recall':>16} "
        f"{'PR-AUC(oof)':>12} {'ROC-AUC(oof)':>13} {'thr':>12}"
    )
    lines = [header, "-" * len(header)]
    for name, r in cv_results.items():
        f1, pr_, rc = r["f1_failure"], r["precision_failure"], r["recall_failure"]
        lines.append(
            f"{name:<22} {f1['mean']:.4f} ± {f1['std']:.4f}  "
            f"{pr_['mean']:.4f} ± {pr_['std']:.3f}  "
            f"{rc['mean']:.4f} ± {rc['std']:.3f}  "
            f"{r['pooled_out_of_fold']['pr_auc']:12.4f} "
            f"{r['pooled_out_of_fold']['roc_auc']:13.4f} "
            f"{r['threshold']['mean']:8.3f}"
        )
    return "\n".join(lines)


__all__ = ["cv_summary_table", "run_grouped_cv"]
