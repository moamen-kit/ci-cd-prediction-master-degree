"""Full training, evaluation and ablation study for the binary task.

Computes the binary-classification metrics that the thesis Results chapter
relies on:

* per-model: accuracy, balanced accuracy, macro F1, precision/recall/F1 on
  the positive class (``failure``), ROC-AUC, PR-AUC, confusion matrix.
* :func:`run_ablation_study` — XGBoost on text-only / structured-only /
  hybrid feature sets so we can quantify the marginal value of each branch.
* :func:`compute_business_metrics` — translate ``failure``-class recall into
  hypothetical dollar savings.
* :func:`identify_best_model` / :func:`save_best_model` — rank by failure
  F1 (more honest than raw accuracy on the imbalanced target) and persist
  the winning pipeline with a metadata sidecar.
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_preparation import (
    ALL_FEATURE_COLUMNS,
    TARGET,
)
from .hybrid_pipeline import (
    LabelEncoderForBinary,
    build_categorical_only_preprocessor,
    build_structured_only_preprocessor,
    build_text_only_preprocessor,
    build_xgboost_with_preprocessor,
    get_all_pipelines,
)
from .threshold_optimization import DEFAULT_FP_COST
from .utils import ensure_dir, get_logger


_LOGGER = get_logger(__name__)

RANDOM_STATE = 42
JOBLIB_COMPRESSION = 3

POSITIVE_LABEL = "failure"  # the class we actually want to detect


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Business cost model (phase4.md section 1.5)
# --------------------------------------------------------------------------- #


# Observed in this dataset: 1,072 failures / 9,772 runs. The previous
# implementation hardcoded 0.30, which inflated every downstream figure by 2.7x.
OBSERVED_FAILURE_RATE = 0.1097

# 8 minutes of wasted compute at $0.008/minute.
COMPUTE_COST_PER_FAILED_BUILD_USD = 0.064

# 15 minutes of developer context-switching at $75/hour.
CONTEXT_SWITCH_COST_PER_FAILURE_USD = 18.75

# 2 minutes of operator triage at $75/hour. Imported rather than redefined so
# the threshold sweep and the business model cannot drift apart.
FALSE_ALARM_COST_USD = DEFAULT_FP_COST


def compute_binary_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba_positive: np.ndarray,
    classes: list[str],
    positive_label: str = POSITIVE_LABEL,
) -> dict[str, Any]:
    """Return the metric set that goes into ``phase4_summary.json``."""
    y_true_str = np.asarray(y_true).astype(str)
    y_pred_str = np.asarray(y_pred).astype(str)

    y_true_bin = (y_true_str == positive_label).astype(int)
    y_pred_bin = (y_pred_str == positive_label).astype(int)

    ordered_classes = list(classes)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_str, y_pred_str)),
        "macro_f1": float(
            f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true_str, y_pred_str, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true_str, y_pred_str, average="macro", zero_division=0
            )
        ),
        # Positive-class (failure) metrics — the ones that matter for ops.
        "failure_precision": float(
            precision_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "failure_recall": float(
            recall_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "failure_f1": float(
            f1_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "positive_label": positive_label,
        "classes_ordered": ordered_classes,
        "confusion_matrix": confusion_matrix(
            y_true_str, y_pred_str, labels=ordered_classes
        ).tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_proba_positive))
    except Exception as exc:  # pragma: no cover — defensive
        _LOGGER.warning("ROC-AUC failed: %s", exc)
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = float(
            average_precision_score(y_true_bin, y_proba_positive)
        )
    except Exception as exc:  # pragma: no cover — defensive
        _LOGGER.warning("PR-AUC failed: %s", exc)
        metrics["pr_auc"] = None

    return metrics


# --------------------------------------------------------------------------- #
# Pipeline probing
# --------------------------------------------------------------------------- #


def get_proba_and_classes(
    pipeline: Any, x_test: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Return ``(proba_matrix, ordered_class_labels)`` for any of the 3 pipelines."""
    if isinstance(pipeline, LabelEncoderForBinary):
        proba = np.asarray(pipeline.estimator.predict_proba(x_test))
        classes = [str(c) for c in pipeline.encoder_.classes_]
    else:
        proba = np.asarray(pipeline.predict_proba(x_test))
        classifier = pipeline.named_steps["classifier"]
        classes = [str(c) for c in classifier.classes_]
    return proba, classes


def evaluate_pipeline(
    pipeline: Any, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    """Predict + all binary metrics for one fitted pipeline."""
    _LOGGER.info("Predicting on %d test rows ...", len(x_test))
    t0 = time.perf_counter()
    y_pred = pipeline.predict(x_test)
    predict_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba_matrix, classes = get_proba_and_classes(pipeline, x_test)
    proba_time = time.perf_counter() - t0

    try:
        pos_idx = classes.index(POSITIVE_LABEL)
    except ValueError:
        # Fall back to the last column.
        pos_idx = proba_matrix.shape[1] - 1

    y_proba_positive = proba_matrix[:, pos_idx]

    metrics = compute_binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba_positive=y_proba_positive,
        classes=classes,
    )

    return {
        "predict_time_sec": round(predict_time, 3),
        "predict_proba_time_sec": round(proba_time, 3),
        "n_test_samples": int(len(x_test)),
        "metrics": metrics,
        # In-memory artefacts kept for plotting — stripped before JSON dump.
        "_y_pred": y_pred,
        "_y_proba_positive": y_proba_positive,
        "_classes": classes,
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_all_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path | None,
) -> dict[str, Any]:
    """Fit each of the three pipelines, persist them, then evaluate.

    ``save_dir=None`` fits and evaluates without persisting, which the
    chronological secondary evaluation uses: those models exist to produce a
    transfer number, not to be shipped, and writing them would overwrite the
    primary models of the same name.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        ensure_dir(save_dir)

    pipelines = get_all_pipelines()
    results: dict[str, Any] = {}

    for name, pipeline in pipelines.items():
        slug = name.lower().replace(" ", "_")
        _LOGGER.info("=" * 70)
        _LOGGER.info("Training %s on %d rows ...", name, len(x_train))
        _LOGGER.info("=" * 70)

        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0
        _LOGGER.info("%s fitted in %.1fs", name, fit_time)

        if save_dir is not None:
            model_path = save_dir / f"{slug}_full.joblib"
            joblib.dump(pipeline, model_path, compress=JOBLIB_COMPRESSION)
            _LOGGER.info("Saved %s → %s", name, model_path)
        else:
            model_path = None

        eval_result = evaluate_pipeline(pipeline, x_test, y_test)
        eval_result["fit_time_sec"] = round(fit_time, 3)
        eval_result["model_path"] = str(model_path) if model_path else None
        eval_result["_pipeline"] = pipeline
        results[name] = eval_result

    return results


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #


def run_ablation_study(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path,
    hybrid_xgb: Any | None = None,
) -> dict[str, Any]:
    """Compare reduced feature sets against the full hybrid, XGBoost throughout.

    Configurations: ``hybrid_full``, ``text_only``, ``structured_only`` and
    ``categorical_only``. The last is the discriminating experiment — see
    :func:`~src.hybrid_pipeline.build_categorical_only_preprocessor`.

    If ``hybrid_xgb`` is supplied it is reused as the ``hybrid_full`` row so
    we don't retrain the same model twice.
    """
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    results: dict[str, Any] = {}

    if hybrid_xgb is None:
        raise ValueError("hybrid_xgb must be supplied to avoid retraining.")

    # 1) hybrid (already trained) — just re-evaluate so the entry exists.
    _LOGGER.info("[ablation] hybrid_full → reusing trained pipeline")
    hybrid_eval = evaluate_pipeline(hybrid_xgb, x_test, y_test)
    hybrid_eval["fit_time_sec"] = 0.0
    hybrid_eval["_pipeline"] = hybrid_xgb
    results["hybrid_full"] = hybrid_eval

    # 2) each reduced feature set, classifier configuration held identical
    for config_name, preprocessor in (
        ("text_only", build_text_only_preprocessor()),
        ("structured_only", build_structured_only_preprocessor()),
        ("categorical_only", build_categorical_only_preprocessor()),
    ):
        _LOGGER.info("[ablation] %s → training XGBoost ...", config_name)
        pipeline = build_xgboost_with_preprocessor(preprocessor)
        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0
        _LOGGER.info("[ablation] %s fitted in %.1fs", config_name, fit_time)
        joblib.dump(
            pipeline,
            save_dir / f"xgb_{config_name}.joblib",
            compress=JOBLIB_COMPRESSION,
        )

        eval_result = evaluate_pipeline(pipeline, x_test, y_test)
        eval_result["fit_time_sec"] = round(fit_time, 3)
        eval_result["_pipeline"] = pipeline
        results[config_name] = eval_result

    return results


# --------------------------------------------------------------------------- #
# Business impact
# --------------------------------------------------------------------------- #


def compute_business_metrics(
    failure_recall: float,
    failure_precision: float,
    pipelines_per_day: int = 1_000,
    failure_rate: float = OBSERVED_FAILURE_RATE,
    avg_latency_ms: float | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Net operational value of the predictor, per the Phase 4 cost model.

    Rebuilt against the specification in ``phase4.md``. The previous
    implementation had three defects that all pushed the estimate upward: it
    assumed a 30 per cent failure rate against an observed 11 per cent, priced
    the benefit as saved triage minutes rather than the specified compute plus
    context-switch cost, and never subtracted the cost of a false alarm, which
    made the estimate a function of recall alone. That last defect is why the
    tuned model previously reported *lower* savings than the untuned one: it
    traded precision for recall, and only recall was being priced.

    Cost model, all per failed build:

    * ``$0.064`` of wasted compute (8 minutes at ``$0.008``/minute);
    * ``$18.75`` of developer context-switching (15 minutes at ``$75``/hour);
    * ``$2.50`` per false alarm (2 minutes of operator triage at ``$75``/hour),
      taken from :data:`~src.threshold_optimization.DEFAULT_FP_COST` so the
      project carries one cost model rather than two.

    A caught failure returns compute plus context-switch cost; a false alarm
    costs triage time. Net savings are the difference, so a model cannot buy a
    better number purely by flagging more builds.
    """
    failures_per_day = pipelines_per_day * failure_rate
    caught_per_day = failures_per_day * failure_recall

    # false alarms = FP, derived from precision: TP/(TP+FP) = precision
    if failure_precision > 0:
        false_alarms_per_day = caught_per_day * (1.0 - failure_precision) / failure_precision
    else:
        false_alarms_per_day = 0.0

    compute_saved = caught_per_day * COMPUTE_COST_PER_FAILED_BUILD_USD
    developer_time_saved = caught_per_day * CONTEXT_SWITCH_COST_PER_FAILURE_USD
    gross_daily_saved = compute_saved + developer_time_saved
    false_alarm_cost = false_alarms_per_day * FALSE_ALARM_COST_USD
    net_daily_saved = gross_daily_saved - false_alarm_cost

    result: dict[str, Any] = {
        "configuration": label,
        "failure_recall": round(float(failure_recall), 4),
        "failure_precision": round(float(failure_precision), 4),
        "daily_failures": round(failures_per_day, 1),
        "daily_caught_by_model": round(caught_per_day, 1),
        "daily_false_alarms": round(false_alarms_per_day, 1),
        "daily_compute_saved_usd": round(compute_saved, 2),
        "daily_developer_time_saved_usd": round(developer_time_saved, 2),
        "daily_gross_saved_usd": round(gross_daily_saved, 2),
        "daily_false_alarm_cost_usd": round(false_alarm_cost, 2),
        "daily_net_saved_usd": round(net_daily_saved, 2),
        "monthly_net_saved_usd": round(net_daily_saved * 30.0, 2),
        "annual_net_saved_usd": round(net_daily_saved * 365.0, 2),
        "assumptions": {
            "pipelines_per_day": pipelines_per_day,
            "failure_rate": round(float(failure_rate), 4),
            "failure_rate_source": "observed in this dataset (1,072 / 9,772)",
            "compute_cost_per_failed_build_usd": COMPUTE_COST_PER_FAILED_BUILD_USD,
            "context_switch_cost_per_failure_usd": CONTEXT_SWITCH_COST_PER_FAILURE_USD,
            "false_alarm_cost_usd": FALSE_ALARM_COST_USD,
            "savings_formula": (
                "net_daily = caught x (compute + context_switch) "
                "- false_alarms x false_alarm_cost"
            ),
        },
    }
    if avg_latency_ms is not None:
        result["average_inference_latency_ms"] = round(float(avg_latency_ms), 3)
    return result


# --------------------------------------------------------------------------- #
# Best-model selection + persistence
# --------------------------------------------------------------------------- #


def identify_best_model(
    model_results: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    """Rank by F1 on the positive (``failure``) class — robust to imbalance."""
    scores: dict[str, float] = {}
    for name, info in model_results.items():
        scores[name] = float(info["metrics"]["failure_f1"])
    best = max(scores, key=lambda k: scores[k])
    return best, scores


def _classifier_hyperparams(pipeline: Any) -> dict[str, str]:
    inner = pipeline.estimator if isinstance(pipeline, LabelEncoderForBinary) else pipeline
    classifier = inner.named_steps["classifier"]
    try:
        params = classifier.get_params()
    except Exception:  # pragma: no cover — defensive
        params = {}
    return {k: str(v) for k, v in params.items()}


def save_best_model(
    pipeline: Any,
    name: str,
    model_results: dict[str, Any],
    save_dir: Path,
) -> dict[str, Any]:
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    best_path = save_dir / "best_default_threshold_rf.joblib"
    joblib.dump(pipeline, best_path, compress=JOBLIB_COMPRESSION)

    test_metrics = {
        k: v for k, v in model_results[name]["metrics"].items()
        if not k.startswith("_")
    }

    metadata = {
        "model_name": name,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": _classifier_hyperparams(pipeline),
        "test_metrics": test_metrics,
        "expected_feature_columns": ALL_FEATURE_COLUMNS,
        "target_column": TARGET,
        "positive_label": POSITIVE_LABEL,
        "fit_time_sec": model_results[name].get("fit_time_sec"),
        "predict_time_sec": model_results[name].get("predict_time_sec"),
        "n_test_samples": model_results[name].get("n_test_samples"),
    }

    metadata_path = save_dir / "best_default_threshold_rf_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    _LOGGER.info("Saved best model → %s", best_path)
    _LOGGER.info("Saved metadata   → %s", metadata_path)

    return {
        "best_model_path": str(best_path),
        "best_model_metadata_path": str(metadata_path),
        "best_model_name": name,
    }


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #


def serializable_results(results: dict[str, Any]) -> dict[str, Any]:
    """Strip in-memory artefacts (pipelines, ndarrays) before JSON dump."""
    clean: dict[str, Any] = {}
    for name, info in results.items():
        clean[name] = {k: v for k, v in info.items() if not k.startswith("_")}
    return clean


__all__ = [
    "POSITIVE_LABEL",
    "compute_binary_metrics",
    "compute_business_metrics",
    "evaluate_pipeline",
    "get_proba_and_classes",
    "identify_best_model",
    "run_ablation_study",
    "save_best_model",
    "serializable_results",
    "train_all_models",
]
