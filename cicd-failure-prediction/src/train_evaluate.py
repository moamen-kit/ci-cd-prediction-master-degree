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
from .utils import ensure_dir, get_logger


_LOGGER = get_logger(__name__)

RANDOM_STATE = 42
JOBLIB_COMPRESSION = 3

POSITIVE_LABEL = "failure"  # the class we actually want to detect


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


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
    model_results: dict[str, Any], best_model_name: str
) -> dict[str, Any]:
    """Translate ``failure``-class detection into operational dollar savings."""
    best = model_results[best_model_name]
    metrics = best["metrics"]
    n_test = int(best["n_test_samples"])
    predict_time_sec = float(best["predict_time_sec"])
    avg_latency_ms = (predict_time_sec * 1000.0) / max(n_test, 1)

    failure_recall = float(metrics["failure_recall"])
    failure_precision = float(metrics["failure_precision"])

    pipelines_per_day = 1_000
    failure_rate = 0.30  # hypothetical mid-size org failure rate
    failures_per_day = pipelines_per_day * failure_rate

    manual_minutes = 5.0
    auto_minutes = 0.5
    devops_hourly_rate_usd = 75.0

    time_saved_per_caught_min = manual_minutes - auto_minutes
    caught_failures_per_day = failures_per_day * failure_recall
    daily_minutes_saved = caught_failures_per_day * time_saved_per_caught_min
    daily_hours_saved = daily_minutes_saved / 60.0
    daily_usd_saved = daily_hours_saved * devops_hourly_rate_usd
    monthly_usd_saved = daily_usd_saved * 30.0
    annual_usd_saved = daily_usd_saved * 365.0

    return {
        "best_model": best_model_name,
        "failure_recall": round(failure_recall, 4),
        "failure_precision": round(failure_precision, 4),
        "failure_f1": round(float(metrics["failure_f1"]), 4),
        "average_inference_latency_ms": round(avg_latency_ms, 3),
        "routing_reduction_per_failure_seconds": round(
            (manual_minutes - auto_minutes) * 60.0, 1
        ),
        "daily_failures_to_triage": int(failures_per_day),
        "daily_caught_by_model": round(caught_failures_per_day, 1),
        "daily_usd_saved": round(daily_usd_saved, 2),
        "monthly_usd_saved": round(monthly_usd_saved, 2),
        "annual_usd_saved": round(annual_usd_saved, 2),
        "assumptions": {
            "pipelines_per_day": pipelines_per_day,
            "failure_rate": failure_rate,
            "manual_minutes_per_failure": manual_minutes,
            "auto_minutes_per_failure": auto_minutes,
            "devops_hourly_rate_usd": devops_hourly_rate_usd,
            "savings_formula": "daily_usd = (failures × recall_failure × Δminutes / 60) × hourly_rate",
        },
    }


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
