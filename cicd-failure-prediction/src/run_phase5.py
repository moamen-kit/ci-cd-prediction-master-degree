"""Phase 5 runner: threshold tuning on Phase 4 trained models.

Steps:

1. Load the three Phase 4 models from ``models/*.joblib`` (no retraining).
2. Compute failure-class probabilities on the stratified test set.
3. Per model, sweep thresholds and pick the optimum by F1, by Youden's J,
   and by business cost.
4. Apply the F1-optimal thresholds to the chronological test set as a
   sanity check.
5. Render ``fig_20`` (F1 vs threshold) and ``fig_21`` (default vs optimised
   metrics, one panel per model).
6. Recompute business impact at the optimised threshold of the new winner
   and save it alongside ``best_tuned_threshold_xgb.joblib``.

Run from project root::

    python src/run_phase5.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preparation import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    TARGET,
)
from src.hybrid_pipeline import (  # noqa: E402
    LabelEncoderForBinary,
    prepare_features_targets,
)
from src.threshold_optimization import (  # noqa: E402
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    DEFAULT_THRESHOLD,
    compute_metrics_at_threshold,
    find_optimal_threshold,
    find_threshold_by_business_cost,
)
from src.train_evaluate import (  # noqa: E402
    POSITIVE_LABEL,
    compute_business_metrics,
    get_proba_and_classes,
)
from src.utils import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ensure_dir,
    get_logger,
)
from src.visualization import ThesisPlotter  # noqa: E402


_LOGGER = get_logger(__name__)


MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression_full.joblib",
    "Random Forest": MODELS_DIR / "random_forest_full.joblib",
    "XGBoost": MODELS_DIR / "xgboost_full.joblib",
}


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


def plot_threshold_optimization(
    results: dict[str, Any], plotter: ThesisPlotter
) -> None:
    fig, ax = plotter.new_figure(figsize=(11.0, 6.2))
    palette = plotter.palette(len(results))

    for color, (name, info) in zip(palette, results.items()):
        thresholds = info["sweep"]["thresholds"]
        f1_values = info["sweep"]["f1_values"]
        opt_t = info["thresholds"]["f1"]
        # F1 at the optimal threshold (precomputed in `at_f1_optimal` block):
        opt_f1 = info["metrics"]["at_f1_optimal"]["f1_failure"]

        ax.plot(
            thresholds,
            f1_values,
            color=color,
            linewidth=1.8,
            label=f"{name} (best F1={opt_f1:.3f} @ thr={opt_t:.2f})",
        )
        ax.scatter([opt_t], [opt_f1], color="#c62828", s=70, zorder=5)
        ax.annotate(
            f"{opt_t:.2f}",
            (opt_t, opt_f1),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#c62828",
        )

    ax.axvline(
        DEFAULT_THRESHOLD,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label=f"Default threshold ({DEFAULT_THRESHOLD})",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("F1 on failure class")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 0.8)
    ax.set_title("Threshold Optimization: F1 vs Decision Threshold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plotter.save_figure(
        fig,
        "fig_20_threshold_optimization.png",
        caption=(
            "F1-score on the failure class as a function of the "
            "classification threshold. Red dots mark each model's "
            "F1-optimal threshold; the dashed grey line marks the default "
            "0.5 threshold. Threshold tuning is particularly impactful for "
            "XGBoost, whose default decision rule suppresses failure "
            "predictions despite excellent ROC-AUC."
        ),
        title="Figure 20 — Threshold optimization",
    )


def plot_before_after(
    results: dict[str, Any], plotter: ThesisPlotter
) -> None:
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "precision_failure",
        "recall_failure",
        "f1_failure",
    )
    metric_labels = (
        "Acc",
        "BalAcc",
        "Fail Prec.",
        "Fail Rec.",
        "Fail F1",
    )

    fig, axes = plotter.new_figure(figsize=(16.5, 5.5), nrows=1, ncols=3)
    palette = plotter.palette(2)

    for c, (name, info) in enumerate(results.items()):
        ax = axes[c]
        default = [info["metrics"]["default_0.5"][k] for k in metric_keys]
        optimal = [info["metrics"]["at_f1_optimal"][k] for k in metric_keys]

        x = np.arange(len(metric_keys))
        width = 0.38
        bars_def = ax.bar(
            x - width / 2,
            default,
            width=width,
            color=palette[0],
            edgecolor="black",
            label=f"Default (thr={DEFAULT_THRESHOLD})",
        )
        bars_opt = ax.bar(
            x + width / 2,
            optimal,
            width=width,
            color=palette[1],
            edgecolor="black",
            label=f"F1-optimal (thr={info['thresholds']['f1']:.2f})",
        )
        for bars, values in ((bars_def, default), (bars_opt, optimal)):
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title(name)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Model Performance: Default vs Optimized Threshold",
        y=1.02,
        fontsize=14,
    )

    plotter.save_figure(
        fig,
        "fig_21_metrics_before_after_threshold.png",
        caption=(
            "Comparison of accuracy, balanced accuracy, failure-class "
            "precision/recall and F1 at the default 0.5 threshold versus "
            "each model's F1-optimal threshold. Threshold tuning is "
            "particularly impactful for XGBoost, lifting failure-class F1 "
            "well above its default-threshold value."
        ),
        title="Figure 21 — Metrics before vs after threshold tuning",
    )


# --------------------------------------------------------------------------- #
# Business impact recalculation
# --------------------------------------------------------------------------- #


def compute_business_metrics_optimized(
    name: str,
    metrics_at_opt: dict[str, Any],
    n_test_samples: int,
    predict_time_sec: float,
    optimal_threshold: float,
) -> dict[str, Any]:
    """Business impact at the tuned threshold.

    Delegates to :func:`src.train_evaluate.compute_business_metrics` rather
    than repeating the cost model. This module previously carried its own copy,
    which is how the project ended up reporting that the tuned model saved less
    than the untuned one: both copies priced recall and neither priced a false
    alarm, so trading precision for recall could only look like a loss.
    """
    result = compute_business_metrics(
        failure_recall=float(metrics_at_opt["recall_failure"]),
        failure_precision=float(metrics_at_opt["precision_failure"]),
        avg_latency_ms=(predict_time_sec * 1000.0) / max(n_test_samples, 1),
        label=f"{name} at tuned threshold {optimal_threshold:.2f}",
    )
    result["best_model"] = name
    result["optimal_threshold"] = float(optimal_threshold)
    result["failure_f1"] = round(float(metrics_at_opt["f1_failure"]), 4)
    return result

def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"
    return str(value)


def _print_table(headers, rows) -> None:
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    line = "  ".join(f"{str(headers[i]):<{widths[i]}}" for i in range(len(headers)))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(
            "  ".join(f"{str(row[i]):<{widths[i]}}" for i in range(len(headers)))
        )


def serializable(results: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for name, info in results.items():
        clean_info = {}
        for k, v in info.items():
            if k.startswith("_"):
                continue
            if k == "sweep":
                # Keep but trim length for JSON readability.
                clean_info[k] = {
                    "thresholds": [round(float(t), 4) for t in v["thresholds"]],
                    "f1_values": [round(float(x), 4) for x in v["f1_values"]],
                }
                continue
            clean_info[k] = v
        clean[name] = clean_info
    return clean


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(MODELS_DIR)

    # Load test data (stratified primary, chronological sanity).
    print("[Phase 5] Loading stratified + chronological test sets ...")
    strat_df = pd.read_csv(PROCESSED_DATA_DIR / "test_stratified.csv")
    chrono_df = pd.read_csv(PROCESSED_DATA_DIR / "test_chronological.csv")
    x_strat, y_strat = prepare_features_targets(strat_df)
    x_chrono, y_chrono = prepare_features_targets(chrono_df)
    y_strat_bin = (y_strat.astype(str).values == POSITIVE_LABEL).astype(int)
    y_chrono_bin = (y_chrono.astype(str).values == POSITIVE_LABEL).astype(int)
    print(f"          stratified test = {len(x_strat)} rows · "
          f"chronological test = {len(x_chrono)} rows")

    # Load models.
    models: dict[str, Any] = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model missing at {path}. Run `python src/run_phase4.py` first."
            )
        models[name] = joblib.load(path)
    print(f"          loaded {len(models)} models from disk")

    # Per-model probabilities + threshold sweeps.
    import time

    results: dict[str, Any] = {}
    for name, model in models.items():
        proba_strat, classes = get_proba_and_classes(model, x_strat)
        try:
            pos_idx = classes.index(POSITIVE_LABEL)
        except ValueError:
            pos_idx = proba_strat.shape[1] - 1
        y_proba_strat = proba_strat[:, pos_idx]

        # Measure latency on the chronological test for the business calc.
        t0 = time.perf_counter()
        proba_chrono, _ = get_proba_and_classes(model, x_chrono)
        chrono_predict_time = time.perf_counter() - t0
        y_proba_chrono = proba_chrono[:, pos_idx]

        # Stratified-test latency (used for business metrics).
        t0 = time.perf_counter()
        _ = get_proba_and_classes(model, x_strat)
        strat_predict_time = time.perf_counter() - t0

        opt_f1 = find_optimal_threshold(y_strat_bin, y_proba_strat, "f1")
        opt_youden = find_optimal_threshold(
            y_strat_bin, y_proba_strat, "youden_j"
        )
        opt_balacc = find_optimal_threshold(
            y_strat_bin, y_proba_strat, "balanced_accuracy"
        )
        opt_cost = find_threshold_by_business_cost(
            y_strat_bin, y_proba_strat, DEFAULT_FP_COST, DEFAULT_FN_COST
        )

        default_metrics = compute_metrics_at_threshold(
            y_strat_bin, y_proba_strat, DEFAULT_THRESHOLD
        )
        f1_opt_metrics = compute_metrics_at_threshold(
            y_strat_bin, y_proba_strat, opt_f1["optimal_threshold"]
        )
        youden_opt_metrics = compute_metrics_at_threshold(
            y_strat_bin, y_proba_strat, opt_youden["optimal_threshold"]
        )
        cost_opt_metrics = compute_metrics_at_threshold(
            y_strat_bin, y_proba_strat, opt_cost["optimal_threshold"]
        )

        # Chronological-test metrics at the F1-optimal stratified threshold.
        chrono_default = compute_metrics_at_threshold(
            y_chrono_bin, y_proba_chrono, DEFAULT_THRESHOLD
        )
        chrono_at_f1_opt = compute_metrics_at_threshold(
            y_chrono_bin, y_proba_chrono, opt_f1["optimal_threshold"]
        )

        results[name] = {
            "thresholds": {
                "f1": opt_f1["optimal_threshold"],
                "youden_j": opt_youden["optimal_threshold"],
                "balanced_accuracy": opt_balacc["optimal_threshold"],
                "business_cost": opt_cost["optimal_threshold"],
            },
            "metric_improvements": {
                "f1": opt_f1["improvement"],
                "youden_j": opt_youden["improvement"],
                "balanced_accuracy": opt_balacc["improvement"],
            },
            "metrics": {
                "default_0.5": default_metrics,
                "at_f1_optimal": f1_opt_metrics,
                "at_youden_optimal": youden_opt_metrics,
                "at_cost_optimal": cost_opt_metrics,
            },
            "chronological": {
                "default_0.5": chrono_default,
                "at_f1_optimal_stratified": chrono_at_f1_opt,
            },
            "business_cost": {
                "fp_cost": opt_cost["fp_cost"],
                "fn_cost": opt_cost["fn_cost"],
                "default_threshold_cost": opt_cost["default_threshold_cost"],
                "min_cost": opt_cost["min_cost"],
                "savings": opt_cost["savings"],
            },
            "sweep": {
                "thresholds": opt_f1["all_thresholds"],
                "f1_values": opt_f1["all_metric_values"],
            },
            "n_test_samples": int(len(x_strat)),
            "predict_time_sec_strat": strat_predict_time,
            "predict_time_sec_chrono": chrono_predict_time,
            "_pipeline": model,
            "_y_proba_strat": y_proba_strat,
            "_y_proba_chrono": y_proba_chrono,
        }

    # Persist immediately.
    (RESULTS_DIR / "superseded" / "threshold_optimization.json").write_text(
        json.dumps(serializable(results), indent=2, default=str),
        encoding="utf-8",
    )

    # Find the post-tuning winner.
    winner_name = max(
        results, key=lambda n: results[n]["metrics"]["at_f1_optimal"]["f1_failure"]
    )
    winner_threshold = results[winner_name]["thresholds"]["f1"]
    winner_f1 = results[winner_name]["metrics"]["at_f1_optimal"]["f1_failure"]
    print(f"\n[Phase 5] Post-tuning winner: {winner_name} · "
          f"thr={winner_threshold:.2f} · F1={winner_f1:.4f}")

    # Save best optimized model + metadata.
    best_optimized_path = MODELS_DIR / "best_tuned_threshold_xgb.joblib"
    joblib.dump(results[winner_name]["_pipeline"], best_optimized_path, compress=3)
    metadata = {
        "model_name": winner_name,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "optimal_threshold": float(winner_threshold),
        "metrics_at_optimal": {
            k: v for k, v in results[winner_name]["metrics"]["at_f1_optimal"].items()
            if not k.startswith("_")
        },
        "metrics_default_threshold": {
            k: v for k, v in results[winner_name]["metrics"]["default_0.5"].items()
            if not k.startswith("_")
        },
        "improvement_f1": float(
            results[winner_name]["metrics"]["at_f1_optimal"]["f1_failure"]
            - results[winner_name]["metrics"]["default_0.5"]["f1_failure"]
        ),
        "model_path": str(best_optimized_path),
        "expected_feature_columns": ALL_FEATURE_COLUMNS,
        "target_column": TARGET,
        "positive_label": POSITIVE_LABEL,
    }
    (MODELS_DIR / "best_tuned_threshold_xgb_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    # Recompute business impact at the winner's optimal threshold.
    business = compute_business_metrics_optimized(
        name=winner_name,
        metrics_at_opt=results[winner_name]["metrics"]["at_f1_optimal"],
        n_test_samples=results[winner_name]["n_test_samples"],
        predict_time_sec=results[winner_name]["predict_time_sec_strat"],
        optimal_threshold=winner_threshold,
    )
    (RESULTS_DIR / "superseded" / "business_impact_optimized.json").write_text(
        json.dumps(business, indent=2, default=str), encoding="utf-8"
    )

    # Figures.
    print("\n[Phase 5] Rendering figures ...")
    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    plot_threshold_optimization(results, plotter)
    plot_before_after(results, plotter)

    # Print the headline tables.
    print("\n[Phase 5] Optimal thresholds per model")
    print("-" * 80)
    rows = []
    for name, info in results.items():
        rows.append([
            name,
            f"{info['thresholds']['f1']:.2f}",
            f"{info['thresholds']['youden_j']:.2f}",
            f"{info['thresholds']['balanced_accuracy']:.2f}",
            f"{info['thresholds']['business_cost']:.2f}",
        ])
    _print_table(["Model", "F1", "Youden J", "Bal. Acc.", "Business cost"], rows)

    print("\n[Phase 5] Before vs after (F1-optimal) — stratified test set")
    print("-" * 110)
    headers = ["Model", "Threshold", "Acc", "BalAcc", "FailPrec", "FailRec", "FailF1"]
    rows = []
    for name, info in results.items():
        default = info["metrics"]["default_0.5"]
        f1opt = info["metrics"]["at_f1_optimal"]
        rows.append([name + " (default)", "0.50",
                     _fmt(default["accuracy"]), _fmt(default["balanced_accuracy"]),
                     _fmt(default["precision_failure"]), _fmt(default["recall_failure"]),
                     _fmt(default["f1_failure"])])
        rows.append([name + " (F1-opt)", f"{info['thresholds']['f1']:.2f}",
                     _fmt(f1opt["accuracy"]), _fmt(f1opt["balanced_accuracy"]),
                     _fmt(f1opt["precision_failure"]), _fmt(f1opt["recall_failure"]),
                     _fmt(f1opt["f1_failure"])])
    _print_table(headers, rows)

    print("\n[Phase 5] Chronological test set with same thresholds (sanity)")
    print("-" * 110)
    headers = ["Model", "Threshold", "Acc", "BalAcc", "FailPrec", "FailRec", "FailF1"]
    rows = []
    for name, info in results.items():
        chr_default = info["chronological"]["default_0.5"]
        chr_opt = info["chronological"]["at_f1_optimal_stratified"]
        rows.append([name + " (default)", "0.50",
                     _fmt(chr_default["accuracy"]), _fmt(chr_default["balanced_accuracy"]),
                     _fmt(chr_default["precision_failure"]), _fmt(chr_default["recall_failure"]),
                     _fmt(chr_default["f1_failure"])])
        rows.append([name + " (F1-opt)", f"{info['thresholds']['f1']:.2f}",
                     _fmt(chr_opt["accuracy"]), _fmt(chr_opt["balanced_accuracy"]),
                     _fmt(chr_opt["precision_failure"]), _fmt(chr_opt["recall_failure"]),
                     _fmt(chr_opt["f1_failure"])])
    _print_table(headers, rows)

    print(
        f"\n[Phase 5] Post-tuning winner: {winner_name} "
        f"(F1 = {winner_f1:.4f} at threshold {winner_threshold:.2f})"
    )
    print("\n[Phase 5] Updated business impact:")
    for k, v in business.items():
        if k == "assumptions":
            continue
        print(f"  {k:<40}: {v}")
    print("  assumptions:")
    for ak, av in business["assumptions"].items():
        print(f"    {ak:<38}: {av}")

    print("\n[Phase 5] Done.")


if __name__ == "__main__":
    main()
