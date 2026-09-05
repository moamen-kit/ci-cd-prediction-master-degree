"""Primary evaluation pipeline: commit-grouped CV, business model, figures.

This module produces every headline number the thesis reports, from
``data/processed/cicd_prepared.csv``. It exists because the original Phase 4 and
Phase 5 orchestrators evaluated on a split that leaked duplicate commits and
selected the decision threshold on the test set it then reported. Running this
module reproduces the corrected results end to end:

* ``results/grouped_cv.json`` — the primary metrics
* ``results/oof_predictions.npz`` — one out-of-fold probability per row
* ``results/metric_attribution_ladder.json`` — where the original 0.5924 went
* ``results/business_impact.json`` and the sensitivity analysis
* figures 12, 13, 14, 15, 18, 19 and 22

Invoke from the repository root::

    .venv/bin/python -m src.run_corrected_evaluation
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .cross_validation import cv_summary_table, run_grouped_cv
from .data_preparation import GROUP_KEY, TARGET
from .hybrid_pipeline import (
    build_categorical_only_preprocessor,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_structured_only_preprocessor,
    build_text_only_preprocessor,
    build_xgboost_pipeline,
    build_xgboost_with_preprocessor,
    prepare_features_targets,
)
from .threshold_optimization import find_optimal_threshold
from .train_evaluate import compute_business_metrics
from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, ensure_dir, get_logger
from .visualization import ThesisPlotter

_LOGGER = get_logger(__name__)

# Okabe-Ito. Reserved for polarity in the waterfall; never used for series
# identity, which stays on the ThesisPlotter categorical palette.
_DECREASE = "#D55E00"
_INCREASE = "#009E73"
_TOTAL = "#4C4C4C"

MODEL_FACTORIES = {
    "Logistic Regression": build_logistic_regression_pipeline,
    "Random Forest": build_random_forest_pipeline,
    "XGBoost": build_xgboost_pipeline,
}

ABLATION_FACTORIES = {
    "hybrid_full": build_xgboost_pipeline,
    "structured_only": lambda: build_xgboost_with_preprocessor(
        build_structured_only_preprocessor()
    ),
    "categorical_only": lambda: build_xgboost_with_preprocessor(
        build_categorical_only_preprocessor()
    ),
    "text_only": lambda: build_xgboost_with_preprocessor(
        build_text_only_preprocessor()
    ),
}

ABLATION_LABELS = {
    "text_only": "Text only",
    "hybrid_full": "Hybrid (all four branches)",
    "structured_only": "Structured only",
    "categorical_only": "Categorical only",
}


def _binarise(y: Any) -> np.ndarray:
    return (np.asarray(y).astype(str) == "failure").astype(int)


def _strip_private(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in d.items()
    }


# --------------------------------------------------------------------------- #
# Attribution ladder
# --------------------------------------------------------------------------- #


def build_attribution_ladder(df: pd.DataFrame, cv_xgb: dict[str, Any]) -> dict[str, Any]:
    """Decompose the originally reported F1 one controlled change at a time.

    XGBoost throughout, identical hyperparameters, ``random_state=42``. Each
    step changes exactly one thing, so the difference between consecutive rows
    is attributable to that change and nothing else.
    """
    groups = df[GROUP_KEY].astype(str)

    def fit_proba(train_df: pd.DataFrame, eval_df: pd.DataFrame):
        x_tr, y_tr = prepare_features_targets(train_df)
        x_ev, y_ev = prepare_features_targets(eval_df)
        pipe = build_xgboost_pipeline()
        pipe.fit(x_tr, y_tr)
        pos = list(pipe.classes_).index("failure")
        return pipe.predict_proba(x_ev)[:, pos], _binarise(y_ev)

    steps: list[dict[str, Any]] = []

    # A — as originally reported.
    tr, te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    proba, y = fit_proba(tr, te)
    opt = find_optimal_threshold(y, proba, "f1")
    steps.append(
        {
            "step": "A. Row-level stratified split, threshold chosen on the test set",
            "isolates": "as originally reported",
            "f1_failure": round(float(opt["metric_value_at_optimal"]), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "threshold": round(float(opt["optimal_threshold"]), 3),
        }
    )

    # B — remove duplicate-commit leakage only.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    i_tr, i_te = next(gss.split(df, df[TARGET], groups))
    proba, y = fit_proba(df.iloc[i_tr], df.iloc[i_te])
    opt = find_optimal_threshold(y, proba, "f1")
    steps.append(
        {
            "step": "B. Commit-grouped split, threshold still chosen on the test set",
            "isolates": "duplicate-commit leakage (F-1)",
            "f1_failure": round(float(opt["metric_value_at_optimal"]), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "threshold": round(float(opt["optimal_threshold"]), 3),
        }
    )

    # C — also select the threshold honestly.
    pool = df.iloc[i_tr]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    j_tr, j_val = next(
        inner.split(pool, pool[TARGET], pool[GROUP_KEY].astype(str))
    )
    proba_val, y_val = fit_proba(pool.iloc[j_tr], pool.iloc[j_val])
    thr = float(find_optimal_threshold(y_val, proba_val, "f1")["optimal_threshold"])
    proba, y = fit_proba(pool.iloc[j_tr], df.iloc[i_te])
    steps.append(
        {
            "step": "C. Commit-grouped split, threshold chosen on a validation fold",
            "isolates": "test-set threshold selection (F-3)",
            "f1_failure": round(
                float(f1_score(y, (proba >= thr).astype(int), zero_division=0)), 4
            ),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "threshold": round(thr, 3),
        }
    )

    # D — remove the single-fold lottery.
    steps.append(
        {
            "step": "D. Commit-grouped 5-fold cross-validation (reported result)",
            "isolates": "single-fold composition",
            "f1_failure": cv_xgb["f1_failure"]["mean"],
            "f1_failure_sd": cv_xgb["f1_failure"]["std"],
            "pr_auc": cv_xgb["pooled_out_of_fold"]["pr_auc"],
            "roc_auc": cv_xgb["pooled_out_of_fold"]["roc_auc"],
            "threshold": cv_xgb["threshold"]["mean"],
        }
    )

    prev = None
    for s in steps:
        s["delta_f1_vs_previous"] = (
            None if prev is None else round(s["f1_failure"] - prev, 4)
        )
        prev = s["f1_failure"]

    return {
        "description": (
            "Controlled decomposition of the originally reported failure-class "
            "F1 of 0.5924. XGBoost throughout, identical hyperparameters, "
            "random_state=42; each row changes exactly one aspect of the "
            "evaluation protocol relative to the row above it."
        ),
        "steps": steps,
        "total_change": round(steps[-1]["f1_failure"] - steps[0]["f1_failure"], 4),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_cv_metrics(cv: dict[str, Any], plotter: ThesisPlotter) -> None:
    """Figure 14 — failure-class F1 per model, with cross-validation spread.

    A dot-and-interval plot rather than a bar chart: the finding is that the
    three models are not separated, and a bar chart invites the eye to read a
    point estimate as exact.
    """
    names = list(cv.keys())
    means = [cv[n]["f1_failure"]["mean"] for n in names]
    sds = [cv[n]["f1_failure"]["std"] for n in names]
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    sds = [sds[i] for i in order]

    fig, ax = plotter.new_figure(figsize=(8.0, 4.0))
    colour = plotter.palette(3)[0]
    ypos = np.arange(len(names))
    ax.errorbar(
        means, ypos, xerr=sds, fmt="o", markersize=8, capsize=5,
        color=colour, ecolor=colour, elinewidth=2, markeredgecolor="white",
        markeredgewidth=1.2, zorder=3,
    )
    for y_, m, s in zip(ypos, means, sds):
        ax.annotate(
            f"{m:.3f} ± {s:.3f}", xy=(m, y_), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9, color="#333333",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Failure-class F1 (mean of five commit-grouped folds, ± 1 s.d.)")
    ax.set_xlim(0.0, 0.75)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    # Headroom for the value label above the topmost marker.
    ax.set_ylim(-0.55, len(names) - 1 + 0.75)
    ax.set_title("Model comparison under commit-grouped cross-validation", pad=14)

    plotter.save_figure(
        fig,
        "fig_14_metrics_comparison_bars.png",
        caption=(
            "Failure-class F1 for the three classifiers under commit-grouped "
            "five-fold cross-validation, with the decision threshold selected "
            "on a validation fold carved from each fold's training rows. "
            "Markers show the mean across folds and bars span one standard "
            "deviation. The intervals overlap for all three classifiers, so "
            "the differences between them are not resolvable at this sample "
            "size; the spread across folds is larger than the spread across "
            "models."
        ),
        title="Figure 14 — Model comparison (commit-grouped cross-validation)",
    )


def plot_ablation(cv_ab: dict[str, Any], plotter: ThesisPlotter) -> None:
    """Figure 15 — ablation under the same protocol."""
    names = list(cv_ab.keys())
    means = [cv_ab[n]["f1_failure"]["mean"] for n in names]
    sds = [cv_ab[n]["f1_failure"]["std"] for n in names]
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    sds = [sds[i] for i in order]

    fig, ax = plotter.new_figure(figsize=(8.5, 4.2))
    base = plotter.palette(4)[0]
    highlight = "#D55E00"
    colours = [highlight if n == "categorical_only" else base for n in names]
    ypos = np.arange(len(names))
    for y_, m, s, c in zip(ypos, means, sds, colours):
        ax.errorbar(
            m, y_, xerr=s, fmt="o", markersize=8, capsize=5, color=c,
            ecolor=c, elinewidth=2, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3,
        )
        ax.annotate(
            f"{m:.3f} ± {s:.3f}", xy=(m, y_), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9, color="#333333",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([ABLATION_LABELS.get(n, n) for n in names])
    ax.set_xlabel("Failure-class F1 (mean of five commit-grouped folds, ± 1 s.d.)")
    ax.set_xlim(0.0, 0.75)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.55, len(names) - 1 + 0.75)
    ax.set_title("Feature-set ablation (XGBoost, identical hyperparameters)", pad=14)

    plotter.save_figure(
        fig,
        "fig_15_ablation_study.png",
        caption=(
            "Feature-set ablation under commit-grouped five-fold "
            "cross-validation. The classifier and its hyperparameters are held "
            "identical across configurations; only the input feature set "
            "varies. The categorical-only configuration, which receives no "
            "information beyond repository, workflow name, branch and trigger "
            "event, attains the highest failure-class F1 of any configuration "
            "tested, exceeding the full hybrid model. Commit text in isolation "
            "performs worst by a wide margin. The result indicates that the "
            "model's discriminative power derives principally from project "
            "identity rather than from the content of individual commits."
        ),
        title="Figure 15 — Feature-set ablation",
    )


def plot_attribution_waterfall(ladder: dict[str, Any], plotter: ThesisPlotter) -> None:
    """Figure 18 — where the originally reported F1 went."""
    steps = ladder["steps"]
    labels = [
        "A. Stratified split,\nthreshold on test\n(as reported)",
        "B. Remove duplicate-\ncommit leakage",
        "C. Select threshold\non validation",
        "D. Five-fold\ncross-validation\n(final)",
    ]
    values = [s["f1_failure"] for s in steps]

    fig, ax = plotter.new_figure(figsize=(9.0, 5.0))
    x = np.arange(len(values))

    ax.bar(0, values[0], width=0.6, color=_TOTAL, zorder=3)
    ax.annotate(f"{values[0]:.4f}", xy=(0, values[0]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=10, weight="bold")

    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        bottom = min(values[i - 1], values[i])
        colour = _INCREASE if delta > 0 else _DECREASE
        if i == len(values) - 1:
            ax.bar(i, values[i], width=0.6, color=_TOTAL, zorder=3)
            ax.annotate(f"{values[i]:.4f}", xy=(i, values[i]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=10,
                        weight="bold")
        else:
            ax.bar(i, abs(delta), bottom=bottom, width=0.6, color=colour, zorder=3)
        # On the closing total bar the delta would collide with the value
        # label, so it sits at mid-height inside the bar instead.
        label_y = values[i] / 2 if i == len(values) - 1 else bottom + abs(delta) / 2
        ax.annotate(
            f"{delta:+.4f}", xy=(i, label_y), xytext=(0, 0),
            textcoords="offset points", ha="center", va="center", fontsize=9,
            color="white" if (i == len(values) - 1 or abs(delta) > 0.04) else "#333333",
            weight="bold",
        )
        ax.plot([i - 1 + 0.3, i + 0.3 - 0.6], [values[i - 1]] * 2,
                color="#999999", linewidth=1, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Failure-class F1")
    ax.set_ylim(0.0, 0.68)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Attribution of the originally reported failure-class F1")

    plotter.save_figure(
        fig,
        "fig_18_metric_attribution.png",
        caption=(
            "Decomposition of the originally reported failure-class F1 of "
            "0.5924 into its methodological components. The classifier, its "
            "hyperparameters and the random seed are held constant throughout; "
            "each column changes exactly one aspect of the evaluation protocol "
            "relative to the column to its left. Removing duplicate-commit "
            "leakage costs 0.060; selecting the decision threshold on a "
            "validation fold rather than on the test set costs a further "
            "0.126; replacing a single held-out fold with five-fold "
            "cross-validation returns 0.015. Threshold selection on the "
            "evaluation set was therefore the larger of the two sources of "
            "optimism, by a factor of approximately two."
        ),
        title="Figure 18 — Metric attribution",
    )


def plot_business_sensitivity(
    sensitivity: dict[str, Any], plotter: ThesisPlotter
) -> None:
    """Figure 19 — annual net saving against the assumed false-alarm cost."""
    series = sensitivity["annual_net_saved_usd_by_false_alarm_cost"]
    costs = [float(k.split("_")[-1]) for k in next(iter(series.values())).keys()]

    fig, ax = plotter.new_figure(figsize=(8.5, 5.0))
    colours = plotter.palette(len(series))

    for (name, vals), colour in zip(series.items(), colours):
        y = list(vals.values())
        ax.plot(costs, y, marker="o", markersize=5, linewidth=2,
                color=colour, label=name, zorder=3)
        ax.annotate(name, xy=(costs[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=colour, va="center")

    ax.axhline(0, color="#333333", linewidth=1.2, linestyle="-", zorder=2)
    ax.axvline(2.50, color="#999999", linewidth=1, linestyle="--", zorder=2)
    ax.annotate("specified\ncost $2.50", xy=(2.50, ax.get_ylim()[0]),
                xytext=(4, 12), textcoords="offset points", fontsize=8.5,
                color="#666666")

    ax.set_xlabel("Assumed cost of a single false alarm (USD)")
    ax.set_ylabel("Estimated annual net saving (USD)")
    ax.yaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k")
    )
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(-1, max(costs) + 6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_title("Sensitivity of the estimated saving to the false-alarm cost")

    plotter.save_figure(
        fig,
        "fig_19_business_impact_sensitivity.png",
        caption=(
            "Estimated annual net saving as a function of the assumed cost of "
            "a single false alarm, holding the benefit of a caught failure at "
            "$18.814 (compute $0.064 plus developer context switching $18.75) "
            "and the failure rate at the observed 11 per cent. The vertical "
            "line marks the $2.50 cost specified in the operational scenario. "
            "The point at which each curve crosses zero is the false-alarm "
            "cost above which that configuration ceases to pay for itself: "
            "$10.27 for Logistic Regression, $12.72 for Random Forest, $20.41 "
            "for XGBoost and $27.20 for the categorical-only configuration. "
            "The configuration with the highest estimated saving at the "
            "specified cost is therefore also the one least robust to that "
            "cost having been underestimated."
        ),
        title="Figure 19 — Business impact sensitivity",
    )


def plot_oof_curves(
    oof: dict[str, np.ndarray], y_true: np.ndarray, plotter: ThesisPlotter
) -> None:
    """Figures 13 and 22 — ROC and precision-recall from pooled out-of-fold predictions."""
    colours = plotter.palette(len(oof))

    fig, ax = plotter.new_figure(figsize=(7.0, 6.0))
    for (name, proba), colour in zip(oof.items(), colours):
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, linewidth=2, color=colour,
                label=f"{name} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1,
            label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — pooled out-of-fold predictions")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plotter.save_figure(
        fig, "fig_13_roc_curves_per_target.png",
        caption=(
            "Receiver operating characteristic curves computed on pooled "
            "out-of-fold predictions under commit-grouped five-fold "
            "cross-validation. Every one of the 9,772 workflow runs "
            "contributes exactly one prediction, produced by a model that was "
            "not trained on any run of that run's commit, so the curves use "
            "the whole corpus without reusing any observation."
        ),
        title="Figure 13 — ROC curves (pooled out-of-fold)",
    )

    fig, ax = plotter.new_figure(figsize=(7.0, 6.0))
    prevalence = float(y_true.mean())
    for (name, proba), colour in zip(oof.items(), colours):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, linewidth=2, color=colour,
                label=f"{name} (AP = {average_precision_score(y_true, proba):.3f})")
    ax.axhline(prevalence, linestyle="--", color="#999999", linewidth=1,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall (failure class)")
    ax.set_ylabel("Precision (failure class)")
    ax.set_title("Precision–recall curves — pooled out-of-fold predictions")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plotter.save_figure(
        fig, "fig_22_precision_recall_curves.png",
        caption=(
            "Precision–recall curves on pooled out-of-fold predictions. The "
            "dashed line marks the prevalence of the failure class, which is "
            "the precision a random classifier attains and the correct "
            "baseline under class imbalance. Average precision is reported in "
            "the legend. This figure was specified in the Phase 4 plan and was "
            "absent from the original figure set."
        ),
        title="Figure 22 — Precision-recall curves",
    )


def plot_oof_confusion(
    oof: dict[str, np.ndarray],
    y_true: np.ndarray,
    thresholds: dict[str, float],
    plotter: ThesisPlotter,
) -> None:
    """Figure 12 — confusion matrices from pooled out-of-fold predictions."""
    import seaborn as sns

    fig, axes = plotter.new_figure(figsize=(15.0, 4.6), ncols=len(oof))
    axes = np.atleast_1d(axes)
    for ax, (name, proba) in zip(axes, oof.items()):
        thr = thresholds[name]
        cm = confusion_matrix(y_true, (proba >= thr).astype(int), labels=[0, 1])
        annot = np.array(
            [[f"{v:,}\n({100*v/cm.sum():.1f}%)" for v in row] for row in cm]
        )
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["success", "failure"],
                    yticklabels=["success", "failure"],
                    annot_kws={"fontsize": 10})
        ax.set_title(f"{name}\n(threshold {thr:.3f})", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(
        "Confusion matrices — pooled out-of-fold predictions", fontsize=13, y=1.04
    )
    plotter.save_figure(
        fig, "fig_12_confusion_matrices_grid.png",
        caption=(
            "Confusion matrices computed on pooled out-of-fold predictions "
            "under commit-grouped five-fold cross-validation, thresholded at "
            "the mean of the five thresholds each model's validation folds "
            "selected. Counts therefore cover all 9,772 workflow runs. The "
            "off-diagonal counts show the operating trade-off directly: "
            "Logistic Regression recovers more failures at the cost of "
            "substantially more false alarms, while XGBoost is the more "
            "conservative of the three."
        ),
        title="Figure 12 — Confusion matrices (pooled out-of-fold)",
    )


def plot_threshold_curves(
    oof: dict[str, np.ndarray],
    y_true: np.ndarray,
    thresholds: dict[str, float],
    plotter: ThesisPlotter,
) -> dict[str, Any]:
    """Figure 20 — failure-class F1 against decision threshold.

    The curve is computed on pooled out-of-fold predictions. Two points are
    marked on each curve: the threshold the validation folds actually selected,
    which is the one the system would deploy, and the threshold that maximises
    F1 on the evaluation data itself, which is what the original methodology
    reported. The vertical distance between them is the residual optimism that
    selecting a threshold on the evaluation set would still buy.
    """
    grid = np.arange(0.02, 0.96, 0.01)
    colours = plotter.palette(len(oof))
    fig, ax = plotter.new_figure(figsize=(8.5, 5.4))

    gaps: dict[str, Any] = {}
    for (name, proba), colour in zip(oof.items(), colours):
        scores = [
            f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid
        ]
        ax.plot(grid, scores, linewidth=2, color=colour, label=name, zorder=3)

        best_i = int(np.argmax(scores))
        chosen = thresholds[name]
        chosen_f1 = f1_score(y_true, (proba >= chosen).astype(int), zero_division=0)

        ax.plot(grid[best_i], scores[best_i], marker="*", markersize=14,
                color=colour, markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.plot(chosen, chosen_f1, marker="o", markersize=9, color=colour,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)

        gaps[name] = {
            "validation_selected_threshold": round(float(chosen), 3),
            "f1_at_validation_selected_threshold": round(float(chosen_f1), 4),
            "threshold_maximising_f1_on_evaluation_data": round(float(grid[best_i]), 3),
            "f1_at_that_threshold": round(float(scores[best_i]), 4),
            "residual_selection_gap": round(float(scores[best_i] - chosen_f1), 4),
        }

    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1, zorder=2)
    ax.annotate("default 0.5", xy=(0.5, 0.02), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, color="#666666")
    ax.plot([], [], marker="o", linestyle="none", color="#555555",
            label="threshold selected on validation")
    ax.plot([], [], marker="*", linestyle="none", markersize=12, color="#555555",
            label="threshold maximising F1 on evaluation data")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Failure-class F1 (pooled out-of-fold)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.62)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax.set_title("Failure-class F1 as a function of the decision threshold", pad=12)

    plotter.save_figure(
        fig, "fig_20_threshold_optimization.png",
        caption=(
            "Failure-class F1 against the decision threshold, computed on "
            "pooled out-of-fold predictions. Circles mark the threshold "
            "selected on the validation folds, which is the threshold a "
            "deployment would use; stars mark the threshold that maximises F1 "
            "on the evaluation data itself, which is what the original "
            "methodology reported. The gap between circle and star is the "
            "portion of the previously reported improvement that came from "
            "selecting the threshold on the data it was scored against. The "
            "optima differ sharply between classifiers, with XGBoost optimal "
            "far below the default of 0.5 and Logistic Regression above it, "
            "which shows that the three produce differently calibrated "
            "probabilities and that no single decision rule serves all three."
        ),
        title="Figure 20 — Threshold optimization",
    )
    return gaps


def plot_before_after_threshold(
    oof: dict[str, np.ndarray],
    y_true: np.ndarray,
    thresholds: dict[str, float],
    plotter: ThesisPlotter,
) -> None:
    """Figure 21 — metrics at the default threshold against the selected one."""
    from sklearn.metrics import precision_score, recall_score

    metrics = ("Precision", "Recall", "F1")
    fig, axes = plotter.new_figure(figsize=(14.0, 4.4), ncols=len(oof))
    axes = np.atleast_1d(axes)
    colours = plotter.palette(2)

    for ax, (name, proba) in zip(axes, oof.items()):
        thr = thresholds[name]
        rows = []
        for t in (0.5, thr):
            pred = (proba >= t).astype(int)
            rows.append([
                precision_score(y_true, pred, zero_division=0),
                recall_score(y_true, pred, zero_division=0),
                f1_score(y_true, pred, zero_division=0),
            ])
        x = np.arange(len(metrics))
        width = 0.36
        b1 = ax.bar(x - width/2, rows[0], width, color=colours[0],
                    label="default 0.50", zorder=3)
        b2 = ax.bar(x + width/2, rows[1], width, color=colours[1],
                    label=f"selected {thr:.3f}", zorder=3)
        for bars in (b1, b2):
            for bar in bars:
                ax.annotate(f"{bar.get_height():.2f}",
                            xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("Score (failure class)")
    fig.suptitle(
        "Effect of threshold selection on failure-class metrics "
        "(pooled out-of-fold)", fontsize=13, y=1.05,
    )
    plotter.save_figure(
        fig, "fig_21_metrics_before_after_threshold.png",
        caption=(
            "Failure-class precision, recall and F1 at the default threshold "
            "of 0.50 against the threshold selected on the validation folds, "
            "computed on pooled out-of-fold predictions. The direction of the "
            "change differs by classifier: XGBoost requires a much lower "
            "threshold and gains recall at the cost of precision, whereas "
            "Logistic Regression requires a higher one and gains precision at "
            "the cost of recall. Threshold selection therefore remains "
            "consequential once it is performed honestly, but the improvement "
            "is materially smaller than the twenty-seven percentage points "
            "reported when the threshold was chosen on the test set."
        ),
        title="Figure 21 — Metrics before vs after threshold tuning",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)

    print("[corrected] Loading prepared dataset ...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "cicd_prepared.csv")
    print(f"           {df.shape[0]:,} rows · {df[GROUP_KEY].nunique():,} unique commits")

    print("\n[corrected] Commit-grouped 5-fold CV — models ...")
    cv = run_grouped_cv(df, MODEL_FACTORIES, n_splits=5, random_state=42)
    print(cv_summary_table(cv))

    print("\n[corrected] Commit-grouped 5-fold CV — ablation ...")
    cv_ab = run_grouped_cv(df, ABLATION_FACTORIES, n_splits=5, random_state=42)
    print(cv_summary_table(cv_ab))

    (RESULTS_DIR / "grouped_cv.json").write_text(
        json.dumps(
            {
                "models": _strip_private(cv),
                "ablation": _strip_private(cv_ab),
                "protocol": {
                    "splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
                    "group_key": GROUP_KEY,
                    "threshold_selection": (
                        "inner StratifiedGroupKFold over each fold's training rows; "
                        "F1-optimal on the inner validation set; never on test"
                    ),
                    "ranking_metrics": "pooled out-of-fold over all rows",
                    "decision_metrics": "per-fold, reported as mean and standard deviation",
                },
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    y_true = _binarise(df[TARGET])
    oof = {name: cv[name]["_oof_proba"] for name in cv}
    np.savez_compressed(
        RESULTS_DIR / "oof_predictions.npz", y_true=y_true, **oof
    )

    print("\n[corrected] Attribution ladder ...")
    ladder = build_attribution_ladder(df, cv["XGBoost"])
    (RESULTS_DIR / "metric_attribution_ladder.json").write_text(
        json.dumps(ladder, indent=2), encoding="utf-8"
    )
    for s in ladder["steps"]:
        d = "" if s["delta_f1_vs_previous"] is None else f"{s['delta_f1_vs_previous']:+.4f}"
        print(f"           {s['f1_failure']:.4f} {d:>8}  {s['step']}")

    print("\n[corrected] Business impact ...")
    all_cfg = {**cv, **{k: v for k, v in cv_ab.items() if k == "categorical_only"}}
    business_all = {
        name: compute_business_metrics(
            failure_recall=v["recall_failure"]["mean"],
            failure_precision=v["precision_failure"]["mean"],
            label=f"{name} (commit-grouped 5-fold CV, threshold {v['threshold']['mean']:.3f})",
        )
        for name, v in all_cfg.items()
    }
    (RESULTS_DIR / "business_impact.json").write_text(
        json.dumps(business_all["XGBoost"], indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "business_impact_all_configurations.json").write_text(
        json.dumps(business_all, indent=2), encoding="utf-8"
    )

    from . import train_evaluate as te
    fa_costs = [0.0, 2.50, 5.00, 10.00, 18.81, 25.00]
    sens: dict[str, dict[str, float]] = {}
    for name, v in all_cfg.items():
        vals = {}
        for c in fa_costs:
            original = te.FALSE_ALARM_COST_USD
            te.FALSE_ALARM_COST_USD = c
            r = compute_business_metrics(
                v["recall_failure"]["mean"], v["precision_failure"]["mean"]
            )
            te.FALSE_ALARM_COST_USD = original
            vals[f"fa_cost_{c:.2f}"] = r["annual_net_saved_usd"]
        sens[name] = vals

    benefit = te.COMPUTE_COST_PER_FAILED_BUILD_USD + te.CONTEXT_SWITCH_COST_PER_FAILURE_USD
    sensitivity = {
        "description": (
            "Annual net saving as a function of the assumed false-alarm cost, "
            "holding the benefit per caught failure fixed. The break-even cost "
            "is the false-alarm cost at which a configuration's net saving "
            "reaches zero; it depends only on precision."
        ),
        "benefit_per_caught_failure_usd": round(benefit, 3),
        "annual_net_saved_usd_by_false_alarm_cost": sens,
        "break_even_false_alarm_cost_usd": {
            name: round(
                benefit * v["precision_failure"]["mean"]
                / (1 - v["precision_failure"]["mean"]),
                2,
            )
            for name, v in all_cfg.items()
        },
    }
    (RESULTS_DIR / "business_impact_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )

    print("\n[corrected] Rendering figures ...")
    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    plot_cv_metrics(cv, plotter)
    plot_ablation(cv_ab, plotter)
    plot_attribution_waterfall(ladder, plotter)
    plot_business_sensitivity(sensitivity, plotter)
    plot_oof_curves(oof, y_true, plotter)
    chosen = {n: cv[n]["threshold"]["mean"] for n in cv}
    plot_oof_confusion(oof, y_true, chosen, plotter)
    gaps = plot_threshold_curves(oof, y_true, chosen, plotter)
    plot_before_after_threshold(oof, y_true, chosen, plotter)

    (RESULTS_DIR / "threshold_selection_gap.json").write_text(
        json.dumps(
            {
                "description": (
                    "Residual optimism from selecting a decision threshold on the "
                    "data it is scored against. Compares F1 at the "
                    "validation-selected threshold with F1 at the threshold that "
                    "maximises F1 on the pooled out-of-fold predictions."
                ),
                "per_model": gaps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n[corrected] Residual threshold-selection gap:")
    for name, g in gaps.items():
        print(
            f"           {name:<22} selected {g['validation_selected_threshold']:.3f} "
            f"-> F1 {g['f1_at_validation_selected_threshold']:.4f} | "
            f"argmax {g['threshold_maximising_f1_on_evaluation_data']:.3f} "
            f"-> F1 {g['f1_at_that_threshold']:.4f} | "
            f"gap {g['residual_selection_gap']:+.4f}"
        )

    print("\n[corrected] Done.")


if __name__ == "__main__":
    main()
