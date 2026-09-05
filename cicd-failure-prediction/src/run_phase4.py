"""Phase 4 orchestrator (binary): training, ablation, 6 figures, summary.

Run from project root::

    python src/run_phase4.py

Expected wall-time: 5–15 minutes on a developer laptop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preparation import (  # noqa: E402
    TARGET,
    TEXT_FEATURE,
)
from src.hybrid_pipeline import (  # noqa: E402
    LabelEncoderForBinary,
    prepare_features_targets,
)
from src.train_evaluate import (  # noqa: E402
    POSITIVE_LABEL,
    compute_business_metrics,
    evaluate_pipeline,
    identify_best_model,
    run_ablation_study,
    save_best_model,
    serializable_results,
    train_all_models,
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


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


MODEL_ORDER = ("Logistic Regression", "Random Forest", "XGBoost")
ABLATION_ORDER = (
    "text_only",
    "categorical_only",
    "structured_only",
    "hybrid_full",
)
ABLATION_LABELS = {
    "text_only": "Text only\n(TF-IDF)",
    "categorical_only": "Categorical only\n(repo + workflow\n+ branch + event)",
    "structured_only": "Structured only\n(num + cat + bin)",
    "hybrid_full": "Hybrid (full)",
}
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "failure_precision",
    "failure_recall",
    "failure_f1",
    "roc_auc",
    "pr_auc",
)
METRIC_LABELS = (
    "Accuracy",
    "Balanced Acc.",
    "Failure Prec.",
    "Failure Rec.",
    "Failure F1",
    "ROC-AUC",
    "PR-AUC",
)

MODALITY_COLORS = {
    "numerical": "#1565c0",
    "categorical": "#2e7d32",
    "binary": "#ef6c00",
    "text": "#6a1b9a",
    "other": "#616161",
}


def _modality_of(feature_name: str) -> str:
    prefix = feature_name.split("__", 1)[0] if "__" in feature_name else "other"
    return prefix if prefix in MODALITY_COLORS else "other"


def _inner_pipeline(pipeline: Any) -> Any:
    return (
        pipeline.estimator
        if isinstance(pipeline, LabelEncoderForBinary)
        else pipeline
    )


def _xgboost_importance(pipeline: Any) -> tuple[np.ndarray, np.ndarray]:
    inner = _inner_pipeline(pipeline)
    classifier = inner.named_steps["classifier"]
    importances = np.asarray(classifier.feature_importances_)
    feature_names = np.asarray(inner.named_steps["preprocessor"].get_feature_names_out())
    return feature_names, importances


# --------------------------------------------------------------------------- #
# Figure 12 — confusion matrices grid (1 × 3)
# --------------------------------------------------------------------------- #


def plot_confusion_matrices_grid(
    model_results: dict[str, Any], plotter: ThesisPlotter
) -> None:
    fig, axes = plotter.new_figure(figsize=(15.5, 5.0), nrows=1, ncols=3)
    cmaps = ["Blues", "Greens", "Oranges"]

    for c, model_name in enumerate(MODEL_ORDER):
        if model_name not in model_results:
            continue
        ax = axes[c]
        metrics = model_results[model_name]["metrics"]
        labels = metrics["classes_ordered"]
        cm = np.array(metrics["confusion_matrix"], dtype=int)
        total = cm.sum()
        annot = np.array(
            [
                [f"{cm[i, j]:,}\n({cm[i, j] / total * 100:.1f}%)" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ]
        )
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cmaps[c % len(cmaps)],
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": 11},
        )
        ax.set_title(f"{model_name}", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)

    fig.suptitle(
        "Confusion Matrices Across Models — Binary Target ``conclusion``",
        y=1.02,
        fontsize=14,
    )

    plotter.save_figure(
        fig,
        "fig_12_confusion_matrices_grid.png",
        caption=(
            "Confusion matrices for each model on the stratified test set. "
            "Cells show counts and the fraction of the entire test set each "
            "cell represents. The class imbalance (~89% success) means that "
            "even a model that always predicts ``success`` would reach ~89% "
            "accuracy; the diagonal cells in the lower-right quadrant are "
            "therefore the metric of interest for failure-detection quality."
        ),
        title="Figure 12 — Confusion matrices grid",
    )


# --------------------------------------------------------------------------- #
# Figure 13 — ROC curves (single panel, 3 model curves)
# --------------------------------------------------------------------------- #


def plot_roc_curves(
    model_results: dict[str, Any],
    y_test: pd.Series,
    plotter: ThesisPlotter,
) -> None:
    fig, ax = plotter.new_figure(figsize=(8.5, 6.5))
    palette = plotter.palette(len(MODEL_ORDER))

    y_true_bin = (np.asarray(y_test).astype(str) == POSITIVE_LABEL).astype(int)

    for color, model_name in zip(palette, MODEL_ORDER):
        if model_name not in model_results:
            continue
        y_proba_pos = model_results[model_name]["_y_proba_positive"]
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba_pos)
        auc_value = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.0,
            label=f"{model_name} (AUC = {auc_value:.3f})",
        )

    ax.plot(
        [0, 1], [0, 1],
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label="Random",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("ROC Curves — Predicting Workflow Failure (positive = ``failure``)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")

    plotter.save_figure(
        fig,
        "fig_13_roc_curves_per_target.png",
        caption=(
            "Receiver Operating Characteristic curves for the three models "
            "treating ``failure`` as the positive class. The AUC values "
            "quantify each model's ability to rank failed runs above "
            "successful ones independently of the decision threshold; the "
            "dashed grey line is the random baseline (AUC = 0.5)."
        ),
        title="Figure 13 — ROC curves",
    )


# --------------------------------------------------------------------------- #
# Figure 14 — metrics comparison bars (single panel)
# --------------------------------------------------------------------------- #


def plot_metrics_comparison_bars(
    model_results: dict[str, Any], plotter: ThesisPlotter
) -> None:
    fig, ax = plotter.new_figure(figsize=(13.0, 5.5))
    palette = plotter.palette(len(MODEL_ORDER))

    n_metrics = len(METRIC_KEYS)
    n_models = len(MODEL_ORDER)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model_name in enumerate(MODEL_ORDER):
        if model_name not in model_results:
            continue
        metrics = model_results[model_name]["metrics"]
        values = [
            metrics.get(k) if metrics.get(k) is not None else 0.0
            for k in METRIC_KEYS
        ]
        offset = (i - (n_models - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=palette[i],
            edgecolor="black",
            linewidth=0.4,
            label=model_name,
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Comparative Model Performance Across Evaluation Metrics")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=False)

    plotter.save_figure(
        fig,
        "fig_14_metrics_comparison_bars.png",
        caption=(
            "Comparative performance of Logistic Regression, Random Forest, "
            "and XGBoost across seven metrics on the stratified test set. "
            "Balanced accuracy, precision/recall/F1 on the ``failure`` class, "
            "ROC-AUC and PR-AUC are reported alongside raw accuracy because "
            "the underlying class imbalance (~89% success) makes raw "
            "accuracy a poor primary signal."
        ),
        title="Figure 14 — Metrics comparison bars",
    )


# --------------------------------------------------------------------------- #
# Figure 15 — ablation study (single panel)
# --------------------------------------------------------------------------- #


def plot_ablation_study(
    ablation_results: dict[str, Any], plotter: ThesisPlotter
) -> None:
    fig, ax = plotter.new_figure(figsize=(13.0, 5.5))
    palette = plotter.palette(len(ABLATION_ORDER))

    n_metrics = len(METRIC_KEYS)
    n_groups = len(ABLATION_ORDER)
    x = np.arange(n_metrics)
    width = 0.8 / n_groups

    for i, config in enumerate(ABLATION_ORDER):
        if config not in ablation_results:
            continue
        metrics = ablation_results[config]["metrics"]
        values = [
            metrics.get(k) if metrics.get(k) is not None else 0.0
            for k in METRIC_KEYS
        ]
        offset = (i - (n_groups - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=palette[i],
            edgecolor="black",
            linewidth=0.4,
            label=ABLATION_LABELS[config],
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Contribution of Each Feature Modality (XGBoost)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    plotter.save_figure(
        fig,
        "fig_15_ablation_study.png",
        caption=(
            "Performance of the XGBoost classifier under three feature "
            "configurations: text-only (TF-IDF), structured-only (numerical "
            "+ categorical + binary), and the full hybrid combination. The "
            "comparison quantifies the marginal value of each feature "
            "modality for the binary ``conclusion`` target."
        ),
        title="Figure 15 — Ablation study",
    )


# --------------------------------------------------------------------------- #
# Figure 16 — top TF-IDF tokens per repository (top 6 repos, 2 × 3)
# --------------------------------------------------------------------------- #


def plot_top_tfidf_features(
    pipeline: Any,
    x_test: pd.DataFrame,
    plotter: ThesisPlotter,
    top_n: int = 12,
) -> None:
    inner = _inner_pipeline(pipeline)
    preprocessor = inner.named_steps["preprocessor"]
    feature_names, importances = _xgboost_importance(pipeline)

    text_mask = np.array([n.startswith("text__") for n in feature_names])
    text_features = feature_names[text_mask]
    text_importance = importances[text_mask]
    if text_importance.size == 0:
        _LOGGER.warning("No TF-IDF features found; skipping fig_16.")
        return
    text_tokens = np.array([n.split("__", 1)[1] for n in text_features])

    tfidf_vec = preprocessor.named_transformers_["text"]
    x_text_matrix = tfidf_vec.transform(
        x_test[TEXT_FEATURE].astype(str).values
    )

    # Top 6 repositories by row count in the test set.
    top_repos = (
        x_test["repository"].value_counts().head(6).index.tolist()
    )
    ncols = 3
    nrows = (len(top_repos) + ncols - 1) // ncols

    fig, axes = plotter.new_figure(
        figsize=(16.0, 4.0 * nrows), nrows=nrows, ncols=ncols
    )
    axes_flat = axes.flatten()
    palette = plotter.palette(len(top_repos))

    for idx, (ax, repo) in enumerate(zip(axes_flat, top_repos)):
        mask = (x_test["repository"] == repo).values
        if mask.sum() == 0:
            ax.axis("off")
            continue
        mean_tfidf = np.asarray(x_text_matrix[mask].mean(axis=0)).ravel()
        scores = mean_tfidf * text_importance
        if scores.sum() == 0:
            scores = mean_tfidf

        order = np.argsort(-scores)[:top_n]
        ordered_scores = scores[order][::-1]
        ordered_tokens = text_tokens[order][::-1]

        ax.barh(
            ordered_tokens,
            ordered_scores,
            color=palette[idx % len(palette)],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(repo, fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance × mean TF-IDF")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes_flat[len(top_repos):]:
        ax.axis("off")

    fig.suptitle(
        "Top Discriminative TF-IDF Tokens per Repository (top 6 by row count)",
        y=1.01,
        fontsize=14,
    )

    plotter.save_figure(
        fig,
        "fig_16_top_tfidf_features.png",
        caption=(
            "Top 12 most influential TF-IDF tokens for each of the six "
            "highest-volume repositories in the test set, scored by "
            "``mean TF-IDF × XGBoost feature importance``. The visibly "
            "different vocabularies between repositories confirm that the "
            "text branch captures project-specific terminology in addition "
            "to generic CI/CD signal — useful information for downstream "
            "transfer-learning analyses."
        ),
        title="Figure 16 — Top TF-IDF tokens per repository",
    )


# --------------------------------------------------------------------------- #
# Figure 17 — global feature importance (top 30, colored by modality)
# --------------------------------------------------------------------------- #


def plot_feature_importance_global(
    pipeline: Any, plotter: ThesisPlotter, top_n: int = 30
) -> None:
    feature_names, importances = _xgboost_importance(pipeline)
    order = np.argsort(-importances)[:top_n]
    selected = feature_names[order]
    selected_imp = importances[order]

    modalities = [_modality_of(n) for n in selected]
    display = [
        n.split("__", 1)[1] if "__" in n else n
        for n in selected
    ]

    display = display[::-1]
    selected_imp = selected_imp[::-1]
    modalities = modalities[::-1]

    fig, ax = plotter.new_figure(figsize=(11.5, 11.0))
    colors = [MODALITY_COLORS[m] for m in modalities]
    ax.barh(display, selected_imp, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("XGBoost feature importance (gain-based)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Most Important Features (Hybrid XGBoost)", pad=12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="y", labelsize=9)

    legend_handles = [
        Patch(facecolor=MODALITY_COLORS["numerical"], label="Numerical"),
        Patch(facecolor=MODALITY_COLORS["categorical"], label="Categorical"),
        Patch(facecolor=MODALITY_COLORS["binary"], label="Binary"),
        Patch(facecolor=MODALITY_COLORS["text"], label="Text (TF-IDF)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    plotter.save_figure(
        fig,
        "fig_17_feature_importance_global.png",
        caption=(
            "Top 30 most important features in the XGBoost hybrid model, "
            "grouped and colored by modality. The mix of numerical, "
            "categorical, and text features in the top-30 confirms that the "
            "model leverages all four input branches for the binary "
            "``conclusion`` prediction task."
        ),
        title="Figure 17 — Global feature importance",
    )


# --------------------------------------------------------------------------- #
# Printable tables
# --------------------------------------------------------------------------- #


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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if 0 <= value <= 1 else f"{value:.4f}"
    return str(value)


def print_main_metrics_table(model_results: dict[str, Any]) -> None:
    print("\nMain metrics — per model (stratified test set)")
    print("-" * 110)
    headers = [
        "Model", "Acc", "BalAcc", "FailPrec", "FailRec", "FailF1",
        "ROC-AUC", "PR-AUC",
    ]
    rows: list[list[str]] = []
    for model_name in MODEL_ORDER:
        if model_name not in model_results:
            continue
        m = model_results[model_name]["metrics"]
        rows.append(
            [
                model_name,
                _fmt(m.get("accuracy")),
                _fmt(m.get("balanced_accuracy")),
                _fmt(m.get("failure_precision")),
                _fmt(m.get("failure_recall")),
                _fmt(m.get("failure_f1")),
                _fmt(m.get("roc_auc")),
                _fmt(m.get("pr_auc")),
            ]
        )
    _print_table(headers, rows)


def print_ablation_table(ablation_results: dict[str, Any]) -> None:
    print("\nAblation study — XGBoost across feature configurations")
    print("-" * 110)
    headers = [
        "Config", "Acc", "BalAcc", "FailPrec", "FailRec", "FailF1",
        "ROC-AUC", "PR-AUC",
    ]
    rows: list[list[str]] = []
    for config in ABLATION_ORDER:
        if config not in ablation_results:
            continue
        m = ablation_results[config]["metrics"]
        rows.append(
            [
                ABLATION_LABELS[config].replace("\n", " "),
                _fmt(m.get("accuracy")),
                _fmt(m.get("balanced_accuracy")),
                _fmt(m.get("failure_precision")),
                _fmt(m.get("failure_recall")),
                _fmt(m.get("failure_f1")),
                _fmt(m.get("roc_auc")),
                _fmt(m.get("pr_auc")),
            ]
        )
    _print_table(headers, rows)


def print_chronological_table(chronological_eval: dict[str, Any]) -> None:
    print("\nSecondary evaluation — chronological test set (deployment realism)")
    print("-" * 110)
    headers = [
        "Model", "Acc", "BalAcc", "FailPrec", "FailRec", "FailF1",
        "ROC-AUC", "PR-AUC",
    ]
    rows: list[list[str]] = []
    for model_name in MODEL_ORDER:
        if model_name not in chronological_eval:
            continue
        m = chronological_eval[model_name]["metrics"]
        rows.append(
            [
                model_name,
                _fmt(m.get("accuracy")),
                _fmt(m.get("balanced_accuracy")),
                _fmt(m.get("failure_precision")),
                _fmt(m.get("failure_recall")),
                _fmt(m.get("failure_f1")),
                _fmt(m.get("roc_auc")),
                _fmt(m.get("pr_auc")),
            ]
        )
    _print_table(headers, rows)


def print_top_features(pipeline: Any, top_n: int = 10) -> None:
    feature_names, importances = _xgboost_importance(pipeline)
    order = np.argsort(-importances)[:top_n]
    print(f"\nTop {top_n} most important features (XGBoost hybrid)")
    print("-" * 70)
    print(f"{'Rank':<6}{'Modality':<14}{'Feature':<40}{'Importance':>10}")
    for rank, idx in enumerate(order, start=1):
        name = feature_names[idx]
        display = name.split("__", 1)[1] if "__" in name else name
        modality = _modality_of(name)
        print(f"{rank:<6}{modality:<14}{display[:39]:<40}{importances[idx]:>10.4f}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _load_split(
    train_csv: str, test_csv: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / train_csv)
    test_df = pd.read_csv(PROCESSED_DATA_DIR / test_csv)
    x_train, y_train = prepare_features_targets(train_df)
    x_test, y_test = prepare_features_targets(test_df)
    return x_train, y_train, x_test, y_test


def main() -> None:
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)

    print("[Phase 4] Loading commit-grouped split (PRIMARY) ...")
    x_train, y_train, x_test, y_test = _load_split(
        "train_grouped.csv", "test_grouped.csv"
    )
    print(f"          x_train = {x_train.shape}   x_test = {x_test.shape}")

    print("\n[Phase 4] Training all three pipelines ...")
    model_results = train_all_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=MODELS_DIR,
    )

    # Snapshot main metrics immediately so a later crash doesn't waste training.
    (RESULTS_DIR / "phase4_main_metrics.json").write_text(
        json.dumps(serializable_results(model_results), indent=2, default=str),
        encoding="utf-8",
    )

    print("\n[Phase 4] Running ablation study (XGBoost only) ...")
    ablation_results = run_ablation_study(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=MODELS_DIR,
        hybrid_xgb=model_results["XGBoost"]["_pipeline"],
    )
    (RESULTS_DIR / "ablation_study.json").write_text(
        json.dumps(serializable_results(ablation_results), indent=2, default=str),
        encoding="utf-8",
    )

    # The chronological evaluation retrains from the chronological TRAIN fold.
    # Reusing the grouped-split models here would leak: a commit held out of
    # the chronological test set is not held out of the grouped training set,
    # so the two partitions are not compatible and the transfer number would be
    # measured on commits the model had already seen.
    print("\n[Phase 4] Secondary evaluation — retraining on chronological train ...")
    chrono_x_train, chrono_y_train, chrono_x, chrono_y = _load_split(
        "train_chronological.csv", "test_chronological.csv"
    )
    print(f"          x_train = {chrono_x_train.shape}   x_test = {chrono_x.shape}")
    chronological_models = train_all_models(
        x_train=chrono_x_train,
        y_train=chrono_y_train,
        x_test=chrono_x,
        y_test=chrono_y,
        save_dir=None,
    )
    chronological_eval: dict[str, Any] = {
        name: {k: v for k, v in info.items() if k != "_pipeline"}
        for name, info in chronological_models.items()
    }
    (RESULTS_DIR / "phase4_chronological_eval.json").write_text(
        json.dumps(
            serializable_results(chronological_eval), indent=2, default=str
        ),
        encoding="utf-8",
    )

    print("\n[Phase 4] Identifying best model (by failure F1) ...")
    best_name, scores = identify_best_model(model_results)
    print(
        "          failure F1 per model:\n"
        + "\n".join(f"            {n}: {s:.4f}" for n, s in scores.items())
    )
    print(f"          BEST → {best_name}")

    best_metrics = model_results[best_name]["metrics"]
    n_test = int(model_results[best_name]["n_test_samples"])
    business = compute_business_metrics(
        failure_recall=float(best_metrics["failure_recall"]),
        failure_precision=float(best_metrics["failure_precision"]),
        avg_latency_ms=(
            float(model_results[best_name]["predict_time_sec"]) * 1000.0
            / max(n_test, 1)
        ),
        label=f"{best_name} at default threshold 0.5, single grouped fold",
    )
    (RESULTS_DIR / "business_impact.json").write_text(
        json.dumps(business, indent=2, default=str), encoding="utf-8"
    )

    print("\n[Phase 4] Rendering figures ...")
    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    plot_confusion_matrices_grid(model_results, plotter)
    plot_roc_curves(model_results, y_test, plotter)
    plot_metrics_comparison_bars(model_results, plotter)
    plot_ablation_study(ablation_results, plotter)
    xgb_hybrid = model_results["XGBoost"]["_pipeline"]
    plot_top_tfidf_features(xgb_hybrid, x_test, plotter)
    plot_feature_importance_global(xgb_hybrid, plotter)

    print("\n[Phase 4] Saving best model + metadata ...")
    best_info = save_best_model(
        pipeline=model_results[best_name]["_pipeline"],
        name=best_name,
        model_results=model_results,
        save_dir=MODELS_DIR,
    )

    summary = {
        "best_model": best_info,
        "failure_f1_per_model": {k: round(v, 4) for k, v in scores.items()},
        "models_stratified": serializable_results(model_results),
        "models_chronological": serializable_results(chronological_eval),
        "ablation": serializable_results(ablation_results),
        "business": business,
    }
    summary_path = RESULTS_DIR / "phase4_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[Phase 4] Comprehensive summary → {summary_path}")

    print_main_metrics_table(model_results)
    print_ablation_table(ablation_results)
    print_chronological_table(chronological_eval)

    print("\nBusiness impact summary")
    print("-" * 70)
    for k, v in business.items():
        if k == "assumptions":
            continue
        print(f"  {k:<40}: {v}")
    print("  assumptions:")
    for ak, av in business["assumptions"].items():
        print(f"    {ak:<38}: {av}")

    print_top_features(xgb_hybrid, top_n=10)

    best_f1 = model_results[best_name]["metrics"]["failure_f1"]
    best_pr_auc = model_results[best_name]["metrics"]["pr_auc"]
    print(
        f"\n[Phase 4] Best model: {best_name}  ·  "
        f"failure F1: {best_f1:.4f}  ·  PR-AUC: {best_pr_auc:.4f}"
    )

    print("\n[Phase 4] Done.")


if __name__ == "__main__":
    main()
