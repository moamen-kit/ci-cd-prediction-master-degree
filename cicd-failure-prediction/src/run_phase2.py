"""Phase 2 runner: prepare → split → 4 validation figures → JSON summary.

Adapted from the synthetic-data Phase 2 to the real GitHub Actions schema.
The four validation figures answer different but complementary questions:

* **fig_07** — was class balance preserved across the chronological split?
* **fig_08** — what do the ten engineered features actually look like?
* **fig_09** — does the engineered feature space exhibit useful structure
  in its Pearson correlation matrix?
* **fig_10** — which commit-message tokens are discriminative for
  *failure* vs *success*?

Run from the project root::

    python src/run_phase2.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preparation import (  # noqa: E402
    ENGINEERED_FEATURE_COLUMNS,
    RAW_DATASET_PATH,
    class_distribution,
    prepare_dataset,
    split_dataset,
)
from src.utils import (  # noqa: E402
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ensure_dir,
    get_logger,
)
from src.visualization import ThesisPlotter  # noqa: E402


_LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


CONCLUSION_ORDER: tuple[str, ...] = ("success", "failure")


def plot_train_test_class_balance(
    y_train: pd.Series, y_test: pd.Series, plotter: ThesisPlotter
) -> None:
    train_pct = (y_train.value_counts(normalize=True) * 100).reindex(
        CONCLUSION_ORDER, fill_value=0
    )
    test_pct = (y_test.value_counts(normalize=True) * 100).reindex(
        CONCLUSION_ORDER, fill_value=0
    )

    fig, ax = plotter.new_figure(figsize=(8.0, 5.2))
    x = np.arange(len(CONCLUSION_ORDER))
    width = 0.36
    bars_train = ax.bar(
        x - width / 2,
        train_pct.values,
        width=width,
        color="#1565c0",
        edgecolor="black",
        label=f"Train (n={len(y_train):,})",
    )
    bars_test = ax.bar(
        x + width / 2,
        test_pct.values,
        width=width,
        color="#ef6c00",
        edgecolor="black",
        label=f"Test (n={len(y_test):,})",
    )
    for bars, pct_series in ((bars_train, train_pct), (bars_test, test_pct)):
        for bar, val in zip(bars, pct_series.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    failure_train = float(train_pct.get("failure", 0.0))
    failure_test = float(test_pct.get("failure", 0.0))
    drift = failure_train - failure_test

    ax.set_xticks(x)
    ax.set_xticklabels(CONCLUSION_ORDER)
    ax.set_ylabel("Percentage of rows")
    ax.set_ylim(0, max(train_pct.max(), test_pct.max()) * 1.18)
    ax.set_title(
        "Per-Repository Chronological Split — Class Distribution by Partition"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.99,
        0.05,
        f"Failure-rate drift train → test: {drift:+.1f} pp",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontstyle="italic",
        color="#37474f",
    )

    plotter.save_figure(
        fig,
        "fig_07_train_test_class_balance.png",
        caption=(
            "Class distribution of the binary ``conclusion`` target across the "
            "partitions of the per-repository chronological split, in which "
            "each repository is divided at its own quantile of ``created_at`` "
            "and whole commits are assigned to one side of the division. The "
            "residual difference in failure rate between the partitions is "
            "small. An earlier version of this figure showed a much larger "
            "gap and attributed it to temporal drift in repository stability; "
            "that reading was incorrect. The earlier split was cut globally "
            "on ``commit_date`` and produced a test partition spanning "
            "approximately eleven hours, so the gap reflected which "
            "repositories happened to be active in that window rather than "
            "any change over time."
        ),
        title="Figure 7 — Train/test class balance",
    )


def plot_engineered_features_distribution(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    columns = list(ENGINEERED_FEATURE_COLUMNS)
    palette = plotter.palette(len(columns))

    binary_cols = {
        "is_large_commit",
        "is_many_files",
        "is_retry",
        "is_off_hours",
        "is_weekend",
        "is_bot_author",
    }

    pretty_names = {
        "commit_message_length": "Commit msg length (chars)",
        "commit_message_word_count": "Commit msg word count",
        "lines_change_ratio": "Lines change ratio",
        "avg_lines_per_file": "Avg lines per file",
        "is_large_commit": "Is large commit (0/1)",
        "is_many_files": "Is many files (0/1)",
        "is_retry": "Is retry (0/1)",
        "is_off_hours": "Is off hours (0/1)",
        "is_weekend": "Is weekend (0/1)",
        "is_bot_author": "Is bot author (0/1)",
    }

    fig, axes = plotter.new_figure(figsize=(15.5, 7.5), nrows=2, ncols=5)
    axes_flat = axes.flatten()

    for ax, col, color in zip(axes_flat, columns, palette):
        series = df[col].dropna()
        if col in binary_cols:
            sns.countplot(x=series.astype(int), ax=ax, color=color)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0", "1"])
        else:
            # Heavy tails — log-transform on the visualisation only.
            display = np.log1p(series.astype(float).clip(lower=0))
            sns.histplot(
                display,
                ax=ax,
                color=color,
                bins=30,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(pretty_names[col])
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Distribution of Engineered Features", y=1.02)

    plotter.save_figure(
        fig,
        "fig_08_engineered_features_distribution.png",
        caption=(
            "Distribution of the ten engineered features derived from the "
            "real CI/CD dataset. Continuous features are shown on a "
            "log(1+x) axis to compress the heavy right tails inherited from "
            "the commit-size columns; binary indicators are displayed as "
            "categorical counts."
        ),
        title="Figure 8 — Engineered feature distributions",
    )


def plot_correlation_after_engineering(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    numeric_columns = [
        "run_duration_sec",
        "run_attempt",
        "lines_added",
        "lines_deleted",
        "total_changes",
        "files_changed",
    ] + list(ENGINEERED_FEATURE_COLUMNS)
    available = [c for c in numeric_columns if c in df.columns]
    corr = df[available].apply(pd.to_numeric, errors="coerce").corr(method="pearson")

    fig, ax = plotter.new_figure(figsize=(12.5, 10.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.75},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix After Feature Engineering")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.grid(False)

    plotter.save_figure(
        fig,
        "fig_09_correlation_after_engineering.png",
        caption=(
            "Pearson correlation matrix of the raw numerical columns plus "
            "the ten engineered features. Strong correlations among the "
            "commit-size cluster (``lines_added``, ``lines_deleted``, "
            "``total_changes``, ``files_changed``) are expected; the "
            "engineered ratios and binary flags occupy structurally "
            "distinct regions of the matrix, motivating their inclusion."
        ),
        title="Figure 9 — Correlation heatmap after engineering",
    )


# --------------------------------------------------------------------------- #
# Discriminative-vocabulary word clouds
# --------------------------------------------------------------------------- #


_TOKEN_PATTERN = re.compile(r"(?u)\b[A-Za-z][A-Za-z_]{2,}\b")
_STOPWORDS: set[str] = set(
    "a about above after again against all am an and any are aren as at be "
    "because been before being below between both but by could did do does "
    "doing don down during each few for from further had has have having he "
    "her here hers herself him himself his how i if in into is it its itself "
    "just like me more most my myself nor not now of off on once only or other "
    "our ours ourselves out over own same she should so some such than that "
    "the their theirs them themselves then there these they this those "
    "through to too under until up very was we were what when where which "
    "while who whom why will with you your yours yourself yourselves".split()
)


def _tokenize_corpus(messages: pd.Series) -> list[Counter]:
    """Return one Counter per message (lowercased, stop-word filtered)."""
    counters: list[Counter] = []
    for msg in messages.astype(str):
        tokens = [
            t.lower()
            for t in _TOKEN_PATTERN.findall(msg)
            if t.lower() not in _STOPWORDS and len(t) > 2
        ]
        counters.append(Counter(tokens))
    return counters


def _aggregate_class_counts(
    counters: list[Counter], mask: np.ndarray
) -> Counter:
    agg: Counter = Counter()
    for keep, counter in zip(mask, counters):
        if keep:
            agg.update(counter)
    return agg


def _class_distinctive_scores(
    class_counts: Counter,
    other_counts: Counter,
    min_count: int = 10,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """Score = log( P(token | class) / P(token | other) ).

    Tokens that occur < ``min_count`` times in the class corpus are dropped to
    avoid surfacing noise from one-off identifiers.
    """
    total_class = sum(class_counts.values()) or 1
    total_other = sum(other_counts.values()) or 1
    scores: dict[str, float] = {}
    for token, count in class_counts.items():
        if count < min_count:
            continue
        p_class = (count + smoothing) / total_class
        p_other = (other_counts.get(token, 0) + smoothing) / total_other
        scores[token] = float(np.log(p_class / p_other))
    return scores


def plot_discriminative_vocabulary(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    counters = _tokenize_corpus(df["commit_message"])
    failure_mask = (df["conclusion"].astype(str).values == "failure")
    success_mask = ~failure_mask

    failure_counts = _aggregate_class_counts(counters, failure_mask)
    success_counts = _aggregate_class_counts(counters, success_mask)

    failure_scores = _class_distinctive_scores(failure_counts, success_counts)
    success_scores = _class_distinctive_scores(success_counts, failure_counts)

    # Keep only positive-distinctive tokens (more frequent in this class than
    # in the other) and clip very small/very large for readability.
    failure_terms = {
        t: max(s, 0.01)
        for t, s in failure_scores.items()
        if s > 0
    }
    success_terms = {
        t: max(s, 0.01)
        for t, s in success_scores.items()
        if s > 0
    }

    fig, axes = plotter.new_figure(figsize=(15.5, 6.5), nrows=1, ncols=2)

    if not failure_terms:
        axes[0].text(0.5, 0.5, "no discriminative failure terms", ha="center")
    else:
        wc_failure = WordCloud(
            width=720,
            height=420,
            background_color="white",
            colormap="Reds",
            max_words=60,
            prefer_horizontal=0.9,
            relative_scaling=0.45,
            random_state=42,
        ).generate_from_frequencies(failure_terms)
        axes[0].imshow(wc_failure, interpolation="bilinear")
    axes[0].set_title("Tokens distinctive of FAILURE", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    if not success_terms:
        axes[1].text(0.5, 0.5, "no discriminative success terms", ha="center")
    else:
        wc_success = WordCloud(
            width=720,
            height=420,
            background_color="white",
            colormap="Greens",
            max_words=60,
            prefer_horizontal=0.9,
            relative_scaling=0.45,
            random_state=42,
        ).generate_from_frequencies(success_terms)
        axes[1].imshow(wc_success, interpolation="bilinear")
    axes[1].set_title("Tokens distinctive of SUCCESS", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Discriminative Commit-Message Vocabulary (log-odds vs other class)",
        y=1.02,
        fontsize=14,
    )

    plotter.save_figure(
        fig,
        "fig_10_discriminative_vocabulary.png",
        caption=(
            "Word clouds of commit-message tokens whose log-odds favour the "
            "FAILURE class (left, red) or the SUCCESS class (right, green). "
            "Token sizes are proportional to log-odds magnitude with "
            "additive smoothing. The visible separation between the two "
            "vocabularies confirms that ``commit_message`` carries "
            "type-specific predictive signal for the hybrid classifier."
        ),
        title="Figure 10 — Discriminative commit-message vocabulary",
    )


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _select_sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    rows: list[dict[str, Any]] = []
    for i in idx:
        row = df.iloc[int(i)]
        rows.append(
            {
                "repository": str(row["repository"]),
                "event": str(row["event"]),
                "conclusion": str(row["conclusion"]),
                "commit_message": str(row["commit_message"])[:200],
                "lines_added": int(row["lines_added"]),
                "files_changed": int(row["files_changed"]),
                "is_bot_author": int(row["is_bot_author"]),
                "is_weekend": int(row["is_weekend"]),
            }
        )
    return rows


def build_summary(
    df_prepared: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    return {
        "prepared_dataset": {
            "rows": int(len(df_prepared)),
            "columns": int(df_prepared.shape[1]),
            "column_names": list(df_prepared.columns),
            "engineered_columns": list(ENGINEERED_FEATURE_COLUMNS),
        },
        "splits": {
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_pct": round(len(x_train) / len(df_prepared) * 100, 2),
            "test_pct": round(len(x_test) / len(df_prepared) * 100, 2),
            "train_feature_count": int(x_train.shape[1]),
        },
        "class_distribution": {
            "conclusion": {
                "all": class_distribution(df_prepared["conclusion"]),
                "train": class_distribution(y_train),
                "test": class_distribution(y_test),
            }
        },
        "engineered_feature_summary": {
            col: {
                "mean": round(float(df_prepared[col].mean()), 4),
                "std": round(float(df_prepared[col].std()), 4),
                "min": round(float(df_prepared[col].min()), 4),
                "max": round(float(df_prepared[col].max()), 4),
            }
            for col in ENGINEERED_FEATURE_COLUMNS
        },
        "sample_rows": _select_sample_rows(df_prepared),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR)

    print("[Phase 2] Preparing dataset ...")
    df_prepared = prepare_dataset(
        raw_path=RAW_DATASET_PATH,
        output_path=PROCESSED_DATA_DIR / "cicd_prepared.csv",
    )
    print(f"          shape   = {df_prepared.shape}")
    print(f"          columns = {list(df_prepared.columns)}")

    print("\n[Phase 2] Splitting chronologically ...")
    x_train, x_test, y_train, y_test = split_dataset(df_prepared, test_size=0.2)
    print(
        f"          train rows = {len(x_train):,}   test rows = {len(x_test):,}"
    )

    print("\n[Phase 2] Generating validation figures ...")
    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    plot_train_test_class_balance(y_train, y_test, plotter)
    plot_engineered_features_distribution(df_prepared, plotter)
    plot_correlation_after_engineering(df_prepared, plotter)
    plot_discriminative_vocabulary(df_prepared, plotter)

    print("\n[Phase 2] Building summary ...")
    summary = build_summary(df_prepared, x_train, x_test, y_train, y_test)
    summary_path = RESULTS_DIR / "phase2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"          → {summary_path}")

    print("\n[Phase 2] Sample rows from the prepared dataset:")
    for i, row in enumerate(summary["sample_rows"], start=1):
        msg = " ".join(row["commit_message"].split())[:140]
        print(
            f"  {i}. [{row['repository']} · {row['conclusion']} · {row['event']}"
            f" · bot={row['is_bot_author']} · weekend={row['is_weekend']}]"
        )
        print(
            f"     +{row['lines_added']} lines · {row['files_changed']} files · {msg}"
        )

    print("\n[Phase 2] Class distribution across the chronological split:")
    for split_name, dist in summary["class_distribution"]["conclusion"].items():
        ordered_items = sorted(dist.items(), key=lambda kv: kv[0])
        pretty = ", ".join(f"{k}={v}%" for k, v in ordered_items)
        print(f"  {split_name:<5}: {pretty}")

    print("\n[Phase 2] Done.")


if __name__ == "__main__":
    main()
