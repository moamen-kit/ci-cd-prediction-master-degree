"""Phase 2.5 runner: clean text, build splits, regenerate fig_10, summarise.

This script does **not** train any model — it produces the model-ready
artefacts that Phase 3 will consume:

* ``data/processed/cicd_prepared.csv`` — full prepared dataset.
* ``data/processed/train_stratified.csv`` / ``test_stratified.csv``
  (primary eval) and ``train_chronological.csv`` / ``test_chronological.csv``
  (secondary eval).
* ``figures/fig_10_discriminative_vocabulary.png`` — regenerated with the
  cleaned vocabulary.
* ``results/phase2_5_summary.json`` — stoplist size, feature counts, class
  distributions per split, top failure/success tokens.

Run from project root::

    python src/run_phase2_5.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preparation import (  # noqa: E402
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RAW_DATASET_PATH,
    TARGET,
    TEXT_FEATURE,
    chronological_split,
    grouped_split,
    split_integrity_report,
    class_distribution,
    prepare_dataset,
    stratified_split,
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
# Discriminative-vocabulary helpers
# --------------------------------------------------------------------------- #


def _aggregate_class_counts(
    df: pd.DataFrame, mask: np.ndarray, col: str = TEXT_FEATURE
) -> Counter:
    counter: Counter = Counter()
    for text in df.loc[mask, col].dropna().astype(str):
        counter.update(text.split())
    return counter


def discriminative_scores(
    class_counts: Counter,
    other_counts: Counter,
    min_count: int = 10,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """log( P(token | class) / P(token | other) ), with additive smoothing."""
    total_class = sum(class_counts.values()) or 1
    total_other = sum(other_counts.values()) or 1
    scores: dict[str, float] = {}
    for token, cnt in class_counts.items():
        if cnt < min_count:
            continue
        p_class = (cnt + smoothing) / total_class
        p_other = (other_counts.get(token, 0) + smoothing) / total_other
        scores[token] = float(np.log(p_class / p_other))
    return scores


def plot_discriminative_vocabulary_clean(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> tuple[dict[str, float], dict[str, float]]:
    failure_mask = (df[TARGET].astype(str).values == "failure")
    success_mask = ~failure_mask

    failure_counts = _aggregate_class_counts(df, failure_mask)
    success_counts = _aggregate_class_counts(df, success_mask)

    failure_scores = discriminative_scores(failure_counts, success_counts)
    success_scores = discriminative_scores(success_counts, failure_counts)

    failure_terms = {t: max(s, 0.01) for t, s in failure_scores.items() if s > 0}
    success_terms = {t: max(s, 0.01) for t, s in success_scores.items() if s > 0}

    fig, axes = plotter.new_figure(figsize=(15.5, 6.5), nrows=1, ncols=2)

    if failure_terms:
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
    else:
        axes[0].text(0.5, 0.5, "no failure terms", ha="center", va="center")
    axes[0].set_title("Tokens distinctive of FAILURE", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    if success_terms:
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
    else:
        axes[1].text(0.5, 0.5, "no success terms", ha="center", va="center")
    axes[1].set_title("Tokens distinctive of SUCCESS", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Discriminative Commit-Message Vocabulary (after Phase 2.5 cleaning)",
        y=1.02,
        fontsize=14,
    )

    plotter.save_figure(
        fig,
        "fig_10_discriminative_vocabulary.png",
        caption=(
            "Word clouds of commit-message tokens whose log-odds favour the "
            "FAILURE class (left, red) or the SUCCESS class (right, green). "
            "Built from the Phase 2.5 cleaned text — URLs, SHAs, PR refs, "
            "version strings, filenames, and author/project identifiers have "
            "been removed. The remaining vocabulary is dominated by "
            "meaningful CI/CD content words and validates that "
            "``commit_message`` carries class-specific predictive signal."
        ),
        title="Figure 10 — Discriminative commit-message vocabulary (cleaned)",
    )

    return failure_scores, success_scores


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #


def _print_top_tokens(
    title: str, scores: dict[str, float], n: int = 20, side: str = "+"
) -> list[tuple[str, float]]:
    reverse = side == "+"
    top = sorted(
        scores.items(), key=lambda kv: kv[1], reverse=reverse
    )[:n]
    print(f"\n{title}")
    print("-" * 50)
    for token, score in top:
        print(f"  {token:<24} {score:+.3f}")
    return top


def _print_dist_table(
    label: str, dist_train: dict[str, float], dist_test: dict[str, float]
) -> None:
    keys = sorted(set(dist_train) | set(dist_test))
    line = f"  {label:<22} | " + " | ".join(f"{k:>10}" for k in keys)
    print(line)
    print("  " + "-" * (len(line) - 2))
    print(
        f"  {'train %':<22} | "
        + " | ".join(f"{dist_train.get(k, 0):>10.2f}" for k in keys)
    )
    print(
        f"  {'test  %':<22} | "
        + " | ".join(f"{dist_test.get(k, 0):>10.2f}" for k in keys)
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR)

    print("[Phase 2.5] Preparing dataset (cleaning + stoplist) ...")
    df, stoplist = prepare_dataset(
        raw_path=RAW_DATASET_PATH,
        output_path=PROCESSED_DATA_DIR / "cicd_prepared.csv",
    )
    print(f"             prepared shape = {df.shape}")

    sorted_stop = sorted(stoplist)
    print(f"\n[Phase 2.5] Stoplist size = {len(stoplist)}")
    print(f"             top 20 entries: {sorted_stop[:20]}")

    print("\n[Phase 2.5] Sample of 5 cleaned commit messages (before / after):")
    sample = df.sample(n=5, random_state=42)
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        before = " ".join(str(row["commit_message"]).split())
        after = str(row[TEXT_FEATURE])
        print(f"  {i}. BEFORE: {before[:160]}")
        print(f"     AFTER : {after[:160]}")

    print("\n[Phase 2.5] Final feature count by branch:")
    print(f"  Numerical   : {len(NUMERICAL_FEATURES)} — {NUMERICAL_FEATURES}")
    print(f"  Categorical : {len(CATEGORICAL_FEATURES)} — {CATEGORICAL_FEATURES}")
    print(f"  Binary      : {len(BINARY_FEATURES)} — {BINARY_FEATURES}")
    print(f"  Text        : 1 — '{TEXT_FEATURE}'")

    print("\n[Phase 2.5] Generating commit-grouped split (PRIMARY) ...")
    train_g, val_g, test_g = grouped_split(
        df, test_size=0.2, val_size=0.2, random_state=42
    )
    dist_train_g = class_distribution(train_g[TARGET])
    dist_test_g = class_distribution(test_g[TARGET])
    print(
        f"             train rows = {len(train_g):,}   "
        f"val rows = {len(val_g):,}   test rows = {len(test_g):,}"
    )
    _print_dist_table("grouped", dist_train_g, dist_test_g)

    print("\n[Phase 2.5] Generating per-repository chronological split (secondary) ...")
    train_c, val_c, test_c = chronological_split(df)
    dist_train_c = class_distribution(train_c[TARGET])
    dist_test_c = class_distribution(test_c[TARGET])
    print(
        f"             train rows = {len(train_c):,}   "
        f"val rows = {len(val_c):,}   test rows = {len(test_c):,}"
    )
    _print_dist_table("chronological", dist_train_c, dist_test_c)

    print("\n[Phase 2.5] Regenerating stratified split (leakage demonstration only) ...")
    train_s, test_s = stratified_split(df, test_size=0.2, random_state=42)
    dist_train_s = class_distribution(train_s[TARGET])
    dist_test_s = class_distribution(test_s[TARGET])
    _print_dist_table("stratified (leaky)", dist_train_s, dist_test_s)

    print("\n[Phase 2.5] Verifying split integrity ...")
    integrity = {
        "grouped": split_integrity_report(train_g, val_g, test_g, "grouped"),
        "chronological": split_integrity_report(
            train_c, val_c, test_c, "chronological", per_repository=True
        ),
        "stratified_leaky": split_integrity_report(
            train_s, None, test_s, "stratified_leaky"
        ),
    }
    integrity_path = RESULTS_DIR / "split_integrity.json"
    integrity_path.write_text(
        json.dumps(integrity, indent=2, default=str), encoding="utf-8"
    )
    for key, rep in integrity.items():
        print(
            f"             {key:<18} commit overlap clean = "
            f"{rep['commit_overlap_clean']}  "
            f"temporal invariant = {rep['temporal_invariant_holds']}  "
            f"overlaps = {rep['commit_overlap']}"
        )
    print(f"             Integrity report written -> {integrity_path}")

    print("\n[Phase 2.5] Regenerating fig_10 with cleaned vocabulary ...")
    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    failure_scores, success_scores = plot_discriminative_vocabulary_clean(df, plotter)

    top_failure = _print_top_tokens(
        "Top 20 FAILURE tokens (log-odds, after cleaning)",
        failure_scores,
        n=20,
        side="+",
    )
    top_success = _print_top_tokens(
        "Top 20 SUCCESS tokens (log-odds, after cleaning)",
        success_scores,
        n=20,
        side="+",
    )

    summary = {
        "prepared_shape": list(df.shape),
        "stoplist_size": len(stoplist),
        "stoplist_sample": sorted_stop[:30],
        "feature_counts": {
            "numerical": len(NUMERICAL_FEATURES),
            "categorical": len(CATEGORICAL_FEATURES),
            "binary": len(BINARY_FEATURES),
            "text": 1,
        },
        "feature_lists": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "binary": BINARY_FEATURES,
            "text_feature": TEXT_FEATURE,
            "target": TARGET,
        },
        "grouped_split": {
            "train_rows": int(len(train_g)),
            "val_rows": int(len(val_g)),
            "test_rows": int(len(test_g)),
            "train": dist_train_g,
            "test": dist_test_g,
        },
        "chronological_split": {
            "train_rows": int(len(train_c)),
            "val_rows": int(len(val_c)),
            "test_rows": int(len(test_c)),
            "train": dist_train_c,
            "test": dist_test_c,
        },
        "stratified_split_leaky": {
            "train_rows": int(len(train_s)),
            "test_rows": int(len(test_s)),
            "train": dist_train_s,
            "test": dist_test_s,
        },
        "split_integrity": integrity,
        "top_failure_tokens": [
            {"token": t, "log_odds": round(s, 4)} for t, s in top_failure
        ],
        "top_success_tokens": [
            {"token": t, "log_odds": round(s, 4)} for t, s in top_success
        ],
    }
    summary_path = RESULTS_DIR / "phase2_5_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[Phase 2.5] Summary written → {summary_path}")
    print("\n[Phase 2.5] Done.")


if __name__ == "__main__":
    main()
