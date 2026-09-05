"""Phase 2.5 data preparation: aggressive NLP cleaning + final feature set.

This replaces the Phase 2 module to address three issues surfaced by review:

1. **Text leakage** — the Phase 2 TF-IDF picked up author logins and project
   names instead of commit content. Solved here by
   :func:`clean_commit_message_for_nlp` (regex normalisation) followed by
   :func:`apply_stoplist` (identifier removal based on a corpus-derived
   stoplist).
2. **Outcome leakage** — ``run_duration_sec``, ``run_attempt`` and
   ``is_retry`` are only known *after* a workflow finishes, so they cannot
   appear in a predictive feature set. They are added to
   :data:`LEAKAGE_OR_REDUNDANT` and dropped before any split.
3. **Multicollinearity** — the Phase 1/2 EDA showed several pairs with
   Pearson r ≥ 0.95. The final feature set keeps one representative per
   redundant cluster.

The module exports the final feature constants (``NUMERICAL_FEATURES``,
``CATEGORICAL_FEATURES``, ``BINARY_FEATURES``, ``TEXT_FEATURE``, ``TARGET``)
so the downstream hybrid pipeline can import them verbatim.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .utils import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_dir,
    get_logger,
)


_LOGGER = get_logger(__name__)


RAW_DATASET_PATH: Path = RAW_DATA_DIR / "github_actions_real.csv"


# --------------------------------------------------------------------------- #
# Final feature set (Phase 2.5 — model-ready)
# --------------------------------------------------------------------------- #


NUMERICAL_FEATURES: list[str] = [
    "log_lines_added",
    "log_lines_deleted",
    "log_files_changed",
    "commit_message_length",
    "avg_lines_per_file",
]


CATEGORICAL_FEATURES: list[str] = [
    "repository",
    "workflow_name",
    "branch",
    "event",
]


BINARY_FEATURES: list[str] = [
    "is_large_commit",
    "is_many_files",
    "is_weekend_commit",
    "is_off_hours_commit",
    "is_bot_author",
    "was_truncated",
]


TEXT_FEATURE: str = "commit_message_clean"
TARGET: str = "conclusion"

# Splitting keys. Neither is a feature.
#
# ``GROUP_KEY`` — rows are workflow runs but features are commit-level
# (3.45 runs per commit), so a row-level split scatters near-duplicates of the
# same commit across train and test. Every split groups on this column.
#
# ``TIME_KEY`` — ``created_at`` is when the workflow run executed, which is the
# moment a deployed predictor would be invoked. ``commit_date`` is when the
# commit was authored, which can precede the run by years for re-runs and
# merge-base commits, and is therefore the wrong axis for a temporal holdout.
GROUP_KEY: str = "commit_sha"
TIME_KEY: str = "created_at"


ALL_FEATURE_COLUMNS: list[str] = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [TEXT_FEATURE]
)


LEAKAGE_OR_REDUNDANT: list[str] = [
    # POST-EXECUTION (leakage)
    "run_duration_sec",
    "run_attempt",
    "is_retry",
    "status",
    "updated_at",
    # REDUNDANT (multicollinearity)
    "total_changes",
    "lines_change_ratio",
    "commit_message_word_count",
    "log_total_changes",
    # IDs / non-features
    "run_id",
    # NOTE: ``commit_sha`` is deliberately NOT dropped here. It is retained
    # through preparation as the grouping key for every split (see
    # :data:`GROUP_KEY`) because each modelled feature is commit-level while
    # each row is a workflow run. It is excluded from
    # ``ALL_FEATURE_COLUMNS``, so it never reaches the ColumnTransformer.
    # Already encoded by other features
    "commit_hour",
    "commit_day_of_week",
    "lines_added",
    "lines_deleted",
    "files_changed",
    # Personal identifier (use is_bot_author instead)
    "commit_author",
    # Always NaN
    "author_association",
]


# --------------------------------------------------------------------------- #
# High-cardinality bucketing
# --------------------------------------------------------------------------- #


_BUCKET_TOP_K = {
    "workflow_name": 20,
    "branch": 15,
}

_OTHER_TOKEN = "__other__"
_UNKNOWN_TOKEN = "__unknown__"


def _bucket_top_k(series: pd.Series, k: int, other: str = _OTHER_TOKEN) -> pd.Series:
    top_values = series.value_counts(dropna=False).head(k).index
    return series.where(series.isin(top_values), other=other)


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #


# Order matters here — multi-line headers and Dependabot patterns are
# removed first because they reference author/project names that the more
# generic regexes would otherwise miss.
_COAUTHOR_RE = re.compile(r"(?im)^(?:co-authored-by|signed-off-by):.*$")
_BUMPS_RE = re.compile(
    r"(?i)\bbumps?\s+[^\s]+\s+from\s+[^\s]+\s+to\s+[^\s]+",
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_SHA40_RE = re.compile(r"\b[0-9a-fA-F]{40}\b")
_PR_REF_RE = re.compile(r"\(?#\d+\)?|gh[-_]\d+", re.IGNORECASE)
_VERSION_RE = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+){0,3}(?:[-.][A-Za-z0-9]+)*\b"
)
_SHA8_RE = re.compile(r"\b[0-9a-fA-F]{8,12}\b")
_FILEPATH_RE = re.compile(r"\b[A-Za-z0-9_./-]+\.[A-Za-z]{1,8}\b")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_commit_message_for_nlp(text: object) -> str:
    """Aggressive cleaning to prevent author / project leakage in TF-IDF.

    Order: structural headers → URLs / emails → identifiers
    (SHAs, PR refs, versions) → filenames → strip non-alphabetic
    characters → collapse whitespace.
    """
    if text is None or not isinstance(text, str):
        return ""
    s = text
    s = _COAUTHOR_RE.sub(" ", s)
    s = _BUMPS_RE.sub(" ", s)
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _SHA40_RE.sub(" ", s)
    s = _PR_REF_RE.sub(" ", s)
    s = _VERSION_RE.sub(" ", s)
    s = _SHA8_RE.sub(" ", s)
    s = _FILEPATH_RE.sub(" ", s)
    s = s.lower()
    s = _NON_ALPHA_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Stoplist
# --------------------------------------------------------------------------- #


# Manually curated additions:
# * Generic CI / Git / bot signatures.
# * Identifier-like noise observed in the Phase 2 word cloud
#   (e.g. ``ptl``, ``kamat``, ``yagiz``, ``iseq``).
# * Short tokens commonly leaked from versioned filenames after cleaning.
_EXTRA_STOPWORDS: set[str] = {
    "dependabot", "renovate", "bors", "github", "actions", "bot", "machine",
    "via", "ref", "rev", "fixup", "lgtm", "wip", "etc", "amd", "arm",
    # Phase 2 observed noise (author logins / domain-specific identifiers)
    "anna", "kamat", "yagiz", "trivikr", "kib", "kamil", "eps", "lon",
    "ptl", "iseq", "vcpkg", "callinto", "subroma", "commenter", "ricky",
    "joshua", "renato", "boba", "ryan", "richharris", "asanas", "asana",
    "krish", "vasco", "hemang", "patel",
    # Common general English noise not in sklearn's default stop list
    "would", "could", "should", "make", "need", "want", "thing", "things",
    "way", "much", "even", "still", "also", "however", "though",
}


def build_stoplist(df: pd.DataFrame) -> set[str]:
    """Return tokens to remove from cleaned commit messages.

    Combines:
    * tokens extracted from every distinct ``commit_author`` value,
    * tokens extracted from every distinct ``repository`` value,
    * the curated :data:`_EXTRA_STOPWORDS` set.
    """
    stoplist: set[str] = set()

    if "commit_author" in df.columns:
        for author in df["commit_author"].dropna().astype(str).unique():
            for raw_tok in re.split(r"[-_\s.]+", author.lower()):
                tok = re.sub(r"[^a-z]", "", raw_tok)
                if len(tok) >= 3 and not tok.isdigit():
                    stoplist.add(tok)

    if "repository" in df.columns:
        for repo in df["repository"].dropna().astype(str).unique():
            for part in repo.split("/"):
                for raw_tok in re.split(r"[-_.]+", part.lower()):
                    tok = re.sub(r"[^a-z]", "", raw_tok)
                    if len(tok) >= 3:
                        stoplist.add(tok)

    stoplist.update(_EXTRA_STOPWORDS)
    return stoplist


def apply_stoplist(text: str, stoplist: set[str]) -> str:
    """Remove every token in ``stoplist`` from already-cleaned ``text``."""
    if not text:
        return ""
    return " ".join(tok for tok in text.split() if tok not in stoplist)


# --------------------------------------------------------------------------- #
# Bot detection (used by engineer_features)
# --------------------------------------------------------------------------- #


_BOT_AUTHOR_REGEX = re.compile(
    r"(?i)(\[bot\]|-bot$|^bot-|^dependabot$|^renovate$|^bors$|machine$)"
)


def _detect_bot(author: object) -> int:
    if author is None or (isinstance(author, float) and np.isnan(author)):
        return 0
    return 1 if _BOT_AUTHOR_REGEX.search(str(author)) else 0


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


_TRUNCATION_THRESHOLD = 1000  # Phase 0 truncated commit_message at 1000 chars


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2.5 engineered features (10 + log family + was_truncated)."""
    df = df.copy()

    if "commit_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["commit_date"]
    ):
        df["commit_date"] = pd.to_datetime(
            df["commit_date"], errors="coerce", utc=True
        )

    # Numeric safety
    for col in ("lines_added", "lines_deleted", "total_changes", "files_changed"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    msg = df["commit_message"].fillna("").astype(str)
    df["commit_message_length"] = msg.str.len()
    df["was_truncated"] = (df["commit_message_length"] >= _TRUNCATION_THRESHOLD).astype(int)

    # Log family (the model-facing numerical features).
    df["log_lines_added"] = np.log1p(df["lines_added"].astype(float))
    df["log_lines_deleted"] = np.log1p(df["lines_deleted"].astype(float))
    df["log_files_changed"] = np.log1p(df["files_changed"].astype(float))

    df["avg_lines_per_file"] = df["total_changes"] / (df["files_changed"] + 1.0)

    median_total = float(df["total_changes"].median())
    median_files = float(df["files_changed"].median())
    df["is_large_commit"] = (df["total_changes"] > median_total).astype(int)
    df["is_many_files"] = (df["files_changed"] > median_files).astype(int)

    hour = df["commit_date"].dt.hour
    df["is_off_hours_commit"] = (
        ((hour >= 18) | (hour < 6)).fillna(False).astype(int)
    )
    df["is_weekend_commit"] = (
        (df["commit_date"].dt.dayofweek >= 5).fillna(False).astype(int)
    )

    if "commit_author" in df.columns:
        df["is_bot_author"] = df["commit_author"].apply(_detect_bot).astype(int)
    else:
        df["is_bot_author"] = 0

    return df


# --------------------------------------------------------------------------- #
# End-to-end preparation
# --------------------------------------------------------------------------- #


def prepare_dataset(
    raw_path: Path, output_path: Path
) -> Tuple[pd.DataFrame, set[str]]:
    """Load → engineer → clean text → apply stoplist → drop leakage → persist."""
    _LOGGER.info("Loading raw dataset from %s", raw_path)
    df = pd.read_csv(
        raw_path,
        parse_dates=["created_at", "updated_at", "commit_date"],
    )
    _LOGGER.info("Raw shape: %s", df.shape)

    # Drop empty commit messages — there's no text signal to learn from.
    initial = len(df)
    df = df[
        df["commit_message"].notna()
        & (df["commit_message"].astype(str).str.len() > 0)
    ]
    dropped = initial - len(df)
    if dropped:
        _LOGGER.info("Dropped %d rows with empty commit_message", dropped)

    df["commit_author"] = df["commit_author"].fillna(_UNKNOWN_TOKEN)

    # Build the identifier stoplist BEFORE we drop commit_author.
    stoplist = build_stoplist(df)
    _LOGGER.info("Built stoplist with %d tokens", len(stoplist))

    # Bucket high-cardinality categoricals so OHE downstream stays tractable.
    for col, k in _BUCKET_TOP_K.items():
        if col in df.columns:
            before = df[col].nunique(dropna=False)
            df[col] = _bucket_top_k(df[col], k=k)
            after = df[col].nunique(dropna=False)
            _LOGGER.info(
                "Bucketed %s: %d → %d unique values (top-%d + '%s')",
                col, before, after, k, _OTHER_TOKEN,
            )

    df = engineer_features(df)

    _LOGGER.info("Cleaning commit messages ...")
    cleaned = df["commit_message"].apply(clean_commit_message_for_nlp)
    df["commit_message_clean"] = cleaned.apply(
        lambda t: apply_stoplist(t, stoplist)
    )

    drop = [c for c in LEAKAGE_OR_REDUNDANT if c in df.columns]
    df = df.drop(columns=drop)
    _LOGGER.info("Dropped %d leakage/redundant columns: %s", len(drop), drop)

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    _LOGGER.info("Wrote prepared dataset → %s (shape=%s)", output_path, df.shape)

    return df, stoplist


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def _to_epoch_ns(series: pd.Series) -> np.ndarray:
    """Return ``series`` as UTC nanoseconds since the epoch, for ordering."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None).astype("int64").to_numpy()


def _assign_whole_commits_to_windows(
    frame: pd.DataFrame,
    boundaries: list[int],
    group_key: str = GROUP_KEY,
    time_key: str = TIME_KEY,
) -> pd.Series:
    """Assign each *commit* to exactly one time window, or discard it.

    ``boundaries`` are ascending epoch-nanosecond cut points, so ``n``
    boundaries define ``n + 1`` windows numbered ``0 .. n``. A commit lands in
    window ``i`` only when **all** of its runs fall inside window ``i``. A
    commit whose runs straddle a boundary is assigned ``-1`` and dropped by the
    caller: keeping it would either place the same commit on both sides of the
    cut (group leakage) or place a run on the wrong side of the cut (temporal
    leakage), and there is no third option.

    Returns a row-aligned Series of window ids.
    """
    times = _to_epoch_ns(frame[time_key])
    spans = (
        pd.DataFrame({group_key: frame[group_key].to_numpy(), "t": times})
        .groupby(group_key)["t"]
        .agg(["min", "max"])
    )
    edges = np.asarray(sorted(boundaries), dtype="int64")
    first = np.searchsorted(edges, spans["min"].to_numpy(), side="right")
    last = np.searchsorted(edges, spans["max"].to_numpy(), side="right")
    window = np.where(first == last, first, -1)
    return frame[group_key].map(pd.Series(window, index=spans.index)).astype(int)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on rows — **retained only as a leakage demonstration**.

    This was the original primary evaluation strategy. It splits rows, but
    every modelled feature is commit-level, so 88.3 per cent of its test rows
    share a commit with the training set. The reported metrics it produces are
    inflated by roughly six points of F1. It is kept so the thesis can quantify
    that inflation against :func:`grouped_split`; it must not be used to report
    headline performance.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    ensure_dir(processed_dir)
    train_df.to_csv(processed_dir / "train_stratified.csv", index=False)
    test_df.to_csv(processed_dir / "test_stratified.csv", index=False)
    _LOGGER.info(
        "Stratified split saved — train=%d test=%d", len(train_df), len(test_df)
    )
    return train_df, test_df


def grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Commit-grouped stratified split — **PRIMARY evaluation strategy**.

    Splits on ``commit_sha`` so that every run of a given commit lands on one
    side of the cut, which is what the modelling grain requires: features are
    commit-level while rows are workflow runs.

    :class:`StratifiedGroupKFold` is used in preference to
    :class:`GroupShuffleSplit` because the latter does not stratify. Grouping
    alone lets the class balance drift between folds (measured: 9.71 per cent
    failure in train against 14.72 per cent in validation), which would
    confound threshold selection — a threshold chosen on a validation fold with
    a different prior does not transfer to the test fold, and the resulting
    error would be indistinguishable from a genuine calibration effect. Taking
    the first fold of a five-fold stratified grouped partition yields the
    intended 20 per cent test share with the class balance preserved.

    ``val_size`` is carved from the training portion under the same scheme, and
    exists so the decision threshold can be selected without consulting the
    test set.

    Returns ``(train, validation, test)``.
    """
    if not np.isclose(test_size, 0.2) or not np.isclose(val_size, 0.2):
        raise ValueError(
            "grouped_split is fixed to a 0.2 / 0.2 five-fold scheme; "
            f"got test_size={test_size}, val_size={val_size}."
        )

    groups = df[GROUP_KEY].astype(str)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_pool_idx, test_idx = next(outer.split(df, df[TARGET], groups))
    train_pool = df.iloc[train_pool_idx]

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(
        inner.split(
            train_pool, train_pool[TARGET], train_pool[GROUP_KEY].astype(str)
        )
    )

    train_df = train_pool.iloc[train_idx].reset_index(drop=True)
    val_df = train_pool.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    ensure_dir(processed_dir)
    train_df.to_csv(processed_dir / "train_grouped.csv", index=False)
    val_df.to_csv(processed_dir / "val_grouped.csv", index=False)
    test_df.to_csv(processed_dir / "test_grouped.csv", index=False)
    _LOGGER.info(
        "Grouped split saved — train=%d val=%d test=%d (unique commits %d/%d/%d)",
        len(train_df), len(val_df), len(test_df),
        train_df[GROUP_KEY].nunique(), val_df[GROUP_KEY].nunique(),
        test_df[GROUP_KEY].nunique(),
    )
    return train_df, val_df, test_df


def chronological_split(
    df: pd.DataFrame,
    test_quantile: float = 0.80,
    val_quantile: float = 0.64,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-repository chronological split — SECONDARY evaluation.

    Each repository is cut at its **own** quantiles of :data:`TIME_KEY`, and
    whole commits are assigned to the resulting windows. Cutting per repository
    rather than globally is necessary because the collector capped every
    repository at 600 runs, so a busy repository contributes 600 runs drawn
    from half a day while a quiet one spreads 600 runs over six months. A
    single global cut therefore places 88.7 per cent of the corpus after the
    boundary and yields a test window of roughly nine hours covering eleven of
    the eighteen repositories, which measures the composition of that
    particular window rather than the passage of time.

    The per-repository cut instead holds, for every repository, that the model
    is trained on that repository's past and evaluated on its future. Commits
    straddling a boundary are discarded.

    Returns ``(train, validation, test)``.
    """
    if TIME_KEY not in df.columns:
        raise KeyError(
            f"{TIME_KEY!r} is required for the chronological split; "
            "it must survive prepare_dataset()."
        )

    parts: dict[int, list[pd.DataFrame]] = {0: [], 1: [], 2: []}
    dropped_rows = 0

    for repository, sub in df.groupby("repository", sort=True):
        times = _to_epoch_ns(sub[TIME_KEY])
        boundaries = [
            int(np.quantile(times, val_quantile)),
            int(np.quantile(times, test_quantile)),
        ]
        window = _assign_whole_commits_to_windows(sub, boundaries)
        for bucket in (0, 1, 2):
            selected = sub[window == bucket]
            if len(selected):
                parts[bucket].append(selected)
        dropped = int((window == -1).sum())
        dropped_rows += dropped
        if dropped:
            _LOGGER.info(
                "%s: dropped %d rows on straddling commits", repository, dropped
            )

    train_df = pd.concat(parts[0]).reset_index(drop=True)
    val_df = pd.concat(parts[1]).reset_index(drop=True)
    test_df = pd.concat(parts[2]).reset_index(drop=True)

    ensure_dir(processed_dir)
    train_df.to_csv(processed_dir / "train_chronological.csv", index=False)
    val_df.to_csv(processed_dir / "val_chronological.csv", index=False)
    test_df.to_csv(processed_dir / "test_chronological.csv", index=False)
    _LOGGER.info(
        "Chronological split saved — train=%d val=%d test=%d "
        "(%d rows dropped on straddling commits)",
        len(train_df), len(val_df), len(test_df), dropped_rows,
    )
    return train_df, val_df, test_df


def split_integrity_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    name: str,
    per_repository: bool = False,
) -> dict[str, Any]:
    """Machine-checkable evidence that a split is what it claims to be.

    Records commit overlap between every pair of folds, the temporal ordering
    of the folds, and the class balance of each. ``per_repository`` selects
    which temporal invariant is asserted: the per-repository chronological
    split cannot satisfy a single global ordering, because one repository's
    future may precede another's past.
    """
    folds = {"train": train_df, "test": test_df}
    if val_df is not None:
        folds["val"] = val_df

    commits = {k: set(v[GROUP_KEY].astype(str)) for k, v in folds.items()}
    overlaps = {
        f"{a}_vs_{b}": len(commits[a] & commits[b])
        for a, b in (("train", "test"), ("train", "val"), ("val", "test"))
        if a in commits and b in commits
    }

    report: dict[str, Any] = {
        "split": name,
        "rows": {k: int(len(v)) for k, v in folds.items()},
        "unique_commits": {k: int(len(v)) for k, v in commits.items()},
        "commit_overlap": overlaps,
        "commit_overlap_clean": all(v == 0 for v in overlaps.values()),
        "failure_rate_pct": {
            k: round(float((v[TARGET] == "failure").mean() * 100.0), 3)
            for k, v in folds.items()
        },
    }

    windows = {}
    for k, v in folds.items():
        t = pd.to_datetime(v[TIME_KEY], errors="coerce", utc=True)
        windows[k] = {
            "min": str(t.min()),
            "max": str(t.max()),
            "span_days": round(float((t.max() - t.min()).total_seconds() / 86400.0), 3),
        }
    report["time_window"] = windows

    if per_repository:
        holds = True
        per_repo: dict[str, Any] = {}
        for repository in sorted(set(train_df["repository"]) & set(test_df["repository"])):
            tr = pd.to_datetime(
                train_df.loc[train_df["repository"] == repository, TIME_KEY],
                errors="coerce", utc=True,
            )
            te = pd.to_datetime(
                test_df.loc[test_df["repository"] == repository, TIME_KEY],
                errors="coerce", utc=True,
            )
            ok = bool(te.min() >= tr.max())
            holds = holds and ok
            per_repo[repository] = {
                "train_max": str(tr.max()),
                "test_min": str(te.min()),
                "test_after_train": ok,
            }
        report["temporal_invariant"] = "per_repository: min(test) >= max(train)"
        report["temporal_invariant_holds"] = holds
        report["per_repository"] = per_repo
    else:
        report["temporal_invariant"] = "none (random split)"
        report["temporal_invariant_holds"] = None

    return report


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def class_distribution(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts(normalize=True) * 100.0
    return {str(k): round(float(v), 3) for k, v in counts.items()}


__all__ = [
    "ALL_FEATURE_COLUMNS",
    "GROUP_KEY",
    "TIME_KEY",
    "BINARY_FEATURES",
    "CATEGORICAL_FEATURES",
    "LEAKAGE_OR_REDUNDANT",
    "NUMERICAL_FEATURES",
    "RAW_DATASET_PATH",
    "TARGET",
    "TEXT_FEATURE",
    "apply_stoplist",
    "build_stoplist",
    "chronological_split",
    "class_distribution",
    "grouped_split",
    "split_integrity_report",
    "clean_commit_message_for_nlp",
    "engineer_features",
    "prepare_dataset",
    "stratified_split",
]


if __name__ == "__main__":
    df_prepared, stoplist = prepare_dataset(
        raw_path=RAW_DATASET_PATH,
        output_path=PROCESSED_DATA_DIR / "cicd_prepared.csv",
    )
    print(df_prepared.head())
    print(json.dumps(class_distribution(df_prepared[TARGET]), indent=2))
