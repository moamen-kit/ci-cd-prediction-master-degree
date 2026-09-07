"""Hybrid binary-classification pipeline for the real GitHub Actions data.

The synthetic-dataset version of this module wrapped every estimator in
:class:`MultiOutputClassifier` because that experiment predicted three
targets at once. The real dataset has a single binary target
(``conclusion``), so the pipelines here are flat:

::

    ColumnTransformer (numerical + categorical + binary + TF-IDF text)
        ↓
    estimator  (LogisticRegression | RandomForestClassifier | XGBClassifier)

Three estimator families are built so Phase 4 can compare them on equal
footing. The two scikit-learn estimators accept the raw string labels
``"success"`` / ``"failure"`` directly, but XGBoost requires integer-encoded
labels — so its pipeline is wrapped in
:class:`LabelEncoderForBinary` to keep the ``fit`` / ``predict`` contract
identical across all three pipelines.

Class imbalance is ~89% / 11% (Phase 1 EDA). LR / RF use
``class_weight='balanced'``; XGBoost uses ``scale_pos_weight = 8.12``.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data_preparation import (
    ALL_FEATURE_COLUMNS,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    TEXT_FEATURE,
)
from .utils import get_logger


_LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Class-imbalance constant
# --------------------------------------------------------------------------- #


# Empirical from Phase 1 EDA: 8,700 success / 1,072 failure.
_SCALE_POS_WEIGHT = 8.12


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #


def build_preprocessor() -> ColumnTransformer:
    """Construct the four-branch feature preprocessor.

    The output is a sparse matrix that fuses StandardScaler-normalised
    numerical features, OneHotEncoded categoricals, passthrough binary flags
    and TF-IDF vectors derived from ``commit_message_clean``.
    """
    numerical = StandardScaler()
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    text = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical, NUMERICAL_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            # ``TEXT_FEATURE`` is passed as a string (not a list) so the
            # column is extracted as a 1-D Series — TfidfVectorizer's
            # expected input shape.
            ("text", text, TEXT_FEATURE),
        ],
        sparse_threshold=0.3,
        n_jobs=-1,
        verbose_feature_names_out=True,
    )


# --------------------------------------------------------------------------- #
# Label-encoding wrapper for XGBoost
# --------------------------------------------------------------------------- #


class LabelEncoderForBinary(BaseEstimator, ClassifierMixin):
    """Wrap a binary classifier so it accepts the raw string target.

    XGBoost requires integer class labels. This wrapper fits a single
    :class:`LabelEncoder` at ``fit`` time, then inverse-transforms predictions
    so the caller never sees the encoded integers.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        **fit_params: Any,
    ) -> "LabelEncoderForBinary":
        y_array = np.asarray(y).astype(str)
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y_array)
        self.estimator.fit(X, y_encoded, **fit_params)
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_encoded = np.asarray(self.estimator.predict(X)).astype(int)
        return self.encoder_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)

    @property
    def named_steps(self) -> Any:
        return self.estimator.named_steps

    @property
    def steps(self) -> Any:
        return self.estimator.steps


# --------------------------------------------------------------------------- #
# Pipeline builders
# --------------------------------------------------------------------------- #


def build_logistic_regression_pipeline() -> Pipeline:
    """Hybrid pipeline with a Logistic Regression classifier (binary)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Hybrid pipeline with a Random Forest classifier (binary)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=25,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost_pipeline() -> LabelEncoderForBinary:
    """Hybrid pipeline with an XGBoost classifier (binary, label-encoded)."""
    _LOGGER.warning(
        "XGBoost requires integer-encoded labels; wrapping pipeline in "
        "LabelEncoderForBinary for transparent encode/decode."
    )
    inner = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=_SCALE_POS_WEIGHT,
                    n_jobs=-1,
                    random_state=42,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return LabelEncoderForBinary(inner)


def get_all_pipelines() -> dict[str, Any]:
    """Return all three hybrid pipelines keyed by display name."""
    return {
        "Logistic Regression": build_logistic_regression_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
        "XGBoost": build_xgboost_pipeline(),
    }


# --------------------------------------------------------------------------- #
# Ablation-study helpers (kept here for Phase 4 to import)
# --------------------------------------------------------------------------- #


def build_text_only_preprocessor() -> ColumnTransformer:
    text = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
    )
    return ColumnTransformer(
        transformers=[("text", text, TEXT_FEATURE)],
        sparse_threshold=0.3,
        n_jobs=-1,
        verbose_feature_names_out=True,
    )


def build_structured_only_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        sparse_threshold=0.3,
        n_jobs=-1,
        verbose_feature_names_out=True,
    )


def build_categorical_only_preprocessor() -> ColumnTransformer:
    """One-hot categorical branch alone: repository, workflow, branch, event.

    This is the configuration that tests the thesis's central claim. If a model
    given nothing but categorical identity is competitive with the full hybrid,
    then the system is largely recognising which repositories and workflows are
    failure-prone rather than reading anything from the content of a commit.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.3,
        n_jobs=-1,
        verbose_feature_names_out=True,
    )


def build_xgboost_with_preprocessor(
    preprocessor: ColumnTransformer,
) -> LabelEncoderForBinary:
    """Build an XGBoost pipeline around a custom preprocessor.

    Used by the Phase 4 ablation study to swap in text-only or
    structured-only feature subsets while keeping classifier config identical.
    """
    inner = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=_SCALE_POS_WEIGHT,
                    n_jobs=-1,
                    random_state=42,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return LabelEncoderForBinary(inner)


# --------------------------------------------------------------------------- #
# Feature / target preparation
# --------------------------------------------------------------------------- #


def prepare_features_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` slicing ``df`` to the columns expected by the pipeline.

    Light coercion is applied so the dataframe is robust to the CSV round
    trip between Phase 2.5 and Phase 3 (binary flags may come back as float,
    text as NaN, etc.).
    """
    missing = [c for c in ALL_FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str)
    for column in BINARY_FEATURES:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        )

    x = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(str)
    return x, y


__all__ = [
    "LabelEncoderForBinary",
    "build_logistic_regression_pipeline",
    "build_preprocessor",
    "build_random_forest_pipeline",
    "build_categorical_only_preprocessor",
    "build_structured_only_preprocessor",
    "build_text_only_preprocessor",
    "build_xgboost_pipeline",
    "build_xgboost_with_preprocessor",
    "get_all_pipelines",
    "prepare_features_targets",
]
