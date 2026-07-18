"""Model development utilities for the frozen SUPPORT2 phenomenon test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PatientSplit:
    """Patient-level train, validation, and test indices."""

    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    stratification_method: str


@dataclass
class LogisticSelectionResult:
    """Selected frozen logistic model and model-selection metadata."""

    model: Pipeline
    selection_table: pd.DataFrame
    best_c: float
    numeric_features: list[str]
    categorical_features: list[str]


def create_feature_manifest(
    data: pd.DataFrame,
    candidate_columns: Sequence[str],
) -> pd.DataFrame:
    """Inspect candidate features and decide which are usable.

    Features are excluded when they are:

    - Missing from the input frame.
    - Completely missing.
    - Constant after excluding missing values.
    - Explicit identifier columns.
    """

    identifier_names = {
        "id",
        "patient_id",
        "source_id",
        "subject_id",
        "hadm_id",
        "stay_id",
    }

    rows: list[dict[str, object]] = []

    for column in candidate_columns:
        if column not in data.columns:
            rows.append(
                {
                    "feature": column,
                    "included": False,
                    "reason": "missing_from_frame",
                    "dtype": None,
                    "feature_type": None,
                    "missing_count": None,
                    "missing_fraction": None,
                    "n_unique_nonmissing": None,
                }
            )
            continue

        series = data[column]
        missing_count = int(series.isna().sum())
        missing_fraction = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))

        if str(column).lower() in identifier_names:
            included = False
            reason = "identifier"
        elif missing_count == len(data):
            included = False
            reason = "all_missing"
        elif n_unique <= 1:
            included = False
            reason = "constant"
        else:
            included = True
            reason = "included"

        if is_numeric_dtype(series) and not is_bool_dtype(series):
            feature_type = "numeric"
        else:
            feature_type = "categorical"

        rows.append(
            {
                "feature": column,
                "included": included,
                "reason": reason,
                "dtype": str(series.dtype),
                "feature_type": feature_type,
                "missing_count": missing_count,
                "missing_fraction": missing_fraction,
                "n_unique_nonmissing": n_unique,
            }
        )

    return pd.DataFrame(rows)


def _candidate_hospital_column(
    data: pd.DataFrame,
) -> str | None:
    """Find a plausible low-cardinality hospital/site column."""

    candidates = (
        "hosp",
        "hospital",
        "hospital_id",
        "site",
        "site_id",
        "center",
        "center_id",
    )

    for column in candidates:
        if column not in data.columns:
            continue

        n_unique = data[column].nunique(dropna=True)

        if 2 <= n_unique <= 25:
            return column

    return None


def _combined_strata(
    data: pd.DataFrame,
    outcome: pd.Series,
) -> tuple[pd.Series, str]:
    """Use outcome plus hospital strata when sufficiently populated."""

    hospital_column = _candidate_hospital_column(data)

    if hospital_column is None:
        return (
            outcome.astype(str),
            "outcome_only",
        )

    hospital = (
        data[hospital_column]
        .astype("string")
        .fillna("missing")
    )

    combined = (
        outcome.astype(str)
        + "|"
        + hospital.astype(str)
    )

    counts = combined.value_counts()

    if (
        combined.nunique() <= 25
        and not counts.empty
        and counts.min() >= 8
    ):
        return (
            combined,
            f"outcome_plus_{hospital_column}",
        )

    return (
        outcome.astype(str),
        "outcome_only",
    )


def create_patient_split(
    data: pd.DataFrame,
    outcome: pd.Series,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_seed: int = 2027,
) -> PatientSplit:
    """Create a reproducible patient-level 60/20/20 split."""

    fractions = (
        train_fraction
        + validation_fraction
        + test_fraction
    )

    if not np.isclose(fractions, 1.0):
        raise ValueError(
            "Train, validation, and test fractions must sum to 1"
        )

    if len(data) != len(outcome):
        raise ValueError(
            "Data and outcome must contain the same number of rows"
        )

    indices = np.arange(
        len(data),
        dtype=int,
    )

    strata, method = _combined_strata(
        data,
        outcome.reset_index(drop=True),
    )

    try:
        train_validation_index, test_index = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_seed,
            shuffle=True,
            stratify=strata,
        )

        relative_validation_fraction = (
            validation_fraction
            / (
                train_fraction
                + validation_fraction
            )
        )

        train_index, validation_index = train_test_split(
            train_validation_index,
            test_size=relative_validation_fraction,
            random_state=random_seed + 1,
            shuffle=True,
            stratify=strata.iloc[
                train_validation_index
            ],
        )
    except ValueError:
        # Guaranteed fallback for unusually sparse site/outcome strata.
        method = "outcome_only_fallback"
        strata = outcome.astype(str).reset_index(drop=True)

        train_validation_index, test_index = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_seed,
            shuffle=True,
            stratify=strata,
        )

        relative_validation_fraction = (
            validation_fraction
            / (
                train_fraction
                + validation_fraction
            )
        )

        train_index, validation_index = train_test_split(
            train_validation_index,
            test_size=relative_validation_fraction,
            random_state=random_seed + 1,
            shuffle=True,
            stratify=strata.iloc[
                train_validation_index
            ],
        )

    train_set = set(train_index.tolist())
    validation_set = set(validation_index.tolist())
    test_set = set(test_index.tolist())

    if train_set & validation_set:
        raise RuntimeError(
            "Train and validation splits overlap"
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train and test splits overlap"
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Validation and test splits overlap"
        )

    if (
        len(train_set)
        + len(validation_set)
        + len(test_set)
        != len(data)
    ):
        raise RuntimeError(
            "Split indices do not cover every patient exactly once"
        )

    return PatientSplit(
        train_index=np.sort(train_index),
        validation_index=np.sort(validation_index),
        test_index=np.sort(test_index),
        stratification_method=method,
    )


def infer_feature_types(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Separate usable features into numeric and categorical lists."""

    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in feature_columns:
        if (
            is_numeric_dtype(data[column])
            and not is_bool_dtype(data[column])
        ):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    if not numeric_features and not categorical_features:
        raise ValueError(
            "No usable model features were supplied"
        )

    return (
        numeric_features,
        categorical_features,
    )


def build_logistic_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    c_value: float,
    random_seed: int,
) -> Pipeline:
    """Construct a leakage-safe preprocessing and logistic pipeline."""

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                list(numeric_features),
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=5,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(categorical_features),
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=random_seed,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def tune_and_refit_logistic(
    data: pd.DataFrame,
    outcome: pd.Series,
    split: PatientSplit,
    feature_columns: Sequence[str],
    *,
    c_grid: Sequence[float] = (
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ),
    random_seed: int = 2027,
) -> LogisticSelectionResult:
    """Tune C on validation log loss and refit on train plus validation."""

    numeric_features, categorical_features = infer_feature_types(
        data,
        feature_columns,
    )

    x_train = data.iloc[
        split.train_index
    ][list(feature_columns)]

    y_train = outcome.iloc[
        split.train_index
    ].astype(int)

    x_validation = data.iloc[
        split.validation_index
    ][list(feature_columns)]

    y_validation = outcome.iloc[
        split.validation_index
    ].astype(int)

    rows: list[dict[str, float]] = []

    for c_value in c_grid:
        model = build_logistic_pipeline(
            numeric_features,
            categorical_features,
            c_value=float(c_value),
            random_seed=random_seed,
        )

        model.fit(
            x_train,
            y_train,
        )

        probability = model.predict_proba(
            x_validation
        )[:, 1]

        rows.append(
            {
                "c_value": float(c_value),
                "validation_log_loss": float(
                    log_loss(
                        y_validation,
                        probability,
                        labels=[0, 1],
                    )
                ),
                "validation_brier": float(
                    brier_score_loss(
                        y_validation,
                        probability,
                    )
                ),
                "validation_roc_auc": float(
                    roc_auc_score(
                        y_validation,
                        probability,
                    )
                ),
                "validation_average_precision": float(
                    average_precision_score(
                        y_validation,
                        probability,
                    )
                ),
            }
        )

    selection_table = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "validation_log_loss",
                "validation_brier",
                "c_value",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    best_c = float(
        selection_table.loc[
            0,
            "c_value",
        ]
    )

    final_model = build_logistic_pipeline(
        numeric_features,
        categorical_features,
        c_value=best_c,
        random_seed=random_seed,
    )

    train_validation_index = np.concatenate(
        [
            split.train_index,
            split.validation_index,
        ]
    )

    final_model.fit(
        data.iloc[
            train_validation_index
        ][list(feature_columns)],
        outcome.iloc[
            train_validation_index
        ].astype(int),
    )

    return LogisticSelectionResult(
        model=final_model,
        selection_table=selection_table,
        best_c=best_c,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
