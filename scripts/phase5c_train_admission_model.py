"""Phase 5C-D: admission-time MIMIC-IV mortality model.

Trains and freezes a 30-day mortality prediction model on the
Phase 5C inpatient cohort.

Only information available at admission or from prior encounters
is used. Discharge destination, index length of stay, index ICU
utilization, restricted outcomes, and post-admission information
are excluded from model features.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from actionaudit.km import (
    reference_quantile_edges,
)


CORE_ACTIONS = [
    "home",
    "home_health",
    "skilled_nursing",
    "rehabilitation",
    "hospice",
    "long_term_or_assisted",
    "acute_hospital_transfer",
]

NUMERIC_FEATURES = [
    "age_at_admission",
    "admission_hour_sin",
    "admission_hour_cos",
    "admission_weekday_sin",
    "admission_weekday_cos",
    "prior_admission_count",
    "prior_emergency_or_urgent_count",
    "prior_total_los_days",
    "days_since_previous_discharge",
    "prior_icu_stay_count",
    "prior_icu_los_days",
    "has_prior_admission",
    "has_prior_icu",
]

CATEGORICAL_FEATURES = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "anchor_year_group",
]

FORBIDDEN_MODEL_FEATURES = {
    "discharge_location",
    "raw_discharge_location",
    "discharge_group",
    "primary_action",
    "analysis_action",
    "dischtime",
    "deathtime",
    "dod",
    "outcome",
    "observed",
    "hidden_event",
    "restricted_event",
    "restricted_time",
    "restricted_censored_by_discharge",
    "exit_time",
    "has_icu_stay",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and freeze the admission-time MIMIC-IV "
            "30-day mortality prediction model."
        )
    )

    parser.add_argument(
        "--cohort-dir",
        default="artifacts/phase5c_frozen_cohort",
    )

    parser.add_argument(
        "--mimic-root",
        default="data/raw/mimiciv/3.1",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase5c_admission_model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--decomposition-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use one configuration per model family for "
            "a preliminary smoke test."
        ),
    )

    return parser.parse_args()


def normalize_admission_type(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna(
            "UNKNOWN"
        )
        .astype(str)
        .str.strip()
        .str.upper()
    )


def assign_analysis_action(
    frame: pd.DataFrame,
) -> pd.Series:
    """Assign fixed action groups without using outcome results."""

    action = (
        frame[
            "action"
        ]
        .fillna(
            "no_exit_before_horizon"
        )
        .astype(str)
    )

    result = pd.Series(
        "no_exit_before_horizon",
        index=frame.index,
        dtype=object,
    )

    censored = frame[
        "observed"
    ].astype(int).eq(
        0
    )

    result.loc[
        censored
        & action.isin(
            CORE_ACTIONS
        )
    ] = action.loc[
        censored
        & action.isin(
            CORE_ACTIONS
        )
    ]

    result.loc[
        censored
        & ~action.isin(
            CORE_ACTIONS
        )
    ] = "other_exit"

    return result


def construct_prior_history(
    *,
    mimic_root: Path,
) -> pd.DataFrame:
    """Construct information available before each inpatient admission."""

    admissions_path = (
        mimic_root
        / "hosp"
        / "admissions.csv.gz"
    )

    icustays_path = (
        mimic_root
        / "icu"
        / "icustays.csv.gz"
    )

    if not admissions_path.exists():
        raise FileNotFoundError(
            f"Missing admissions table: {admissions_path}"
        )

    if not icustays_path.exists():
        raise FileNotFoundError(
            f"Missing ICU stays table: {icustays_path}"
        )

    admissions = pd.read_csv(
        admissions_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
        ],
        low_memory=False,
    )

    admissions[
        "admittime"
    ] = pd.to_datetime(
        admissions[
            "admittime"
        ],
        errors="coerce",
    )

    admissions[
        "dischtime"
    ] = pd.to_datetime(
        admissions[
            "dischtime"
        ],
        errors="coerce",
    )

    admission_type = (
        normalize_admission_type(
            admissions[
                "admission_type"
            ]
        )
    )

    observation_mask = (
        admission_type.str.contains(
            "OBSERVATION",
            regex=False,
        )
    )

    valid_time_mask = (
        admissions[
            "admittime"
        ].notna()
        & admissions[
            "dischtime"
        ].notna()
        & (
            admissions[
                "dischtime"
            ]
            >= admissions[
                "admittime"
            ]
        )
    )

    admissions = (
        admissions.loc[
            ~observation_mask
            & valid_time_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    admissions[
        "admission_type_normalized"
    ] = normalize_admission_type(
        admissions[
            "admission_type"
        ]
    )

    admissions[
        "los_days"
    ] = (
        admissions[
            "dischtime"
        ]
        - admissions[
            "admittime"
        ]
    ).dt.total_seconds() / 86400.0

    admissions[
        "is_emergency_or_urgent"
    ] = (
        admissions[
            "admission_type_normalized"
        ]
        .str.contains(
            "EMER",
            regex=False,
        )
        | admissions[
            "admission_type_normalized"
        ]
        .str.contains(
            "URGENT",
            regex=False,
        )
    ).astype(
        int
    )

    icustays = pd.read_csv(
        icustays_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "los",
        ],
        low_memory=False,
    )

    icustays[
        "los"
    ] = pd.to_numeric(
        icustays[
            "los"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    icu_by_admission = (
        icustays.groupby(
            [
                "subject_id",
                "hadm_id",
            ],
            as_index=False,
        )
        .agg(
            admission_icu_stay_count=(
                "stay_id",
                "nunique",
            ),
            admission_icu_los_days=(
                "los",
                "sum",
            ),
        )
    )

    admissions = admissions.merge(
        icu_by_admission,
        on=[
            "subject_id",
            "hadm_id",
        ],
        how="left",
        validate="one_to_one",
    )

    admissions[
        "admission_icu_stay_count"
    ] = (
        admissions[
            "admission_icu_stay_count"
        ]
        .fillna(
            0
        )
        .astype(int)
    )

    admissions[
        "admission_icu_los_days"
    ] = (
        admissions[
            "admission_icu_los_days"
        ]
        .fillna(
            0.0
        )
        .astype(float)
    )

    admissions = admissions.sort_values(
        [
            "subject_id",
            "admittime",
            "hadm_id",
        ]
    ).reset_index(
        drop=True
    )

    subject_group = admissions.groupby(
        "subject_id",
        sort=False,
    )

    admissions[
        "prior_admission_count"
    ] = subject_group.cumcount()

    admissions[
        "prior_total_los_days"
    ] = (
        subject_group[
            "los_days"
        ].cumsum()
        - admissions[
            "los_days"
        ]
    )

    admissions[
        "prior_emergency_or_urgent_count"
    ] = (
        subject_group[
            "is_emergency_or_urgent"
        ].cumsum()
        - admissions[
            "is_emergency_or_urgent"
        ]
    )

    admissions[
        "prior_icu_stay_count"
    ] = (
        subject_group[
            "admission_icu_stay_count"
        ].cumsum()
        - admissions[
            "admission_icu_stay_count"
        ]
    )

    admissions[
        "prior_icu_los_days"
    ] = (
        subject_group[
            "admission_icu_los_days"
        ].cumsum()
        - admissions[
            "admission_icu_los_days"
        ]
    )

    previous_discharge = subject_group[
        "dischtime"
    ].shift(
        1
    )

    admissions[
        "days_since_previous_discharge"
    ] = (
        admissions[
            "admittime"
        ]
        - previous_discharge
    ).dt.total_seconds() / 86400.0

    admissions[
        "days_since_previous_discharge"
    ] = admissions[
        "days_since_previous_discharge"
    ].clip(
        lower=0.0
    )

    admissions[
        "has_prior_admission"
    ] = admissions[
        "prior_admission_count"
    ].gt(
        0
    ).astype(
        int
    )

    admissions[
        "has_prior_icu"
    ] = admissions[
        "prior_icu_stay_count"
    ].gt(
        0
    ).astype(
        int
    )

    return admissions[
        [
            "subject_id",
            "hadm_id",
            "prior_admission_count",
            "prior_emergency_or_urgent_count",
            "prior_total_los_days",
            "days_since_previous_discharge",
            "prior_icu_stay_count",
            "prior_icu_los_days",
            "has_prior_admission",
            "has_prior_icu",
        ]
    ]


def construct_feature_frame(
    *,
    cohort_path: Path,
    mimic_root: Path,
) -> pd.DataFrame:
    cohort = pd.read_csv(
        cohort_path,
        low_memory=False,
    )

    history = construct_prior_history(
        mimic_root=mimic_root
    )

    frame = cohort.merge(
        history,
        on=[
            "subject_id",
            "hadm_id",
        ],
        how="left",
        validate="one_to_one",
    )

    frame[
        "admittime"
    ] = pd.to_datetime(
        frame[
            "admittime"
        ],
        errors="coerce",
    )

    admission_hour = (
        frame[
            "admittime"
        ].dt.hour
        + frame[
            "admittime"
        ].dt.minute
        / 60.0
    )

    frame[
        "admission_hour_sin"
    ] = np.sin(
        2.0
        * np.pi
        * admission_hour
        / 24.0
    )

    frame[
        "admission_hour_cos"
    ] = np.cos(
        2.0
        * np.pi
        * admission_hour
        / 24.0
    )

    admission_weekday = (
        frame[
            "admittime"
        ].dt.dayofweek.astype(
            float
        )
    )

    frame[
        "admission_weekday_sin"
    ] = np.sin(
        2.0
        * np.pi
        * admission_weekday
        / 7.0
    )

    frame[
        "admission_weekday_cos"
    ] = np.cos(
        2.0
        * np.pi
        * admission_weekday
        / 7.0
    )

    frame[
        "analysis_action"
    ] = assign_analysis_action(
        frame
    )

    return frame


def build_logistic_model(
    *,
    c_value: float,
    seed: int,
) -> Pipeline:
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

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )

    classifier = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        tol=1e-6,
        random_state=seed,
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


def build_gbm_model(
    *,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
    seed: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )

    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=300,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=50,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=25,
        random_state=seed,
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


def model_candidates(
    *,
    fast: bool,
) -> list[
    tuple[
        str,
        dict[str, float | int],
    ]
]:
    if fast:
        return [
            (
                "logistic",
                {
                    "C": 1.0,
                },
            ),
            (
                "gbm",
                {
                    "learning_rate": 0.05,
                    "max_leaf_nodes": 31,
                    "l2_regularization": 1.0,
                },
            ),
        ]

    candidates: list[
        tuple[
            str,
            dict[str, float | int],
        ]
    ] = []

    for c_value in [
        0.1,
        1.0,
        10.0,
    ]:
        candidates.append(
            (
                "logistic",
                {
                    "C": c_value,
                },
            )
        )

    for learning_rate in [
        0.03,
        0.05,
    ]:
        for max_leaf_nodes in [
            15,
            31,
        ]:
            for l2_regularization in [
                1.0,
                10.0,
            ]:
                candidates.append(
                    (
                        "gbm",
                        {
                            "learning_rate": (
                                learning_rate
                            ),
                            "max_leaf_nodes": (
                                max_leaf_nodes
                            ),
                            "l2_regularization": (
                                l2_regularization
                            ),
                        },
                    )
                )

    return candidates


def construct_model(
    *,
    family: str,
    parameters: dict[str, float | int],
    seed: int,
) -> Pipeline:
    if family == "logistic":
        return build_logistic_model(
            c_value=float(
                parameters[
                    "C"
                ]
            ),
            seed=seed,
        )

    if family == "gbm":
        return build_gbm_model(
            learning_rate=float(
                parameters[
                    "learning_rate"
                ]
            ),
            max_leaf_nodes=int(
                parameters[
                    "max_leaf_nodes"
                ]
            ),
            l2_regularization=float(
                parameters[
                    "l2_regularization"
                ]
            ),
            seed=seed,
        )

    raise ValueError(
        f"Unknown family: {family}"
    )


def probability_metrics(
    outcome: pd.Series | np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(
        outcome,
        dtype=int,
    )

    probability = np.clip(
        np.asarray(
            prediction,
            dtype=float,
        ),
        1e-8,
        1.0 - 1e-8,
    )

    return {
        "log_loss": float(
            log_loss(
                y,
                probability,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y,
                probability,
            )
        ),
    }


def select_model(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fast: bool,
    seed: int,
) -> tuple[
    str,
    dict[str, float | int],
    pd.DataFrame,
]:
    model_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    rows: list[
        dict[str, object]
    ] = []

    best_key: tuple[
        float,
        float,
    ] | None = None

    best_family: str | None = None
    best_parameters: dict[
        str,
        float | int
    ] | None = None

    for candidate_index, (
        family,
        parameters,
    ) in enumerate(
        model_candidates(
            fast=fast
        )
    ):
        model = construct_model(
            family=family,
            parameters=parameters,
            seed=(
                seed
                + candidate_index
            ),
        )

        model.fit(
            train[
                model_features
            ],
            train[
                "outcome"
            ].astype(int),
        )

        probability = model.predict_proba(
            validation[
                model_features
            ]
        )[:, 1]

        metrics = probability_metrics(
            validation[
                "outcome"
            ],
            probability,
        )

        row = {
            "candidate_index": (
                candidate_index
            ),
            "family": family,
            "parameters": json.dumps(
                parameters,
                sort_keys=True,
            ),
            **{
                f"validation_{key}": value
                for key, value in metrics.items()
            },
        }

        rows.append(
            row
        )

        candidate_key = (
            metrics[
                "log_loss"
            ],
            metrics[
                "brier"
            ],
        )

        if (
            best_key is None
            or candidate_key
            < best_key
        ):
            best_key = candidate_key
            best_family = family
            best_parameters = dict(
                parameters
            )

    if (
        best_family is None
        or best_parameters is None
    ):
        raise RuntimeError(
            "No valid prediction model was selected"
        )

    selection = pd.DataFrame(
        rows
    )

    selected_parameters_text = json.dumps(
        best_parameters,
        sort_keys=True,
    )

    selection[
        "selected"
    ] = (
        selection[
            "family"
        ].eq(
            best_family
        )
        & selection[
            "parameters"
        ].eq(
            selected_parameters_text
        )
    )

    return (
        best_family,
        best_parameters,
        selection,
    )


def calibration_table(
    frame: pd.DataFrame,
    *,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    float,
]:
    edges = reference_quantile_edges(
        frame[
            "prediction"
        ].to_numpy(),
        n_bins,
    )

    bin_index = (
        np.digitize(
            frame[
                "prediction"
            ].to_numpy(),
            edges[
                1:-1
            ],
            right=True,
        )
        + 1
    )

    working = frame.copy()

    working[
        "risk_bin"
    ] = bin_index

    summary = (
        working.groupby(
            "risk_bin",
            as_index=False,
            sort=True,
        )
        .agg(
            n=(
                "subject_id",
                "size",
            ),
            prediction_mean=(
                "prediction",
                "mean",
            ),
            prediction_min=(
                "prediction",
                "min",
            ),
            prediction_max=(
                "prediction",
                "max",
            ),
            outcome_rate=(
                "outcome",
                "mean",
            ),
            restricted_observed_fraction=(
                "observed",
                "mean",
            ),
            hidden_event_rate=(
                "hidden_event",
                "mean",
            ),
        )
    )

    summary[
        "signed_calibration_gap"
    ] = (
        summary[
            "outcome_rate"
        ]
        - summary[
            "prediction_mean"
        ]
    )

    summary[
        "absolute_calibration_gap"
    ] = summary[
        "signed_calibration_gap"
    ].abs()

    summary[
        "weight"
    ] = (
        summary[
            "n"
        ]
        / summary[
            "n"
        ].sum()
    )

    ici = float(
        (
            summary[
                "weight"
            ]
            * summary[
                "absolute_calibration_gap"
            ]
        ).sum()
    )

    return (
        summary,
        ici,
    )


def action_test_summary(
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    exits = test_predictions.loc[
        test_predictions[
            "observed"
        ].eq(
            0
        )
    ].copy()

    return (
        exits.groupby(
            "analysis_action",
            as_index=False,
            sort=False,
        )
        .agg(
            n=(
                "subject_id",
                "size",
            ),
            hidden_events=(
                "hidden_event",
                "sum",
            ),
            hidden_event_rate=(
                "hidden_event",
                "mean",
            ),
            mean_prediction=(
                "prediction",
                "mean",
            ),
            mean_exit_time=(
                "exit_time",
                "mean",
            ),
        )
        .sort_values(
            "n",
            ascending=False,
        )
    )


def make_stratification_label(
    frame: pd.DataFrame,
) -> pd.Series:
    label = (
        frame[
            "analysis_action"
        ].astype(str)
        + "__y"
        + frame[
            "outcome"
        ].astype(int)
        .astype(str)
    )

    counts = label.value_counts()

    rare = set(
        counts.loc[
            counts
            < 2
        ].index
    )

    if rare:
        label = frame[
            "outcome"
        ].astype(int).astype(str)

    return label


def split_cohort(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if (
        validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction
        + test_fraction
        >= 1
    ):
        raise ValueError(
            "Validation and test fractions must be "
            "positive and sum to less than one"
        )

    development, test = train_test_split(
        frame,
        test_size=test_fraction,
        random_state=seed,
        stratify=make_stratification_label(
            frame
        ),
    )

    relative_validation_fraction = (
        validation_fraction
        / (
            1.0
            - test_fraction
        )
    )

    train, validation = train_test_split(
        development,
        test_size=(
            relative_validation_fraction
        ),
        random_state=seed + 1,
        stratify=make_stratification_label(
            development
        ),
    )

    result = frame.copy()

    result[
        "split"
    ] = None

    result.loc[
        result[
            "subject_id"
        ].isin(
            train[
                "subject_id"
            ]
        ),
        "split",
    ] = "train"

    result.loc[
        result[
            "subject_id"
        ].isin(
            validation[
                "subject_id"
            ]
        ),
        "split",
    ] = "validation"

    result.loc[
        result[
            "subject_id"
        ].isin(
            test[
                "subject_id"
            ]
        ),
        "split",
    ] = "test"

    if result[
        "split"
    ].isna().any():
        raise RuntimeError(
            "Some patients were not assigned to a split"
        )

    return result


def build_validation_checks(
    *,
    split_frame: pd.DataFrame,
    test_predictions: pd.DataFrame,
    action_summary: pd.DataFrame,
    validation_selection: pd.DataFrame,
    validation_null_log_loss: float,
    selected_validation_log_loss: float,
    test_metrics: dict[str, float],
) -> tuple[
    pd.DataFrame,
    str,
]:
    feature_set = set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    leakage_overlap = sorted(
        feature_set.intersection(
            FORBIDDEN_MODEL_FEATURES
        )
    )

    split_subject_overlap = (
        split_frame.groupby(
            "subject_id"
        )[
            "split"
        ]
        .nunique()
        .gt(
            1
        )
        .sum()
    )

    core_action_rows = (
        action_summary.loc[
            action_summary[
                "analysis_action"
            ].isin(
                CORE_ACTIONS
            )
        ]
    )

    present_core_actions = set(
        core_action_rows[
            "analysis_action"
        ]
    )

    minimum_core_n = int(
        core_action_rows[
            "n"
        ].min()
    ) if not core_action_rows.empty else 0

    minimum_core_hidden = int(
        core_action_rows[
            "hidden_events"
        ].min()
    ) if not core_action_rows.empty else 0

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "no forbidden post-admission model features"
                ),
                "value": (
                    ",".join(
                        leakage_overlap
                    )
                    if leakage_overlap
                    else "none"
                ),
                "threshold": "none",
                "comparison": "==",
                "passed": (
                    len(
                        leakage_overlap
                    )
                    == 0
                ),
            },
            {
                "check": (
                    "patients do not cross data splits"
                ),
                "value": int(
                    split_subject_overlap
                ),
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    split_subject_overlap
                    == 0
                ),
            },
            {
                "check": (
                    "test set contains at least 25000 patients"
                ),
                "value": len(
                    test_predictions
                ),
                "threshold": 25000,
                "comparison": ">=",
                "passed": (
                    len(
                        test_predictions
                    )
                    >= 25000
                ),
            },
            {
                "check": (
                    "selected model beats validation null log loss"
                ),
                "value": (
                    selected_validation_log_loss
                ),
                "threshold": (
                    validation_null_log_loss
                ),
                "comparison": "<",
                "passed": (
                    selected_validation_log_loss
                    < validation_null_log_loss
                ),
            },
            {
                "check": (
                    "test ROC-AUC exceeds 0.60"
                ),
                "value": test_metrics[
                    "roc_auc"
                ],
                "threshold": 0.60,
                "comparison": ">",
                "passed": (
                    test_metrics[
                        "roc_auc"
                    ]
                    > 0.60
                ),
            },
            {
                "check": (
                    "test predictions have meaningful spread"
                ),
                "value": float(
                    test_predictions[
                        "prediction"
                    ].std()
                ),
                "threshold": 0.01,
                "comparison": ">",
                "passed": (
                    test_predictions[
                        "prediction"
                    ].std()
                    > 0.01
                ),
            },
            {
                "check": (
                    "all seven fixed core actions occur in test"
                ),
                "value": len(
                    present_core_actions
                ),
                "threshold": len(
                    CORE_ACTIONS
                ),
                "comparison": "==",
                "passed": (
                    present_core_actions
                    == set(
                        CORE_ACTIONS
                    )
                ),
            },
            {
                "check": (
                    "each core action has 100 test exits"
                ),
                "value": minimum_core_n,
                "threshold": 100,
                "comparison": ">=",
                "passed": (
                    minimum_core_n
                    >= 100
                ),
            },
            {
                "check": (
                    "each core action has five hidden test deaths"
                ),
                "value": minimum_core_hidden,
                "threshold": 5,
                "comparison": ">=",
                "passed": (
                    minimum_core_hidden
                    >= 5
                ),
            },
            {
                "check": (
                    "exactly one model configuration selected"
                ),
                "value": int(
                    validation_selection[
                        "selected"
                    ].sum()
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    validation_selection[
                        "selected"
                    ].sum()
                    == 1
                ),
            },
        ]
    )

    status = (
        "phase5c_admission_prediction_model_frozen"
        if checks[
            "passed"
        ].all()
        else "phase5c_admission_model_requires_review"
    )

    return (
        checks,
        status,
    )


def main() -> None:
    args = parse_arguments()

    cohort_dir = Path(
        args.cohort_dir
    )

    mimic_root = Path(
        args.mimic_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_path = (
        cohort_dir
        / "primary_inpatient_cohort.csv.gz"
    )

    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Missing frozen cohort: {cohort_path}"
        )

    print(
        "Constructing admission-time features..."
    )

    frame = construct_feature_frame(
        cohort_path=cohort_path,
        mimic_root=mimic_root,
    )

    print(
        f"  patients={len(frame):,}"
    )

    print(
        f"  event_rate="
        f"{frame['outcome'].mean():.6f}"
    )

    split_frame = split_cohort(
        frame,
        validation_fraction=(
            args.validation_fraction
        ),
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    train = (
        split_frame.loc[
            split_frame[
                "split"
            ].eq(
                "train"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation = (
        split_frame.loc[
            split_frame[
                "split"
            ].eq(
                "validation"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        split_frame.loc[
            split_frame[
                "split"
            ].eq(
                "test"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Frozen split:"
    )

    print(
        f"  train={len(train):,}, "
        f"validation={len(validation):,}, "
        f"test={len(test):,}"
    )

    (
        selected_family,
        selected_parameters,
        model_selection,
    ) = select_model(
        train=train,
        validation=validation,
        fast=args.fast,
        seed=args.seed + 100,
    )

    selected_row = (
        model_selection.loc[
            model_selection[
                "selected"
            ]
        ]
        .iloc[
            0
        ]
    )

    selected_validation_log_loss = float(
        selected_row[
            "validation_log_loss"
        ]
    )

    train_event_rate = float(
        train[
            "outcome"
        ].mean()
    )

    validation_null_probability = np.full(
        len(
            validation
        ),
        train_event_rate,
        dtype=float,
    )

    validation_null_log_loss = float(
        log_loss(
            validation[
                "outcome"
            ].astype(int),
            validation_null_probability,
            labels=[
                0,
                1,
            ],
        )
    )

    print(
        "Validation-selected model:"
    )

    print(
        f"  family={selected_family}"
    )

    print(
        f"  parameters={selected_parameters}"
    )

    development = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    model = construct_model(
        family=selected_family,
        parameters=selected_parameters,
        seed=args.seed + 10000,
    )

    model_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    model.fit(
        development[
            model_features
        ],
        development[
            "outcome"
        ].astype(int),
    )

    test_probability = model.predict_proba(
        test[
            model_features
        ]
    )[:, 1]

    test_predictions = test[
        [
            "subject_id",
            "hadm_id",
            "outcome",
            "observed",
            "hidden_event",
            "restricted_event",
            "restricted_time",
            "restricted_censored_by_discharge",
            "exit_time",
            "action",
            "primary_action",
            "analysis_action",
            "has_icu_stay",
        ]
    ].copy()

    test_predictions[
        "prediction"
    ] = test_probability

    test_metrics = probability_metrics(
        test_predictions[
            "outcome"
        ],
        test_predictions[
            "prediction"
        ].to_numpy(),
    )

    (
        calibration_10,
        ici_10,
    ) = calibration_table(
        test_predictions,
        n_bins=args.calibration_bins,
    )

    (
        calibration_5,
        ici_5,
    ) = calibration_table(
        test_predictions,
        n_bins=args.decomposition_bins,
    )

    test_metrics[
        "reference_ici_10_bins"
    ] = ici_10

    test_metrics[
        "reference_ici_5_bins"
    ] = ici_5

    test_metrics[
        "mean_prediction"
    ] = float(
        test_predictions[
            "prediction"
        ].mean()
    )

    test_metrics[
        "event_rate"
    ] = float(
        test_predictions[
            "outcome"
        ].mean()
    )

    test_metrics[
        "overall_signed_calibration_gap"
    ] = (
        test_metrics[
            "event_rate"
        ]
        - test_metrics[
            "mean_prediction"
        ]
    )

    action_summary = action_test_summary(
        test_predictions
    )

    (
        checks,
        status,
    ) = build_validation_checks(
        split_frame=split_frame,
        test_predictions=test_predictions,
        action_summary=action_summary,
        validation_selection=(
            model_selection
        ),
        validation_null_log_loss=(
            validation_null_log_loss
        ),
        selected_validation_log_loss=(
            selected_validation_log_loss
        ),
        test_metrics=test_metrics,
    )

    if args.fast:
        preliminary_status = status
        status = (
            "phase5c_admission_model_smoke_completed"
        )

        checks[
            "smoke_run"
        ] = True
    else:
        preliminary_status = None
        checks[
            "smoke_run"
        ] = False

    split_frame[
        [
            "subject_id",
            "hadm_id",
            "split",
            "outcome",
            "analysis_action",
        ]
    ].to_csv(
        output_dir
        / "split_manifest.csv.gz",
        index=False,
        compression="gzip",
    )

    model_selection.to_csv(
        output_dir
        / "model_selection.csv",
        index=False,
    )

    test_predictions.to_csv(
        output_dir
        / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    calibration_10.to_csv(
        output_dir
        / "reference_calibration_10_bins.csv",
        index=False,
    )

    calibration_5.to_csv(
        output_dir
        / "reference_calibration_5_bins.csv",
        index=False,
    )

    action_summary.to_csv(
        output_dir
        / "test_action_summary.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "model_freeze_checks.csv",
        index=False,
    )

    feature_manifest = pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_type": "numeric",
                "available_at_admission": True,
                "included": True,
            }
            for feature in NUMERIC_FEATURES
        ]
        + [
            {
                "feature": feature,
                "feature_type": "categorical",
                "available_at_admission": True,
                "included": True,
            }
            for feature in CATEGORICAL_FEATURES
        ]
    )

    feature_manifest.to_csv(
        output_dir
        / "feature_manifest.csv",
        index=False,
    )

    joblib.dump(
        model,
        output_dir
        / "selected_model.joblib",
    )

    metadata: dict[str, Any] = {
        "phase": "5C-D",
        "experiment": (
            "Frozen admission-time MIMIC-IV "
            "30-day mortality prediction model"
        ),
        "cohort_n": int(
            len(
                split_frame
            )
        ),
        "train_n": int(
            len(
                train
            )
        ),
        "validation_n": int(
            len(
                validation
            )
        ),
        "test_n": int(
            len(
                test
            )
        ),
        "split_seed": int(
            args.seed
        ),
        "selected_family": (
            selected_family
        ),
        "selected_parameters": (
            selected_parameters
        ),
        "validation_null_log_loss": (
            validation_null_log_loss
        ),
        "selected_validation_log_loss": (
            selected_validation_log_loss
        ),
        "test_metrics": (
            test_metrics
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "fixed_core_actions": (
            CORE_ACTIONS
        ),
        "residual_action_group": (
            "other_exit"
        ),
        "post_admission_features_excluded": sorted(
            FORBIDDEN_MODEL_FEATURES
        ),
        "fast": bool(
            args.fast
        ),
        "preliminary_status": (
            preliminary_status
        ),
        "status": status,
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
        },
    }

    (
        output_dir
        / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5c_model_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nModel selection"
    )

    print(
        model_selection.to_string(
            index=False
        )
    )

    print(
        "\nTest metrics"
    )

    for key, value in test_metrics.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nTest action summary"
    )

    print(
        action_summary.to_string(
            index=False
        )
    )

    print(
        "\nModel freeze checks"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
