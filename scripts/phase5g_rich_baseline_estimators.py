"""Phase 5G-B: rich admission-time adjusted practitioner estimators.

Uses the exact 21 frozen Phase 5C admission-time features.

The linked reference outcome is never used during nuisance fitting.
It is used only after estimation to evaluate curve recovery.

Methods:
  * restricted complete case;
  * grouped Kaplan-Meier;
  * score-only OR, IPCW, and AIPW from Phase 5G-A;
  * rich baseline pooled survival OR;
  * rich baseline IPCW at multiple propensity clips;
  * rich baseline AIPW at multiple propensity clips.

Inference uses a patient bootstrap conditional on the repeated
cross-fitted nuisance predictions.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDENTIFIERS = [
    "subject_id",
    "hadm_id",
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

PREDICTION_FEATURES = [
    "prediction_logit",
    "prediction_logit_squared",
    "prediction_logit_cubed",
]

FORBIDDEN_MODEL_FEATURES = {
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
    "dischtime",
    "dod",
    "deathtime",
    "discharge_location",
}

PROPENSITY_CLIPS = [
    0.01,
    0.025,
    0.05,
]


def clip_label(
    value: float,
) -> str:
    return (
        f"{value:.3f}"
        .rstrip("0")
        .rstrip(".")
        .replace(
            ".",
            "p",
        )
    )


def method_columns() -> list[str]:
    methods = [
        "reference_oracle",
        "restricted_complete_case",
        "grouped_km",
        "score_pooled_survival_or",
        "score_ipcw",
        "score_aipw",
        "rich_pooled_survival_or",
    ]

    for clip in PROPENSITY_CLIPS:
        label = clip_label(
            clip
        )

        methods.extend(
            [
                f"rich_ipcw_clip_{label}",
                f"rich_aipw_clip_{label}",
            ]
        )

    return methods


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark rich baseline-adjusted "
            "practitioner estimators."
        )
    )

    parser.add_argument(
        "--rich-input",
        default=(
            "artifacts/phase5g_baseline_features/"
            "rich_test_analysis.csv.gz"
        ),
    )

    parser.add_argument(
        "--feature-manifest",
        default=(
            "artifacts/phase5g_baseline_features/"
            "baseline_feature_manifest.csv"
        ),
    )

    parser.add_argument(
        "--score-nuisance",
        default=(
            "artifacts/phase5g_practitioner_estimators/"
            "cross_fitted_nuisance_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--score-summary",
        default=(
            "artifacts/phase5g_practitioner_estimators/"
            "estimator_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5g_rich_baseline_estimators"
        ),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--crossfit-repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--model-c",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing columns: {missing}"
        )


def validate_unique(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    duplicated = frame.duplicated(
        IDENTIFIERS,
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            f"{label} has duplicate admission keys"
        )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float64,
        )


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
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
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="MISSING",
                ),
            ),
            (
                "one_hot",
                make_one_hot_encoder(),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
                + PREDICTION_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
        remainder="drop",
    )


def load_inputs(
    *,
    rich_path: Path,
    manifest_path: Path,
    score_nuisance_path: Path,
    score_summary_path: Path,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    rich = pd.read_csv(
        rich_path,
        low_memory=False,
    )

    manifest = pd.read_csv(
        manifest_path
    )

    score_nuisance = pd.read_csv(
        score_nuisance_path,
        low_memory=False,
    )

    score_summary = pd.read_csv(
        score_summary_path
    )

    required_analysis = (
        IDENTIFIERS
        + [
            "prediction",
            "outcome",
            "observed",
            "restricted_event",
            "restricted_time",
        ]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    require_columns(
        rich,
        required_analysis,
        label="Rich analysis input",
    )

    require_columns(
        score_nuisance,
        IDENTIFIERS
        + [
            "outcome_regression_risk",
            "observation_probability",
        ],
        label="Score-only nuisance input",
    )

    validate_unique(
        rich,
        label="Rich analysis input",
    )

    validate_unique(
        score_nuisance,
        label="Score-only nuisance input",
    )

    score_subset = score_nuisance[
        IDENTIFIERS
        + [
            "outcome_regression_risk",
            "observation_probability",
        ]
    ].rename(
        columns={
            "outcome_regression_risk": (
                "score_outcome_regression_risk"
            ),
            "observation_probability": (
                "score_observation_probability"
            ),
        }
    )

    data = rich.merge(
        score_subset,
        on=IDENTIFIERS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_score_nuisance = int(
        data[
            "_merge"
        ].ne(
            "both"
        ).sum()
    )

    data = data.drop(
        columns=[
            "_merge",
        ]
    )

    if unmatched_score_nuisance:
        raise ValueError(
            "Some rich rows did not match the "
            "Phase 5G-A nuisance predictions"
        )

    prediction = np.clip(
        pd.to_numeric(
            data[
                "prediction"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        ),
        1e-6,
        1.0 - 1e-6,
    )

    prediction_score = logit(
        prediction
    )

    data[
        "prediction"
    ] = prediction

    data[
        "prediction_logit"
    ] = prediction_score

    data[
        "prediction_logit_squared"
    ] = prediction_score**2

    data[
        "prediction_logit_cubed"
    ] = prediction_score**3

    edges = np.unique(
        np.quantile(
            prediction,
            np.linspace(
                0.0,
                1.0,
                n_bins + 1,
            ),
        )
    )

    if len(
        edges
    ) != n_bins + 1:
        raise ValueError(
            "Prediction values do not support "
            "the requested number of bins"
        )

    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    data[
        "risk_bin"
    ] = pd.cut(
        prediction,
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(
        int
    )

    expected_manifest = set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    actual_manifest = set(
        manifest.loc[
            manifest[
                "nuisance_model_feature"
            ].astype(
                bool
            ),
            "feature",
        ]
    )

    if expected_manifest != actual_manifest:
        raise ValueError(
            "Baseline feature manifest differs "
            "from the frozen 21-feature specification"
        )

    return (
        data.reset_index(
            drop=True
        ),
        manifest,
        score_summary,
        edges,
    )


def assign_subject_folds(
    subject_id: pd.Series,
    *,
    seed: int,
) -> np.ndarray:
    subjects = pd.Index(
        pd.unique(
            subject_id
        )
    )

    rng = np.random.default_rng(
        seed
    )

    shuffled = subjects.to_numpy(
        copy=True
    )

    rng.shuffle(
        shuffled
    )

    lookup = {
        subject: int(
            index % 2
        )
        for index, subject in enumerate(
            shuffled
        )
    }

    return subject_id.map(
        lookup
    ).to_numpy(
        dtype=int
    )


def time_design(
    prediction: np.ndarray,
    day: np.ndarray,
    *,
    horizon: int,
) -> sparse.csr_matrix:
    prediction_score = logit(
        np.clip(
            np.asarray(
                prediction,
                dtype=float,
            ),
            1e-6,
            1.0 - 1e-6,
        )
    )

    scaled_time = (
        np.asarray(
            day,
            dtype=float,
        )
        / float(
            horizon
        )
    )

    design = np.column_stack(
        [
            scaled_time,
            scaled_time**2,
            scaled_time**3,
            np.log1p(
                np.asarray(
                    day,
                    dtype=float,
                )
            ),
            prediction_score
            * scaled_time,
            prediction_score
            * scaled_time**2,
        ]
    )

    return sparse.csr_matrix(
        design
    )


def build_person_period_design(
    frame: pd.DataFrame,
    baseline_matrix: sparse.csr_matrix,
    *,
    horizon: int,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
]:
    followup_day = np.ceil(
        frame[
            "restricted_time"
        ].to_numpy(
            dtype=float
        )
    ).astype(
        int
    )

    followup_day = np.clip(
        followup_day,
        1,
        horizon,
    )

    patient_index = np.repeat(
        np.arange(
            len(
                frame
            ),
            dtype=int,
        ),
        followup_day,
    )

    days = np.concatenate(
        [
            np.arange(
                1,
                length + 1,
                dtype=int,
            )
            for length in followup_day
        ]
    )

    baseline_repeated = baseline_matrix[
        patient_index
    ]

    temporal = time_design(
        frame[
            "prediction"
        ].to_numpy(
            dtype=float
        )[
            patient_index
        ],
        days,
        horizon=horizon,
    )

    design = sparse.hstack(
        [
            baseline_repeated,
            temporal,
        ],
        format="csr",
    )

    outcome = np.zeros(
        len(
            patient_index
        ),
        dtype=int,
    )

    row_ends = (
        np.cumsum(
            followup_day
        )
        - 1
    )

    outcome[
        row_ends
    ] = frame[
        "restricted_event"
    ].to_numpy(
        dtype=int
    )

    return (
        design,
        outcome,
    )


def build_horizon_prediction_design(
    frame: pd.DataFrame,
    baseline_matrix: sparse.csr_matrix,
    *,
    horizon: int,
) -> sparse.csr_matrix:
    n = len(
        frame
    )

    patient_index = np.repeat(
        np.arange(
            n,
            dtype=int,
        ),
        horizon,
    )

    days = np.tile(
        np.arange(
            1,
            horizon + 1,
            dtype=int,
        ),
        n,
    )

    baseline_repeated = baseline_matrix[
        patient_index
    ]

    temporal = time_design(
        frame[
            "prediction"
        ].to_numpy(
            dtype=float
        )[
            patient_index
        ],
        days,
        horizon=horizon,
    )

    return sparse.hstack(
        [
            baseline_repeated,
            temporal,
        ],
        format="csr",
    )


def fit_logistic_model(
    design: sparse.csr_matrix,
    outcome: np.ndarray,
    *,
    model_c: float,
    max_iter: int,
    seed: int,
) -> tuple[
    LogisticRegression,
    bool,
]:
    if len(
        np.unique(
            outcome
        )
    ) != 2:
        raise ValueError(
            "Logistic model requires two outcome classes"
        )

    model = LogisticRegression(
        C=model_c,
        penalty="l2",
        solver="saga",
        max_iter=max_iter,
        tol=1e-4,
        n_jobs=-1,
        random_state=seed,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            design,
            outcome,
        )

    convergence_warning = any(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning in caught
    )

    return (
        model,
        not convergence_warning,
    )


def predict_horizon_risk(
    model: LogisticRegression,
    frame: pd.DataFrame,
    baseline_matrix: sparse.csr_matrix,
    *,
    horizon: int,
) -> np.ndarray:
    design = build_horizon_prediction_design(
        frame,
        baseline_matrix,
        horizon=horizon,
    )

    hazard = model.predict_proba(
        design
    )[
        :,
        1
    ].reshape(
        len(
            frame
        ),
        horizon,
    )

    hazard = np.clip(
        hazard,
        1e-8,
        1.0 - 1e-8,
    )

    return np.clip(
        1.0
        - np.prod(
            1.0 - hazard,
            axis=1,
        ),
        0.0,
        1.0,
    )


def cross_fitted_rich_predictions(
    data: pd.DataFrame,
    *,
    horizon: int,
    repeats: int,
    model_c: float,
    max_iter: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    n = len(
        data
    )

    outcome_risk_sum = np.zeros(
        n,
        dtype=float,
    )

    observation_probability_sum = np.zeros(
        n,
        dtype=float,
    )

    prediction_count = np.zeros(
        n,
        dtype=int,
    )

    diagnostics = []

    model_columns = (
        NUMERIC_FEATURES
        + PREDICTION_FEATURES
        + CATEGORICAL_FEATURES
    )

    for repeat in range(
        repeats
    ):
        folds = assign_subject_folds(
            data[
                "subject_id"
            ],
            seed=(
                seed
                + 1009
                * repeat
            ),
        )

        for held_out_fold in [
            0,
            1,
        ]:
            train_mask = (
                folds
                != held_out_fold
            )

            test_mask = (
                folds
                == held_out_fold
            )

            train = data.loc[
                train_mask
            ].reset_index(
                drop=True
            )

            test = data.loc[
                test_mask
            ].reset_index(
                drop=True
            )

            preprocessor = make_preprocessor()

            train_baseline = (
                preprocessor.fit_transform(
                    train[
                        model_columns
                    ]
                )
            )

            test_baseline = (
                preprocessor.transform(
                    test[
                        model_columns
                    ]
                )
            )

            if not sparse.issparse(
                train_baseline
            ):
                train_baseline = (
                    sparse.csr_matrix(
                        train_baseline
                    )
                )
            else:
                train_baseline = (
                    train_baseline.tocsr()
                )

            if not sparse.issparse(
                test_baseline
            ):
                test_baseline = (
                    sparse.csr_matrix(
                        test_baseline
                    )
                )
            else:
                test_baseline = (
                    test_baseline.tocsr()
                )

            (
                survival_design,
                survival_outcome,
            ) = build_person_period_design(
                train,
                train_baseline,
                horizon=horizon,
            )

            (
                survival_model,
                survival_converged,
            ) = fit_logistic_model(
                survival_design,
                survival_outcome,
                model_c=model_c,
                max_iter=max_iter,
                seed=(
                    seed
                    + repeat
                    * 100
                    + held_out_fold
                ),
            )

            test_outcome_risk = (
                predict_horizon_risk(
                    survival_model,
                    test,
                    test_baseline,
                    horizon=horizon,
                )
            )

            (
                observation_model,
                observation_converged,
            ) = fit_logistic_model(
                train_baseline,
                train[
                    "observed"
                ].to_numpy(
                    dtype=int
                ),
                model_c=model_c,
                max_iter=max_iter,
                seed=(
                    seed
                    + 5000
                    + repeat
                    * 100
                    + held_out_fold
                ),
            )

            test_observation_probability = (
                observation_model.predict_proba(
                    test_baseline
                )[
                    :,
                    1
                ]
            )

            original_test_index = np.flatnonzero(
                test_mask
            )

            outcome_risk_sum[
                original_test_index
            ] += test_outcome_risk

            observation_probability_sum[
                original_test_index
            ] += np.clip(
                test_observation_probability,
                1e-6,
                1.0 - 1e-6,
            )

            prediction_count[
                original_test_index
            ] += 1

            diagnostics.append(
                {
                    "repeat": repeat,
                    "held_out_fold": (
                        held_out_fold
                    ),
                    "n_train": len(
                        train
                    ),
                    "n_test": len(
                        test
                    ),
                    "baseline_feature_dimension": int(
                        train_baseline.shape[
                            1
                        ]
                    ),
                    "person_period_rows": int(
                        survival_design.shape[
                            0
                        ]
                    ),
                    "person_period_columns": int(
                        survival_design.shape[
                            1
                        ]
                    ),
                    "restricted_events_train": int(
                        train[
                            "restricted_event"
                        ].sum()
                    ),
                    "observed_train": int(
                        train[
                            "observed"
                        ].sum()
                    ),
                    "survival_model_converged": (
                        survival_converged
                    ),
                    "observation_model_converged": (
                        observation_converged
                    ),
                    "survival_iterations": int(
                        np.max(
                            survival_model.n_iter_
                        )
                    ),
                    "observation_iterations": int(
                        np.max(
                            observation_model.n_iter_
                        )
                    ),
                }
            )

    if not np.all(
        prediction_count
        == repeats
    ):
        raise RuntimeError(
            "Repeated cross-fitting did not "
            "cover every admission"
        )

    return (
        outcome_risk_sum
        / prediction_count,
        observation_probability_sum
        / prediction_count,
        prediction_count,
        pd.DataFrame(
            diagnostics
        ),
    )


def kaplan_meier_risk(
    time: np.ndarray,
    event: np.ndarray,
    *,
    horizon: int,
) -> float:
    followup = np.asarray(
        time,
        dtype=float,
    )

    event_indicator = np.asarray(
        event,
        dtype=int,
    )

    event_times = np.unique(
        followup[
            (
                event_indicator
                == 1
            )
            & (
                followup
                <= horizon
            )
        ]
    )

    survival = 1.0

    for event_time in np.sort(
        event_times
    ):
        at_risk = int(
            np.sum(
                followup
                >= event_time
            )
        )

        deaths = int(
            np.sum(
                (
                    followup
                    == event_time
                )
                & (
                    event_indicator
                    == 1
                )
            )
        )

        if at_risk > 0:
            survival *= (
                1.0
                - deaths
                / at_risk
            )

    return float(
        np.clip(
            1.0 - survival,
            0.0,
            1.0,
        )
    )


def weighted_estimators(
    group: pd.DataFrame,
    *,
    outcome_model_column: str,
    propensity_column: str,
    clip: float | None,
) -> tuple[
    float,
    float,
    float,
]:
    observed = group[
        "observed"
    ].to_numpy(
        dtype=float
    )

    visible_outcome = group[
        "restricted_event"
    ].to_numpy(
        dtype=float
    )

    outcome_model = group[
        outcome_model_column
    ].to_numpy(
        dtype=float
    )

    propensity = group[
        propensity_column
    ].to_numpy(
        dtype=float
    )

    if clip is not None:
        propensity = np.clip(
            propensity,
            clip,
            1.0 - clip,
        )
    else:
        propensity = np.clip(
            propensity,
            1e-6,
            1.0 - 1e-6,
        )

    denominator = float(
        np.sum(
            observed
            / propensity
        )
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid IPCW denominator"
        )

    ipcw = float(
        np.sum(
            observed
            * visible_outcome
            / propensity
        )
        / denominator
    )

    pseudo_outcome = (
        outcome_model
        + observed
        / propensity
        * (
            visible_outcome
            - outcome_model
        )
    )

    aipw_raw = float(
        np.mean(
            pseudo_outcome
        )
    )

    return (
        float(
            np.clip(
                ipcw,
                0.0,
                1.0,
            )
        ),
        float(
            np.clip(
                aipw_raw,
                0.0,
                1.0,
            )
        ),
        aipw_raw,
    )


def estimate_bin_curves(
    data: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    rows = []

    for risk_bin, group in data.groupby(
        "risk_bin",
        sort=True,
    ):
        observed = group.loc[
            group[
                "observed"
            ].eq(
                1
            )
        ]

        if observed.empty:
            raise ValueError(
                f"No observed outcomes in bin {risk_bin}"
            )

        (
            score_ipcw,
            score_aipw,
            score_aipw_raw,
        ) = weighted_estimators(
            group,
            outcome_model_column=(
                "score_outcome_regression_risk"
            ),
            propensity_column=(
                "score_observation_probability"
            ),
            clip=None,
        )

        row = {
            "risk_bin": int(
                risk_bin
            ),
            "n": int(
                len(
                    group
                )
            ),
            "reference_weight": float(
                len(
                    group
                )
                / len(
                    data
                )
            ),
            "mean_prediction": float(
                group[
                    "prediction"
                ].mean()
            ),
            "reference_oracle": float(
                group[
                    "outcome"
                ].mean()
            ),
            "restricted_complete_case": float(
                observed[
                    "restricted_event"
                ].mean()
            ),
            "grouped_km": kaplan_meier_risk(
                group[
                    "restricted_time"
                ].to_numpy(
                    dtype=float
                ),
                group[
                    "restricted_event"
                ].to_numpy(
                    dtype=int
                ),
                horizon=horizon,
            ),
            "score_pooled_survival_or": float(
                group[
                    "score_outcome_regression_risk"
                ].mean()
            ),
            "score_ipcw": score_ipcw,
            "score_aipw": score_aipw,
            "score_aipw_raw": (
                score_aipw_raw
            ),
            "rich_pooled_survival_or": float(
                group[
                    "rich_outcome_regression_risk"
                ].mean()
            ),
            "observed_fraction": float(
                group[
                    "observed"
                ].mean()
            ),
            "mean_score_observation_probability": float(
                group[
                    "score_observation_probability"
                ].mean()
            ),
            "mean_rich_observation_probability": float(
                group[
                    "rich_observation_probability"
                ].mean()
            ),
        }

        for clip in PROPENSITY_CLIPS:
            label = clip_label(
                clip
            )

            (
                rich_ipcw,
                rich_aipw,
                rich_aipw_raw,
            ) = weighted_estimators(
                group,
                outcome_model_column=(
                    "rich_outcome_regression_risk"
                ),
                propensity_column=(
                    "rich_observation_probability"
                ),
                clip=clip,
            )

            row[
                f"rich_ipcw_clip_{label}"
            ] = rich_ipcw

            row[
                f"rich_aipw_clip_{label}"
            ] = rich_aipw

            row[
                f"rich_aipw_raw_clip_{label}"
            ] = rich_aipw_raw

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "risk_bin"
    ).reset_index(
        drop=True
    )


def calculate_metrics(
    curves: pd.DataFrame,
) -> pd.DataFrame:
    weights = curves[
        "reference_weight"
    ].to_numpy(
        dtype=float
    )

    weights = (
        weights
        / weights.sum()
    )

    prediction = curves[
        "mean_prediction"
    ].to_numpy(
        dtype=float
    )

    reference = curves[
        "reference_oracle"
    ].to_numpy(
        dtype=float
    )

    reference_event_rate = float(
        np.sum(
            weights
            * reference
        )
    )

    reference_ici = float(
        np.sum(
            weights
            * np.abs(
                reference
                - prediction
            )
        )
    )

    rows = []

    for method in method_columns():
        estimate = curves[
            method
        ].to_numpy(
            dtype=float
        )

        estimated_event_rate = float(
            np.sum(
                weights
                * estimate
            )
        )

        estimated_ici = float(
            np.sum(
                weights
                * np.abs(
                    estimate
                    - prediction
                )
            )
        )

        rows.append(
            {
                "method": method,
                "estimated_event_rate": (
                    estimated_event_rate
                ),
                "reference_event_rate": (
                    reference_event_rate
                ),
                "event_rate_error": abs(
                    estimated_event_rate
                    - reference_event_rate
                ),
                "estimated_ici": (
                    estimated_ici
                ),
                "reference_ici": (
                    reference_ici
                ),
                "ici_error": abs(
                    estimated_ici
                    - reference_ici
                ),
                "bin_curve_mae": float(
                    np.sum(
                        weights
                        * np.abs(
                            estimate
                            - reference
                        )
                    )
                ),
                "signed_curve_error": float(
                    np.sum(
                        weights
                        * (
                            estimate
                            - reference
                        )
                    )
                ),
                "maximum_bin_error": float(
                    np.max(
                        np.abs(
                            estimate
                            - reference
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def cluster_bootstrap_indices(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = [
        group.index.to_numpy(
            dtype=int
        )
        for _, group in data.groupby(
            "subject_id",
            sort=False,
        )
    ]

    sampled = rng.integers(
        0,
        len(
            groups
        ),
        size=len(
            groups
        ),
    )

    return np.concatenate(
        [
            groups[
                index
            ]
            for index in sampled
        ]
    )


def run_bootstrap(
    data: pd.DataFrame,
    *,
    horizon: int,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    rows = []

    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        indices = cluster_bootstrap_indices(
            data,
            rng,
        )

        sample = (
            data.loc[
                indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        try:
            curves = estimate_bin_curves(
                sample,
                horizon=horizon,
            )

            metrics = calculate_metrics(
                curves
            )

        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            invalid += 1
            continue

        for _, row in metrics.iterrows():
            rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    **row.to_dict(),
                }
            )

        if (
            bootstrap_id
            + 1
        ) % 100 == 0:
            print(
                f"  completed "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

    return (
        pd.DataFrame(
            rows
        ),
        invalid,
    )


def summarize_intervals(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "estimated_event_rate",
        "event_rate_error",
        "estimated_ici",
        "ici_error",
        "bin_curve_mae",
        "signed_curve_error",
        "maximum_bin_error",
    ]

    rows = []

    for method, group in bootstrap.groupby(
        "method",
        sort=True,
    ):
        for metric in metrics:
            values = (
                group[
                    metric
                ]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "q025": float(
                        values.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        values.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        values.quantile(
                            0.975
                        )
                    ),
                    "n_valid": int(
                        len(
                            values
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def paired_difference_tables(
    point_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pairs = [
        (
            "rich_pooled_survival_or",
            "score_pooled_survival_or",
        ),
        (
            "rich_ipcw_clip_0p01",
            "score_ipcw",
        ),
        (
            "rich_aipw_clip_0p01",
            "score_aipw",
        ),
        (
            "rich_pooled_survival_or",
            "grouped_km",
        ),
    ]

    comparison_metrics = [
        "bin_curve_mae",
        "event_rate_error",
        "ici_error",
        "maximum_bin_error",
    ]

    point_lookup = point_metrics.set_index(
        "method"
    )

    point_rows = []

    interval_rows = []

    for rich_method, comparator in pairs:
        for metric in comparison_metrics:
            point_rows.append(
                {
                    "method": rich_method,
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(
                        point_lookup.loc[
                            rich_method,
                            metric,
                        ]
                        - point_lookup.loc[
                            comparator,
                            metric,
                        ]
                    ),
                }
            )

            rich_values = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        rich_method
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: "rich_value",
                    }
                )
            )

            comparator_values = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        comparator
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: (
                            "comparator_value"
                        ),
                    }
                )
            )

            paired = rich_values.merge(
                comparator_values,
                on="bootstrap_id",
                how="inner",
                validate="one_to_one",
            )

            difference = (
                paired[
                    "rich_value"
                ]
                - paired[
                    "comparator_value"
                ]
            )

            interval_rows.append(
                {
                    "method": rich_method,
                    "comparator": comparator,
                    "metric": metric,
                    "q025": float(
                        difference.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        difference.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        difference.quantile(
                            0.975
                        )
                    ),
                    "n_valid": int(
                        len(
                            difference
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(
            point_rows
        ),
        pd.DataFrame(
            interval_rows
        ),
    )


def positivity_diagnostics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    observed = data[
        "observed"
    ].to_numpy(
        dtype=float
    )

    raw_probability = data[
        "rich_observation_probability"
    ].to_numpy(
        dtype=float
    )

    for clip in PROPENSITY_CLIPS:
        propensity = np.clip(
            raw_probability,
            clip,
            1.0 - clip,
        )

        weight = (
            observed
            / propensity
        )

        effective_sample_size = float(
            weight.sum() ** 2
            / (
                np.sum(
                    weight**2
                )
                + 1e-12
            )
        )

        rows.append(
            {
                "propensity_clip": clip,
                "raw_minimum_probability": float(
                    np.min(
                        raw_probability
                    )
                ),
                "raw_q01_probability": float(
                    np.quantile(
                        raw_probability,
                        0.01,
                    )
                ),
                "raw_median_probability": float(
                    np.median(
                        raw_probability
                    )
                ),
                "maximum_inverse_weight": float(
                    1.0
                    / propensity.min()
                ),
                "weighted_observed_effective_sample_size": (
                    effective_sample_size
                ),
                "fraction_clipped_low": float(
                    np.mean(
                        raw_probability
                        < clip
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_validation_checks(
    *,
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    score_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    prediction_count: np.ndarray,
    repeats: int,
    bootstrap_rows: int,
    invalid_bootstraps: int,
    requested_bootstraps: int,
) -> pd.DataFrame:
    model_features = set(
        NUMERIC_FEATURES
        + PREDICTION_FEATURES
        + CATEGORICAL_FEATURES
    )

    forbidden_overlap = sorted(
        model_features.intersection(
            FORBIDDEN_MODEL_FEATURES
        )
    )

    score_mapping = {
        "pooled_survival_or": (
            "score_pooled_survival_or"
        ),
        "score_ipcw": "score_ipcw",
        "score_aipw": "score_aipw",
        "grouped_km": "grouped_km",
        "restricted_complete_case": (
            "restricted_complete_case"
        ),
    }

    current_lookup = metrics.set_index(
        "method"
    )

    previous_lookup = score_summary.set_index(
        "method"
    )

    score_differences = []

    for previous_name, current_name in (
        score_mapping.items()
    ):
        for metric in [
            "bin_curve_mae",
            "event_rate_error",
            "ici_error",
            "maximum_bin_error",
        ]:
            score_differences.append(
                abs(
                    float(
                        current_lookup.loc[
                            current_name,
                            metric,
                        ]
                    )
                    - float(
                        previous_lookup.loc[
                            previous_name,
                            metric,
                        ]
                    )
                )
            )

    maximum_score_reproduction_error = max(
        score_differences
    )

    minimum_curve = float(
        curves[
            method_columns()
        ].min().min()
    )

    maximum_curve = float(
        curves[
            method_columns()
        ].max().max()
    )

    expected_bootstrap_rows = (
        requested_bootstraps
        * len(
            method_columns()
        )
    )

    manifest_feature_count = int(
        manifest[
            "nuisance_model_feature"
        ]
        .astype(
            bool
        )
        .sum()
    )

    rich_risk_minimum = float(
        data[
            "rich_outcome_regression_risk"
        ].min()
    )

    rich_risk_maximum = float(
        data[
            "rich_outcome_regression_risk"
        ].max()
    )

    rich_probability_minimum = float(
        data[
            "rich_observation_probability"
        ].min()
    )

    rich_probability_maximum = float(
        data[
            "rich_observation_probability"
        ].max()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "all 21 frozen baseline "
                    "features available"
                ),
                "value": (
                    manifest_feature_count
                ),
                "threshold": 21,
                "comparison": "==",
                "passed": (
                    manifest_feature_count
                    == 21
                ),
            },
            {
                "check": (
                    "no forbidden variables in "
                    "nuisance feature matrix"
                ),
                "value": len(
                    forbidden_overlap
                ),
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    len(
                        forbidden_overlap
                    )
                    == 0
                ),
            },
            {
                "check": (
                    "all cross-fitted rich nuisance "
                    "predictions complete"
                ),
                "value": int(
                    prediction_count.min()
                ),
                "threshold": repeats,
                "comparison": "==",
                "passed": bool(
                    np.all(
                        prediction_count
                        == repeats
                    )
                ),
            },
            {
                "check": (
                    "rich outcome-regression risks "
                    "within probability bounds"
                ),
                "value": (
                    f"{rich_risk_minimum},"
                    f"{rich_risk_maximum}"
                ),
                "threshold": "[0,1]",
                "comparison": "within",
                "passed": bool(
                    data[
                        "rich_outcome_regression_risk"
                    ]
                    .between(
                        0.0,
                        1.0,
                    )
                    .all()
                ),
            },
            {
                "check": (
                    "rich observation probabilities "
                    "strictly within bounds"
                ),
                "value": (
                    f"{rich_probability_minimum},"
                    f"{rich_probability_maximum}"
                ),
                "threshold": "(0,1)",
                "comparison": "within",
                "passed": bool(
                    (
                        data[
                            "rich_observation_probability"
                        ]
                        > 0
                    ).all()
                    and (
                        data[
                            "rich_observation_probability"
                        ]
                        < 1
                    ).all()
                ),
            },
            {
                "check": (
                    "score-only benchmark reproduced"
                ),
                "value": (
                    maximum_score_reproduction_error
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    maximum_score_reproduction_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all estimator curves remain "
                    "within probability bounds"
                ),
                "value": (
                    f"{minimum_curve},"
                    f"{maximum_curve}"
                ),
                "threshold": "[0,1]",
                "comparison": "within",
                "passed": (
                    minimum_curve
                    >= 0.0
                    and maximum_curve
                    <= 1.0
                ),
            },
            {
                "check": (
                    "all requested bootstrap rows completed"
                ),
                "value": bootstrap_rows,
                "threshold": expected_bootstrap_rows,
                "comparison": "==",
                "passed": (
                    bootstrap_rows
                    == expected_bootstrap_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        data,
        manifest,
        score_summary,
        bin_edges,
    ) = load_inputs(
        rich_path=Path(
            args.rich_input
        ),
        manifest_path=Path(
            args.feature_manifest
        ),
        score_nuisance_path=Path(
            args.score_nuisance
        ),
        score_summary_path=Path(
            args.score_summary
        ),
        n_bins=args.n_bins,
    )

    print(
        "Phase 5G-B rich baseline estimators"
    )

    print(
        f"  admissions={len(data):,}"
    )

    print(
        f"  baseline features="
        f"{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"  restricted observed fraction="
        f"{data['observed'].mean():.6f}"
    )

    print(
        f"  cross-fit repeats="
        f"{args.crossfit_repeats}"
    )

    print(
        "\nFitting repeated cross-fitted "
        "rich nuisance models..."
    )

    (
        rich_outcome_risk,
        rich_observation_probability,
        prediction_count,
        fold_diagnostics,
    ) = cross_fitted_rich_predictions(
        data,
        horizon=args.horizon,
        repeats=args.crossfit_repeats,
        model_c=args.model_c,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    data[
        "rich_outcome_regression_risk"
    ] = rich_outcome_risk

    data[
        "rich_observation_probability"
    ] = rich_observation_probability

    curves = estimate_bin_curves(
        data,
        horizon=args.horizon,
    )

    metrics = calculate_metrics(
        curves
    )

    ranking = (
        metrics.loc[
            ~metrics[
                "method"
            ].eq(
                "reference_oracle"
            )
        ]
        .sort_values(
            [
                "bin_curve_mae",
                "event_rate_error",
                "ici_error",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "rank"
    ] = np.arange(
        1,
        len(
            ranking
        )
        + 1,
    )

    print(
        "\nRunning paired patient bootstrap "
        "conditional on nuisance predictions..."
    )

    (
        bootstrap,
        invalid_bootstraps,
    ) = run_bootstrap(
        data,
        horizon=args.horizon,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
    )

    intervals = summarize_intervals(
        bootstrap
    )

    (
        point_differences,
        difference_intervals,
    ) = paired_difference_tables(
        metrics,
        bootstrap,
    )

    positivity = positivity_diagnostics(
        data
    )

    checks = build_validation_checks(
        data=data,
        manifest=manifest,
        score_summary=score_summary,
        metrics=metrics,
        curves=curves,
        prediction_count=(
            prediction_count
        ),
        repeats=args.crossfit_repeats,
        bootstrap_rows=len(
            bootstrap
        ),
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
    )

    preliminary_status = (
        "phase5g_rich_baseline_estimators_benchmarked"
        if checks[
            "passed"
        ].all()
        else (
            "phase5g_rich_baseline_estimators_"
            "require_review"
        )
    )

    status = (
        "phase5g_rich_baseline_estimators_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    nuisance_output = data[
        IDENTIFIERS
        + [
            "prediction",
            "risk_bin",
            "observed",
            "restricted_event",
            "restricted_time",
            "score_outcome_regression_risk",
            "score_observation_probability",
            "rich_outcome_regression_risk",
            "rich_observation_probability",
        ]
    ].copy()

    nuisance_output.to_csv(
        output_dir
        / "cross_fitted_rich_nuisance_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    curves.to_csv(
        output_dir
        / "rich_estimator_bin_curves.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "rich_estimator_metric_summary.csv",
        index=False,
    )

    ranking.to_csv(
        output_dir
        / "rich_estimator_ranking.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_metric_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    intervals.to_csv(
        output_dir
        / "bootstrap_metric_intervals.csv",
        index=False,
    )

    point_differences.to_csv(
        output_dir
        / "paired_method_point_differences.csv",
        index=False,
    )

    difference_intervals.to_csv(
        output_dir
        / "paired_method_difference_intervals.csv",
        index=False,
    )

    positivity.to_csv(
        output_dir
        / "positivity_diagnostics.csv",
        index=False,
    )

    fold_diagnostics.to_csv(
        output_dir
        / "crossfit_fold_diagnostics.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "edge_index": np.arange(
                len(
                    bin_edges
                )
            ),
            "edge": bin_edges,
        }
    ).to_csv(
        output_dir
        / "reference_bin_edges.csv",
        index=False,
    )

    evidence = {
        "phase": "5G-B",
        "status": status,
        "preliminary_status": (
            preliminary_status
            if args.smoke
            else None
        ),
        "n_admissions": int(
            len(
                data
            )
        ),
        "n_numeric_features": len(
            NUMERIC_FEATURES
        ),
        "n_categorical_features": len(
            CATEGORICAL_FEATURES
        ),
        "prediction_summary_features": (
            PREDICTION_FEATURES
        ),
        "crossfit_repeats": (
            args.crossfit_repeats
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "model_family": (
            "L2-regularized sparse logistic models"
        ),
        "outcome_regression": (
            "Discrete-time pooled logistic survival "
            "model using restricted time and event."
        ),
        "observation_model": (
            "Logistic probability of having the "
            "30-day outcome observed."
        ),
        "propensity_clips": (
            PROPENSITY_CLIPS
        ),
        "oracle_use": (
            "Linked outcome used only for final "
            "retrospective curve evaluation."
        ),
        "inference": (
            "Patient bootstrap conditional on repeated "
            "cross-fitted nuisance predictions."
        ),
        "scientific_scope": (
            "This tests whether exact frozen baseline "
            "covariate adjustment repairs the score-only "
            "estimator failures. It does not yet test "
            "more flexible nonlinear nuisance models."
        ),
    }

    (
        output_dir
        / "rich_estimator_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5g_rich_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nEstimator ranking"
    )

    print(
        ranking[
            [
                "rank",
                "method",
                "bin_curve_mae",
                "event_rate_error",
                "ici_error",
                "maximum_bin_error",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPaired method differences"
    )

    print(
        point_differences.loc[
            point_differences[
                "metric"
            ].eq(
                "bin_curve_mae"
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nPositivity diagnostics"
    )

    print(
        positivity.to_string(
            index=False
        )
    )

    print(
        "\nValidation checks"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        f"\nPreliminary status: "
        f"{preliminary_status}"
    )

    print(
        f"Saved status: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
