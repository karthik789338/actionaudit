"""Phase 5G-C: nonlinear practitioner-available nuisance estimators.

Primary nonlinear family:
    sklearn HistGradientBoostingClassifier.

The linked post-discharge outcome is never used for nuisance fitting.
It is used only after estimator construction to evaluate recovery of
the retrospective reference calibration curve.

Inference uses a patient bootstrap conditional on repeated
cross-fitted nuisance predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier


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

FORBIDDEN_FEATURES = {
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
    "deathtime",
    "dod",
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
        "rich_logistic_survival_or",
        "hgb_survival_or",
        "score_ipcw",
        "score_aipw",
    ]

    for clip in PROPENSITY_CLIPS:
        label = clip_label(
            clip
        )

        methods.extend(
            [
                (
                    "rich_logistic_ipcw_"
                    f"clip_{label}"
                ),
                (
                    "rich_logistic_aipw_"
                    f"clip_{label}"
                ),
                f"hgb_ipcw_clip_{label}",
                f"hgb_aipw_clip_{label}",
            ]
        )

    return methods


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark nonlinear admission-time "
            "nuisance estimators."
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
        "--logistic-nuisance",
        default=(
            "artifacts/"
            "phase5g_rich_baseline_estimators/"
            "cross_fitted_rich_nuisance_predictions.csv.gz"
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
        "--rich-summary",
        default=(
            "artifacts/"
            "phase5g_rich_baseline_estimators/"
            "rich_estimator_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5g_nonlinear_estimators"
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
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=1.0,
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
    if frame.duplicated(
        IDENTIFIERS
    ).any():
        raise ValueError(
            f"{label} contains duplicate admission keys"
        )


class BaselineEncoder:
    """Training-fold categorical encoder for native HGB splits."""

    def __init__(
        self,
    ) -> None:
        self.category_maps: dict[
            str,
            dict[str, int],
        ] = {}

    def fit(
        self,
        frame: pd.DataFrame,
    ) -> "BaselineEncoder":
        for feature in CATEGORICAL_FEATURES:
            categories = sorted(
                frame[
                    feature
                ]
                .dropna()
                .astype(
                    str
                )
                .unique()
                .tolist()
            )

            if len(
                categories
            ) > 255:
                raise ValueError(
                    f"Categorical feature {feature} "
                    "has more than 255 levels"
                )

            self.category_maps[
                feature
            ] = {
                category: index
                for index, category in enumerate(
                    categories
                )
            }

        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        columns = []

        for feature in (
            NUMERIC_FEATURES
            + PREDICTION_FEATURES
        ):
            columns.append(
                pd.to_numeric(
                    frame[
                        feature
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
            )

        for feature in CATEGORICAL_FEATURES:
            mapping = self.category_maps[
                feature
            ]

            encoded = (
                frame[
                    feature
                ]
                .astype(
                    "string"
                )
                .map(
                    mapping
                )
            )

            columns.append(
                pd.to_numeric(
                    encoded,
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
            )

        return np.column_stack(
            columns
        )

    @property
    def categorical_mask(
        self,
    ) -> np.ndarray:
        return np.asarray(
            [
                False
            ]
            * (
                len(
                    NUMERIC_FEATURES
                )
                + len(
                    PREDICTION_FEATURES
                )
            )
            + [
                True
            ]
            * len(
                CATEGORICAL_FEATURES
            ),
            dtype=bool,
        )


def load_inputs(
    *,
    rich_path: Path,
    manifest_path: Path,
    nuisance_path: Path,
    score_summary_path: Path,
    rich_summary_path: Path,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
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

    nuisance = pd.read_csv(
        nuisance_path,
        low_memory=False,
    )

    score_summary = pd.read_csv(
        score_summary_path
    )

    rich_summary = pd.read_csv(
        rich_summary_path
    )

    required_rich = (
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
        required_rich,
        label="Rich analysis input",
    )

    nuisance_columns = (
        IDENTIFIERS
        + [
            "score_outcome_regression_risk",
            "score_observation_probability",
            "rich_outcome_regression_risk",
            "rich_observation_probability",
        ]
    )

    require_columns(
        nuisance,
        nuisance_columns,
        label="Frozen logistic nuisance input",
    )

    validate_unique(
        rich,
        label="Rich analysis input",
    )

    validate_unique(
        nuisance,
        label="Frozen logistic nuisance input",
    )

    nuisance_subset = nuisance[
        nuisance_columns
    ].rename(
        columns={
            "rich_outcome_regression_risk": (
                "rich_logistic_outcome_risk"
            ),
            "rich_observation_probability": (
                "rich_logistic_observation_probability"
            ),
        }
    )

    data = rich.merge(
        nuisance_subset,
        on=IDENTIFIERS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        data[
            "_merge"
        ].ne(
            "both"
        ).sum()
    )

    if unmatched:
        raise ValueError(
            f"{unmatched} admissions failed "
            "the nuisance join"
        )

    data = data.drop(
        columns=[
            "_merge",
        ]
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

    score = logit(
        prediction
    )

    data[
        "prediction"
    ] = prediction

    data[
        "prediction_logit"
    ] = score

    data[
        "prediction_logit_squared"
    ] = score**2

    data[
        "prediction_logit_cubed"
    ] = score**3

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
            "Predictions do not support "
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

    expected_features = set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    actual_features = set(
        manifest.loc[
            manifest[
                "nuisance_model_feature"
            ].astype(
                bool
            ),
            "feature",
        ]
    )

    if expected_features != actual_features:
        raise ValueError(
            "Feature manifest differs from "
            "the frozen 21-feature specification"
        )

    return (
        data.reset_index(
            drop=True
        ),
        manifest,
        score_summary,
        rich_summary,
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


def temporal_features(
    prediction: np.ndarray,
    day: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    score = logit(
        np.clip(
            np.asarray(
                prediction,
                dtype=float,
            ),
            1e-6,
            1.0 - 1e-6,
        )
    )

    day_array = np.asarray(
        day,
        dtype=float,
    )

    scaled_time = (
        day_array
        / float(
            horizon
        )
    )

    return np.column_stack(
        [
            scaled_time,
            scaled_time**2,
            scaled_time**3,
            np.log1p(
                day_array
            ),
            score
            * scaled_time,
            score
            * scaled_time**2,
        ]
    )


def build_person_period_design(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    *,
    horizon: int,
) -> tuple[
    np.ndarray,
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

    repeated_baseline = baseline[
        patient_index
    ]

    temporal = temporal_features(
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

    design = np.column_stack(
        [
            repeated_baseline,
            temporal,
        ]
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


def make_hgb_model(
    *,
    categorical_mask: np.ndarray,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    seed: int,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_bins=255,
        categorical_features=(
            categorical_mask
        ),
        early_stopping=False,
        random_state=seed,
    )


def predict_horizon_risk(
    model: HistGradientBoostingClassifier,
    frame: pd.DataFrame,
    baseline: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    survival = np.ones(
        len(
            frame
        ),
        dtype=float,
    )

    prediction = frame[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    for day in range(
        1,
        horizon + 1,
    ):
        days = np.full(
            len(
                frame
            ),
            day,
            dtype=int,
        )

        temporal = temporal_features(
            prediction,
            days,
            horizon=horizon,
        )

        design = np.column_stack(
            [
                baseline,
                temporal,
            ]
        )

        hazard = model.predict_proba(
            design
        )[
            :,
            1
        ]

        hazard = np.clip(
            hazard,
            1e-8,
            1.0 - 1e-8,
        )

        survival *= (
            1.0
            - hazard
        )

    return np.clip(
        1.0 - survival,
        0.0,
        1.0,
    )


def cross_fitted_hgb_predictions(
    data: pd.DataFrame,
    *,
    horizon: int,
    repeats: int,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
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

            encoder = BaselineEncoder().fit(
                train[
                    model_columns
                ]
            )

            train_baseline = encoder.transform(
                train[
                    model_columns
                ]
            )

            test_baseline = encoder.transform(
                test[
                    model_columns
                ]
            )

            baseline_mask = (
                encoder.categorical_mask
            )

            (
                survival_design,
                survival_outcome,
            ) = build_person_period_design(
                train,
                train_baseline,
                horizon=horizon,
            )

            survival_mask = np.concatenate(
                [
                    baseline_mask,
                    np.zeros(
                        6,
                        dtype=bool,
                    ),
                ]
            )

            survival_model = make_hgb_model(
                categorical_mask=(
                    survival_mask
                ),
                learning_rate=(
                    learning_rate
                ),
                max_iter=max_iter,
                max_leaf_nodes=(
                    max_leaf_nodes
                ),
                min_samples_leaf=(
                    min_samples_leaf
                ),
                l2_regularization=(
                    l2_regularization
                ),
                seed=(
                    seed
                    + repeat
                    * 100
                    + held_out_fold
                ),
            )

            survival_model.fit(
                survival_design,
                survival_outcome,
            )

            test_outcome_risk = (
                predict_horizon_risk(
                    survival_model,
                    test,
                    test_baseline,
                    horizon=horizon,
                )
            )

            observation_model = make_hgb_model(
                categorical_mask=(
                    baseline_mask
                ),
                learning_rate=(
                    learning_rate
                ),
                max_iter=max_iter,
                max_leaf_nodes=(
                    max_leaf_nodes
                ),
                min_samples_leaf=(
                    min_samples_leaf
                ),
                l2_regularization=(
                    l2_regularization
                ),
                seed=(
                    seed
                    + 5000
                    + repeat
                    * 100
                    + held_out_fold
                ),
            )

            observation_model.fit(
                train_baseline,
                train[
                    "observed"
                ].to_numpy(
                    dtype=int
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

            test_observation_probability = np.clip(
                test_observation_probability,
                1e-6,
                1.0 - 1e-6,
            )

            original_indices = np.flatnonzero(
                test_mask
            )

            outcome_risk_sum[
                original_indices
            ] += test_outcome_risk

            observation_probability_sum[
                original_indices
            ] += test_observation_probability

            prediction_count[
                original_indices
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
                    "baseline_columns": int(
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
                    "survival_iterations": int(
                        survival_model.n_iter_
                    ),
                    "observation_iterations": int(
                        observation_model.n_iter_
                    ),
                    "survival_hit_max_iter": bool(
                        survival_model.n_iter_
                        >= max_iter
                    ),
                    "observation_hit_max_iter": bool(
                        observation_model.n_iter_
                        >= max_iter
                    ),
                    "test_risk_min": float(
                        test_outcome_risk.min()
                    ),
                    "test_risk_max": float(
                        test_outcome_risk.max()
                    ),
                    "test_observation_probability_min": float(
                        test_observation_probability.min()
                    ),
                    "test_observation_probability_max": float(
                        test_observation_probability.max()
                    ),
                }
            )

    if not np.all(
        prediction_count
        == repeats
    ):
        raise RuntimeError(
            "Repeated nonlinear cross-fitting "
            "did not cover every admission"
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

    if clip is None:
        propensity = np.clip(
            propensity,
            1e-6,
            1.0 - 1e-6,
        )
    else:
        propensity = np.clip(
            propensity,
            clip,
            1.0 - clip,
        )

    denominator = float(
        np.sum(
            observed
            / propensity
        )
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid weighted denominator"
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
        observed_group = group.loc[
            group[
                "observed"
            ].eq(
                1
            )
        ]

        if observed_group.empty:
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
                observed_group[
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
            "rich_logistic_survival_or": float(
                group[
                    "rich_logistic_outcome_risk"
                ].mean()
            ),
            "hgb_survival_or": float(
                group[
                    "hgb_outcome_regression_risk"
                ].mean()
            ),
            "score_ipcw": score_ipcw,
            "score_aipw": score_aipw,
            "score_aipw_raw": (
                score_aipw_raw
            ),
            "observed_fraction": float(
                group[
                    "observed"
                ].mean()
            ),
            "mean_hgb_observation_probability": float(
                group[
                    "hgb_observation_probability"
                ].mean()
            ),
        }

        for clip in PROPENSITY_CLIPS:
            label = clip_label(
                clip
            )

            (
                logistic_ipcw,
                logistic_aipw,
                logistic_aipw_raw,
            ) = weighted_estimators(
                group,
                outcome_model_column=(
                    "rich_logistic_outcome_risk"
                ),
                propensity_column=(
                    "rich_logistic_observation_probability"
                ),
                clip=clip,
            )

            (
                hgb_ipcw,
                hgb_aipw,
                hgb_aipw_raw,
            ) = weighted_estimators(
                group,
                outcome_model_column=(
                    "hgb_outcome_regression_risk"
                ),
                propensity_column=(
                    "hgb_observation_probability"
                ),
                clip=clip,
            )

            row[
                (
                    "rich_logistic_ipcw_"
                    f"clip_{label}"
                )
            ] = logistic_ipcw

            row[
                (
                    "rich_logistic_aipw_"
                    f"clip_{label}"
                )
            ] = logistic_aipw

            row[
                (
                    "rich_logistic_aipw_raw_"
                    f"clip_{label}"
                )
            ] = logistic_aipw_raw

            row[
                f"hgb_ipcw_clip_{label}"
            ] = hgb_ipcw

            row[
                f"hgb_aipw_clip_{label}"
            ] = hgb_aipw

            row[
                f"hgb_aipw_raw_clip_{label}"
            ] = hgb_aipw_raw

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
            "hgb_survival_or",
            "score_pooled_survival_or",
        ),
        (
            "hgb_survival_or",
            "rich_logistic_survival_or",
        ),
        (
            "hgb_survival_or",
            "grouped_km",
        ),
    ]

    for clip in PROPENSITY_CLIPS:
        label = clip_label(
            clip
        )

        pairs.extend(
            [
                (
                    f"hgb_ipcw_clip_{label}",
                    (
                        "rich_logistic_ipcw_"
                        f"clip_{label}"
                    ),
                ),
                (
                    f"hgb_ipcw_clip_{label}",
                    "score_ipcw",
                ),
                (
                    f"hgb_ipcw_clip_{label}",
                    "restricted_complete_case",
                ),
                (
                    f"hgb_aipw_clip_{label}",
                    (
                        "rich_logistic_aipw_"
                        f"clip_{label}"
                    ),
                ),
                (
                    f"hgb_aipw_clip_{label}",
                    "score_aipw",
                ),
                (
                    f"hgb_aipw_clip_{label}",
                    "restricted_complete_case",
                ),
            ]
        )

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

    for method, comparator in pairs:
        for metric in comparison_metrics:
            point_rows.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(
                        point_lookup.loc[
                            method,
                            metric,
                        ]
                        - point_lookup.loc[
                            comparator,
                            metric,
                        ]
                    ),
                }
            )

            left = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        method
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: "method_value",
                    }
                )
            )

            right = (
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

            paired = left.merge(
                right,
                on="bootstrap_id",
                validate="one_to_one",
            )

            difference = (
                paired[
                    "method_value"
                ]
                - paired[
                    "comparator_value"
                ]
            )

            interval_rows.append(
                {
                    "method": method,
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
                        difference.notna().sum()
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
        "hgb_observation_probability"
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
                    raw_probability.min()
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


def maximum_reproduction_error(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    mapping: dict[str, str],
) -> float:
    current_lookup = current.set_index(
        "method"
    )

    previous_lookup = previous.set_index(
        "method"
    )

    differences = []

    for previous_method, current_method in (
        mapping.items()
    ):
        for metric in [
            "bin_curve_mae",
            "event_rate_error",
            "ici_error",
            "maximum_bin_error",
        ]:
            differences.append(
                abs(
                    float(
                        current_lookup.loc[
                            current_method,
                            metric,
                        ]
                    )
                    - float(
                        previous_lookup.loc[
                            previous_method,
                            metric,
                        ]
                    )
                )
            )

    return max(
        differences
    )


def build_validation_checks(
    *,
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    score_summary: pd.DataFrame,
    rich_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    diagnostics: pd.DataFrame,
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
            FORBIDDEN_FEATURES
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

    rich_mapping = {
        "rich_pooled_survival_or": (
            "rich_logistic_survival_or"
        ),
        "rich_ipcw_clip_0p01": (
            "rich_logistic_ipcw_clip_0p01"
        ),
        "rich_ipcw_clip_0p025": (
            "rich_logistic_ipcw_clip_0p025"
        ),
        "rich_ipcw_clip_0p05": (
            "rich_logistic_ipcw_clip_0p05"
        ),
        "rich_aipw_clip_0p01": (
            "rich_logistic_aipw_clip_0p01"
        ),
        "rich_aipw_clip_0p025": (
            "rich_logistic_aipw_clip_0p025"
        ),
        "rich_aipw_clip_0p05": (
            "rich_logistic_aipw_clip_0p05"
        ),
    }

    score_error = maximum_reproduction_error(
        metrics,
        score_summary,
        score_mapping,
    )

    rich_error = maximum_reproduction_error(
        metrics,
        rich_summary,
        rich_mapping,
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

    manifest_count = int(
        manifest[
            "nuisance_model_feature"
        ]
        .astype(
            bool
        )
        .sum()
    )

    iterations_positive = bool(
        (
            diagnostics[
                [
                    "survival_iterations",
                    "observation_iterations",
                ]
            ]
            > 0
        ).all().all()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "all 21 frozen baseline "
                    "features available"
                ),
                "value": manifest_count,
                "threshold": 21,
                "comparison": "==",
                "passed": (
                    manifest_count
                    == 21
                ),
            },
            {
                "check": (
                    "no forbidden variables in "
                    "nonlinear feature matrix"
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
                    "all nonlinear cross-fitted "
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
                    "HGB outcome risks remain "
                    "within probability bounds"
                ),
                "value": (
                    f"{data['hgb_outcome_regression_risk'].min()},"
                    f"{data['hgb_outcome_regression_risk'].max()}"
                ),
                "threshold": "[0,1]",
                "comparison": "within",
                "passed": bool(
                    data[
                        "hgb_outcome_regression_risk"
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
                    "HGB observation probabilities "
                    "strictly within bounds"
                ),
                "value": (
                    f"{data['hgb_observation_probability'].min()},"
                    f"{data['hgb_observation_probability'].max()}"
                ),
                "threshold": "(0,1)",
                "comparison": "within",
                "passed": bool(
                    (
                        data[
                            "hgb_observation_probability"
                        ]
                        > 0
                    ).all()
                    and (
                        data[
                            "hgb_observation_probability"
                        ]
                        < 1
                    ).all()
                ),
            },
            {
                "check": (
                    "all nonlinear models performed "
                    "at least one boosting iteration"
                ),
                "value": int(
                    iterations_positive
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    iterations_positive
                ),
            },
            {
                "check": (
                    "score-only benchmark reproduced"
                ),
                "value": score_error,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    score_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "rich logistic benchmark reproduced"
                ),
                "value": rich_error,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    rich_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all estimator curves within "
                    "probability bounds"
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
        rich_summary,
        bin_edges,
    ) = load_inputs(
        rich_path=Path(
            args.rich_input
        ),
        manifest_path=Path(
            args.feature_manifest
        ),
        nuisance_path=Path(
            args.logistic_nuisance
        ),
        score_summary_path=Path(
            args.score_summary
        ),
        rich_summary_path=Path(
            args.rich_summary
        ),
        n_bins=args.n_bins,
    )

    print(
        "Phase 5G-C nonlinear estimators"
    )

    print(
        f"  admissions={len(data):,}"
    )

    print(
        f"  frozen baseline features="
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
        "histogram-gradient-boosting models..."
    )

    (
        hgb_outcome_risk,
        hgb_observation_probability,
        prediction_count,
        fold_diagnostics,
    ) = cross_fitted_hgb_predictions(
        data,
        horizon=args.horizon,
        repeats=args.crossfit_repeats,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=(
            args.max_leaf_nodes
        ),
        min_samples_leaf=(
            args.min_samples_leaf
        ),
        l2_regularization=(
            args.l2_regularization
        ),
        seed=args.seed,
    )

    data[
        "hgb_outcome_regression_risk"
    ] = hgb_outcome_risk

    data[
        "hgb_observation_probability"
    ] = hgb_observation_probability

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
        rich_summary=rich_summary,
        metrics=metrics,
        curves=curves,
        diagnostics=fold_diagnostics,
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
        "phase5g_nonlinear_estimators_benchmarked"
        if checks[
            "passed"
        ].all()
        else (
            "phase5g_nonlinear_estimators_"
            "require_review"
        )
    )

    status = (
        "phase5g_nonlinear_estimators_smoke_completed"
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
            "hgb_outcome_regression_risk",
            "hgb_observation_probability",
        ]
    ].copy()

    nuisance_output.to_csv(
        output_dir
        / "cross_fitted_hgb_nuisance_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    curves.to_csv(
        output_dir
        / "nonlinear_estimator_bin_curves.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "nonlinear_estimator_metric_summary.csv",
        index=False,
    )

    ranking.to_csv(
        output_dir
        / "nonlinear_estimator_ranking.csv",
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
        / "hgb_positivity_diagnostics.csv",
        index=False,
    )

    fold_diagnostics.to_csv(
        output_dir
        / "hgb_crossfit_fold_diagnostics.csv",
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
        "phase": "5G-C",
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
        "crossfit_repeats": (
            args.crossfit_repeats
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "nonlinear_family": (
            "HistGradientBoostingClassifier"
        ),
        "hyperparameters": {
            "early_stopping": False,
            "learning_rate": (
                args.learning_rate
            ),
            "max_iter": args.max_iter,
            "max_leaf_nodes": (
                args.max_leaf_nodes
            ),
            "min_samples_leaf": (
                args.min_samples_leaf
            ),
            "l2_regularization": (
                args.l2_regularization
            ),
        },
        "oracle_use": (
            "Linked outcome used only for final "
            "retrospective curve evaluation."
        ),
        "inference": (
            "Patient bootstrap conditional on repeated "
            "cross-fitted nonlinear nuisance predictions."
        ),
        "scope": (
            "Tests whether nonlinear admission-time "
            "nuisance models materially improve on the "
            "frozen score-only and regularized-logistic "
            "benchmarks."
        ),
    }

    (
        output_dir
        / "nonlinear_estimator_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5g_nonlinear_status.txt"
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
        "\nPaired curve-MAE differences"
    )

    print(
        difference_intervals.loc[
            difference_intervals[
                "metric"
            ].eq(
                "bin_curve_mae"
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nHGB positivity diagnostics"
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
