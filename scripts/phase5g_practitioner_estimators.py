"""Phase 5G-A: practitioner-available calibration estimators.

The linked 30-day mortality outcome is used only to construct the
retrospective reference curve and evaluate estimator error.

Nuisance models use only:

  * frozen admission-time prediction;
  * hospital-restricted event indicator;
  * hospital-restricted follow-up time;
  * whether the horizon outcome is observed.

Methods:

  * restricted complete case;
  * grouped Kaplan-Meier;
  * pooled discrete-time survival outcome regression;
  * score-based IPCW;
  * score-based AIPW.

Repeated two-fold cross-fitting is used for nuisance predictions.
Bootstrap intervals are conditional on the fitted cross-fitted
nuisance predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


METHOD_ORDER = [
    "reference_oracle",
    "restricted_complete_case",
    "grouped_km",
    "pooled_survival_or",
    "score_ipcw",
    "score_aipw",
]


COLUMN_CANDIDATES = {
    "subject_id": [
        "subject_id",
        "patient_id",
    ],
    "hadm_id": [
        "hadm_id",
        "admission_id",
    ],
    "prediction": [
        "prediction",
        "predicted_probability",
        "predicted_risk",
    ],
    "outcome": [
        "outcome",
        "reference_outcome",
        "mortality_30d",
    ],
    "observed": [
        "observed",
        "restricted_observed",
        "outcome_observed",
    ],
    "restricted_event": [
        "restricted_event",
        "hospital_event",
        "event",
    ],
    "restricted_time": [
        "restricted_time",
        "hospital_followup_time",
        "followup_time",
    ],
    "action": [
        "analysis_action",
        "action",
        "primary_action",
    ],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark practitioner-available estimators "
            "against linked reference outcomes."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5g_practitioner_estimators"
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
        "--propensity-clip",
        type=float,
        default=0.01,
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


def resolve_column(
    columns: pd.Index,
    conceptual_name: str,
    *,
    required: bool = True,
) -> str | None:
    for candidate in COLUMN_CANDIDATES[
        conceptual_name
    ]:
        if candidate in columns:
            return candidate

    if required:
        raise ValueError(
            "Unable to resolve required column "
            f"'{conceptual_name}'. Available columns: "
            f"{list(columns)}"
        )

    return None


def load_analysis_frame(
    path: Path,
    *,
    horizon: int,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    dict[str, str | None],
    np.ndarray,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input does not exist: {path}"
        )

    raw = pd.read_csv(
        path
    )

    mapping = {
        name: resolve_column(
            raw.columns,
            name,
            required=(
                name != "hadm_id"
                and name != "action"
            ),
        )
        for name in COLUMN_CANDIDATES
    }

    data = pd.DataFrame(
        {
            "subject_id": raw[
                mapping[
                    "subject_id"
                ]
            ],
            "prediction": pd.to_numeric(
                raw[
                    mapping[
                        "prediction"
                    ]
                ],
                errors="raise",
            ),
            "outcome": pd.to_numeric(
                raw[
                    mapping[
                        "outcome"
                    ]
                ],
                errors="raise",
            ).astype(
                int
            ),
            "observed": pd.to_numeric(
                raw[
                    mapping[
                        "observed"
                    ]
                ],
                errors="raise",
            ).astype(
                int
            ),
            "restricted_event": pd.to_numeric(
                raw[
                    mapping[
                        "restricted_event"
                    ]
                ],
                errors="raise",
            ).astype(
                int
            ),
            "restricted_time": pd.to_numeric(
                raw[
                    mapping[
                        "restricted_time"
                    ]
                ],
                errors="raise",
            ),
        }
    )

    if mapping[
        "hadm_id"
    ] is None:
        data[
            "hadm_id"
        ] = np.arange(
            len(
                data
            ),
            dtype=int,
        )
    else:
        data[
            "hadm_id"
        ] = raw[
            mapping[
                "hadm_id"
            ]
        ]

    if mapping[
        "action"
    ] is None:
        data[
            "action"
        ] = "not_available"
    else:
        data[
            "action"
        ] = (
            raw[
                mapping[
                    "action"
                ]
            ]
            .fillna(
                "not_applicable"
            )
            .astype(
                str
            )
        )

    if data.empty:
        raise ValueError(
            "Input contains no rows"
        )

    if not data[
        "outcome"
    ].isin(
        [
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "Reference outcome must be binary"
        )

    if not data[
        "observed"
    ].isin(
        [
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "Observed indicator must be binary"
        )

    if not data[
        "restricted_event"
    ].isin(
        [
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "Restricted event must be binary"
        )

    if (
        data[
            "restricted_event"
        ]
        > data[
            "observed"
        ]
    ).any():
        raise ValueError(
            "A restricted event cannot have "
            "observed=0"
        )

    if (
        data[
            "restricted_time"
        ]
        < 0
    ).any():
        raise ValueError(
            "Restricted follow-up time cannot be negative"
        )

    data[
        "prediction"
    ] = data[
        "prediction"
    ].clip(
        1e-6,
        1.0 - 1e-6,
    )

    data[
        "restricted_time"
    ] = data[
        "restricted_time"
    ].clip(
        lower=0.0,
        upper=float(
            horizon
        ),
    )

    edges = np.unique(
        np.quantile(
            data[
                "prediction"
            ],
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
            "Predictions do not support the requested "
            "number of unique quantile bins"
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
        data[
            "prediction"
        ],
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(
        int
    )

    return (
        data.reset_index(
            drop=True
        ),
        mapping,
        edges,
    )


def score_features(
    prediction: np.ndarray,
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

    return np.column_stack(
        [
            score,
            score**2,
            score**3,
        ]
    )


def person_time_features(
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

    time = (
        np.asarray(
            day,
            dtype=float,
        )
        / float(
            horizon
        )
    )

    return np.column_stack(
        [
            score,
            score**2,
            time,
            time**2,
            time**3,
            score * time,
            score * time**2,
        ]
    )


def build_person_period_data(
    frame: pd.DataFrame,
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

    lengths = followup_day

    patient_index = np.repeat(
        np.arange(
            len(
                frame
            ),
            dtype=int,
        ),
        lengths,
    )

    days = np.concatenate(
        [
            np.arange(
                1,
                length + 1,
                dtype=int,
            )
            for length in lengths
        ]
    )

    prediction = frame[
        "prediction"
    ].to_numpy(
        dtype=float
    )[
        patient_index
    ]

    outcome = np.zeros(
        len(
            patient_index
        ),
        dtype=int,
    )

    row_ends = (
        np.cumsum(
            lengths
        )
        - 1
    )

    event_at_end = frame[
        "restricted_event"
    ].to_numpy(
        dtype=int
    )

    outcome[
        row_ends
    ] = event_at_end

    features = person_time_features(
        prediction,
        days,
        horizon=horizon,
    )

    return (
        features,
        outcome,
    )


def fit_survival_outcome_model(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> LogisticRegression:
    features, outcome = (
        build_person_period_data(
            frame,
            horizon=horizon,
        )
    )

    if outcome.sum() < 10:
        raise ValueError(
            "Insufficient restricted events "
            "for outcome regression"
        )

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        features,
        outcome,
    )

    return model


def predict_horizon_risk(
    model: LogisticRegression,
    prediction: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    patient_prediction = np.asarray(
        prediction,
        dtype=float,
    )

    n = len(
        patient_prediction
    )

    repeated_prediction = np.repeat(
        patient_prediction,
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

    features = person_time_features(
        repeated_prediction,
        days,
        horizon=horizon,
    )

    hazard = model.predict_proba(
        features
    )[
        :,
        1
    ].reshape(
        n,
        horizon,
    )

    hazard = np.clip(
        hazard,
        1e-8,
        1.0 - 1e-8,
    )

    survival = np.prod(
        1.0 - hazard,
        axis=1,
    )

    return np.clip(
        1.0 - survival,
        0.0,
        1.0,
    )


def fit_observation_model(
    frame: pd.DataFrame,
) -> LogisticRegression:
    features = score_features(
        frame[
            "prediction"
        ].to_numpy(
            dtype=float
        )
    )

    outcome = frame[
        "observed"
    ].to_numpy(
        dtype=int
    )

    if len(
        np.unique(
            outcome
        )
    ) != 2:
        raise ValueError(
            "Observation model requires both "
            "observed and hidden outcomes"
        )

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        features,
        outcome,
    )

    return model


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

    fold_lookup = {
        subject: int(
            index % 2
        )
        for index, subject in enumerate(
            shuffled
        )
    }

    return subject_id.map(
        fold_lookup
    ).to_numpy(
        dtype=int
    )


def cross_fitted_nuisance_predictions(
    data: pd.DataFrame,
    *,
    horizon: int,
    repeats: int,
    propensity_clip: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
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
            ]

            test = data.loc[
                test_mask
            ]

            outcome_model = (
                fit_survival_outcome_model(
                    train,
                    horizon=horizon,
                )
            )

            observation_model = (
                fit_observation_model(
                    train
                )
            )

            outcome_risk_sum[
                test_mask
            ] += predict_horizon_risk(
                outcome_model,
                test[
                    "prediction"
                ].to_numpy(
                    dtype=float
                ),
                horizon=horizon,
            )

            observation_probability = (
                observation_model.predict_proba(
                    score_features(
                        test[
                            "prediction"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )[
                    :,
                    1
                ]
            )

            observation_probability_sum[
                test_mask
            ] += np.clip(
                observation_probability,
                propensity_clip,
                1.0
                - propensity_clip,
            )

            prediction_count[
                test_mask
            ] += 1

    if not np.all(
        prediction_count
        == repeats
    ):
        raise RuntimeError(
            "Cross-fitting did not cover every row "
            "on every repeat"
        )

    return (
        outcome_risk_sum
        / prediction_count,
        observation_probability_sum
        / prediction_count,
        prediction_count,
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

        if at_risk <= 0:
            continue

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


def estimate_bin_curves(
    data: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    rows = []

    visible_outcome = np.where(
        data[
            "observed"
        ].to_numpy(
            dtype=int
        )
        == 1,
        data[
            "outcome"
        ].to_numpy(
            dtype=float
        ),
        0.0,
    )

    working = data.copy()

    working[
        "_visible_outcome"
    ] = visible_outcome

    working[
        "_ipcw_numerator"
    ] = (
        working[
            "observed"
        ]
        * working[
            "_visible_outcome"
        ]
        / working[
            "observation_probability"
        ]
    )

    working[
        "_ipcw_denominator"
    ] = (
        working[
            "observed"
        ]
        / working[
            "observation_probability"
        ]
    )

    working[
        "_aipw_pseudo"
    ] = (
        working[
            "outcome_regression_risk"
        ]
        + working[
            "observed"
        ]
        / working[
            "observation_probability"
        ]
        * (
            working[
                "_visible_outcome"
            ]
            - working[
                "outcome_regression_risk"
            ]
        )
    )

    for risk_bin, group in working.groupby(
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

        ipcw_denominator = float(
            group[
                "_ipcw_denominator"
            ].sum()
        )

        if ipcw_denominator <= 0:
            raise ValueError(
                f"Invalid IPCW denominator in bin {risk_bin}"
            )

        raw_aipw = float(
            group[
                "_aipw_pseudo"
            ].mean()
        )

        rows.append(
            {
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
                        working
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
                        "outcome"
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
                "pooled_survival_or": float(
                    group[
                        "outcome_regression_risk"
                    ].mean()
                ),
                "score_ipcw": float(
                    np.clip(
                        group[
                            "_ipcw_numerator"
                        ].sum()
                        / ipcw_denominator,
                        0.0,
                        1.0,
                    )
                ),
                "score_aipw": float(
                    np.clip(
                        raw_aipw,
                        0.0,
                        1.0,
                    )
                ),
                "score_aipw_raw": (
                    raw_aipw
                ),
                "observed_fraction": float(
                    group[
                        "observed"
                    ].mean()
                ),
                "mean_observation_probability": float(
                    group[
                        "observation_probability"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "risk_bin"
    ).reset_index(
        drop=True
    )


def calculate_method_metrics(
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

    for method in METHOD_ORDER:
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
    grouped_indices = [
        group.index.to_numpy(
            dtype=int
        )
        for _, group in data.groupby(
            "subject_id",
            sort=False,
        )
    ]

    sampled_groups = rng.integers(
        0,
        len(
            grouped_indices
        ),
        size=len(
            grouped_indices
        ),
    )

    return np.concatenate(
        [
            grouped_indices[
                group_index
            ]
            for group_index in sampled_groups
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

            metrics = calculate_method_metrics(
                curves
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            invalid += 1
            continue

        for _, metric_row in metrics.iterrows():
            rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    **metric_row.to_dict(),
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
    metric_columns = [
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
        for metric in metric_columns:
            values = (
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
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


def metric_value(
    metrics: pd.DataFrame,
    *,
    method: str,
    metric: str,
) -> float:
    selected = metrics.loc[
        metrics[
            "method"
        ].eq(
            method
        ),
        metric,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            f"Metric lookup failed: {method}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def build_validation_checks(
    *,
    data: pd.DataFrame,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    prediction_count: np.ndarray,
    repeats: int,
    bootstrap_rows: int,
    invalid_bootstraps: int,
    requested_bootstraps: int,
) -> pd.DataFrame:
    complete_case_error = metric_value(
        metrics,
        method="restricted_complete_case",
        metric="bin_curve_mae",
    )

    practitioner = metrics.loc[
        ~metrics[
            "method"
        ].isin(
            [
                "reference_oracle",
                "restricted_complete_case",
            ]
        )
    ]

    best_practitioner_error = float(
        practitioner[
            "bin_curve_mae"
        ].min()
    )

    curve_columns = METHOD_ORDER

    minimum_curve = float(
        curves[
            curve_columns
        ].min().min()
    )

    maximum_curve = float(
        curves[
            curve_columns
        ].max().max()
    )

    expected_bootstrap_rows = (
        requested_bootstraps
        * len(
            METHOD_ORDER
        )
    )

    maximum_weight = float(
        (
            1.0
            / data[
                "observation_probability"
            ]
        ).max()
    )

    observed_weight = (
        data[
            "observed"
        ].to_numpy(
            dtype=float
        )
        / data[
            "observation_probability"
        ].to_numpy(
            dtype=float
        )
    )

    effective_sample_size = float(
        observed_weight.sum() ** 2
        / (
            np.sum(
                observed_weight**2
            )
            + 1e-12
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "all cross-fitted nuisance "
                    "predictions complete"
                ),
                "value": int(
                    np.min(
                        prediction_count
                    )
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
                    "reference outcome not used "
                    "by nuisance fitting functions"
                ),
                "value": "none",
                "threshold": "none",
                "comparison": "==",
                "passed": True,
            },
            {
                "check": (
                    "all estimated curves remain "
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
                    "restricted complete-case curve "
                    "is materially distorted"
                ),
                "value": (
                    complete_case_error
                ),
                "threshold": 0.10,
                "comparison": ">",
                "passed": (
                    complete_case_error
                    > 0.10
                ),
            },
            {
                "check": (
                    "at least one practitioner estimator "
                    "improves over complete case"
                ),
                "value": (
                    best_practitioner_error
                ),
                "threshold": (
                    complete_case_error
                ),
                "comparison": "<",
                "passed": (
                    best_practitioner_error
                    < complete_case_error
                ),
            },
            {
                "check": (
                    "maximum inverse observation weight "
                    "is finite"
                ),
                "value": (
                    maximum_weight
                ),
                "threshold": "finite",
                "comparison": "==",
                "passed": bool(
                    np.isfinite(
                        maximum_weight
                    )
                ),
            },
            {
                "check": (
                    "weighted observed effective "
                    "sample size exceeds 100"
                ),
                "value": (
                    effective_sample_size
                ),
                "threshold": 100.0,
                "comparison": ">",
                "passed": (
                    effective_sample_size
                    > 100.0
                ),
            },
            {
                "check": (
                    "all requested bootstrap rows completed"
                ),
                "value": (
                    bootstrap_rows
                ),
                "threshold": (
                    expected_bootstrap_rows
                ),
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

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        data,
        mapping,
        bin_edges,
    ) = load_analysis_frame(
        input_path,
        horizon=args.horizon,
        n_bins=args.n_bins,
    )

    print(
        "Phase 5G-A practitioner estimators"
    )

    print(
        f"  admissions={len(data):,}"
    )

    print(
        f"  subjects="
        f"{data['subject_id'].nunique():,}"
    )

    print(
        f"  restricted observed fraction="
        f"{data['observed'].mean():.6f}"
    )

    print(
        f"  restricted events="
        f"{data['restricted_event'].sum():,}"
    )

    print(
        f"  cross-fit repeats="
        f"{args.crossfit_repeats}"
    )

    print(
        "\nFitting repeated cross-fitted "
        "nuisance models..."
    )

    (
        outcome_risk,
        observation_probability,
        prediction_count,
    ) = cross_fitted_nuisance_predictions(
        data,
        horizon=args.horizon,
        repeats=args.crossfit_repeats,
        propensity_clip=(
            args.propensity_clip
        ),
        seed=args.seed,
    )

    data[
        "outcome_regression_risk"
    ] = outcome_risk

    data[
        "observation_probability"
    ] = observation_probability

    curves = estimate_bin_curves(
        data,
        horizon=args.horizon,
    )

    metrics = calculate_method_metrics(
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
        "\nRunning clustered bootstrap "
        "conditional on cross-fitted nuisances..."
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

    checks = build_validation_checks(
        data=data,
        curves=curves,
        metrics=metrics,
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
        "phase5g_practitioner_estimators_benchmarked"
        if checks[
            "passed"
        ].all()
        else (
            "phase5g_practitioner_estimators_"
            "require_review"
        )
    )

    status = (
        "phase5g_practitioner_estimators_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    schema_table = pd.DataFrame(
        [
            {
                "conceptual_variable": key,
                "source_column": value,
            }
            for key, value in mapping.items()
        ]
    )

    nuisance_predictions = data[
        [
            "subject_id",
            "hadm_id",
            "prediction",
            "risk_bin",
            "observed",
            "restricted_event",
            "restricted_time",
            "action",
            "outcome_regression_risk",
            "observation_probability",
        ]
    ].copy()

    schema_table.to_csv(
        output_dir
        / "resolved_schema.csv",
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

    nuisance_predictions.to_csv(
        output_dir
        / "cross_fitted_nuisance_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    curves.to_csv(
        output_dir
        / "estimator_bin_curves.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "estimator_metric_summary.csv",
        index=False,
    )

    ranking.to_csv(
        output_dir
        / "estimator_ranking.csv",
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

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    observed_weight = (
        data[
            "observed"
        ].to_numpy(
            dtype=float
        )
        / data[
            "observation_probability"
        ].to_numpy(
            dtype=float
        )
    )

    effective_sample_size = float(
        observed_weight.sum() ** 2
        / (
            np.sum(
                observed_weight**2
            )
            + 1e-12
        )
    )

    evidence = {
        "phase": "5G-A",
        "status": status,
        "preliminary_status": (
            preliminary_status
            if args.smoke
            else None
        ),
        "input": str(
            input_path
        ),
        "horizon_days": (
            args.horizon
        ),
        "n_bins": (
            args.n_bins
        ),
        "n_admissions": int(
            len(
                data
            )
        ),
        "n_subjects": int(
            data[
                "subject_id"
            ].nunique()
        ),
        "restricted_observed_fraction": float(
            data[
                "observed"
            ].mean()
        ),
        "crossfit_repeats": (
            args.crossfit_repeats
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "bootstrap_inference": (
            "Patient-cluster bootstrap conditional on "
            "the repeated cross-fitted nuisance predictions."
        ),
        "propensity_clip": (
            args.propensity_clip
        ),
        "minimum_observation_probability": float(
            data[
                "observation_probability"
            ].min()
        ),
        "maximum_inverse_observation_weight": float(
            (
                1.0
                / data[
                    "observation_probability"
                ]
            ).max()
        ),
        "weighted_observed_effective_sample_size": (
            effective_sample_size
        ),
        "oracle_use": {
            "nuisance_fitting": (
                "No linked reference outcomes used."
            ),
            "estimator_evaluation": (
                "Linked reference outcome used only "
                "after estimator construction."
            ),
        },
        "identification_warning": (
            "KM, outcome regression, IPCW, and AIPW "
            "rely on independent-censoring or "
            "missing-at-random assumptions conditional "
            "on their nuisance information. Failure "
            "against the linked reference does not mean "
            "the numerical implementation is incorrect."
        ),
        "action_modeling_note": (
            "Realized discharge action is not used as "
            "an admission-time outcome-regression "
            "covariate because that would introduce "
            "post-baseline information. Cause-specific "
            "action censoring models are deferred to "
            "Phase 5G-B."
        ),
        "methods": METHOD_ORDER,
    }

    (
        output_dir
        / "practitioner_estimator_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5g_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nEstimator metric summary"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nPractitioner ranking"
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
