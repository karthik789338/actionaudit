"""Discrete-time nuisance models and right-censoring IPCW estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .km import kaplan_meier_event_risk


TIME_FEATURES = [
    "time_day_scaled",
    "time_day_squared",
    "time_day_log",
]


@dataclass
class SurvivalEstimatorResult:
    """Outputs from corrected right-censoring estimators."""

    bin_summary: pd.DataFrame
    scalar_summary: pd.Series
    diagnostics: pd.Series
    bin_edges: np.ndarray


def expand_person_period(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    horizon: int,
    target: str,
) -> pd.DataFrame:
    """Expand one patient row into discrete at-risk day intervals."""

    if target not in {
        "event",
        "censor",
    }:
        raise ValueError(
            "target must be 'event' or 'censor'"
        )

    required = {
        "restricted_time",
        "restricted_event",
        "restricted_censored_by_discharge",
    }

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing survival columns: {missing}"
        )

    if horizon < 1:
        raise ValueError(
            "horizon must be positive"
        )

    end_day = np.ceil(
        np.minimum(
            frame["restricted_time"]
            .astype(float)
            .to_numpy(),
            horizon,
        )
    ).astype(int)

    end_day = np.clip(
        end_day,
        1,
        horizon,
    )

    repeated_indices = np.repeat(
        np.arange(len(frame)),
        end_day,
    )

    expanded = (
        frame.iloc[
            repeated_indices
        ][list(feature_columns)]
        .reset_index(drop=True)
    )

    days = np.concatenate(
        [
            np.arange(
                1,
                patient_end_day + 1,
                dtype=int,
            )
            for patient_end_day in end_day
        ]
    )

    expanded[
        "time_day_scaled"
    ] = days / horizon

    expanded[
        "time_day_squared"
    ] = np.square(
        days / horizon
    )

    expanded[
        "time_day_log"
    ] = np.log1p(days)

    hazard_target = np.zeros(
        len(expanded),
        dtype=int,
    )

    final_row_indices = (
        np.cumsum(end_day)
        - 1
    )

    if target == "event":
        final_targets = (
            frame[
                "restricted_event"
            ]
            .astype(int)
            .to_numpy()
        )
    else:
        final_targets = (
            frame[
                "restricted_censored_by_discharge"
            ]
            .astype(int)
            .to_numpy()
        )

    hazard_target[
        final_row_indices
    ] = final_targets

    expanded[
        "hazard_target"
    ] = hazard_target

    expanded[
        "source_patient_row"
    ] = repeated_indices

    expanded[
        "time_day"
    ] = days

    return expanded


def predict_hazard_grid(
    model,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    horizon: int,
    batch_size: int = 50000,
) -> np.ndarray:
    """Predict daily hazards for every patient through the horizon."""

    n = len(frame)

    patient_indices = np.repeat(
        np.arange(n),
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

    prediction_frame = (
        frame.iloc[
            patient_indices
        ][list(feature_columns)]
        .reset_index(drop=True)
    )

    prediction_frame[
        "time_day_scaled"
    ] = days / horizon

    prediction_frame[
        "time_day_squared"
    ] = np.square(
        days / horizon
    )

    prediction_frame[
        "time_day_log"
    ] = np.log1p(days)

    model_features = (
        list(feature_columns)
        + TIME_FEATURES
    )

    probability_parts: list[np.ndarray] = []

    for start in range(
        0,
        len(prediction_frame),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(prediction_frame),
        )

        probability_parts.append(
            model.predict_proba(
                prediction_frame.iloc[
                    start:stop
                ][model_features]
            )[:, 1]
        )

    hazards = np.concatenate(
        probability_parts
    ).reshape(
        n,
        horizon,
    )

    return np.clip(
        hazards,
        1e-8,
        1.0 - 1e-8,
    )


def construct_survival_nuisance_predictions(
    frame: pd.DataFrame,
    *,
    event_hazards: np.ndarray,
    censor_hazards: np.ndarray,
    horizon: int,
    censor_survival_clip: float,
) -> pd.DataFrame:
    """Create survival outcome-regression and IPCW quantities."""

    n = len(frame)

    expected_shape = (
        n,
        horizon,
    )

    if event_hazards.shape != expected_shape:
        raise ValueError(
            "event_hazards has incorrect dimensions"
        )

    if censor_hazards.shape != expected_shape:
        raise ValueError(
            "censor_hazards has incorrect dimensions"
        )

    if not 0 < censor_survival_clip < 1:
        raise ValueError(
            "censor_survival_clip must lie in (0, 1)"
        )

    result = frame.copy()

    event_survival = np.cumprod(
        1.0 - event_hazards,
        axis=1,
    )

    event_risk = (
        1.0
        - event_survival[:, -1]
    )

    censor_survival_through_day = np.cumprod(
        1.0 - censor_hazards,
        axis=1,
    )

    censor_survival_before_day = np.ones_like(
        censor_survival_through_day
    )

    censor_survival_before_day[
        :,
        1:
    ] = censor_survival_through_day[
        :,
        :-1
    ]

    observed_day = np.ceil(
        np.minimum(
            result[
                "restricted_time"
            ].astype(float).to_numpy(),
            horizon,
        )
    ).astype(int)

    observed_day = np.clip(
        observed_day,
        1,
        horizon,
    )

    row_index = np.arange(n)

    raw_g_before = censor_survival_before_day[
        row_index,
        observed_day - 1,
    ]

    clipped_g_before = np.clip(
        raw_g_before,
        censor_survival_clip,
        1.0,
    )

    restricted_event = (
        result[
            "restricted_event"
        ]
        .astype(int)
        .to_numpy()
    )

    ipcw_weight = np.where(
        restricted_event == 1,
        1.0 / clipped_g_before,
        0.0,
    )

    ipcw_event_contribution = (
        restricted_event
        * ipcw_weight
    )

    result[
        "survival_outcome_regression"
    ] = event_risk

    result[
        "raw_censor_survival_before_observed_time"
    ] = raw_g_before

    result[
        "clipped_censor_survival_before_observed_time"
    ] = clipped_g_before

    result[
        "ipcw_event_weight"
    ] = ipcw_weight

    result[
        "ipcw_event_contribution"
    ] = ipcw_event_contribution

    result[
        "censor_survival_at_horizon"
    ] = censor_survival_through_day[
        :,
        -1
    ]

    result[
        "event_hazard_risk_at_horizon"
    ] = event_risk

    return result


def _weighted_metric(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Return normalized weighted average over available values."""

    valid = (
        values.notna()
        & weights.notna()
    )

    if not valid.any():
        return float("nan")

    selected_weights = weights.loc[
        valid
    ]

    selected_weights = (
        selected_weights
        / selected_weights.sum()
    )

    return float(
        (
            values.loc[valid]
            * selected_weights
        ).sum()
    )


def evaluate_survival_estimators(
    frame: pd.DataFrame,
    *,
    horizon: int,
    bin_edges: np.ndarray,
    censor_survival_clip: float,
) -> SurvivalEstimatorResult:
    """Compare corrected IPCW and survival OR against reference."""

    required = {
        "prediction",
        "outcome",
        "observed",
        "restricted_time",
        "restricted_event",
        "restricted_censored_by_discharge",
        "survival_outcome_regression",
        "raw_censor_survival_before_observed_time",
        "ipcw_event_weight",
        "ipcw_event_contribution",
    }

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing estimator columns: {missing}"
        )

    data = frame.copy()

    data["_bin"] = pd.cut(
        data["prediction"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    if data["_bin"].isna().any():
        raise ValueError(
            "Some predictions were not assigned to bins"
        )

    n_total = len(data)
    rows: list[dict[str, object]] = []

    for bin_id, group in data.groupby(
        "_bin",
        sort=True,
        observed=True,
    ):
        observed_group = group.loc[
            group[
                "observed"
            ].astype(int).eq(1)
        ]

        event_group = group.loc[
            group[
                "restricted_event"
            ].astype(int).eq(1)
        ]

        reference_risk = float(
            group["outcome"].mean()
        )

        complete_case_risk = (
            float(
                observed_group[
                    "outcome"
                ].mean()
            )
            if not observed_group.empty
            else float("nan")
        )

        km_risk = kaplan_meier_event_risk(
            group[
                "restricted_time"
            ].to_numpy(),
            group[
                "restricted_event"
            ].to_numpy(),
            horizon,
        )

        ipcw_event_risk = float(
            group[
                "ipcw_event_contribution"
            ].mean()
        )

        survival_or_risk = float(
            group[
                "survival_outcome_regression"
            ].mean()
        )

        event_weights = event_group[
            "ipcw_event_weight"
        ]

        squared_weight_sum = float(
            np.square(
                event_weights
            ).sum()
        )

        event_weight_ess = (
            float(
                np.square(
                    event_weights.sum()
                )
                / squared_weight_sum
            )
            if (
                not event_weights.empty
                and squared_weight_sum > 0
            )
            else 0.0
        )

        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(group)),
                "reference_weight": float(
                    len(group) / n_total
                ),
                "mean_prediction": float(
                    group["prediction"].mean()
                ),
                "reference_risk": reference_risk,
                "complete_case_risk": complete_case_risk,
                "kaplan_meier_risk": km_risk,
                "survival_or_risk": survival_or_risk,
                "ipcw_event_risk": ipcw_event_risk,
                "ipcw_event_risk_clipped_for_plot": float(
                    np.clip(
                        ipcw_event_risk,
                        0.0,
                        1.0,
                    )
                ),
                "n_observed_at_horizon": int(
                    group["observed"].sum()
                ),
                "n_restricted_events": int(
                    group[
                        "restricted_event"
                    ].sum()
                ),
                "n_discharge_censored": int(
                    group[
                        "restricted_censored_by_discharge"
                    ].sum()
                ),
                "mean_censor_survival_before_event": (
                    float(
                        event_group[
                            "raw_censor_survival_before_observed_time"
                        ].mean()
                    )
                    if not event_group.empty
                    else float("nan")
                ),
                "maximum_event_weight": (
                    float(
                        event_weights.max()
                    )
                    if not event_weights.empty
                    else float("nan")
                ),
                "event_weight_effective_sample_size": (
                    event_weight_ess
                ),
            }
        )

    bin_summary = pd.DataFrame(rows)

    estimator_columns = {
        "reference": "reference_risk",
        "complete_case": "complete_case_risk",
        "kaplan_meier": "kaplan_meier_risk",
        "survival_or": "survival_or_risk",
        "ipcw_event": "ipcw_event_risk",
    }

    weights = bin_summary[
        "reference_weight"
    ]

    prediction = bin_summary[
        "mean_prediction"
    ]

    reference = bin_summary[
        "reference_risk"
    ]

    scalar_values: dict[str, float] = {}

    for estimator_name, column in estimator_columns.items():
        estimate = bin_summary[column]

        scalar_values[
            f"{estimator_name}_ici"
        ] = _weighted_metric(
            (estimate - prediction).abs(),
            weights,
        )

        if estimator_name != "reference":
            scalar_values[
                f"{estimator_name}_mae_to_reference"
            ] = _weighted_metric(
                (estimate - reference).abs(),
                weights,
            )

            scalar_values[
                f"{estimator_name}_signed_gap"
            ] = _weighted_metric(
                estimate - reference,
                weights,
            )

    complete_case_mae = scalar_values[
        "complete_case_mae_to_reference"
    ]

    for estimator_name in (
        "kaplan_meier",
        "survival_or",
        "ipcw_event",
    ):
        scalar_values[
            f"{estimator_name}_improvement_over_complete_case"
        ] = (
            complete_case_mae
            - scalar_values[
                f"{estimator_name}_mae_to_reference"
            ]
        )

    restricted_events = data.loc[
        data[
            "restricted_event"
        ].astype(int).eq(1)
    ]

    raw_g_event = restricted_events[
        "raw_censor_survival_before_observed_time"
    ]

    event_weights = restricted_events[
        "ipcw_event_weight"
    ]

    event_weight_square_sum = float(
        np.square(
            event_weights
        ).sum()
    )

    overall_ess = (
        float(
            np.square(
                event_weights.sum()
            )
            / event_weight_square_sum
        )
        if event_weight_square_sum > 0
        else 0.0
    )

    diagnostics = pd.Series(
        {
            "n": float(len(data)),
            "restricted_event_count": float(
                len(restricted_events)
            ),
            "discharge_censored_count": float(
                data[
                    "restricted_censored_by_discharge"
                ].sum()
            ),
            "censor_survival_clip": float(
                censor_survival_clip
            ),
            "event_g_min": float(
                raw_g_event.min()
            ),
            "event_g_q01": float(
                raw_g_event.quantile(0.01)
            ),
            "event_g_q05": float(
                raw_g_event.quantile(0.05)
            ),
            "event_g_median": float(
                raw_g_event.median()
            ),
            "event_g_q95": float(
                raw_g_event.quantile(0.95)
            ),
            "event_g_max": float(
                raw_g_event.max()
            ),
            "fraction_event_g_clipped": float(
                (
                    raw_g_event
                    < censor_survival_clip
                ).mean()
            ),
            "maximum_event_weight": float(
                event_weights.max()
            ),
            "event_weight_effective_sample_size": (
                overall_ess
            ),
            "ipcw_event_risk_min_bin": float(
                bin_summary[
                    "ipcw_event_risk"
                ].min()
            ),
            "ipcw_event_risk_max_bin": float(
                bin_summary[
                    "ipcw_event_risk"
                ].max()
            ),
        },
        name="value",
    )

    return SurvivalEstimatorResult(
        bin_summary=bin_summary,
        scalar_summary=pd.Series(
            scalar_values,
            name="value",
        ),
        diagnostics=diagnostics,
        bin_edges=bin_edges,
    )
