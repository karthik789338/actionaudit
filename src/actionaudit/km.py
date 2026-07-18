"""Kaplan-Meier calibration utilities for artificially truncated outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BinnedEstimatorResult:
    """Bin-level and scalar estimator comparisons."""

    bin_summary: pd.DataFrame
    scalar_summary: pd.Series
    bin_edges: np.ndarray


def reference_quantile_edges(
    prediction: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Create fixed quantile edges from the original test cohort."""

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    if prediction.ndim != 1:
        raise ValueError(
            "prediction must be one-dimensional"
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "prediction contains non-finite values"
        )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2"
        )

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.unique(
        np.quantile(
            prediction,
            quantiles,
        )
    )

    if len(edges) < 3:
        raise ValueError(
            "Predictions do not have enough variation"
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def kaplan_meier_event_risk(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
) -> float:
    """Estimate cumulative event risk by the requested horizon.

    Parameters
    ----------
    time:
        Observed event, censoring, or administrative follow-up time.

    event:
        One when the event occurred at `time`; zero when censored.

    horizon:
        Evaluation horizon.

    Returns
    -------
    float
        Kaplan-Meier estimate of 1 - S(horizon).
    """

    time = np.asarray(
        time,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    if len(time) != len(event):
        raise ValueError(
            "time and event lengths differ"
        )

    if len(time) == 0:
        return float("nan")

    if not np.isfinite(time).all():
        raise ValueError(
            "time contains non-finite values"
        )

    if not np.isin(
        event,
        [0, 1],
    ).all():
        raise ValueError(
            "event must contain only 0 and 1"
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive"
        )

    survival = 1.0

    event_times = np.unique(
        time[
            (event == 1)
            & (time <= horizon)
        ]
    )

    for event_time in np.sort(event_times):
        at_risk = int(
            np.sum(time >= event_time)
        )

        deaths = int(
            np.sum(
                (time == event_time)
                & (event == 1)
            )
        )

        if at_risk <= 0:
            continue

        survival *= (
            1.0
            - deaths / at_risk
        )

    return float(
        1.0 - survival
    )


def make_hospital_restricted_survival(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Construct artificial hospital-only survival follow-up.

    An event is observed only when death occurs in hospital by the horizon.
    Alive discharge before the horizon becomes a censoring event.
    Everyone else receives administrative censoring at the horizon.
    """

    required = {
        "death",
        "d.time",
        "hospdead",
        "slos",
    }

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive"
        )

    result = frame.copy()

    death = pd.to_numeric(
        result["death"],
        errors="coerce",
    )

    death_time = pd.to_numeric(
        result["d.time"],
        errors="coerce",
    )

    hospital_death = pd.to_numeric(
        result["hospdead"],
        errors="coerce",
    )

    discharge_time = pd.to_numeric(
        result["slos"],
        errors="coerce",
    )

    if any(
        series.isna().any()
        for series in (
            death,
            death_time,
            hospital_death,
            discharge_time,
        )
    ):
        raise ValueError(
            "Restricted survival variables contain missing values"
        )

    in_hospital_event = (
        death.eq(1)
        & hospital_death.eq(1)
        & death_time.le(horizon)
    )

    alive_discharge_censoring = (
        hospital_death.eq(0)
        & discharge_time.lt(horizon)
    )

    restricted_time = np.where(
        in_hospital_event,
        death_time,
        np.where(
            alive_discharge_censoring,
            discharge_time,
            horizon,
        ),
    )

    result["restricted_time"] = (
        np.minimum(
            restricted_time,
            horizon,
        )
    )

    result["restricted_event"] = (
        in_hospital_event.astype(int)
    )

    result["restricted_censored_by_discharge"] = (
        alive_discharge_censoring.astype(int)
    )

    result["restricted_observed_at_horizon"] = (
        (
            in_hospital_event
            | ~alive_discharge_censoring
        )
        .astype(int)
    )

    return result


def evaluate_binned_estimators(
    frame: pd.DataFrame,
    *,
    horizon: int,
    bin_edges: np.ndarray,
) -> BinnedEstimatorResult:
    """Compare reference, complete-case, and KM calibration."""

    required = {
        "prediction",
        "outcome",
        "observed",
        "restricted_time",
        "restricted_event",
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
            "Some predictions could not be assigned to bins"
        )

    n_total = len(data)
    rows: list[dict[str, object]] = []

    for bin_id, group in data.groupby(
        "_bin",
        sort=True,
        observed=True,
    ):
        observed_group = group.loc[
            group["observed"].astype(int).eq(1)
        ]

        reference_risk = float(
            group["outcome"].mean()
        )

        complete_case_risk = (
            float(
                observed_group["outcome"].mean()
            )
            if not observed_group.empty
            else float("nan")
        )

        km_risk = kaplan_meier_event_risk(
            group["restricted_time"].to_numpy(),
            group["restricted_event"].to_numpy(),
            horizon,
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
                "n_complete_case": int(
                    len(observed_group)
                ),
                "complete_case_fraction": float(
                    len(observed_group)
                    / len(group)
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
                "complete_case_minus_reference": (
                    complete_case_risk
                    - reference_risk
                    if np.isfinite(
                        complete_case_risk
                    )
                    else float("nan")
                ),
                "kaplan_meier_minus_reference": (
                    km_risk
                    - reference_risk
                ),
            }
        )

    bin_summary = pd.DataFrame(rows)

    scalar_summary = summarize_binned_estimators(
        bin_summary
    )

    return BinnedEstimatorResult(
        bin_summary=bin_summary,
        scalar_summary=scalar_summary,
        bin_edges=bin_edges,
    )


def _weighted_metric(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[float, float]:
    """Calculate normalized weighted metric and available coverage."""

    valid = (
        values.notna()
        & weights.notna()
    )

    if not valid.any():
        return (
            float("nan"),
            0.0,
        )

    available_weights = weights.loc[
        valid
    ]

    coverage = float(
        available_weights.sum()
    )

    normalized_weights = (
        available_weights
        / coverage
    )

    estimate = float(
        (
            values.loc[valid]
            * normalized_weights
        ).sum()
    )

    return estimate, coverage


def summarize_binned_estimators(
    bin_summary: pd.DataFrame,
) -> pd.Series:
    """Summarize estimator calibration and reference-recovery error."""

    weights = bin_summary[
        "reference_weight"
    ]

    prediction = bin_summary[
        "mean_prediction"
    ]

    reference = bin_summary[
        "reference_risk"
    ]

    complete_case = bin_summary[
        "complete_case_risk"
    ]

    kaplan_meier = bin_summary[
        "kaplan_meier_risk"
    ]

    reference_ici, reference_coverage = _weighted_metric(
        (reference - prediction).abs(),
        weights,
    )

    complete_case_ici, complete_case_coverage = _weighted_metric(
        (complete_case - prediction).abs(),
        weights,
    )

    km_ici, km_coverage = _weighted_metric(
        (kaplan_meier - prediction).abs(),
        weights,
    )

    cc_mae, _ = _weighted_metric(
        (complete_case - reference).abs(),
        weights,
    )

    km_mae, _ = _weighted_metric(
        (kaplan_meier - reference).abs(),
        weights,
    )

    cc_signed_gap, _ = _weighted_metric(
        complete_case - reference,
        weights,
    )

    km_signed_gap, _ = _weighted_metric(
        kaplan_meier - reference,
        weights,
    )

    return pd.Series(
        {
            "reference_ici": reference_ici,
            "complete_case_ici": complete_case_ici,
            "kaplan_meier_ici": km_ici,
            "complete_case_mae_to_reference": cc_mae,
            "kaplan_meier_mae_to_reference": km_mae,
            "complete_case_signed_gap": cc_signed_gap,
            "kaplan_meier_signed_gap": km_signed_gap,
            "complete_case_coverage": complete_case_coverage,
            "kaplan_meier_coverage": km_coverage,
            "reference_coverage": reference_coverage,
            "cc_minus_km_mae_improvement": (
                cc_mae - km_mae
            ),
            "minimum_complete_case_bin_n": float(
                bin_summary[
                    "n_complete_case"
                ].min()
            ),
            "maximum_complete_case_bin_n": float(
                bin_summary[
                    "n_complete_case"
                ].max()
            ),
        },
        name="value",
    )
