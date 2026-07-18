"""SUPPORT2 ingestion and feasibility checks for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = (
    "death",
    "d.time",
    "hospdead",
    "slos",
)


# Exclude outcomes, future information, previous predictions, physician
# prognoses, costs, DNR decisions, and post-landmark variables from the
# primary prediction model.
DEFAULT_EXCLUDED_PREDICTORS = {
    "death",
    "hospdead",
    "d.time",
    "slos",
    "surv2m",
    "surv6m",
    "sps",
    "aps",
    "prg2m",
    "prg6m",
    "dnr",
    "dnrday",
    "charges",
    "totcst",
    "totmcst",
    "sfdm2",
}


@dataclass(frozen=True)
class Support2HorizonView:
    """Reference and hospital-restricted outcomes for one horizon."""

    frame: pd.DataFrame
    horizon_days: int


@dataclass(frozen=True)
class Support2FeasibilityResult:
    """Tables produced by the SUPPORT2 feasibility audit."""

    overview: pd.Series
    horizon_summary: pd.DataFrame
    consistency_checks: pd.DataFrame
    missingness: pd.DataFrame


def load_support2_csv(
    path: str | Path,
) -> pd.DataFrame:
    """Load SUPPORT2 and normalize common CSV formatting issues."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"SUPPORT2 CSV not found at {path}. "
            "Run python scripts/fetch_support2.py first."
        )

    data = pd.read_csv(
        path,
        skipinitialspace=True,
    )

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    unnamed = [
        column
        for column in data.columns
        if column.lower().startswith("unnamed:")
    ]

    if unnamed:
        data = data.drop(
            columns=unnamed
        )

    # Some copies include an explicit row-number column.
    first_column = data.columns[0]

    if first_column.lower() in {
        "row",
        "row_id",
        "index",
        "id",
    }:
        candidate = pd.to_numeric(
            data[first_column],
            errors="coerce",
        )

        if (
            candidate.notna().all()
            and candidate.nunique() == len(data)
        ):
            data = data.drop(
                columns=[first_column]
            )

    missing = [
        column
        for column in KEY_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"SUPPORT2 file is missing key columns: {missing}"
        )

    for column in KEY_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data.reset_index(drop=True)


def prediction_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    """Return baseline candidate predictors after leakage exclusions."""

    return [
        column
        for column in data.columns
        if column not in DEFAULT_EXCLUDED_PREDICTORS
    ]


def build_support2_horizon_view(
    data: pd.DataFrame,
    horizon_days: int,
) -> Support2HorizonView:
    """Construct reference and artificially hospital-restricted outcomes.

    The reference outcome is known when:

    1. Death occurs on or before the horizon, or
    2. Follow-up reaches or passes the horizon.

    Patients who are alive at their final follow-up but whose follow-up ends
    before the horizon are reference-unknown.

    The hospital-restricted view considers a horizon outcome observable when:

    1. The patient remains hospitalized through the horizon, or
    2. The patient dies in hospital on or before the horizon.

    Patients discharged alive before the horizon are treated as unobserved
    under the restricted hospital-only observation boundary.
    """

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be positive"
        )

    missing = [
        column
        for column in KEY_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing key SUPPORT2 columns: {missing}"
        )

    result = data.copy()

    death = pd.to_numeric(
        result["death"],
        errors="coerce",
    )

    followup = pd.to_numeric(
        result["d.time"],
        errors="coerce",
    )

    hospital_death = pd.to_numeric(
        result["hospdead"],
        errors="coerce",
    )

    study_length_of_stay = pd.to_numeric(
        result["slos"],
        errors="coerce",
    )

    event_by_horizon = (
        death.eq(1)
        & followup.le(horizon_days)
    )

    reference_known = (
        event_by_horizon
        | followup.ge(horizon_days)
    )

    restricted_observed = (
        study_length_of_stay.ge(horizon_days)
        | (
            hospital_death.eq(1)
            & followup.le(horizon_days)
        )
    )

    result["patient_id"] = np.arange(
        len(result),
        dtype=int,
    )

    result["horizon_days"] = horizon_days

    result["reference_known"] = (
        reference_known.astype(int)
    )

    result["outcome"] = np.where(
        reference_known,
        event_by_horizon.astype(int),
        np.nan,
    )

    result["observed"] = np.where(
        reference_known,
        restricted_observed.astype(int),
        np.nan,
    )

    result["action"] = np.where(
        reference_known
        & ~restricted_observed,
        "alive_hospital_discharge",
        None,
    )

    result["exit_time"] = np.where(
        reference_known
        & ~restricted_observed,
        study_length_of_stay,
        np.nan,
    )

    result["post_discharge_event_by_horizon"] = (
        reference_known
        & event_by_horizon
        & hospital_death.eq(0)
        & study_length_of_stay.lt(horizon_days)
    ).astype(int)

    return Support2HorizonView(
        frame=result,
        horizon_days=horizon_days,
    )


def _check_row(
    name: str,
    count: int,
    total: int,
    severity: str = "warning",
) -> dict[str, object]:
    """Create a standardized consistency-check result."""

    return {
        "check": name,
        "count": int(count),
        "fraction": (
            float(count / total)
            if total
            else float("nan")
        ),
        "severity": severity,
        "passed": bool(count == 0),
    }


def support2_feasibility_audit(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (60, 180),
) -> Support2FeasibilityResult:
    """Audit reference follow-up and discharge-truncation prevalence."""

    n = len(data)

    death = pd.to_numeric(
        data["death"],
        errors="coerce",
    )

    followup = pd.to_numeric(
        data["d.time"],
        errors="coerce",
    )

    hospital_death = pd.to_numeric(
        data["hospdead"],
        errors="coerce",
    )

    study_length_of_stay = pd.to_numeric(
        data["slos"],
        errors="coerce",
    )

    consistency = pd.DataFrame(
        [
            _check_row(
                "negative_followup_time",
                int(followup.lt(0).sum()),
                n,
                "error",
            ),
            _check_row(
                "negative_study_length_of_stay",
                int(study_length_of_stay.lt(0).sum()),
                n,
                "error",
            ),
            _check_row(
                "hospital_death_without_any_death",
                int(
                    (
                        hospital_death.eq(1)
                        & ~death.eq(1)
                    ).sum()
                ),
                n,
                "error",
            ),
            _check_row(
                "death_time_before_hospital_exit_for_hospital_survivor",
                int(
                    (
                        hospital_death.eq(0)
                        & death.eq(1)
                        & followup.lt(study_length_of_stay)
                    ).sum()
                ),
                n,
                "error",
            ),
            _check_row(
                "hospital_death_time_differs_from_exit_time_over_one_day",
                int(
                    (
                        hospital_death.eq(1)
                        & (
                            followup
                            - study_length_of_stay
                        )
                        .abs()
                        .gt(1)
                    ).sum()
                ),
                n,
                "warning",
            ),
            _check_row(
                "nonbinary_death",
                int(
                    (
                        death.notna()
                        & ~death.isin([0, 1])
                    ).sum()
                ),
                n,
                "error",
            ),
            _check_row(
                "nonbinary_hospdead",
                int(
                    (
                        hospital_death.notna()
                        & ~hospital_death.isin([0, 1])
                    ).sum()
                ),
                n,
                "error",
            ),
            _check_row(
                "missing_death",
                int(death.isna().sum()),
                n,
                "error",
            ),
            _check_row(
                "missing_followup_time",
                int(followup.isna().sum()),
                n,
                "error",
            ),
            _check_row(
                "missing_hospital_death",
                int(hospital_death.isna().sum()),
                n,
                "error",
            ),
            _check_row(
                "missing_study_length_of_stay",
                int(study_length_of_stay.isna().sum()),
                n,
                "error",
            ),
        ]
    )

    horizon_rows: list[dict[str, object]] = []

    for horizon in horizons:
        view = build_support2_horizon_view(
            data,
            horizon,
        ).frame

        known = view["reference_known"].eq(1)

        restricted_missing = (
            known
            & view["observed"].eq(0)
        )

        outcomes = view.loc[
            known,
            "outcome",
        ]

        hidden_events = view.loc[
            restricted_missing,
            "post_discharge_event_by_horizon",
        ]

        known_count = int(known.sum())

        horizon_rows.append(
            {
                "horizon_days": horizon,
                "n_total": n,
                "n_reference_known": known_count,
                "reference_known_fraction": float(
                    known.mean()
                ),
                "n_reference_unknown": int(
                    (~known).sum()
                ),
                "reference_event_rate": (
                    float(outcomes.mean())
                    if not outcomes.empty
                    else np.nan
                ),
                "n_restricted_unobserved": int(
                    restricted_missing.sum()
                ),
                "restricted_unobserved_fraction_among_known": (
                    float(
                        restricted_missing.sum()
                        / known_count
                    )
                    if known_count
                    else np.nan
                ),
                "n_hidden_events": int(
                    hidden_events.sum()
                ),
                "hidden_event_rate": (
                    float(hidden_events.mean())
                    if not hidden_events.empty
                    else np.nan
                ),
                "n_in_hospital_events": int(
                    (
                        known
                        & view["outcome"].eq(1)
                        & hospital_death.eq(1)
                    ).sum()
                ),
            }
        )

    missingness = (
        data
        .isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(
            missing_fraction=lambda frame: (
                frame["missing_count"]
                / n
            )
        )
        .sort_values(
            [
                "missing_fraction",
                "missing_count",
            ],
            ascending=False,
        )
        .reset_index(
            names="variable"
        )
    )

    overview = pd.Series(
        {
            "n_rows": n,
            "n_columns": data.shape[1],
            "expected_row_count": 9105,
            "row_count_matches_expected": bool(
                n == 9105
            ),
            "landmark": (
                "SUPPORT study-entry/day-3 assessment"
            ),
            "hday_interpretation": (
                "hospital day of study entry; "
                "do not filter to hday == 3"
            ),
            "candidate_prediction_features": len(
                prediction_feature_columns(data)
            ),
            "error_check_failures": int(
                (
                    consistency["severity"].eq("error")
                    & ~consistency["passed"]
                ).sum()
            ),
            "warning_check_failures": int(
                (
                    consistency["severity"].eq("warning")
                    & ~consistency["passed"]
                ).sum()
            ),
        },
        name="value",
    )

    return Support2FeasibilityResult(
        overview=overview,
        horizon_summary=pd.DataFrame(
            horizon_rows
        ),
        consistency_checks=consistency,
        missingness=missingness,
    )
