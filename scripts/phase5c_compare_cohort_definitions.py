"""Compare candidate MIMIC-IV cohort definitions for Phase 5C."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase5c_mimic_readiness import (
    action_summary_table,
)


READINESS_DIR = Path(
    "artifacts/phase5c_mimic_readiness"
)

MIMIC_ROOT = Path(
    "data/raw/mimiciv/3.1"
)

OUTPUT_DIR = Path(
    "artifacts/phase5c_cohort_comparison"
)


def candidate_scalar_summary(
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Summarize one proposed cohort."""

    censored = frame.loc[
        frame[
            "observed"
        ].eq(
            0
        )
    ]

    action_summary = action_summary_table(
        frame
    )

    unknown_fraction = float(
        censored[
            "action"
        ].isin(
            [
                "unknown",
                "other_or_unmapped",
            ]
        ).mean()
    )

    if np.isnan(
        unknown_fraction
    ):
        unknown_fraction = 1.0

    return {
        "candidate": name,
        "n_patients": int(
            len(
                frame
            )
        ),
        "reference_event_rate": float(
            frame[
                "outcome"
            ].mean()
        ),
        "restricted_observed_fraction": float(
            frame[
                "observed"
            ].mean()
        ),
        "discharge_censored_fraction": float(
            frame[
                "observed"
            ].eq(
                0
            ).mean()
        ),
        "hidden_event_rate": float(
            frame[
                "hidden_event"
            ].mean()
        ),
        "unknown_action_fraction_among_censored": (
            unknown_fraction
        ),
        "n_action_groups": int(
            action_summary[
                "action"
            ].nunique()
        ),
        "n_actions_at_least_200": int(
            action_summary[
                "n"
            ].ge(
                200
            ).sum()
        ),
        "n_actions_at_least_20_hidden_events": int(
            action_summary[
                "hidden_events"
            ].ge(
                20
            ).sum()
        ),
    }


def diagnostic_unknown_breakdown(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Determine where missing discharge destinations occur."""

    unknown = cohort.loc[
        cohort[
            "observed"
        ].eq(
            0
        )
        & cohort[
            "action"
        ].isin(
            [
                "unknown",
                "other_or_unmapped",
            ]
        )
    ].copy()

    if unknown.empty:
        return pd.DataFrame()

    unknown[
        "admission_type_normalized"
    ] = (
        unknown[
            "admission_type"
        ]
        .fillna(
            "UNKNOWN"
        )
        .astype(str)
        .str.strip()
        .str.upper()
    )

    summary = (
        unknown.groupby(
            [
                "admission_type_normalized",
                "admission_location",
            ],
            dropna=False,
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
            mean_exit_day=(
                "exit_time",
                "mean",
            ),
            median_exit_day=(
                "exit_time",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            "n",
            ascending=False,
        )
    )

    summary[
        "fraction_of_unknown"
    ] = (
        summary[
            "n"
        ]
        / len(
            unknown
        )
    )

    return summary


def candidate_checks(
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate prespecified readiness requirements."""

    rows: list[
        dict[str, object]
    ] = []

    for _, row in candidate_summary.iterrows():
        candidate = str(
            row[
                "candidate"
            ]
        )

        checks = [
            (
                "at least 5000 patients",
                row[
                    "n_patients"
                ],
                5000,
                ">=",
                row[
                    "n_patients"
                ]
                >= 5000,
            ),
            (
                "unknown actions below 5 percent",
                row[
                    "unknown_action_fraction_among_censored"
                ],
                0.05,
                "<",
                row[
                    "unknown_action_fraction_among_censored"
                ]
                < 0.05,
            ),
            (
                "at least four actions with 200 patients",
                row[
                    "n_actions_at_least_200"
                ],
                4,
                ">=",
                row[
                    "n_actions_at_least_200"
                ]
                >= 4,
            ),
            (
                "at least two actions with 20 hidden deaths",
                row[
                    "n_actions_at_least_20_hidden_events"
                ],
                2,
                ">=",
                row[
                    "n_actions_at_least_20_hidden_events"
                ]
                >= 2,
            ),
        ]

        for (
            check,
            value,
            threshold,
            comparison,
            passed,
        ) in checks:
            rows.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "value": value,
                    "threshold": threshold,
                    "comparison": comparison,
                    "passed": bool(
                        passed
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    cohort_path = (
        READINESS_DIR
        / "mimic_phase5c_cohort.csv.gz"
    )

    icustays_path = (
        MIMIC_ROOT
        / "icu"
        / "icustays.csv.gz"
    )

    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Missing readiness cohort: {cohort_path}"
        )

    if not icustays_path.exists():
        raise FileNotFoundError(
            f"Missing ICU stays: {icustays_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort = pd.read_csv(
        cohort_path,
        low_memory=False,
    )

    admission_type = (
        cohort[
            "admission_type"
        ]
        .fillna(
            "UNKNOWN"
        )
        .astype(str)
        .str.strip()
        .str.upper()
    )

    observation_mask = (
        admission_type.str.contains(
            "OBSERVATION",
            regex=False,
        )
    )

    inpatient_only = (
        cohort.loc[
            ~observation_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    icustays = pd.read_csv(
        icustays_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ],
        low_memory=False,
    )

    icu_admissions = (
        icustays[
            [
                "subject_id",
                "hadm_id",
            ]
        ]
        .drop_duplicates()
    )

    icu_associated = (
        cohort.merge(
            icu_admissions.assign(
                has_icu_stay=1
            ),
            on=[
                "subject_id",
                "hadm_id",
            ],
            how="inner",
            validate="one_to_one",
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    candidates = {
        "all_last_admissions": (
            cohort
        ),
        "inpatient_only_last_admissions": (
            inpatient_only
        ),
        "icu_associated_last_admissions": (
            icu_associated
        ),
    }

    scalar_rows = [
        candidate_scalar_summary(
            name,
            frame,
        )
        for name, frame in candidates.items()
    ]

    scalar_summary = pd.DataFrame(
        scalar_rows
    )

    checks = candidate_checks(
        scalar_summary
    )

    unknown_breakdown = (
        diagnostic_unknown_breakdown(
            cohort
        )
    )

    scalar_summary.to_csv(
        OUTPUT_DIR
        / "candidate_scalar_summary.csv",
        index=False,
    )

    checks.to_csv(
        OUTPUT_DIR
        / "candidate_readiness_checks.csv",
        index=False,
    )

    unknown_breakdown.to_csv(
        OUTPUT_DIR
        / "unknown_action_breakdown.csv",
        index=False,
    )

    for name, frame in candidates.items():
        action_summary = (
            action_summary_table(
                frame
            )
        )

        action_summary.to_csv(
            OUTPUT_DIR
            / f"action_summary_{name}.csv",
            index=False,
        )

    inpatient_passes = bool(
        checks.loc[
            checks[
                "candidate"
            ].eq(
                "inpatient_only_last_admissions"
            ),
            "passed",
        ].all()
    )

    icu_passes = bool(
        checks.loc[
            checks[
                "candidate"
            ].eq(
                "icu_associated_last_admissions"
            ),
            "passed",
        ].all()
    )

    if inpatient_passes:
        recommended_primary = (
            "inpatient_only_last_admissions"
        )
        status = (
            "phase5c_inpatient_primary_cohort_ready"
        )
    elif icu_passes:
        recommended_primary = (
            "icu_associated_last_admissions"
        )
        status = (
            "phase5c_icu_primary_cohort_ready"
        )
    else:
        recommended_primary = None
        status = (
            "phase5c_cohort_definition_requires_review"
        )

    metadata = {
        "phase": "5C-B",
        "experiment": (
            "Comparison of MIMIC-IV cohort definitions"
        ),
        "current_unknown_count": int(
            cohort.loc[
                cohort[
                    "observed"
                ].eq(
                    0
                )
                & cohort[
                    "action"
                ].isin(
                    [
                        "unknown",
                        "other_or_unmapped",
                    ]
                )
            ].shape[
                0
            ]
        ),
        "observation_admissions_excluded": int(
            observation_mask.sum()
        ),
        "recommended_primary": (
            recommended_primary
        ),
        "recommended_sensitivity": (
            "icu_associated_last_admissions"
            if recommended_primary
            != "icu_associated_last_admissions"
            else "inpatient_only_last_admissions"
        ),
        "status": status,
    }

    (
        OUTPUT_DIR
        / "comparison_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5c_cohort_comparison_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Candidate cohort summary"
    )

    print(
        scalar_summary.to_string(
            index=False
        )
    )

    print(
        "\nCandidate readiness checks"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        "\nLargest unknown-action sources"
    )

    print(
        unknown_breakdown.head(
            25
        ).to_string(
            index=False
        )
    )

    print(
        f"\nRecommended primary: "
        f"{recommended_primary}"
    )

    print(
        f"Status: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
