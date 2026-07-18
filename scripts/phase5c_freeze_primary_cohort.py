"""Phase 5C-C: freeze the MIMIC-IV primary and sensitivity cohorts.

Primary cohort:
    1. Exclude observation-type encounters.
    2. Retain valid adult inpatient admissions.
    3. Select the last eligible inpatient admission per patient.
    4. Use 30-day all-cause mortality as the reference outcome.
    5. Preserve discharge destinations as exit actions.

Sensitivity cohort:
    Primary-cohort admissions containing at least one ICU stay.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from phase5c_mimic_readiness import (
    action_summary_table,
    construct_cohort,
    find_table,
    read_selected_columns,
)


EXCLUDED_ADMISSION_PATTERN = "OBSERVATION"

UNCLASSIFIED_ACTIONS = {
    "unknown",
    "other_or_unmapped",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the last eligible inpatient admission "
            "MIMIC-IV cohort for Phase 5C."
        )
    )

    parser.add_argument(
        "--mimic-root",
        default="data/raw/mimiciv/3.1",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase5c_frozen_cohort",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--minimum-age",
        type=float,
        default=18.0,
    )

    parser.add_argument(
        "--minimum-action-n",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--minimum-hidden-events",
        type=int,
        default=20,
    )

    return parser.parse_args()


def read_mimic_tables(
    root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
]:
    """Read the required MIMIC-IV tables."""

    admissions_path = find_table(
        root,
        "admissions",
    )

    patients_path = find_table(
        root,
        "patients",
    )

    icustays_path = find_table(
        root,
        "icustays",
    )

    admissions = read_selected_columns(
        admissions_path,
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        ],
    )

    patients = read_selected_columns(
        patients_path,
        [
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
            "dod",
        ],
    )

    icustays = read_selected_columns(
        icustays_path,
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ],
    )

    paths = {
        "admissions": str(
            admissions_path
        ),
        "patients": str(
            patients_path
        ),
        "icustays": str(
            icustays_path
        ),
    }

    return (
        admissions,
        patients,
        icustays,
        paths,
    )


def exclude_observation_encounters(
    admissions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Remove all observation-type encounters before patient selection."""

    admission_type = (
        admissions[
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
            EXCLUDED_ADMISSION_PATTERN,
            regex=False,
        )
    )

    eligible = (
        admissions.loc[
            ~observation_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    excluded = (
        admissions.loc[
            observation_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        eligible,
        excluded,
    )


def add_icu_indicator(
    cohort: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one ICU-stay indicator to each selected admission."""

    icu_admissions = (
        icustays[
            [
                "subject_id",
                "hadm_id",
            ]
        ]
        .drop_duplicates()
        .assign(
            has_icu_stay=1
        )
    )

    result = cohort.merge(
        icu_admissions,
        on=[
            "subject_id",
            "hadm_id",
        ],
        how="left",
        validate="one_to_one",
    )

    result[
        "has_icu_stay"
    ] = (
        result[
            "has_icu_stay"
        ]
        .fillna(
            0
        )
        .astype(int)
    )

    return result


def define_primary_actions(
    cohort: pd.DataFrame,
    *,
    minimum_action_n: int,
    minimum_hidden_events: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """Select stable named actions without using calibration results."""

    raw_summary = action_summary_table(
        cohort
    )

    eligible_rows = raw_summary.loc[
        raw_summary[
            "n"
        ].ge(
            minimum_action_n
        )
        & raw_summary[
            "hidden_events"
        ].ge(
            minimum_hidden_events
        )
        & ~raw_summary[
            "action"
        ].isin(
            UNCLASSIFIED_ACTIONS
        )
    ].copy()

    eligible_actions = (
        eligible_rows[
            "action"
        ]
        .astype(str)
        .tolist()
    )

    result = cohort.copy()

    censored = result[
        "observed"
    ].eq(
        0
    )

    result[
        "primary_action"
    ] = None

    result.loc[
        censored
        & result[
            "action"
        ].isin(
            eligible_actions
        ),
        "primary_action",
    ] = result.loc[
        censored
        & result[
            "action"
        ].isin(
            eligible_actions
        ),
        "action",
    ]

    result.loc[
        censored
        & ~result[
            "action"
        ].isin(
            eligible_actions
        ),
        "primary_action",
    ] = "rare_or_unclassified"

    primary_summary_input = (
        result.copy()
    )

    primary_summary_input[
        "action"
    ] = primary_summary_input[
        "primary_action"
    ]

    primary_summary = action_summary_table(
        primary_summary_input
    )

    return (
        result,
        raw_summary,
        eligible_actions,
    )


def previous_postfilter_comparison(
    cohort: pd.DataFrame,
) -> dict[str, int | None]:
    """Compare against the earlier filter-after-selection candidate."""

    previous_path = Path(
        "artifacts/phase5c_mimic_readiness/"
        "mimic_phase5c_cohort.csv.gz"
    )

    if not previous_path.exists():
        return {
            "previous_postfilter_n": None,
            "new_last_inpatient_n": int(
                len(
                    cohort
                )
            ),
            "subjects_added_by_correct_order": None,
            "subjects_removed_by_correct_order": None,
        }

    previous = pd.read_csv(
        previous_path,
        usecols=[
            "subject_id",
            "admission_type",
        ],
        low_memory=False,
    )

    previous_type = (
        previous[
            "admission_type"
        ]
        .fillna(
            "UNKNOWN"
        )
        .astype(str)
        .str.strip()
        .str.upper()
    )

    previous_postfilter = previous.loc[
        ~previous_type.str.contains(
            EXCLUDED_ADMISSION_PATTERN,
            regex=False,
        )
    ]

    previous_subjects = set(
        previous_postfilter[
            "subject_id"
        ].astype(int)
    )

    new_subjects = set(
        cohort[
            "subject_id"
        ].astype(int)
    )

    return {
        "previous_postfilter_n": int(
            len(
                previous_postfilter
            )
        ),
        "new_last_inpatient_n": int(
            len(
                cohort
            )
        ),
        "subjects_added_by_correct_order": int(
            len(
                new_subjects.difference(
                    previous_subjects
                )
            )
        ),
        "subjects_removed_by_correct_order": int(
            len(
                previous_subjects.difference(
                    new_subjects
                )
            )
        ),
    }


def build_checks(
    *,
    cohort: pd.DataFrame,
    icu_cohort: pd.DataFrame,
    raw_action_summary: pd.DataFrame,
    eligible_actions: list[str],
) -> tuple[
    pd.DataFrame,
    str,
]:
    """Run prespecified cohort-freeze checks."""

    censored = cohort.loc[
        cohort[
            "observed"
        ].eq(
            0
        )
    ]

    unknown_fraction = float(
        censored[
            "action"
        ].isin(
            UNCLASSIFIED_ACTIONS
        ).mean()
    )

    residual_fraction = float(
        censored[
            "primary_action"
        ].eq(
            "rare_or_unclassified"
        ).mean()
    )

    observation_count = int(
        cohort[
            "admission_type"
        ]
        .fillna(
            ""
        )
        .astype(str)
        .str.upper()
        .str.contains(
            EXCLUDED_ADMISSION_PATTERN,
            regex=False,
        )
        .sum()
    )

    duplicate_subjects = int(
        cohort[
            "subject_id"
        ].duplicated().sum()
    )

    eligible_summary = (
        raw_action_summary.loc[
            raw_action_summary[
                "action"
            ].isin(
                eligible_actions
            )
        ]
    )

    all_named_actions_stable = bool(
        eligible_summary[
            "n"
        ].ge(
            200
        ).all()
        and eligible_summary[
            "hidden_events"
        ].ge(
            20
        ).all()
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "at least 100000 primary-cohort patients"
                ),
                "value": len(
                    cohort
                ),
                "threshold": 100000,
                "comparison": ">=",
                "passed": (
                    len(
                        cohort
                    )
                    >= 100000
                ),
            },
            {
                "check": (
                    "one selected admission per patient"
                ),
                "value": duplicate_subjects,
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    duplicate_subjects
                    == 0
                ),
            },
            {
                "check": (
                    "no observation encounters remain"
                ),
                "value": observation_count,
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    observation_count
                    == 0
                ),
            },
            {
                "check": (
                    "unknown raw actions below one percent"
                ),
                "value": unknown_fraction,
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    unknown_fraction
                    < 0.01
                ),
            },
            {
                "check": (
                    "at least six stable named actions"
                ),
                "value": len(
                    eligible_actions
                ),
                "threshold": 6,
                "comparison": ">=",
                "passed": (
                    len(
                        eligible_actions
                    )
                    >= 6
                ),
            },
            {
                "check": (
                    "all named actions meet size requirements"
                ),
                "value": (
                    all_named_actions_stable
                ),
                "threshold": True,
                "comparison": "==",
                "passed": (
                    all_named_actions_stable
                ),
            },
            {
                "check": (
                    "residual pooled action below three percent"
                ),
                "value": residual_fraction,
                "threshold": 0.03,
                "comparison": "<",
                "passed": (
                    residual_fraction
                    < 0.03
                ),
            },
            {
                "check": (
                    "ICU sensitivity cohort contains 20000 patients"
                ),
                "value": len(
                    icu_cohort
                ),
                "threshold": 20000,
                "comparison": ">=",
                "passed": (
                    len(
                        icu_cohort
                    )
                    >= 20000
                ),
            },
        ]
    )

    status = (
        "phase5c_primary_and_icu_cohorts_frozen"
        if checks[
            "passed"
        ].all()
        else "phase5c_cohort_freeze_requires_review"
    )

    return (
        checks,
        status,
    )


def main() -> None:
    args = parse_arguments()

    root = Path(
        args.mimic_root
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        admissions,
        patients,
        icustays,
        table_paths,
    ) = read_mimic_tables(
        root
    )

    (
        eligible_admissions,
        excluded_observations,
    ) = exclude_observation_encounters(
        admissions
    )

    (
        cohort,
        cohort_counts,
    ) = construct_cohort(
        eligible_admissions,
        patients,
        horizon=args.horizon,
        minimum_age=args.minimum_age,
    )

    cohort = add_icu_indicator(
        cohort,
        icustays,
    )

    (
        cohort,
        raw_action_summary,
        eligible_actions,
    ) = define_primary_actions(
        cohort,
        minimum_action_n=(
            args.minimum_action_n
        ),
        minimum_hidden_events=(
            args.minimum_hidden_events
        ),
    )

    primary_summary_input = (
        cohort.copy()
    )

    primary_summary_input[
        "action"
    ] = primary_summary_input[
        "primary_action"
    ]

    primary_action_summary = (
        action_summary_table(
            primary_summary_input
        )
    )

    icu_cohort = (
        cohort.loc[
            cohort[
                "has_icu_stay"
            ].eq(
                1
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    (
        checks,
        status,
    ) = build_checks(
        cohort=cohort,
        icu_cohort=icu_cohort,
        raw_action_summary=(
            raw_action_summary
        ),
        eligible_actions=(
            eligible_actions
        ),
    )

    comparison = (
        previous_postfilter_comparison(
            cohort
        )
    )

    cohort.to_csv(
        output_dir
        / "primary_inpatient_cohort.csv.gz",
        index=False,
        compression="gzip",
    )

    icu_cohort.to_csv(
        output_dir
        / "icu_sensitivity_cohort.csv.gz",
        index=False,
        compression="gzip",
    )

    raw_action_summary.to_csv(
        output_dir
        / "raw_action_summary.csv",
        index=False,
    )

    primary_action_summary.to_csv(
        output_dir
        / "primary_action_summary.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "cohort_freeze_checks.csv",
        index=False,
    )

    (
        output_dir
        / "eligible_named_actions.json"
    ).write_text(
        json.dumps(
            eligible_actions,
            indent=2,
        ),
        encoding="utf-8",
    )

    scalar_summary = {
        "raw_admissions": int(
            len(
                admissions
            )
        ),
        "excluded_observation_admissions": int(
            len(
                excluded_observations
            )
        ),
        "eligible_nonobservation_admissions": int(
            len(
                eligible_admissions
            )
        ),
        "primary_cohort_n": int(
            len(
                cohort
            )
        ),
        "icu_sensitivity_n": int(
            len(
                icu_cohort
            )
        ),
        "reference_event_rate": float(
            cohort[
                "outcome"
            ].mean()
        ),
        "restricted_observed_fraction": float(
            cohort[
                "observed"
            ].mean()
        ),
        "hidden_event_rate": float(
            cohort[
                "hidden_event"
            ].mean()
        ),
        "named_action_count": int(
            len(
                eligible_actions
            )
        ),
        "named_actions": (
            eligible_actions
        ),
        **comparison,
    }

    metadata = {
        "phase": "5C-C",
        "experiment": (
            "Frozen MIMIC-IV primary and ICU "
            "sensitivity cohorts"
        ),
        "selection_order": [
            "exclude observation-type encounters",
            "retain valid adult admissions",
            "select last eligible inpatient admission per patient",
        ],
        "reference_outcome": (
            "30-day all-cause mortality"
        ),
        "boundary": "hospital",
        "minimum_action_n": int(
            args.minimum_action_n
        ),
        "minimum_hidden_events": int(
            args.minimum_hidden_events
        ),
        "rare_action_policy": (
            "Pool censored actions failing the prespecified "
            "size thresholds as rare_or_unclassified"
        ),
        "table_paths": table_paths,
        "construct_cohort_counts": (
            cohort_counts
        ),
        "scalar_summary": (
            scalar_summary
        ),
        "status": status,
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
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
        / "phase5c_cohort_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Frozen cohort summary"
    )

    for key, value in scalar_summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nPrimary action summary"
    )

    print(
        primary_action_summary.to_string(
            index=False
        )
    )

    print(
        "\nFreeze checks"
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
