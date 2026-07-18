"""Phase 5C-A: MIMIC-IV multi-action cohort readiness audit.

Constructs a last-admission adult cohort for 30-day all-cause
mortality and maps alive hospital discharge destinations into
clinically interpretable exit-action groups.

This phase does not fit or evaluate a prediction model. It verifies
that the local MIMIC data support the planned real-data experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TABLE_SUFFIXES = [
    ".csv.gz",
    ".csv",
    ".parquet",
]


OPTIONAL_TABLES = [
    "icustays",
    "oasis",
    "sapsii",
    "apsiii",
    "first_day_sofa",
    "first_day_lab",
    "charlson",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit local MIMIC-IV files and construct the "
            "Phase 5C multi-action cohort."
        )
    )

    parser.add_argument(
        "--mimic-root",
        required=True,
        help=(
            "Directory containing the MIMIC-IV hosp/icu "
            "tables or a parent directory above them."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase5c_mimic_readiness",
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

    return parser.parse_args()


def find_table(
    root: Path,
    table_name: str,
    *,
    required: bool = True,
) -> Path | None:
    """Find one MIMIC table recursively."""

    candidates: list[Path] = []

    for suffix in TABLE_SUFFIXES:
        candidates.extend(
            root.rglob(
                f"{table_name}{suffix}"
            )
        )

    candidates = sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_file()
        },
        key=lambda path: (
            len(path.parts),
            len(str(path)),
            str(path),
        ),
    )

    if not candidates:
        if required:
            raise FileNotFoundError(
                f"Could not find {table_name} under {root}"
            )

        return None

    if len(candidates) > 1:
        print(
            f"Multiple {table_name} candidates found; "
            f"using {candidates[0]}"
        )

    return candidates[0]


def read_selected_columns(
    path: Path,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Read selected columns from CSV, CSV.GZ, or Parquet."""

    requested = list(
        dict.fromkeys(
            columns
        )
    )

    path_text = str(
        path
    ).lower()

    if path_text.endswith(
        ".parquet"
    ):
        try:
            frame = pd.read_parquet(
                path,
                columns=requested,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not read parquet table {path}: {error}"
            ) from error
    else:
        frame = pd.read_csv(
            path,
            usecols=lambda column: (
                column in requested
            ),
            low_memory=False,
        )

    missing = sorted(
        set(
            requested
        ).difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    return frame


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna(
            "UNKNOWN"
        )
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )


def discharge_action_group(
    location: object,
) -> str:
    """Map raw MIMIC discharge destinations to stable actions."""

    value = str(
        location
    ).strip().upper()

    if (
        not value
        or value
        in {
            "NAN",
            "NONE",
            "UNKNOWN",
        }
    ):
        return "unknown"

    if "DIED" in value:
        return "died"

    if "HOSPICE" in value:
        return "hospice"

    if "HOME HEALTH" in value:
        return "home_health"

    if value == "HOME":
        return "home"

    if (
        "SKILLED NURSING" in value
        or value == "SNF"
    ):
        return "skilled_nursing"

    if "REHAB" in value:
        return "rehabilitation"

    if (
        "ACUTE HOSPITAL" in value
        or "TRANSFER TO HOSPITAL" in value
    ):
        return "acute_hospital_transfer"

    if (
        "CHRONIC" in value
        or "LONG TERM" in value
        or "ASSISTED LIVING" in value
        or "HEALTHCARE FACILITY" in value
    ):
        return "long_term_or_assisted"

    if "PSYCH" in value:
        return "psychiatric_facility"

    if (
        "AGAINST" in value
        or "AMA" in value
    ):
        return "against_medical_advice"

    if (
        "OTHER FACILITY" in value
        or value == "OTHER"
    ):
        return "other_facility"

    return "other_or_unmapped"


def construct_cohort(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    *,
    horizon: int,
    minimum_age: float,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """Construct one last adult admission per patient."""

    admissions = admissions.copy()
    patients = patients.copy()

    for column in [
        "admittime",
        "dischtime",
        "deathtime",
    ]:
        admissions[
            column
        ] = pd.to_datetime(
            admissions[
                column
            ],
            errors="coerce",
        )

    patients[
        "dod"
    ] = pd.to_datetime(
        patients[
            "dod"
        ],
        errors="coerce",
    )

    initial_counts = {
        "raw_admissions": int(
            len(
                admissions
            )
        ),
        "raw_patients": int(
            len(
                patients
            )
        ),
    }

    frame = admissions.merge(
        patients,
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )

    frame = frame.loc[
        frame[
            "admittime"
        ].notna()
        & frame[
            "dischtime"
        ].notna()
        & (
            frame[
                "dischtime"
            ]
            >= frame[
                "admittime"
            ]
        )
    ].copy()

    initial_counts[
        "valid_time_admissions"
    ] = int(
        len(
            frame
        )
    )

    frame[
        "age_at_admission"
    ] = (
        pd.to_numeric(
            frame[
                "anchor_age"
            ],
            errors="coerce",
        )
        + (
            frame[
                "admittime"
            ].dt.year
            - pd.to_numeric(
                frame[
                    "anchor_year"
                ],
                errors="coerce",
            )
        )
    )

    frame = frame.loc[
        frame[
            "age_at_admission"
        ].ge(
            minimum_age
        )
    ].copy()

    initial_counts[
        "adult_admissions"
    ] = int(
        len(
            frame
        )
    )

    # One independent observation per patient. The last admission
    # also gives the clearest interpretation of the documented
    # post-discharge mortality follow-up window.
    frame = (
        frame.sort_values(
            [
                "subject_id",
                "admittime",
                "hadm_id",
            ]
        )
        .groupby(
            "subject_id",
            as_index=False,
            sort=False,
        )
        .tail(
            1
        )
        .reset_index(
            drop=True
        )
    )

    initial_counts[
        "last_admission_adults"
    ] = int(
        len(
            frame
        )
    )

    frame[
        "raw_discharge_location"
    ] = normalize_text(
        frame[
            "discharge_location"
        ]
    )

    frame[
        "discharge_group"
    ] = frame[
        "raw_discharge_location"
    ].map(
        discharge_action_group
    )

    horizon_delta = pd.to_timedelta(
        horizon,
        unit="D",
    )

    frame[
        "horizon_time"
    ] = (
        frame[
            "admittime"
        ]
        + horizon_delta
    )

    ordinary_in_hospital_death = (
        frame[
            "deathtime"
        ].notna()
        & (
            frame[
                "deathtime"
            ]
            <= frame[
                "horizon_time"
            ]
        )
    )

    died_location_fallback = (
        frame[
            "discharge_group"
        ].eq(
            "died"
        )
        & (
            frame[
                "dischtime"
            ]
            <= frame[
                "horizon_time"
            ]
        )
    )

    frame[
        "restricted_event"
    ] = (
        ordinary_in_hospital_death
        | died_location_fallback
    ).astype(
        int
    )

    # DOD is date-level. Compare calendar dates rather than
    # treating midnight as the exact death time.
    dod_day = frame[
        "dod"
    ].dt.normalize()

    horizon_day = frame[
        "horizon_time"
    ].dt.normalize()

    frame[
        "outcome"
    ] = (
        frame[
            "restricted_event"
        ].eq(
            1
        )
        | (
            dod_day.notna()
            & (
                dod_day
                <= horizon_day
            )
        )
    ).astype(
        int
    )

    alive_discharge_before_horizon = (
        frame[
            "dischtime"
        ]
        < frame[
            "horizon_time"
        ]
    ) & frame[
        "restricted_event"
    ].eq(
        0
    )

    frame[
        "observed"
    ] = (
        ~alive_discharge_before_horizon
    ).astype(
        int
    )

    frame[
        "action"
    ] = np.where(
        frame[
            "observed"
        ].eq(
            0
        ),
        frame[
            "discharge_group"
        ],
        None,
    )

    frame[
        "hidden_event"
    ] = (
        frame[
            "outcome"
        ].eq(
            1
        )
        & frame[
            "observed"
        ].eq(
            0
        )
    ).astype(
        int
    )

    length_of_stay_days = (
        frame[
            "dischtime"
        ]
        - frame[
            "admittime"
        ]
    ).dt.total_seconds() / 86400.0

    frame[
        "exit_time"
    ] = np.minimum(
        length_of_stay_days,
        float(
            horizon
        ),
    )

    frame[
        "restricted_time"
    ] = np.where(
        frame[
            "restricted_event"
        ].eq(
            1
        ),
        (
            frame[
                "deathtime"
            ].fillna(
                frame[
                    "dischtime"
                ]
            )
            - frame[
                "admittime"
            ]
        ).dt.total_seconds()
        / 86400.0,
        frame[
            "exit_time"
        ],
    )

    frame[
        "restricted_time"
    ] = (
        pd.to_numeric(
            frame[
                "restricted_time"
            ],
            errors="coerce",
        )
        .clip(
            lower=0.0,
            upper=float(
                horizon
            ),
        )
    )

    frame[
        "restricted_censored_by_discharge"
    ] = frame[
        "observed"
    ].eq(
        0
    ).astype(
        int
    )

    return (
        frame,
        initial_counts,
    )


def action_summary_table(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    censored = cohort.loc[
        cohort[
            "observed"
        ].eq(
            0
        )
    ].copy()

    if censored.empty:
        return pd.DataFrame(
            columns=[
                "action",
                "n",
                "cohort_fraction",
                "censored_fraction",
                "hidden_events",
                "hidden_event_rate",
                "mean_exit_day",
                "median_exit_day",
            ]
        )

    summary = (
        censored.groupby(
            "action",
            dropna=False,
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
    )

    summary[
        "cohort_fraction"
    ] = (
        summary[
            "n"
        ]
        / len(
            cohort
        )
    )

    summary[
        "censored_fraction"
    ] = (
        summary[
            "n"
        ]
        / len(
            censored
        )
    )

    return summary[
        [
            "action",
            "n",
            "cohort_fraction",
            "censored_fraction",
            "hidden_events",
            "hidden_event_rate",
            "mean_exit_day",
            "median_exit_day",
        ]
    ].sort_values(
        "n",
        ascending=False,
    )


def readiness_checks(
    cohort: pd.DataFrame,
    action_summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    str,
]:
    duplicate_subjects = int(
        cohort[
            "subject_id"
        ].duplicated().sum()
    )

    overlap = int(
        (
            cohort[
                "restricted_event"
            ].eq(
                1
            )
            & cohort[
                "restricted_censored_by_discharge"
            ].eq(
                1
            )
        ).sum()
    )

    died_used_as_action = int(
        cohort.loc[
            cohort[
                "observed"
            ].eq(
                0
            ),
            "action",
        ].eq(
            "died"
        ).sum()
    )

    sufficiently_large_actions = int(
        action_summary[
            "n"
        ].ge(
            200
        ).sum()
    )

    actions_with_hidden_events = int(
        (
            action_summary[
                "hidden_events"
            ]
            >= 20
        ).sum()
    )

    unknown_fraction = float(
        cohort.loc[
            cohort[
                "observed"
            ].eq(
                0
            ),
            "action",
        ]
        .isin(
            [
                "unknown",
                "other_or_unmapped",
            ]
        )
        .mean()
    )

    if np.isnan(
        unknown_fraction
    ):
        unknown_fraction = 1.0

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "at least 5000 independent adult patients"
                ),
                "value": len(
                    cohort
                ),
                "threshold": 5000,
                "comparison": ">=",
                "passed": (
                    len(
                        cohort
                    )
                    >= 5000
                ),
            },
            {
                "check": (
                    "one admission per patient"
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
                    "event and discharge censoring do not overlap"
                ),
                "value": overlap,
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    overlap
                    == 0
                ),
            },
            {
                "check": (
                    "death is never treated as an exit action"
                ),
                "value": died_used_as_action,
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    died_used_as_action
                    == 0
                ),
            },
            {
                "check": (
                    "at least four action groups contain 200 patients"
                ),
                "value": (
                    sufficiently_large_actions
                ),
                "threshold": 4,
                "comparison": ">=",
                "passed": (
                    sufficiently_large_actions
                    >= 4
                ),
            },
            {
                "check": (
                    "at least two actions contain 20 hidden deaths"
                ),
                "value": (
                    actions_with_hidden_events
                ),
                "threshold": 2,
                "comparison": ">=",
                "passed": (
                    actions_with_hidden_events
                    >= 2
                ),
            },
            {
                "check": (
                    "unknown or unmapped actions below 5 percent"
                ),
                "value": unknown_fraction,
                "threshold": 0.05,
                "comparison": "<",
                "passed": (
                    unknown_fraction
                    < 0.05
                ),
            },
        ]
    )

    status = (
        "phase5c_mimic_ready_for_modeling"
        if checks[
            "passed"
        ].all()
        else "phase5c_mimic_readiness_requires_review"
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

    if not root.exists():
        raise FileNotFoundError(
            f"MIMIC root does not exist: {root}"
        )

    if args.horizon < 2:
        raise ValueError(
            "horizon must be at least two days"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    admissions_path = find_table(
        root,
        "admissions",
    )

    patients_path = find_table(
        root,
        "patients",
    )

    print(
        f"Admissions: {admissions_path}"
    )

    print(
        f"Patients:   {patients_path}"
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

    (
        cohort,
        cohort_counts,
    ) = construct_cohort(
        admissions,
        patients,
        horizon=args.horizon,
        minimum_age=args.minimum_age,
    )

    action_summary = action_summary_table(
        cohort
    )

    (
        checks,
        status,
    ) = readiness_checks(
        cohort,
        action_summary,
    )

    optional_tables = {
        table: (
            str(
                find_table(
                    root,
                    table,
                    required=False,
                )
            )
            if find_table(
                root,
                table,
                required=False,
            )
            is not None
            else None
        )
        for table in OPTIONAL_TABLES
    }

    cohort.to_csv(
        output_dir
        / "mimic_phase5c_cohort.csv.gz",
        index=False,
        compression="gzip",
    )

    action_summary.to_csv(
        output_dir
        / "action_readiness_summary.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "readiness_checks.csv",
        index=False,
    )

    cohort[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "age_at_admission",
            "gender",
            "raw_discharge_location",
            "discharge_group",
            "outcome",
            "observed",
            "action",
            "hidden_event",
            "exit_time",
        ]
    ].head(
        100
    ).to_csv(
        output_dir
        / "cohort_preview.csv",
        index=False,
    )

    scalar_summary = {
        **cohort_counts,
        "analysis_cohort_n": int(
            len(
                cohort
            )
        ),
        "reference_30d_event_rate": float(
            cohort[
                "outcome"
            ].mean()
        ),
        "restricted_observed_fraction": float(
            cohort[
                "observed"
            ].mean()
        ),
        "alive_discharge_censored_fraction": float(
            cohort[
                "observed"
            ].eq(
                0
            ).mean()
        ),
        "hidden_30d_event_rate": float(
            cohort[
                "hidden_event"
            ].mean()
        ),
        "n_action_groups": int(
            action_summary[
                "action"
            ].nunique()
        ),
    }

    metadata = {
        "phase": "5C-A",
        "experiment": (
            "MIMIC-IV multi-action cohort readiness"
        ),
        "mimic_root": str(
            root
        ),
        "admissions_path": str(
            admissions_path
        ),
        "patients_path": str(
            patients_path
        ),
        "horizon_days": int(
            args.horizon
        ),
        "boundary": "hospital",
        "cohort_unit": (
            "last adult hospital admission per patient"
        ),
        "reference_outcome": (
            "all-cause mortality within horizon "
            "using DOD or in-hospital death"
        ),
        "restricted_outcome": (
            "in-hospital event or continued hospitalization "
            "through the horizon"
        ),
        "optional_tables": (
            optional_tables
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
        / "phase5c_readiness_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nCohort summary"
    )

    for key, value in scalar_summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nAction summary"
    )

    print(
        action_summary.to_string(
            index=False
        )
    )

    print(
        "\nReadiness checks"
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
        f"\nArtifacts saved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
