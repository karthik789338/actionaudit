"""Phase 5H-D: read-only MIMIC heterogeneity provenance audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_ROOTS = [
    "artifacts/phase5c_frozen_cohort",
    "artifacts/phase5c_admission_model",
    "artifacts/phase5c_empirical_action_decomposition",
    "artifacts/phase5c_frozen_for_manuscript",
    "artifacts/phase5g_baseline_features",
]

DEFAULT_SCRIPT_GLOBS = [
    "scripts/phase5c*.py",
    "scripts/phase5g*.py",
]

ROLE_PATTERNS = {
    "identifier": [
        r"^subject_id$",
        r"^hadm_id$",
        r"^stay_id$",
        r"patient",
        r"admission_id",
    ],
    "prediction_or_risk": [
        r"predict",
        r"risk",
        r"prob",
        r"score",
        r"logit",
        r"risk_bin",
    ],
    "reference_outcome": [
        r"linked",
        r"reference",
        r"mortality",
        r"death",
        r"outcome",
        r"event",
    ],
    "restriction_or_observation": [
        r"observed",
        r"restricted",
        r"censor",
        r"follow",
        r"discharge",
        r"exit",
    ],
    "action": [
        r"action",
        r"discharge_disposition",
        r"discharge_location",
        r"disposition",
        r"destination",
        r"hospice",
        r"rehab",
        r"snf",
        r"home_health",
        r"transfer",
    ],
    "timing": [
        r"admission_hour",
        r"admission_weekday",
        r"admittime",
        r"dischtime",
        r"anchor_year",
        r"year_group",
        r"month",
        r"weekday",
        r"hour",
        r"timing",
        r"length_of_stay",
        r"\blos\b",
    ],
    "disease_or_diagnosis": [
        r"diagn",
        r"disease",
        r"drg",
        r"icd",
        r"comorb",
        r"charlson",
        r"elix",
        r"service",
        r"admission_type",
        r"admission_location",
    ],
    "care_setting": [
        r"careunit",
        r"care_unit",
        r"ward",
        r"icu",
        r"service",
        r"location",
        r"unit",
    ],
    "institution": [
        r"hospital_id",
        r"site_id",
        r"institution",
        r"facility",
        r"center_id",
        r"provider_id",
    ],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen MIMIC artifacts for "
            "cross-dataset heterogeneity variables."
        )
    )

    parser.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
    )

    parser.add_argument(
        "--script-globs",
        nargs="*",
        default=DEFAULT_SCRIPT_GLOBS,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5h_mimic_heterogeneity_audit"
        ),
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=500,
    )

    return parser.parse_args()


def normalize_name(
    value: str,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.lower(),
    ).strip(
        "_"
    )


def detect_roles(
    column: str,
) -> list[
    tuple[str, str]
]:
    normalized = normalize_name(
        column
    )

    matches = []

    for role, patterns in ROLE_PATTERNS.items():
        for pattern in patterns:
            if re.search(
                pattern,
                normalized,
            ):
                matches.append(
                    (
                        role,
                        pattern,
                    )
                )

    return matches


def discover_files(
    roots: list[str],
) -> pd.DataFrame:
    rows = []

    allowed_suffixes = (
        ".csv",
        ".csv.gz",
        ".json",
        ".txt",
    )

    for root_text in roots:
        root = Path(
            root_text
        )

        if not root.exists():
            rows.append(
                {
                    "root": root_text,
                    "path": root_text,
                    "exists": False,
                    "file_type": "missing_root",
                    "size_bytes": 0,
                }
            )

            continue

        for path in sorted(
            root.rglob(
                "*"
            )
        ):
            if not path.is_file():
                continue

            path_text = str(
                path
            ).lower()

            if not path_text.endswith(
                allowed_suffixes
            ):
                continue

            if path_text.endswith(
                ".csv.gz"
            ):
                file_type = "csv.gz"
            else:
                file_type = (
                    path.suffix.lower()
                    .lstrip(
                        "."
                    )
                )

            rows.append(
                {
                    "root": root_text,
                    "path": str(
                        path
                    ),
                    "exists": True,
                    "file_type": file_type,
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def read_csv_sample(
    path: Path,
    sample_rows: int,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        nrows=sample_rows,
        low_memory=False,
    )


def audit_csv_schemas(
    inventory: pd.DataFrame,
    sample_rows: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    schema_rows = []
    candidate_rows = []

    csv_inventory = inventory.loc[
        inventory[
            "exists"
        ].eq(
            True
        )
        & inventory[
            "file_type"
        ].isin(
            [
                "csv",
                "csv.gz",
            ]
        )
    ]

    for _, file_row in csv_inventory.iterrows():
        path = Path(
            file_row[
                "path"
            ]
        )

        try:
            sample = read_csv_sample(
                path,
                sample_rows,
            )

            read_error = ""
        except Exception as error:
            sample = pd.DataFrame()

            read_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        if read_error:
            schema_rows.append(
                {
                    "path": str(
                        path
                    ),
                    "column": "",
                    "dtype": "",
                    "sample_nonmissing": 0,
                    "sample_missing_fraction": 1.0,
                    "sample_unique": 0,
                    "sample_values": "",
                    "read_error": read_error,
                }
            )

            continue

        for column in sample.columns:
            series = sample[
                column
            ]

            sample_values = (
                series.dropna()
                .astype(
                    str
                )
                .drop_duplicates()
                .head(
                    8
                )
                .tolist()
            )

            schema_rows.append(
                {
                    "path": str(
                        path
                    ),
                    "column": column,
                    "dtype": str(
                        series.dtype
                    ),
                    "sample_nonmissing": int(
                        series.notna().sum()
                    ),
                    "sample_missing_fraction": float(
                        series.isna().mean()
                    ),
                    "sample_unique": int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                    "sample_values": " | ".join(
                        sample_values
                    ),
                    "read_error": "",
                }
            )

            for role, pattern in detect_roles(
                column
            ):
                candidate_rows.append(
                    {
                        "path": str(
                            path
                        ),
                        "column": column,
                        "candidate_role": role,
                        "matched_pattern": pattern,
                        "dtype": str(
                            series.dtype
                        ),
                        "sample_missing_fraction": float(
                            series.isna().mean()
                        ),
                        "sample_unique": int(
                            series.nunique(
                                dropna=True
                            )
                        ),
                        "sample_values": " | ".join(
                            sample_values
                        ),
                    }
                )

    return (
        pd.DataFrame(
            schema_rows
        ),
        pd.DataFrame(
            candidate_rows
        ),
    )


def discover_scripts(
    patterns: list[str],
) -> list[
    Path
]:
    paths = []

    for pattern in patterns:
        paths.extend(
            Path(
                "."
            ).glob(
                pattern
            )
        )

    return sorted(
        {
            path
            for path in paths
            if path.is_file()
        }
    )


def audit_scripts(
    script_patterns: list[str],
) -> pd.DataFrame:
    rows = []

    scripts = discover_scripts(
        script_patterns
    )

    search_terms = {
        role: patterns
        for role, patterns in ROLE_PATTERNS.items()
        if role
        in {
            "action",
            "timing",
            "disease_or_diagnosis",
            "care_setting",
            "institution",
        }
    }

    for path in scripts:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lines = content.splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            normalized_line = normalize_name(
                line
            )

            for role, patterns in search_terms.items():
                matched = [
                    pattern
                    for pattern in patterns
                    if re.search(
                        pattern,
                        normalized_line,
                    )
                ]

                if not matched:
                    continue

                rows.append(
                    {
                        "script": str(
                            path
                        ),
                        "line_number": (
                            line_number
                        ),
                        "candidate_role": role,
                        "matched_patterns": " | ".join(
                            matched
                        ),
                        "line": line.strip(),
                    }
                )

    return pd.DataFrame(
        rows
    )


def summarize_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "candidate_role",
                "column",
                "files_containing_column",
                "paths",
                "sample_unique_min",
                "sample_unique_max",
                "sample_values",
            ]
        )

    rows = []

    grouped = candidates.groupby(
        [
            "candidate_role",
            "column",
        ],
        sort=True,
    )

    for (
        role,
        column,
    ), group in grouped:
        rows.append(
            {
                "candidate_role": role,
                "column": column,
                "files_containing_column": int(
                    group[
                        "path"
                    ].nunique()
                ),
                "paths": " || ".join(
                    sorted(
                        group[
                            "path"
                        ].unique()
                    )
                ),
                "sample_unique_min": int(
                    group[
                        "sample_unique"
                    ].min()
                ),
                "sample_unique_max": int(
                    group[
                        "sample_unique"
                    ].max()
                ),
                "sample_values": " || ".join(
                    group[
                        "sample_values"
                    ]
                    .drop_duplicates()
                    .head(
                        5
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
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

    inventory = discover_files(
        args.roots
    )

    (
        schemas,
        candidates,
    ) = audit_csv_schemas(
        inventory,
        args.sample_rows,
    )

    script_evidence = audit_scripts(
        args.script_globs
    )

    candidate_summary = (
        summarize_candidates(
            candidates
        )
    )

    available_roles = set(
        candidates[
            "candidate_role"
        ].unique()
        if not candidates.empty
        else []
    )

    required_roles = {
        "identifier",
        "prediction_or_risk",
        "reference_outcome",
        "restriction_or_observation",
        "action",
    }

    recommended_roles = {
        "timing",
        "disease_or_diagnosis",
        "care_setting",
    }

    missing_required = sorted(
        required_roles
        - available_roles
    )

    missing_recommended = sorted(
        recommended_roles
        - available_roles
    )

    institution_available = (
        "institution"
        in available_roles
    )

    readable_csv_count = int(
        schemas[
            "read_error"
        ].eq(
            ""
        ).groupby(
            schemas[
                "path"
            ]
        ).any().sum()
        if not schemas.empty
        else 0
    )

    validations = pd.DataFrame(
        [
            {
                "check": (
                    "all requested artifact roots exist"
                ),
                "value": int(
                    inventory.loc[
                        inventory[
                            "file_type"
                        ].eq(
                            "missing_root"
                        )
                    ].empty
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": bool(
                    inventory.loc[
                        inventory[
                            "file_type"
                        ].eq(
                            "missing_root"
                        )
                    ].empty
                ),
            },
            {
                "check": (
                    "at least one readable CSV "
                    "artifact found"
                ),
                "value": (
                    readable_csv_count
                ),
                "threshold": 1,
                "comparison": ">=",
                "passed": bool(
                    readable_csv_count
                    >= 1
                ),
            },
            {
                "check": (
                    "all required analysis roles found"
                ),
                "value": (
                    ",".join(
                        missing_required
                    )
                    or "none"
                ),
                "threshold": "none",
                "comparison": "==",
                "passed": not bool(
                    missing_required
                ),
            },
            {
                "check": (
                    "timing disease and care-setting "
                    "candidates found"
                ),
                "value": (
                    ",".join(
                        missing_recommended
                    )
                    or "none"
                ),
                "threshold": "none",
                "comparison": "==",
                "passed": not bool(
                    missing_recommended
                ),
            },
        ]
    )

    status = (
        "phase5h_mimic_heterogeneity_"
        "provenance_audited"
        if validations[
            "passed"
        ].all()
        else (
            "phase5h_mimic_heterogeneity_"
            "audit_requires_review"
        )
    )

    summary = {
        "phase": "5H-D",
        "status": status,
        "roots": args.roots,
        "artifact_file_count": int(
            inventory.loc[
                inventory[
                    "exists"
                ].eq(
                    True
                )
            ].shape[
                0
            ]
        ),
        "readable_csv_count": (
            readable_csv_count
        ),
        "available_roles": sorted(
            available_roles
        ),
        "missing_required_roles": (
            missing_required
        ),
        "missing_recommended_roles": (
            missing_recommended
        ),
        "institution_candidate_available": (
            institution_available
        ),
        "institution_interpretation": (
            "A candidate field requires semantic "
            "validation before claiming institution "
            "heterogeneity."
            if institution_available
            else (
                "No institution identifier was "
                "detected in the frozen artifacts."
            )
        ),
        "h3_role": (
            "Cancellation is retained as a secondary "
            "diagnostic and is not a confirmatory "
            "Phase 5H success criterion."
        ),
    }

    inventory.to_csv(
        output_dir
        / "mimic_artifact_inventory.csv",
        index=False,
    )

    schemas.to_csv(
        output_dir
        / "mimic_csv_schema_inventory.csv",
        index=False,
    )

    candidates.to_csv(
        output_dir
        / "mimic_candidate_columns.csv",
        index=False,
    )

    candidate_summary.to_csv(
        output_dir
        / "mimic_candidate_column_summary.csv",
        index=False,
    )

    script_evidence.to_csv(
        output_dir
        / "mimic_script_provenance.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    (
        output_dir
        / "mimic_heterogeneity_audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5h_mimic_heterogeneity_audit_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5H-D MIMIC heterogeneity audit"
    )

    print(
        "\nValidation"
    )

    print(
        validations.to_string(
            index=False
        )
    )

    print(
        "\nCandidate columns"
    )

    if candidate_summary.empty:
        print(
            "No candidate columns detected."
        )
    else:
        print(
            candidate_summary.to_string(
                index=False
            )
        )

    print(
        "\nInstitution candidate available: "
        f"{institution_available}"
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
