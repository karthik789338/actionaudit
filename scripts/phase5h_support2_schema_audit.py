"""Read-only SUPPORT2 schema and provenance audit for Phase 5H."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCRIPTS = [
    "scripts/fetch_support2.py",
    "scripts/phase1_support2_feasibility.py",
    "scripts/phase1b_support2_logistic.py",
    "scripts/phase2a_support2_km.py",
    "scripts/phase2b_support2_dr.py",
]

ROLE_PATTERNS = {
    "identifier_candidate": [
        r"(^|_)(id|pid|patient|subject|case)($|_)",
    ],
    "mortality_or_event_candidate": [
        r"death",
        r"dead",
        r"mort",
        r"surv",
        r"outcome",
        r"event",
    ],
    "time_or_followup_candidate": [
        r"time",
        r"day",
        r"follow",
        r"duration",
        r"los",
        r"stay",
    ],
    "restriction_or_discharge_candidate": [
        r"disch",
        r"hospital",
        r"hosp",
        r"los",
        r"stay",
        r"censor",
        r"observ",
    ],
    "action_or_policy_candidate": [
        r"dnr",
        r"treat",
        r"action",
        r"intervention",
        r"procedure",
        r"therapy",
        r"transfer",
        r"disch",
    ],
    "institution_candidate": [
        r"site",
        r"center",
        r"hospital_id",
        r"hosp_id",
        r"institution",
        r"provider",
    ],
    "disease_candidate": [
        r"dz",
        r"disease",
        r"diagn",
        r"cancer",
        r"comorb",
        r"class",
        r"group",
    ],
    "risk_or_prediction_candidate": [
        r"risk",
        r"score",
        r"predict",
        r"prob",
        r"prg",
    ],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit SUPPORT2 schema and prior "
            "ActionAudit SUPPORT2 code."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/support2.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase5h_support2_audit",
    )

    parser.add_argument(
        "--scripts",
        nargs="*",
        default=DEFAULT_SCRIPTS,
    )

    return parser.parse_args()


def normalized_name(
    value: str,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.lower(),
    ).strip(
        "_"
    )


def compact_samples(
    series: pd.Series,
    limit: int = 6,
) -> str:
    values = (
        series.dropna()
        .astype(
            str
        )
        .drop_duplicates()
        .head(
            limit
        )
        .tolist()
    )

    return " | ".join(
        values
    )


def build_schema(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in data.columns:
        series = data[
            column
        ]

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric_count = int(
            numeric.notna().sum()
        )

        original_nonmissing = int(
            series.notna().sum()
        )

        mostly_numeric = bool(
            original_nonmissing > 0
            and numeric_count
            / original_nonmissing
            >= 0.95
        )

        rows.append(
            {
                "column": column,
                "normalized_column": (
                    normalized_name(
                        column
                    )
                ),
                "dtype": str(
                    series.dtype
                ),
                "n_nonmissing": (
                    original_nonmissing
                ),
                "n_missing": int(
                    series.isna().sum()
                ),
                "missing_fraction": float(
                    series.isna().mean()
                ),
                "n_unique_nonmissing": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "mostly_numeric": (
                    mostly_numeric
                ),
                "numeric_min": (
                    float(
                        numeric.min()
                    )
                    if numeric_count
                    else np.nan
                ),
                "numeric_max": (
                    float(
                        numeric.max()
                    )
                    if numeric_count
                    else np.nan
                ),
                "sample_values": (
                    compact_samples(
                        series
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_role_candidates(
    schema: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in schema.iterrows():
        name = row[
            "normalized_column"
        ]

        for role, patterns in (
            ROLE_PATTERNS.items()
        ):
            matched_patterns = [
                pattern
                for pattern in patterns
                if re.search(
                    pattern,
                    name,
                )
            ]

            if matched_patterns:
                rows.append(
                    {
                        "column": row[
                            "column"
                        ],
                        "candidate_role": role,
                        "matched_patterns": (
                            " | ".join(
                                matched_patterns
                            )
                        ),
                        "missing_fraction": row[
                            "missing_fraction"
                        ],
                        "n_unique_nonmissing": row[
                            "n_unique_nonmissing"
                        ],
                        "sample_values": row[
                            "sample_values"
                        ],
                    }
                )

    return pd.DataFrame(
        rows
    )


def audit_prior_scripts(
    columns: list[str],
    script_paths: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    usage_rows = []
    artifact_rows = []

    artifact_pattern = re.compile(
        r"""["']([^"']*artifacts/[^"']+)["']"""
    )

    for script_text in script_paths:
        path = Path(
            script_text
        )

        exists = path.exists()

        content = (
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            if exists
            else ""
        )

        for column in columns:
            count = len(
                re.findall(
                    re.escape(
                        column
                    ),
                    content,
                    flags=re.IGNORECASE,
                )
            )

            if count:
                matching_lines = [
                    f"{line_number}:{line.strip()}"
                    for line_number, line in enumerate(
                        content.splitlines(),
                        start=1,
                    )
                    if column.lower()
                    in line.lower()
                ]

                usage_rows.append(
                    {
                        "script": script_text,
                        "script_exists": exists,
                        "column": column,
                        "occurrence_count": count,
                        "matching_lines": (
                            " || ".join(
                                matching_lines[
                                    :12
                                ]
                            )
                        ),
                    }
                )

        for match in artifact_pattern.finditer(
            content
        ):
            artifact_rows.append(
                {
                    "script": script_text,
                    "artifact_reference": (
                        match.group(
                            1
                        )
                    ),
                }
            )

        if not exists:
            usage_rows.append(
                {
                    "script": script_text,
                    "script_exists": False,
                    "column": "",
                    "occurrence_count": 0,
                    "matching_lines": "",
                }
            )

    return (
        pd.DataFrame(
            usage_rows
        ),
        pd.DataFrame(
            artifact_rows
        ).drop_duplicates(),
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

    if not input_path.exists():
        raise FileNotFoundError(
            f"SUPPORT2 input not found: "
            f"{input_path}"
        )

    data = pd.read_csv(
        input_path,
        low_memory=False,
    )

    if data.empty:
        raise ValueError(
            "SUPPORT2 input contains no rows"
        )

    schema = build_schema(
        data
    )

    role_candidates = (
        build_role_candidates(
            schema
        )
    )

    (
        script_usage,
        artifact_references,
    ) = audit_prior_scripts(
        data.columns.tolist(),
        args.scripts,
    )

    script_exists = {
        script: Path(
            script
        ).exists()
        for script in args.scripts
    }

    unnamed_columns = [
        column
        for column in data.columns
        if normalized_name(
            column
        ).startswith(
            "unnamed"
        )
    ]

    summary = {
        "phase": "5H-A",
        "input": str(
            input_path
        ),
        "n_rows": int(
            len(
                data
            )
        ),
        "n_columns": int(
            data.shape[
                1
            ]
        ),
        "duplicate_full_rows": int(
            data.duplicated().sum()
        ),
        "unnamed_columns": (
            unnamed_columns
        ),
        "scripts": script_exists,
        "columns_referenced_by_prior_scripts": int(
            script_usage.loc[
                script_usage[
                    "column"
                ].ne(
                    ""
                ),
                "column",
            ].nunique()
        ),
        "high_missingness_columns": (
            schema.loc[
                schema[
                    "missing_fraction"
                ]
                >= 0.50,
                "column",
            ].tolist()
        ),
        "note": (
            "Candidate roles are keyword-based "
            "audit signals, not frozen definitions."
        ),
    }

    schema.to_csv(
        output_dir
        / "support2_schema.csv",
        index=False,
    )

    role_candidates.to_csv(
        output_dir
        / "support2_keyword_role_candidates.csv",
        index=False,
    )

    script_usage.to_csv(
        output_dir
        / "support2_prior_script_column_usage.csv",
        index=False,
    )

    artifact_references.to_csv(
        output_dir
        / "support2_prior_artifact_references.csv",
        index=False,
    )

    (
        output_dir
        / "support2_audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = (
        "phase5h_support2_schema_and_provenance_audited"
        if all(
            script_exists.values()
        )
        else (
            "phase5h_support2_schema_audited_"
            "with_missing_prior_scripts"
        )
    )

    (
        output_dir
        / "phase5h_support2_audit_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5H-A SUPPORT2 schema "
        "and provenance audit"
    )

    print(
        f"\nRows: {len(data):,}"
    )

    print(
        f"Columns: {data.shape[1]}"
    )

    print(
        f"Duplicate full rows: "
        f"{data.duplicated().sum():,}"
    )

    print(
        "\nColumns used by prior SUPPORT2 scripts"
    )

    if script_usage.empty:
        print(
            "No exact column references found."
        )
    else:
        usage_summary = (
            script_usage.loc[
                script_usage[
                    "column"
                ].ne(
                    ""
                )
            ]
            .groupby(
                "column",
                as_index=False,
            )
            .agg(
                scripts_using_column=(
                    "script",
                    "nunique",
                ),
                total_occurrences=(
                    "occurrence_count",
                    "sum",
                ),
            )
            .sort_values(
                [
                    "scripts_using_column",
                    "total_occurrences",
                ],
                ascending=False,
            )
        )

        print(
            usage_summary.to_string(
                index=False
            )
        )

    print(
        "\nKeyword role candidates"
    )

    if role_candidates.empty:
        print(
            "No keyword candidates found."
        )
    else:
        print(
            role_candidates[
                [
                    "column",
                    "candidate_role",
                    "missing_fraction",
                    "n_unique_nonmissing",
                    "sample_values",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nHighest-missingness columns"
    )

    print(
        schema.sort_values(
            "missing_fraction",
            ascending=False,
        )[
            [
                "column",
                "dtype",
                "missing_fraction",
                "n_unique_nonmissing",
                "sample_values",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
