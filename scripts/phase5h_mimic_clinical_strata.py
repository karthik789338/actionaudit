"""Phase 5H-E: construct MIMIC clinical heterogeneity strata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TEST_N = 31012

EXPECTED_PHASE5C_STATUS = (
    "phase5c_frozen_for_manuscript"
)

RAW_TABLES = {
    "diagnoses": "hosp/diagnoses_icd.csv.gz",
    "diagnosis_dictionary": (
        "hosp/d_icd_diagnoses.csv.gz"
    ),
    "services": "hosp/services.csv.gz",
    "drg": "hosp/drgcodes.csv.gz",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct diagnosis and clinical-service "
            "strata for the frozen MIMIC test cohort."
        )
    )

    parser.add_argument(
        "--predictions",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--rich-input",
        default=(
            "artifacts/phase5g_baseline_features/"
            "rich_test_analysis.csv.gz"
        ),
    )

    parser.add_argument(
        "--phase5c-status",
        default=(
            "artifacts/phase5c_frozen_for_manuscript/"
            "phase5c_freeze_status.txt"
        ),
    )

    parser.add_argument(
        "--mimic-root",
        default="data/raw/mimiciv/3.1",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5h_mimic_clinical_strata"
        ),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
    )

    return parser.parse_args()


def require_columns(
    data: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing columns: {missing}"
        )


def load_test_cohort(
    predictions_path: Path,
    rich_path: Path,
) -> pd.DataFrame:
    predictions = pd.read_csv(
        predictions_path,
        low_memory=False,
    )

    require_columns(
        predictions,
        [
            "subject_id",
            "hadm_id",
            "prediction",
            "outcome",
            "observed",
        ],
        label="Frozen test predictions",
    )

    if predictions[
        [
            "subject_id",
            "hadm_id",
        ]
    ].duplicated().any():
        raise ValueError(
            "Frozen predictions contain "
            "duplicate admissions"
        )

    rich = pd.read_csv(
        rich_path,
        low_memory=False,
    )

    require_columns(
        rich,
        [
            "subject_id",
            "hadm_id",
            "anchor_year_group",
            "admission_type",
            "admission_location",
            "has_icu_stay",
            "admission_hour_sin",
            "admission_hour_cos",
            "admission_weekday_sin",
            "admission_weekday_cos",
        ],
        label="Rich MIMIC analysis input",
    )

    rich_columns = [
        "subject_id",
        "hadm_id",
        "anchor_year_group",
        "admission_type",
        "admission_location",
        "has_icu_stay",
        "admission_hour_sin",
        "admission_hour_cos",
        "admission_weekday_sin",
        "admission_weekday_cos",
    ]

    join_keys = [
        "subject_id",
        "hadm_id",
    ]

    rich_additional_columns = [
        column
        for column in rich_columns
        if (
            column in join_keys
            or column not in predictions.columns
        )
    ]

    cohort = predictions.merge(
        rich[
            rich_additional_columns
        ],
        on=join_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        cohort[
            "_merge"
        ].ne(
            "both"
        ).sum()
    )

    if unmatched:
        raise ValueError(
            f"{unmatched} admissions failed "
            "the rich-feature join"
        )

    return (
        cohort.drop(
            columns=[
                "_merge",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def load_filtered_table(
    path: Path,
    *,
    hadm_ids: set[int],
    chunksize: int,
) -> pd.DataFrame:
    pieces = []

    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        low_memory=False,
    ):
        if "hadm_id" not in chunk.columns:
            raise ValueError(
                f"{path} has no hadm_id column"
            )

        filtered = chunk.loc[
            chunk[
                "hadm_id"
            ].isin(
                hadm_ids
            )
        ]

        if not filtered.empty:
            pieces.append(
                filtered
            )

    if not pieces:
        return pd.DataFrame()

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def clean_icd_code(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(
            value
        ).upper(),
    )


def icd9_chapter(
    code: str,
) -> str:
    if not code:
        return "missing_or_unclassified"

    if code.startswith(
        "E"
    ):
        return "external_causes"

    if code.startswith(
        "V"
    ):
        return "supplementary_factors"

    match = re.match(
        r"(\d{3})",
        code,
    )

    if not match:
        return "missing_or_unclassified"

    number = int(
        match.group(
            1
        )
    )

    ranges = [
        (
            1,
            139,
            "infectious_and_parasitic",
        ),
        (
            140,
            239,
            "neoplasms",
        ),
        (
            240,
            279,
            "endocrine_metabolic_immunity",
        ),
        (
            280,
            289,
            "blood_disorders",
        ),
        (
            290,
            319,
            "mental_disorders",
        ),
        (
            320,
            389,
            "nervous_and_sense_organs",
        ),
        (
            390,
            459,
            "circulatory",
        ),
        (
            460,
            519,
            "respiratory",
        ),
        (
            520,
            579,
            "digestive",
        ),
        (
            580,
            629,
            "genitourinary",
        ),
        (
            630,
            679,
            "pregnancy_childbirth",
        ),
        (
            680,
            709,
            "skin_subcutaneous",
        ),
        (
            710,
            739,
            "musculoskeletal",
        ),
        (
            740,
            759,
            "congenital_anomalies",
        ),
        (
            760,
            779,
            "perinatal_conditions",
        ),
        (
            780,
            799,
            "symptoms_and_ill_defined",
        ),
        (
            800,
            999,
            "injury_and_poisoning",
        ),
    ]

    for lower, upper, label in ranges:
        if lower <= number <= upper:
            return label

    return "missing_or_unclassified"


def icd10_chapter(
    code: str,
) -> str:
    if not code:
        return "missing_or_unclassified"

    letter = code[
        0
    ]

    two_digits_match = re.match(
        r"[A-Z](\d{2})",
        code,
    )

    two_digits = (
        int(
            two_digits_match.group(
                1
            )
        )
        if two_digits_match
        else None
    )

    if letter in {
        "A",
        "B",
    }:
        return "infectious_and_parasitic"

    if letter == "C":
        return "neoplasms"

    if letter == "D":
        if (
            two_digits is not None
            and two_digits <= 49
        ):
            return "neoplasms"

        return "blood_and_immune_disorders"

    mapping = {
        "E": "endocrine_nutritional_metabolic",
        "F": "mental_behavioral",
        "G": "nervous_system",
        "H": "eye_and_ear",
        "I": "circulatory",
        "J": "respiratory",
        "K": "digestive",
        "L": "skin_subcutaneous",
        "M": "musculoskeletal",
        "N": "genitourinary",
        "O": "pregnancy_childbirth",
        "P": "perinatal_conditions",
        "Q": "congenital_anomalies",
        "R": "symptoms_and_abnormal_findings",
        "S": "injury_poisoning",
        "T": "injury_poisoning",
        "V": "external_causes",
        "W": "external_causes",
        "X": "external_causes",
        "Y": "external_causes",
        "Z": "factors_influencing_health",
        "U": "special_purpose_codes",
    }

    return mapping.get(
        letter,
        "missing_or_unclassified",
    )


def derive_icd_chapter(
    code: object,
    version: object,
) -> str:
    clean_code = clean_icd_code(
        code
    )

    try:
        numeric_version = int(
            version
        )
    except (
        TypeError,
        ValueError,
    ):
        return "missing_or_unclassified"

    if numeric_version == 9:
        return icd9_chapter(
            clean_code
        )

    if numeric_version == 10:
        return icd10_chapter(
            clean_code
        )

    return "missing_or_unclassified"


def reconstruct_hour(
    sine: pd.Series,
    cosine: pd.Series,
) -> pd.Series:
    angle = np.arctan2(
        sine.to_numpy(
            dtype=float
        ),
        cosine.to_numpy(
            dtype=float
        ),
    )

    hour = (
        angle
        % (
            2.0
            * np.pi
        )
    ) * 24.0 / (
        2.0
        * np.pi
    )

    return pd.Series(
        hour,
        index=sine.index,
    )


def hour_stratum(
    hour: pd.Series,
) -> pd.Series:
    return pd.cut(
        hour,
        bins=[
            -np.inf,
            5.999999,
            11.999999,
            17.999999,
            np.inf,
        ],
        labels=[
            "overnight_00_to_05",
            "morning_06_to_11",
            "afternoon_12_to_17",
            "evening_18_to_23",
        ],
    ).astype(
        "string"
    )


def reconstruct_weekday(
    sine: pd.Series,
    cosine: pd.Series,
) -> pd.Series:
    angle = np.arctan2(
        sine.to_numpy(
            dtype=float
        ),
        cosine.to_numpy(
            dtype=float
        ),
    )

    day = np.rint(
        (
            angle
            % (
                2.0
                * np.pi
            )
        )
        * 7.0
        / (
            2.0
            * np.pi
        )
    ).astype(
        int
    ) % 7

    return pd.Series(
        day,
        index=sine.index,
    )


def build_principal_diagnosis(
    diagnoses: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        diagnoses,
        [
            "subject_id",
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
        ],
        label="MIMIC diagnoses",
    )

    require_columns(
        dictionary,
        [
            "icd_code",
            "icd_version",
            "long_title",
        ],
        label="MIMIC diagnosis dictionary",
    )

    diagnoses = diagnoses.copy()

    diagnoses[
        "seq_num"
    ] = pd.to_numeric(
        diagnoses[
            "seq_num"
        ],
        errors="coerce",
    )

    diagnoses = diagnoses.sort_values(
        [
            "hadm_id",
            "seq_num",
            "icd_code",
        ],
        kind="stable",
    )

    principal = (
        diagnoses.loc[
            diagnoses[
                "seq_num"
            ].eq(
                1
            )
        ]
        .drop_duplicates(
            subset=[
                "hadm_id",
            ],
            keep="first",
        )
        .copy()
    )

    dictionary = dictionary.drop_duplicates(
        subset=[
            "icd_code",
            "icd_version",
        ],
        keep="first",
    )

    principal = principal.merge(
        dictionary[
            [
                "icd_code",
                "icd_version",
                "long_title",
            ]
        ],
        on=[
            "icd_code",
            "icd_version",
        ],
        how="left",
        validate="many_to_one",
    )

    principal[
        "principal_icd_chapter"
    ] = [
        derive_icd_chapter(
            code,
            version,
        )
        for code, version in zip(
            principal[
                "icd_code"
            ],
            principal[
                "icd_version"
            ],
            strict=False,
        )
    ]

    diagnosis_counts = (
        diagnoses.groupby(
            "hadm_id"
        )
        .size()
        .rename(
            "diagnosis_count"
        )
        .reset_index()
    )

    principal = principal.merge(
        diagnosis_counts,
        on="hadm_id",
        how="left",
        validate="one_to_one",
    )

    return principal[
        [
            "subject_id",
            "hadm_id",
            "icd_code",
            "icd_version",
            "long_title",
            "principal_icd_chapter",
            "diagnosis_count",
        ]
    ]


def build_service_strata(
    services: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        services,
        [
            "subject_id",
            "hadm_id",
            "transfertime",
            "curr_service",
        ],
        label="MIMIC services",
    )

    services = services.copy()

    services[
        "transfertime"
    ] = pd.to_datetime(
        services[
            "transfertime"
        ],
        errors="coerce",
    )

    services = services.sort_values(
        [
            "hadm_id",
            "transfertime",
        ],
        kind="stable",
    )

    rows = []

    for hadm_id, group in services.groupby(
        "hadm_id",
        sort=False,
    ):
        rows.append(
            {
                "hadm_id": hadm_id,
                "first_service": (
                    group[
                        "curr_service"
                    ].iloc[
                        0
                    ]
                ),
                "last_service": (
                    group[
                        "curr_service"
                    ].iloc[
                        -1
                    ]
                ),
                "service_record_count": int(
                    len(
                        group
                    )
                ),
                "service_transition_count": int(
                    max(
                        len(
                            group
                        )
                        - 1,
                        0,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_drg_strata(
    drg: pd.DataFrame,
) -> pd.DataFrame:
    if drg.empty:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "drg_type",
                "drg_code",
                "description",
                "drg_severity",
                "drg_mortality",
            ]
        )

    require_columns(
        drg,
        [
            "hadm_id",
            "drg_type",
            "drg_code",
            "description",
        ],
        label="MIMIC DRG",
    )

    preferred_order = {
        "APR": 0,
        "HCFA": 1,
        "MS": 2,
    }

    drg = drg.copy()

    drg[
        "_preference"
    ] = (
        drg[
            "drg_type"
        ]
        .astype(
            str
        )
        .str.upper()
        .map(
            preferred_order
        )
        .fillna(
            99
        )
    )

    drg = (
        drg.sort_values(
            [
                "hadm_id",
                "_preference",
                "drg_code",
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "hadm_id",
            ],
            keep="first",
        )
    )

    output_columns = [
        "hadm_id",
        "drg_type",
        "drg_code",
        "description",
    ]

    for optional in [
        "drg_severity",
        "drg_mortality",
    ]:
        if optional not in drg.columns:
            drg[
                optional
            ] = np.nan

        output_columns.append(
            optional
        )

    return drg[
        output_columns
    ]


def coverage_row(
    data: pd.DataFrame,
    column: str,
    label: str,
) -> dict[str, object]:
    present = data[
        column
    ].notna()

    return {
        "stratum": label,
        "column": column,
        "n_available": int(
            present.sum()
        ),
        "coverage": float(
            present.mean()
        ),
        "n_unique": int(
            data.loc[
                present,
                column,
            ].nunique()
        ),
        "sample_values": " | ".join(
            data.loc[
                present,
                column,
            ]
            .astype(
                str
            )
            .drop_duplicates()
            .head(
                12
            )
            .tolist()
        ),
    }


def build_validations(
    *,
    data: pd.DataFrame,
    phase5c_status: str,
    raw_paths: dict[
        str,
        Path,
    ],
) -> pd.DataFrame:
    principal_coverage = float(
        data[
            "principal_icd_chapter"
        ].notna().mean()
    )

    classified_coverage = float(
        data[
            "principal_icd_chapter"
        ]
        .ne(
            "missing_or_unclassified"
        )
        .fillna(
            False
        )
        .mean()
    )

    service_coverage = float(
        data[
            "first_service"
        ].notna().mean()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "Phase 5C benchmark already frozen"
                ),
                "value": phase5c_status,
                "threshold": (
                    EXPECTED_PHASE5C_STATUS
                ),
                "comparison": "==",
                "passed": (
                    phase5c_status
                    == EXPECTED_PHASE5C_STATUS
                ),
            },
            {
                "check": (
                    "all required raw MIMIC "
                    "tables exist"
                ),
                "value": int(
                    all(
                        path.exists()
                        for path in raw_paths.values()
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": all(
                    path.exists()
                    for path in raw_paths.values()
                ),
            },
            {
                "check": (
                    "frozen test cohort has "
                    "31012 unique admissions"
                ),
                "value": len(
                    data
                ),
                "threshold": EXPECTED_TEST_N,
                "comparison": "==",
                "passed": bool(
                    len(
                        data
                    )
                    == EXPECTED_TEST_N
                    and not data[
                        "hadm_id"
                    ].duplicated().any()
                ),
            },
            {
                "check": (
                    "principal diagnosis coverage "
                    "exceeds 95 percent"
                ),
                "value": principal_coverage,
                "threshold": 0.95,
                "comparison": ">=",
                "passed": (
                    principal_coverage
                    >= 0.95
                ),
            },
            {
                "check": (
                    "classified ICD chapter coverage "
                    "exceeds 95 percent"
                ),
                "value": classified_coverage,
                "threshold": 0.95,
                "comparison": ">=",
                "passed": (
                    classified_coverage
                    >= 0.95
                ),
            },
            {
                "check": (
                    "first-service coverage "
                    "exceeds 95 percent"
                ),
                "value": service_coverage,
                "threshold": 0.95,
                "comparison": ">=",
                "passed": (
                    service_coverage
                    >= 0.95
                ),
            },
            {
                "check": (
                    "admission timing strata complete"
                ),
                "value": int(
                    data[
                        [
                            "admission_hour_stratum",
                            "admission_weekpart",
                            "anchor_year_group",
                        ]
                    ].notna().all().all()
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": bool(
                    data[
                        [
                            "admission_hour_stratum",
                            "admission_weekpart",
                            "anchor_year_group",
                        ]
                    ].notna().all().all()
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    mimic_root = Path(
        args.mimic_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_paths = {
        name: (
            mimic_root
            / relative
        )
        for name, relative in RAW_TABLES.items()
    }

    missing_files = [
        str(
            path
        )
        for path in raw_paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing MIMIC tables: "
            f"{missing_files}"
        )

    phase5c_status = Path(
        args.phase5c_status
    ).read_text(
        encoding="utf-8"
    ).strip()

    cohort = load_test_cohort(
        Path(
            args.predictions
        ),
        Path(
            args.rich_input
        ),
    )

    hadm_ids = set(
        cohort[
            "hadm_id"
        ].astype(
            int
        )
    )

    print(
        "Phase 5H-E MIMIC clinical strata"
    )

    print(
        f"\nFrozen test admissions: "
        f"{len(cohort):,}"
    )

    print(
        "\nLoading diagnosis records..."
    )

    diagnoses = load_filtered_table(
        raw_paths[
            "diagnoses"
        ],
        hadm_ids=hadm_ids,
        chunksize=args.chunksize,
    )

    dictionary = pd.read_csv(
        raw_paths[
            "diagnosis_dictionary"
        ],
        low_memory=False,
    )

    principal = build_principal_diagnosis(
        diagnoses,
        dictionary,
    )

    print(
        "Loading service records..."
    )

    services = load_filtered_table(
        raw_paths[
            "services"
        ],
        hadm_ids=hadm_ids,
        chunksize=args.chunksize,
    )

    service_strata = build_service_strata(
        services
    )

    print(
        "Loading DRG records..."
    )

    drg = load_filtered_table(
        raw_paths[
            "drg"
        ],
        hadm_ids=hadm_ids,
        chunksize=args.chunksize,
    )

    drg_strata = build_drg_strata(
        drg
    )

    analysis = cohort.merge(
        principal,
        on=[
            "subject_id",
            "hadm_id",
        ],
        how="left",
        validate="one_to_one",
    )

    analysis = analysis.merge(
        service_strata,
        on="hadm_id",
        how="left",
        validate="one_to_one",
    )

    analysis = analysis.merge(
        drg_strata,
        on="hadm_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_drg",
        ),
    )

    analysis[
        "principal_icd_chapter"
    ] = (
        analysis[
            "principal_icd_chapter"
        ]
        .fillna(
            "missing_or_unclassified"
        )
    )

    admission_hour = reconstruct_hour(
        analysis[
            "admission_hour_sin"
        ],
        analysis[
            "admission_hour_cos"
        ],
    )

    admission_weekday = reconstruct_weekday(
        analysis[
            "admission_weekday_sin"
        ],
        analysis[
            "admission_weekday_cos"
        ],
    )

    analysis[
        "admission_hour"
    ] = admission_hour

    analysis[
        "admission_hour_stratum"
    ] = hour_stratum(
        admission_hour
    )

    analysis[
        "admission_weekday_index"
    ] = admission_weekday

    analysis[
        "admission_weekpart"
    ] = np.where(
        admission_weekday.isin(
            [
                5,
                6,
            ]
        ),
        "weekend",
        "weekday",
    )

    coverage = pd.DataFrame(
        [
            coverage_row(
                analysis,
                "principal_icd_chapter",
                "principal ICD chapter",
            ),
            coverage_row(
                analysis,
                "long_title",
                "principal diagnosis title",
            ),
            coverage_row(
                analysis,
                "first_service",
                "first hospital service",
            ),
            coverage_row(
                analysis,
                "last_service",
                "last hospital service",
            ),
            coverage_row(
                analysis,
                "drg_type",
                "DRG type",
            ),
            coverage_row(
                analysis,
                "description",
                "DRG description",
            ),
            coverage_row(
                analysis,
                "anchor_year_group",
                "admission era",
            ),
            coverage_row(
                analysis,
                "admission_hour_stratum",
                "admission hour",
            ),
            coverage_row(
                analysis,
                "admission_weekpart",
                "admission weekpart",
            ),
            coverage_row(
                analysis,
                "has_icu_stay",
                "ICU involvement",
            ),
        ]
    )

    validations = build_validations(
        data=analysis,
        phase5c_status=phase5c_status,
        raw_paths=raw_paths,
    )

    status = (
        "phase5h_mimic_clinical_strata_ready"
        if validations[
            "passed"
        ].all()
        else (
            "phase5h_mimic_clinical_strata_"
            "require_review"
        )
    )

    analysis.to_csv(
        output_dir
        / "mimic_harmonized_clinical_test.csv.gz",
        index=False,
        compression="gzip",
    )

    principal.to_csv(
        output_dir
        / "principal_diagnosis_strata.csv.gz",
        index=False,
        compression="gzip",
    )

    service_strata.to_csv(
        output_dir
        / "service_strata.csv.gz",
        index=False,
        compression="gzip",
    )

    drg_strata.to_csv(
        output_dir
        / "drg_strata.csv.gz",
        index=False,
        compression="gzip",
    )

    coverage.to_csv(
        output_dir
        / "clinical_strata_coverage.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5H-E",
        "status": status,
        "dataset": "MIMIC-IV",
        "test_n": int(
            len(
                analysis
            )
        ),
        "disease_definition": (
            "Principal diagnosis, seq_num=1, "
            "mapped to deterministic ICD-9/10 "
            "chapter-level strata."
        ),
        "service_definition": (
            "First and last curr_service ordered "
            "by service transfer time."
        ),
        "timing_definitions": {
            "era": "anchor_year_group",
            "hour": (
                "Four reconstructed admission-hour "
                "strata from frozen cyclic features."
            ),
            "weekpart": (
                "Weekday versus weekend reconstructed "
                "from frozen cyclic features."
            ),
        },
        "drg_role": (
            "Secondary descriptive sensitivity only; "
            "not the primary disease definition."
        ),
        "institution_heterogeneity": (
            "Unavailable."
        ),
    }

    (
        output_dir
        / "mimic_clinical_strata_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5h_mimic_clinical_strata_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nCoverage"
    )

    print(
        coverage.to_string(
            index=False
        )
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
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
