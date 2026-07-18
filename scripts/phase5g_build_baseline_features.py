"""Build the exact frozen admission-time covariates for Phase 5G-B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase5c_train_admission_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    construct_feature_frame,
)


IDENTIFIERS = [
    "subject_id",
    "hadm_id",
]

REQUIRED_ANALYSIS_COLUMNS = [
    "subject_id",
    "hadm_id",
    "prediction",
    "outcome",
    "observed",
    "restricted_event",
    "restricted_time",
    "analysis_action",
]

FORBIDDEN_NUISANCE_FEATURES = {
    "analysis_action",
    "deathtime",
    "discharge_group",
    "discharge_location",
    "dischtime",
    "dod",
    "exit_time",
    "has_icu_stay",
    "hidden_event",
    "observed",
    "outcome",
    "primary_action",
    "raw_discharge_location",
    "restricted_censored_by_discharge",
    "restricted_event",
    "restricted_time",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct exact frozen admission-time "
            "covariates for Phase 5G-B."
        )
    )

    parser.add_argument(
        "--cohort",
        default=(
            "artifacts/phase5c_frozen_cohort/"
            "primary_inpatient_cohort.csv.gz"
        ),
    )

    parser.add_argument(
        "--predictions",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
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
            "phase5g_baseline_features"
        ),
    )

    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing columns: {missing}"
        )


def validate_unique_keys(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    duplicated = frame.duplicated(
        IDENTIFIERS,
        keep=False,
    )

    if duplicated.any():
        examples = (
            frame.loc[
                duplicated,
                IDENTIFIERS,
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            f"{label} has duplicate admission keys: "
            f"{examples}"
        )


def main() -> None:
    args = parse_arguments()

    cohort_path = Path(
        args.cohort
    )

    prediction_path = Path(
        args.predictions
    )

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

    predictions = pd.read_csv(
        prediction_path,
        low_memory=False,
    )

    require_columns(
        predictions,
        REQUIRED_ANALYSIS_COLUMNS,
        label="Frozen test predictions",
    )

    validate_unique_keys(
        predictions,
        label="Frozen test predictions",
    )

    print(
        "Constructing exact Phase 5C "
        "admission-time features..."
    )

    full_feature_frame = construct_feature_frame(
        cohort_path=cohort_path,
        mimic_root=mimic_root,
    )

    baseline_features = (
        list(
            NUMERIC_FEATURES
        )
        + list(
            CATEGORICAL_FEATURES
        )
    )

    require_columns(
        full_feature_frame,
        IDENTIFIERS
        + baseline_features,
        label="Constructed feature frame",
    )

    validate_unique_keys(
        full_feature_frame,
        label="Constructed feature frame",
    )

    forbidden_overlap = sorted(
        set(
            baseline_features
        )
        .intersection(
            FORBIDDEN_NUISANCE_FEATURES
        )
    )

    if forbidden_overlap:
        raise ValueError(
            "Frozen baseline feature manifest contains "
            "post-admission variables: "
            f"{forbidden_overlap}"
        )

    feature_subset = full_feature_frame[
        IDENTIFIERS
        + baseline_features
    ].copy()

    merged = predictions.merge(
        feature_subset,
        on=IDENTIFIERS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        merged[
            "_merge"
        ].ne(
            "both"
        ).sum()
    )

    merged = merged.drop(
        columns=[
            "_merge",
        ]
    )

    numeric_features = list(
        NUMERIC_FEATURES
    )

    categorical_features = list(
        CATEGORICAL_FEATURES
    )

    feature_manifest = pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_type": "numeric",
                "practitioner_available": True,
                "nuisance_model_feature": True,
            }
            for feature in numeric_features
        ]
        + [
            {
                "feature": feature,
                "feature_type": "categorical",
                "practitioner_available": True,
                "nuisance_model_feature": True,
            }
            for feature in categorical_features
        ]
    )

    missingness_rows = []

    for feature in baseline_features:
        series = merged[
            feature
        ]

        missingness_rows.append(
            {
                "feature": feature,
                "feature_type": (
                    "numeric"
                    if feature
                    in numeric_features
                    else "categorical"
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_fraction": float(
                    series.isna().mean()
                ),
                "unique_nonmissing": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    missingness = pd.DataFrame(
        missingness_rows
    )

    numeric_finite_fraction = {}

    for feature in numeric_features:
        numeric = pd.to_numeric(
            merged[
                feature
            ],
            errors="coerce",
        )

        finite_or_missing = (
            np.isfinite(
                numeric.fillna(
                    0.0
                )
            )
        )

        numeric_finite_fraction[
            feature
        ] = float(
            finite_or_missing.mean()
        )

    validation = pd.DataFrame(
        [
            {
                "check": (
                    "test admission count preserved"
                ),
                "value": len(
                    merged
                ),
                "threshold": len(
                    predictions
                ),
                "comparison": "==",
                "passed": (
                    len(
                        merged
                    )
                    == len(
                        predictions
                    )
                ),
            },
            {
                "check": (
                    "all frozen test admissions matched"
                ),
                "value": unmatched,
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    unmatched
                    == 0
                ),
            },
            {
                "check": (
                    "all 13 numeric features present"
                ),
                "value": len(
                    numeric_features
                ),
                "threshold": 13,
                "comparison": "==",
                "passed": (
                    len(
                        numeric_features
                    )
                    == 13
                ),
            },
            {
                "check": (
                    "all 8 categorical features present"
                ),
                "value": len(
                    categorical_features
                ),
                "threshold": 8,
                "comparison": "==",
                "passed": (
                    len(
                        categorical_features
                    )
                    == 8
                ),
            },
            {
                "check": (
                    "no forbidden post-admission "
                    "nuisance features"
                ),
                "value": len(
                    forbidden_overlap
                ),
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    len(
                        forbidden_overlap
                    )
                    == 0
                ),
            },
            {
                "check": (
                    "all numeric values finite "
                    "after preserving missingness"
                ),
                "value": min(
                    numeric_finite_fraction.values()
                ),
                "threshold": 1.0,
                "comparison": "==",
                "passed": bool(
                    all(
                        value
                        == 1.0
                        for value
                        in numeric_finite_fraction.values()
                    )
                ),
            },
            {
                "check": (
                    "admission identifiers remain unique"
                ),
                "value": int(
                    merged.duplicated(
                        IDENTIFIERS
                    ).sum()
                ),
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    not merged.duplicated(
                        IDENTIFIERS
                    ).any()
                ),
            },
            {
                "check": (
                    "current-admission ICU indicator "
                    "excluded from nuisance features"
                ),
                "value": (
                    "has_icu_stay"
                    in baseline_features
                ),
                "threshold": False,
                "comparison": "==",
                "passed": (
                    "has_icu_stay"
                    not in baseline_features
                ),
            },
        ]
    )

    status = (
        "phase5g_exact_baseline_features_ready"
        if validation[
            "passed"
        ].all()
        else (
            "phase5g_baseline_features_"
            "require_review"
        )
    )

    merged.to_csv(
        output_dir
        / "rich_test_analysis.csv.gz",
        index=False,
        compression="gzip",
    )

    feature_manifest.to_csv(
        output_dir
        / "baseline_feature_manifest.csv",
        index=False,
    )

    missingness.to_csv(
        output_dir
        / "baseline_feature_missingness.csv",
        index=False,
    )

    validation.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5G-B feature assembly",
        "status": status,
        "source_constructor": (
            "scripts/"
            "phase5c_train_admission_model.py::"
            "construct_feature_frame"
        ),
        "cohort": str(
            cohort_path
        ),
        "predictions": str(
            prediction_path
        ),
        "mimic_root": str(
            mimic_root
        ),
        "n_rows": int(
            len(
                merged
            )
        ),
        "n_numeric_features": len(
            numeric_features
        ),
        "n_categorical_features": len(
            categorical_features
        ),
        "numeric_features": (
            numeric_features
        ),
        "categorical_features": (
            categorical_features
        ),
        "nuisance_feature_policy": (
            "Only frozen admission-time features and "
            "the frozen prediction may enter nuisance "
            "models. Outcomes, follow-up, discharge, "
            "current ICU use, and action variables are "
            "excluded from the feature matrix."
        ),
    }

    (
        output_dir
        / "baseline_feature_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5g_baseline_feature_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nFeature assembly validation"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nFeature missingness"
    )

    print(
        missingness.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"Artifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
