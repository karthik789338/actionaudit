"""Freeze SUPPORT2 definitions for cross-dataset Phase 5H."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from actionaudit.support2 import (
    build_support2_horizon_view,
    load_support2_csv,
)


EXPECTED_HORIZON = 60
EXPECTED_TEST_N = 1821
EXPECTED_REFERENCE_EVENT_RATE = (
    0.3470620538165843
)
EXPECTED_OBSERVED_FRACTION = (
    0.3025809994508512
)
EXPECTED_HIDDEN_N = 1270
EXPECTED_HIDDEN_EVENT_N = 159
EXPECTED_BOUNDARY_ACTION = (
    "alive_hospital_discharge"
)

PROGNOSTIC_OR_POST_BASELINE_FIELDS = {
    "death",
    "hospdead",
    "slos",
    "d.time",
    "surv2m",
    "surv6m",
    "prg2m",
    "prg6m",
    "dnr",
    "dnrday",
    "charges",
    "totcst",
    "totmcst",
    "sfdm2",
}

HASH_INPUTS = [
    "data/raw/support2.csv",
    (
        "artifacts/phase1b_logistic/"
        "test_predictions.csv"
    ),
    (
        "artifacts/phase1b_logistic/"
        "split_manifest.csv"
    ),
    (
        "artifacts/phase1b_logistic/"
        "feature_manifest.csv"
    ),
    (
        "artifacts/phase1b_logistic/"
        "experiment_metadata.json"
    ),
    (
        "artifacts/phase1b_logistic/"
        "scalar_summary.csv"
    ),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the SUPPORT2 60-day "
            "cross-dataset analysis definitions."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/support2.csv",
    )

    parser.add_argument(
        "--phase1b-dir",
        default="artifacts/phase1b_logistic",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5h_support2_definitions"
        ),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing freeze input: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype(
            "string"
        )
        .str.strip()
        .str.lower()
    )


def dnr_policy_stratum(
    series: pd.Series,
) -> pd.Series:
    normalized = normalize_text(
        series
    )

    mapping = {
        "no dnr": "no_dnr",
        "dnr before sadm": (
            "dnr_before_study_admission"
        ),
        "dnr after sadm": (
            "dnr_after_study_admission"
        ),
    }

    return (
        normalized.map(
            mapping
        )
        .fillna(
            "dnr_missing_or_unknown"
        )
    )


def hospital_timing_stratum(
    hday: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        hday,
        errors="coerce",
    )

    return pd.cut(
        numeric,
        bins=[
            -np.inf,
            1,
            3,
            7,
            np.inf,
        ],
        labels=[
            "hospital_day_1",
            "hospital_days_2_to_3",
            "hospital_days_4_to_7",
            "hospital_day_8_plus",
        ],
        include_lowest=True,
    ).astype(
        "string"
    ).fillna(
        "hospital_day_missing"
    )


def equal_text(
    left: pd.Series,
    right: pd.Series,
) -> bool:
    return bool(
        left.astype(
            "string"
        )
        .fillna(
            "<MISSING>"
        )
        .eq(
            right.astype(
                "string"
            )
            .fillna(
                "<MISSING>"
            )
        )
        .all()
    )


def summarize_stratum(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    rows = []

    for value, group in data.groupby(
        column,
        dropna=False,
        sort=True,
    ):
        hidden = group[
            "observed"
        ].eq(
            0
        )

        rows.append(
            {
                "stratum_type": column,
                "stratum": str(
                    value
                ),
                "n": int(
                    len(
                        group
                    )
                ),
                "fraction": float(
                    len(
                        group
                    )
                    / len(
                        data
                    )
                ),
                "reference_event_rate": float(
                    group[
                        "outcome"
                    ].mean()
                ),
                "observed_fraction": float(
                    group[
                        "observed"
                    ].mean()
                ),
                "hidden_n": int(
                    hidden.sum()
                ),
                "hidden_event_n": int(
                    group.loc[
                        hidden,
                        "outcome",
                    ].sum()
                ),
                "hidden_event_rate": (
                    float(
                        group.loc[
                            hidden,
                            "outcome",
                        ].mean()
                    )
                    if hidden.any()
                    else np.nan
                ),
                "mean_prediction": float(
                    group[
                        "prediction"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_arguments()

    if args.horizon != EXPECTED_HORIZON:
        raise ValueError(
            "Phase 5H primary SUPPORT2 horizon "
            "must remain the frozen 60 days"
        )

    phase1b_dir = Path(
        args.phase1b_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_support2_csv(
        args.input
    )

    horizon_result = (
        build_support2_horizon_view(
            raw,
            args.horizon,
        )
    )

    horizon = (
        horizon_result.frame.loc[
            horizon_result.frame[
                "reference_known"
            ].eq(
                1
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    frozen = pd.read_csv(
        phase1b_dir
        / "test_predictions.csv"
    )

    split_manifest = pd.read_csv(
        phase1b_dir
        / "split_manifest.csv"
    )

    feature_manifest = pd.read_csv(
        phase1b_dir
        / "feature_manifest.csv"
    )

    metadata = json.loads(
        (
            phase1b_dir
            / "experiment_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    scalar_summary = (
        pd.read_csv(
            phase1b_dir
            / "scalar_summary.csv"
        )
        .set_index(
            "metric"
        )[
            "value"
        ]
    )

    if horizon[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Reconstructed SUPPORT2 horizon "
            "view contains duplicate patients"
        )

    if frozen[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Frozen Phase 1B test predictions "
            "contain duplicate patients"
        )

    join_columns = [
        "patient_id",
        "outcome",
        "observed",
        "action",
        "exit_time",
        "post_discharge_event_by_horizon",
        "dnr",
        "dnrday",
        "hday",
        "dzgroup",
        "dzclass",
        "age",
        "sex",
        "ca",
    ]

    missing_join_columns = [
        column
        for column in join_columns
        if column not in horizon.columns
    ]

    if missing_join_columns:
        raise ValueError(
            "Horizon view is missing required "
            f"columns: {missing_join_columns}"
        )

    enriched = frozen.merge(
        horizon[
            join_columns
        ],
        on="patient_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "_frozen",
            "_reconstructed",
        ),
        indicator=True,
    )

    unmatched = int(
        enriched[
            "_merge"
        ].ne(
            "both"
        ).sum()
    )

    if unmatched:
        raise ValueError(
            f"{unmatched} frozen test patients "
            "did not match the reconstructed cohort"
        )

    enriched = enriched.drop(
        columns=[
            "_merge",
        ]
    )

    enriched[
        "dnr_policy_stratum"
    ] = dnr_policy_stratum(
        enriched[
            "dnr"
        ]
    )

    enriched[
        "hospital_timing_stratum"
    ] = hospital_timing_stratum(
        enriched[
            "hday"
        ]
    )

    enriched[
        "dnr_relative_to_study_day"
    ] = (
        pd.to_numeric(
            enriched[
                "dnrday"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            enriched[
                "hday"
            ],
            errors="coerce",
        )
    )

    hidden = enriched[
        "observed_frozen"
    ].eq(
        0
    )

    enriched[
        "dnr_partition_action"
    ] = np.where(
        hidden,
        (
            "alive_hospital_discharge__"
            + enriched[
                "dnr_policy_stratum"
            ].astype(
                str
            )
        ),
        "observed_within_hospital_boundary",
    )

    enriched[
        "disease_partition_action"
    ] = np.where(
        hidden,
        (
            "alive_hospital_discharge__"
            + enriched[
                "dzgroup"
            ]
            .astype(
                "string"
            )
            .fillna(
                "disease_missing"
            )
            .str.lower()
            .str.replace(
                r"[^a-z0-9]+",
                "_",
                regex=True,
            )
            .str.strip(
                "_"
            )
        ),
        "observed_within_hospital_boundary",
    )

    enriched[
        "timing_partition_action"
    ] = np.where(
        hidden,
        (
            "alive_hospital_discharge__"
            + enriched[
                "hospital_timing_stratum"
            ].astype(
                str
            )
        ),
        "observed_within_hospital_boundary",
    )

    included_features = set(
        metadata.get(
            "included_features",
            [],
        )
    )

    if not included_features:
        included_features = set(
            feature_manifest.loc[
                feature_manifest[
                    "included"
                ].astype(
                    bool
                ),
                "feature",
            ].astype(
                str
            )
        )

    outcome_match = bool(
        enriched[
            "outcome_frozen"
        ]
        .astype(
            int
        )
        .eq(
            enriched[
                "outcome_reconstructed"
            ].astype(
                int
            )
        )
        .all()
    )

    observed_match = bool(
        enriched[
            "observed_frozen"
        ]
        .astype(
            int
        )
        .eq(
            enriched[
                "observed_reconstructed"
            ].astype(
                int
            )
        )
        .all()
    )

    action_match = equal_text(
        enriched[
            "action_frozen"
        ],
        enriched[
            "action_reconstructed"
        ],
    )

    exit_time_match = bool(
        np.allclose(
            enriched[
                "exit_time_frozen"
            ].to_numpy(
                dtype=float
            ),
            enriched[
                "exit_time_reconstructed"
            ].to_numpy(
                dtype=float
            ),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        )
    )

    post_discharge_match = bool(
        enriched[
            (
                "post_discharge_event_by_"
                "horizon_frozen"
            )
        ]
        .astype(
            int
        )
        .eq(
            enriched[
                (
                    "post_discharge_event_by_"
                    "horizon_reconstructed"
                )
            ].astype(
                int
            )
        )
        .all()
    )

    hidden = enriched[
        "observed_frozen"
    ].eq(
        0
    )

    hidden_n = int(
        hidden.sum()
    )

    hidden_event_n = int(
        enriched.loc[
            hidden,
            "outcome_frozen",
        ].sum()
    )

    hidden_actions = set(
        enriched.loc[
            hidden,
            "action_frozen",
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
    )

    institution_candidates = {
        "hosp",
        "hospital",
        "site",
        "center",
        "institution",
    }.intersection(
        horizon.columns
    )

    validations = pd.DataFrame(
        [
            {
                "check": (
                    "frozen metadata horizon is 60 days"
                ),
                "value": metadata.get(
                    "horizon_days"
                ),
                "threshold": 60,
                "comparison": "==",
                "passed": (
                    int(
                        metadata.get(
                            "horizon_days"
                        )
                    )
                    == 60
                ),
            },
            {
                "check": (
                    "frozen test predictions remain frozen"
                ),
                "value": metadata.get(
                    "test_predictions_frozen"
                ),
                "threshold": True,
                "comparison": "==",
                "passed": bool(
                    metadata.get(
                        "test_predictions_frozen"
                    )
                ),
            },
            {
                "check": (
                    "test cohort contains 1821 patients"
                ),
                "value": len(
                    enriched
                ),
                "threshold": EXPECTED_TEST_N,
                "comparison": "==",
                "passed": (
                    len(
                        enriched
                    )
                    == EXPECTED_TEST_N
                ),
            },
            {
                "check": (
                    "reconstructed outcome exactly "
                    "matches frozen outcome"
                ),
                "value": int(
                    outcome_match
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": outcome_match,
            },
            {
                "check": (
                    "reconstructed observation indicator "
                    "exactly matches frozen indicator"
                ),
                "value": int(
                    observed_match
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": observed_match,
            },
            {
                "check": (
                    "reconstructed boundary action "
                    "exactly matches frozen action"
                ),
                "value": int(
                    action_match
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": action_match,
            },
            {
                "check": (
                    "reconstructed exit time exactly "
                    "matches frozen exit time"
                ),
                "value": int(
                    exit_time_match
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": exit_time_match,
            },
            {
                "check": (
                    "post-discharge event indicator "
                    "matches frozen indicator"
                ),
                "value": int(
                    post_discharge_match
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": post_discharge_match,
            },
            {
                "check": (
                    "reference event rate reproduced"
                ),
                "value": float(
                    enriched[
                        "outcome_frozen"
                    ].mean()
                ),
                "threshold": (
                    EXPECTED_REFERENCE_EVENT_RATE
                ),
                "comparison": "allclose",
                "passed": bool(
                    np.isclose(
                        enriched[
                            "outcome_frozen"
                        ].mean(),
                        EXPECTED_REFERENCE_EVENT_RATE,
                        atol=1e-12,
                    )
                ),
            },
            {
                "check": (
                    "restricted observed fraction reproduced"
                ),
                "value": float(
                    enriched[
                        "observed_frozen"
                    ].mean()
                ),
                "threshold": (
                    EXPECTED_OBSERVED_FRACTION
                ),
                "comparison": "allclose",
                "passed": bool(
                    np.isclose(
                        enriched[
                            "observed_frozen"
                        ].mean(),
                        EXPECTED_OBSERVED_FRACTION,
                        atol=1e-12,
                    )
                ),
            },
            {
                "check": (
                    "hidden count reproduced"
                ),
                "value": hidden_n,
                "threshold": EXPECTED_HIDDEN_N,
                "comparison": "==",
                "passed": (
                    hidden_n
                    == EXPECTED_HIDDEN_N
                ),
            },
            {
                "check": (
                    "hidden event count reproduced"
                ),
                "value": hidden_event_n,
                "threshold": (
                    EXPECTED_HIDDEN_EVENT_N
                ),
                "comparison": "==",
                "passed": (
                    hidden_event_n
                    == EXPECTED_HIDDEN_EVENT_N
                ),
            },
            {
                "check": (
                    "original hidden boundary action "
                    "is alive hospital discharge"
                ),
                "value": ",".join(
                    sorted(
                        hidden_actions
                    )
                ),
                "threshold": (
                    EXPECTED_BOUNDARY_ACTION
                ),
                "comparison": "==",
                "passed": (
                    hidden_actions
                    == {
                        EXPECTED_BOUNDARY_ACTION
                    }
                ),
            },
            {
                "check": (
                    "frozen prediction model has "
                    "31 baseline features"
                ),
                "value": len(
                    included_features
                ),
                "threshold": 31,
                "comparison": "==",
                "passed": (
                    len(
                        included_features
                    )
                    == 31
                ),
            },
            {
                "check": (
                    "no prognostic or post-baseline "
                    "variables entered frozen predictor"
                ),
                "value": ",".join(
                    sorted(
                        included_features.intersection(
                            PROGNOSTIC_OR_POST_BASELINE_FIELDS
                        )
                    )
                )
                or "none",
                "threshold": "none",
                "comparison": "==",
                "passed": not bool(
                    included_features.intersection(
                        PROGNOSTIC_OR_POST_BASELINE_FIELDS
                    )
                ),
            },
            {
                "check": (
                    "all three substantive DNR strata "
                    "occur in test cohort"
                ),
                "value": len(
                    set(
                        enriched[
                            "dnr_policy_stratum"
                        ].unique()
                    ).intersection(
                        {
                            "no_dnr",
                            (
                                "dnr_before_"
                                "study_admission"
                            ),
                            (
                                "dnr_after_"
                                "study_admission"
                            ),
                        }
                    )
                ),
                "threshold": 3,
                "comparison": "==",
                "passed": {
                    "no_dnr",
                    (
                        "dnr_before_"
                        "study_admission"
                    ),
                    (
                        "dnr_after_"
                        "study_admission"
                    ),
                }.issubset(
                    set(
                        enriched[
                            "dnr_policy_stratum"
                        ].unique()
                    )
                ),
            },
            {
                "check": (
                    "SUPPORT2 has no usable "
                    "institution identifier"
                ),
                "value": ",".join(
                    sorted(
                        institution_candidates
                    )
                )
                or "none",
                "threshold": "none",
                "comparison": "==",
                "passed": not bool(
                    institution_candidates
                ),
            },
        ]
    )

    status = (
        "phase5h_support2_definitions_frozen"
        if validations[
            "passed"
        ].all()
        else (
            "phase5h_support2_definitions_"
            "require_review"
        )
    )

    final_columns = [
        "patient_id",
        "prediction",
        "outcome_frozen",
        "observed_frozen",
        "action_frozen",
        "exit_time_frozen",
        (
            "post_discharge_event_by_"
            "horizon_frozen"
        ),
        "dnr",
        "dnrday",
        "hday",
        "dnr_relative_to_study_day",
        "dnr_policy_stratum",
        "hospital_timing_stratum",
        "dzgroup",
        "dzclass",
        "age",
        "sex",
        "ca",
        "dnr_partition_action",
        "disease_partition_action",
        "timing_partition_action",
    ]

    final = enriched[
        final_columns
    ].rename(
        columns={
            "outcome_frozen": "outcome",
            "observed_frozen": "observed",
            "action_frozen": (
                "boundary_action"
            ),
            "exit_time_frozen": (
                "exit_time"
            ),
            (
                "post_discharge_event_by_"
                "horizon_frozen"
            ): (
                "post_discharge_event_"
                "by_horizon"
            ),
        }
    )

    stratum_tables = [
        summarize_stratum(
            final,
            "dnr_policy_stratum",
        ),
        summarize_stratum(
            final,
            "hospital_timing_stratum",
        ),
        summarize_stratum(
            final,
            "dzgroup",
        ),
        summarize_stratum(
            final,
            "dzclass",
        ),
    ]

    stratum_summary = pd.concat(
        stratum_tables,
        ignore_index=True,
    )

    manifest = {
        "phase": "5H-B",
        "status": status,
        "dataset": "SUPPORT2",
        "primary_horizon_days": 60,
        "boundary": "hospital",
        "reference_outcome": (
            "60-day all-cause mortality as "
            "constructed by "
            "build_support2_horizon_view"
        ),
        "restricted_observation": (
            "Outcome observability under the "
            "hospital boundary as constructed by "
            "build_support2_horizon_view"
        ),
        "original_boundary_action": (
            EXPECTED_BOUNDARY_ACTION
        ),
        "secondary_descriptive_partitions": [
            "DNR policy status",
            "SUPPORT2 disease group",
            "hospital day at study admission",
        ],
        "dnr_interpretation": (
            "Realized care-policy stratum used only "
            "for descriptive partitioning; not a "
            "causal treatment and not a predictor."
        ),
        "institution_heterogeneity_available": (
            False
        ),
        "institution_note": (
            "No usable institution identifier is "
            "present in the available SUPPORT2 file."
        ),
        "primary_disease_stratum": "dzgroup",
        "secondary_disease_stratum": "dzclass",
        "timing_cutpoints": [
            1,
            3,
            7,
        ],
        "frozen_predictor_feature_count": len(
            included_features
        ),
        "frozen_predictor_features": sorted(
            included_features
        ),
        "phase1b_scalar_reproduction": {
            "reference_event_rate": float(
                enriched[
                    "outcome_frozen"
                ].mean()
            ),
            "observed_fraction": float(
                enriched[
                    "observed_frozen"
                ].mean()
            ),
            "hidden_n": hidden_n,
            "hidden_event_n": hidden_event_n,
            "standardized_signed_gap": float(
                scalar_summary.loc[
                    "standardized_signed_gap"
                ]
            ),
        },
        "sha256": {
            item: sha256_file(
                Path(
                    item
                )
            )
            for item in HASH_INPUTS
        },
    }

    final.to_csv(
        output_dir
        / "support2_harmonized_test.csv.gz",
        index=False,
        compression="gzip",
    )

    stratum_summary.to_csv(
        output_dir
        / "support2_stratum_summary.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    (
        output_dir
        / "support2_definition_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5h_support2_definition_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5H-B SUPPORT2 definition freeze"
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
        "\nStratum summary"
    )

    print(
        stratum_summary.to_string(
            index=False
        )
    )

    print(
        "\nInstitution heterogeneity: "
        "not available in SUPPORT2"
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
