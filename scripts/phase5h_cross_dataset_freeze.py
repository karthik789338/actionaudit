"""Freeze cross-dataset Phase 5H evidence for H1, H2, and H4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SUPPORT2_STATUS = (
    "phase5h_support2_action_decomposition_benchmarked"
)

EXPECTED_MIMIC_STATUS = (
    "phase5h_mimic_heterogeneity_benchmarked"
)

EXPECTED_PHASE5G_STATUS = (
    "phase5g_random_forest_benchmark_"
    "frozen_for_manuscript"
)


SUPPORT2_H2_VIEWS = {
    "dnr_policy_partition": (
        "DNR policy partition"
    ),
    "disease_partition": (
        "disease-group partition"
    ),
    "hospital_timing_partition": (
        "hospital-timing partition"
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the cross-dataset evidence "
            "matrix for H1, H2, and H4."
        )
    )

    parser.add_argument(
        "--support2-dir",
        default=(
            "artifacts/"
            "phase5h_support2_action_decomposition"
        ),
    )

    parser.add_argument(
        "--mimic-dir",
        default=(
            "artifacts/"
            "phase5h_mimic_heterogeneity_benchmark"
        ),
    )

    parser.add_argument(
        "--support2-h4-dir",
        default=(
            "artifacts/"
            "phase2b_right_censoring_10bins_clip005"
        ),
    )

    parser.add_argument(
        "--artifact-root",
        default="artifacts",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5h_cross_dataset_frozen_for_manuscript"
        ),
    )

    return parser.parse_args()


def read_status(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing status file: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


def find_exact_status(
    root: Path,
    expected: str,
) -> list[str]:
    matches = []

    for path in root.rglob(
        "*.txt"
    ):
        try:
            value = path.read_text(
                encoding="utf-8"
            ).strip()
        except Exception:
            continue

        if value == expected:
            matches.append(
                str(
                    path
                )
            )

    return sorted(
        matches
    )


def require_single_row(
    data: pd.DataFrame,
    mask: pd.Series,
    *,
    label: str,
) -> pd.Series:
    selected = data.loc[
        mask
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            f"{label}: expected one row, "
            f"found {len(selected)}"
        )

    return selected.iloc[
        0
    ]


def bool_value(
    value: object,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def main() -> None:
    args = parse_arguments()

    support2_dir = Path(
        args.support2_dir
    )

    mimic_dir = Path(
        args.mimic_dir
    )

    support2_h4_dir = Path(
        args.support2_h4_dir
    )

    artifact_root = Path(
        args.artifact_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    support2_status = read_status(
        support2_dir
        / (
            "phase5h_support2_action_"
            "decomposition_status.txt"
        )
    )

    mimic_status = read_status(
        mimic_dir
        / (
            "phase5h_mimic_"
            "heterogeneity_status.txt"
        )
    )

    phase5g_status_paths = find_exact_status(
        artifact_root,
        EXPECTED_PHASE5G_STATUS,
    )

    support2_views = pd.read_csv(
        support2_dir
        / "support2_view_summary.csv"
    )

    support2_view_intervals = pd.read_csv(
        support2_dir
        / "bootstrap_view_metric_intervals.csv"
    )

    support2_actions = pd.read_csv(
        support2_dir
        / "support2_action_summary.csv"
    )

    mimic_points = pd.read_csv(
        mimic_dir
        / "mimic_stratum_point_estimates.csv"
    )

    mimic_contrasts = pd.read_csv(
        mimic_dir
        / "prespecified_contrasts.csv"
    )

    mimic_extremes = pd.read_csv(
        mimic_dir
        / "descriptive_extreme_contrasts.csv"
    )

    support2_h4_ranking = pd.read_csv(
        support2_h4_dir
        / "estimator_ranking.csv"
    )

    support2_h4_intervals = pd.read_csv(
        support2_h4_dir
        / "bootstrap_intervals.csv"
    )

    support2_original = require_single_row(
        support2_views,
        support2_views[
            "view"
        ].eq(
            "original_boundary_action"
        ),
        label="SUPPORT2 original view",
    )

    mimic_overall = require_single_row(
        mimic_points,
        mimic_points[
            "dimension"
        ].eq(
            "overall"
        ),
        label="MIMIC overall result",
    )

    support2_h2_rows = []

    for view, label in (
        SUPPORT2_H2_VIEWS.items()
    ):
        point = require_single_row(
            support2_views,
            support2_views[
                "view"
            ].eq(
                view
            ),
            label=(
                f"SUPPORT2 {view}"
            ),
        )

        interval = require_single_row(
            support2_view_intervals,
            support2_view_intervals[
                "view"
            ].eq(
                view
            )
            & support2_view_intervals[
                "metric"
            ].eq(
                "action_sab_range_eligible"
            ),
            label=(
                f"SUPPORT2 {view} interval"
            ),
        )

        support2_h2_rows.append(
            {
                "dataset": "SUPPORT2",
                "dimension": label,
                "point_range": float(
                    point[
                        "action_sab_range_eligible"
                    ]
                ),
                "q025": float(
                    interval[
                        "q025"
                    ]
                ),
                "median": float(
                    interval[
                        "median"
                    ]
                ),
                "q975": float(
                    interval[
                        "q975"
                    ]
                ),
                "n_valid": int(
                    interval[
                        "n_valid"
                    ]
                ),
                "evidence_type": (
                    "descriptive discharge-action "
                    "partition"
                ),
            }
        )

    support2_h2 = pd.DataFrame(
        support2_h2_rows
    )

    mimic_primary_contrasts = (
        mimic_contrasts.loc[
            mimic_contrasts[
                "metric"
            ].eq(
                "standardized_signed_gap"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    mimic_primary_extremes = (
        mimic_extremes.loc[
            mimic_extremes[
                "dimension"
            ].isin(
                [
                    "clinical_chapter",
                    "first_hospital_service",
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    survival_or = require_single_row(
        support2_h4_ranking,
        support2_h4_ranking[
            "estimator"
        ].eq(
            "survival_or"
        ),
        label="SUPPORT2 survival OR",
    )

    complete_case = require_single_row(
        support2_h4_ranking,
        support2_h4_ranking[
            "estimator"
        ].eq(
            "complete_case"
        ),
        label="SUPPORT2 complete case",
    )

    survival_or_interval = require_single_row(
        support2_h4_intervals,
        support2_h4_intervals[
            "metric"
        ].eq(
            "survival_or_mae_to_reference"
        ),
        label="SUPPORT2 survival OR interval",
    )

    significant_mimic_contrasts = (
        mimic_primary_contrasts[
            "interval_excludes_zero"
        ].map(
            bool_value
        )
    )

    weekend_row = require_single_row(
        mimic_primary_contrasts,
        mimic_primary_contrasts[
            "contrast"
        ].eq(
            "weekend_minus_weekday"
        ),
        label="MIMIC weekend contrast",
    )

    support2_positive_ranges = int(
        (
            support2_h2[
                "q025"
            ]
            > 0
        ).sum()
    )

    clinical_extreme = require_single_row(
        mimic_primary_extremes,
        mimic_primary_extremes[
            "dimension"
        ].eq(
            "clinical_chapter"
        ),
        label="MIMIC clinical chapter extreme",
    )

    service_extreme = require_single_row(
        mimic_primary_extremes,
        mimic_primary_extremes[
            "dimension"
        ].eq(
            "first_hospital_service"
        ),
        label="MIMIC service extreme",
    )

    claim_matrix = pd.DataFrame(
        [
            {
                "claim": "H1",
                "claim_text": (
                    "Restricting follow-up at the "
                    "hospital boundary materially "
                    "changes measured calibration."
                ),
                "support2_evidence": (
                    "Direct: standardized signed "
                    f"distortion "
                    f"{float(support2_original['signed_distortion']):.6f}; "
                    f"reference ICI "
                    f"{float(support2_original['reference_ici']):.6f}; "
                    f"restricted ICI "
                    f"{float(support2_original['restricted_ici']):.6f}."
                ),
                "mimic_evidence": (
                    "Direct: standardized signed "
                    f"distortion "
                    f"{float(mimic_overall['standardized_signed_gap']):.6f}; "
                    f"reference ICI "
                    f"{float(mimic_overall['reference_ici']):.6f}; "
                    f"restricted ICI "
                    f"{float(mimic_overall['restricted_ici']):.6f}."
                ),
                "final_status": (
                    "strong cross-dataset support"
                ),
                "manuscript_role": (
                    "primary confirmatory claim"
                ),
            },
            {
                "claim": "H2",
                "claim_text": (
                    "Observation-induced distortion "
                    "is heterogeneous across risk and "
                    "clinically meaningful strata."
                ),
                "support2_evidence": (
                    "Direct descriptive heterogeneity "
                    "across DNR policy, disease group, "
                    "and hospital timing partitions."
                ),
                "mimic_evidence": (
                    "Direct heterogeneity across "
                    "clinical chapter, service, ICU "
                    "involvement, era, and admission "
                    "hour; weekend versus weekday "
                    "was not supported."
                ),
                "final_status": (
                    "strong bounded cross-dataset support"
                ),
                "manuscript_role": (
                    "primary confirmatory claim with "
                    "dimension-specific nulls retained"
                ),
            },
            {
                "claim": "H4",
                "claim_text": (
                    "Adjustment performance depends "
                    "on the identification regime, "
                    "nuisance specification, and "
                    "positivity."
                ),
                "support2_evidence": (
                    "Survival outcome regression "
                    f"reduced MAE from "
                    f"{float(complete_case['mae_to_reference']):.6f} "
                    f"to "
                    f"{float(survival_or['mae_to_reference']):.6f}, "
                    "but did not eliminate residual "
                    "error."
                ),
                "mimic_evidence": (
                    "Frozen Phase 5G benchmark: "
                    "score-only, logistic, HGB, and RF "
                    "nuisance families produced only "
                    "partial recovery; weighting was "
                    "limited by severe positivity."
                ),
                "final_status": (
                    "strong bounded cross-dataset support"
                ),
                "manuscript_role": (
                    "primary methodological claim"
                ),
            },
            {
                "claim": (
                    "cancellation_secondary"
                ),
                "claim_text": (
                    "Opposing action distortions can "
                    "cancel in principle."
                ),
                "support2_evidence": (
                    "Meaningful empirical cancellation "
                    "was not observed."
                ),
                "mimic_evidence": (
                    "Meaningful empirical cancellation "
                    "was not observed."
                ),
                "final_status": (
                    "synthetic mechanism supported; "
                    "negative empirical result"
                ),
                "manuscript_role": (
                    "secondary diagnostic, not a "
                    "confirmatory hypothesis"
                ),
            },
            {
                "claim": "H5",
                "claim_text": (
                    "Action-specific sensitivity is "
                    "more informative than global "
                    "sensitivity."
                ),
                "support2_evidence": (
                    "Phase 5I pending."
                ),
                "mimic_evidence": (
                    "Supported by frozen Phase 5D; "
                    "cross-dataset replication pending."
                ),
                "final_status": (
                    "not yet cross-dataset frozen"
                ),
                "manuscript_role": (
                    "next experimental phase"
                ),
            },
        ]
    )

    validations = pd.DataFrame(
        [
            {
                "check": (
                    "SUPPORT2 Phase 5H benchmark passed"
                ),
                "value": support2_status,
                "threshold": (
                    EXPECTED_SUPPORT2_STATUS
                ),
                "comparison": "==",
                "passed": (
                    support2_status
                    == EXPECTED_SUPPORT2_STATUS
                ),
            },
            {
                "check": (
                    "MIMIC Phase 5H benchmark passed"
                ),
                "value": mimic_status,
                "threshold": (
                    EXPECTED_MIMIC_STATUS
                ),
                "comparison": "==",
                "passed": (
                    mimic_status
                    == EXPECTED_MIMIC_STATUS
                ),
            },
            {
                "check": (
                    "MIMIC Phase 5G benchmark frozen"
                ),
                "value": len(
                    phase5g_status_paths
                ),
                "threshold": 1,
                "comparison": ">=",
                "passed": bool(
                    phase5g_status_paths
                ),
            },
            {
                "check": (
                    "H1 distortion exceeds 0.30 "
                    "in both datasets"
                ),
                "value": (
                    f"SUPPORT2="
                    f"{float(support2_original['signed_distortion']):.6f};"
                    f"MIMIC="
                    f"{float(mimic_overall['standardized_signed_gap']):.6f}"
                ),
                "threshold": (
                    "both > 0.30"
                ),
                "comparison": ">",
                "passed": bool(
                    float(
                        support2_original[
                            "signed_distortion"
                        ]
                    )
                    > 0.30
                    and float(
                        mimic_overall[
                            "standardized_signed_gap"
                        ]
                    )
                    > 0.30
                ),
            },
            {
                "check": (
                    "at least two SUPPORT2 H2 "
                    "partition-range intervals "
                    "exclude zero"
                ),
                "value": (
                    support2_positive_ranges
                ),
                "threshold": 2,
                "comparison": ">=",
                "passed": (
                    support2_positive_ranges
                    >= 2
                ),
            },
            {
                "check": (
                    "at least three MIMIC "
                    "prespecified H2 contrasts "
                    "exclude zero"
                ),
                "value": int(
                    significant_mimic_contrasts.sum()
                ),
                "threshold": 3,
                "comparison": ">=",
                "passed": bool(
                    significant_mimic_contrasts.sum()
                    >= 3
                ),
            },
            {
                "check": (
                    "weekend versus weekday null "
                    "is retained"
                ),
                "value": bool_value(
                    weekend_row[
                        "interval_excludes_zero"
                    ]
                ),
                "threshold": False,
                "comparison": "==",
                "passed": not bool_value(
                    weekend_row[
                        "interval_excludes_zero"
                    ]
                ),
            },
            {
                "check": (
                    "MIMIC disease and service "
                    "descriptive ranges have "
                    "positive bootstrap lower bounds"
                ),
                "value": (
                    f"chapter="
                    f"{float(clinical_extreme['q025']):.6f};"
                    f"service="
                    f"{float(service_extreme['q025']):.6f}"
                ),
                "threshold": (
                    "both > 0"
                ),
                "comparison": ">",
                "passed": bool(
                    float(
                        clinical_extreme[
                            "q025"
                        ]
                    )
                    > 0
                    and float(
                        service_extreme[
                            "q025"
                        ]
                    )
                    > 0
                ),
            },
            {
                "check": (
                    "SUPPORT2 survival outcome "
                    "regression materially improves "
                    "over complete case"
                ),
                "value": float(
                    complete_case[
                        "mae_to_reference"
                    ]
                    - survival_or[
                        "mae_to_reference"
                    ]
                ),
                "threshold": 0.25,
                "comparison": ">",
                "passed": bool(
                    float(
                        complete_case[
                            "mae_to_reference"
                        ]
                        - survival_or[
                            "mae_to_reference"
                        ]
                    )
                    > 0.25
                    and float(
                        survival_or_interval[
                            "bootstrap_q975"
                        ]
                    )
                    < float(
                        complete_case[
                            "mae_to_reference"
                        ]
                    )
                ),
            },
            {
                "check": (
                    "institution heterogeneity "
                    "is not claimed"
                ),
                "value": "not_available",
                "threshold": "not_available",
                "comparison": "==",
                "passed": True,
            },
            {
                "check": (
                    "H3 cancellation removed from "
                    "primary hypothesis list"
                ),
                "value": (
                    claim_matrix.loc[
                        claim_matrix[
                            "claim"
                        ].eq(
                            "cancellation_secondary"
                        ),
                        "manuscript_role",
                    ].iloc[
                        0
                    ]
                ),
                "threshold": (
                    "secondary diagnostic"
                ),
                "comparison": "contains",
                "passed": bool(
                    claim_matrix.loc[
                        claim_matrix[
                            "claim"
                        ].eq(
                            "cancellation_secondary"
                        ),
                        "manuscript_role",
                    ].iloc[
                        0
                    ].startswith(
                        "secondary diagnostic"
                    )
                ),
            },
        ]
    )

    status = (
        "phase5h_cross_dataset_h1_h2_h4_"
        "frozen_for_manuscript"
        if validations[
            "passed"
        ].all()
        else (
            "phase5h_cross_dataset_freeze_"
            "requires_review"
        )
    )

    claim_matrix.to_csv(
        output_dir
        / "cross_dataset_claim_matrix.csv",
        index=False,
    )

    support2_h2.to_csv(
        output_dir
        / "support2_h2_partition_evidence.csv",
        index=False,
    )

    mimic_primary_contrasts.to_csv(
        output_dir
        / "mimic_h2_prespecified_contrasts.csv",
        index=False,
    )

    mimic_primary_extremes.to_csv(
        output_dir
        / "mimic_h2_descriptive_extremes.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "freeze_validation_checks.csv",
        index=False,
    )

    manifest = {
        "phase": "5H-G",
        "status": status,
        "claims_frozen": [
            "H1",
            "H2",
            "H4",
        ],
        "hypothesis_reorganization": {
            "H3": (
                "Removed from primary confirmatory "
                "hypotheses. Retained as a synthetic "
                "mechanism and secondary empirical "
                "negative result."
            ),
            "H5": (
                "Cross-dataset replication deferred "
                "to Phase 5I."
            ),
        },
        "support2": {
            "horizon_days": 60,
            "signed_distortion": float(
                support2_original[
                    "signed_distortion"
                ]
            ),
            "reference_ici": float(
                support2_original[
                    "reference_ici"
                ]
            ),
            "restricted_ici": float(
                support2_original[
                    "restricted_ici"
                ]
            ),
            "survival_or_mae": float(
                survival_or[
                    "mae_to_reference"
                ]
            ),
            "complete_case_mae": float(
                complete_case[
                    "mae_to_reference"
                ]
            ),
        },
        "mimic": {
            "horizon_days": 30,
            "signed_distortion": float(
                mimic_overall[
                    "standardized_signed_gap"
                ]
            ),
            "reference_ici": float(
                mimic_overall[
                    "reference_ici"
                ]
            ),
            "restricted_ici": float(
                mimic_overall[
                    "restricted_ici"
                ]
            ),
            "significant_prespecified_h2_contrasts": int(
                significant_mimic_contrasts.sum()
            ),
            "weekend_weekday_supported": bool_value(
                weekend_row[
                    "interval_excludes_zero"
                ]
            ),
        },
        "phase5g_status_paths": (
            phase5g_status_paths
        ),
        "interpretation_boundaries": [
            (
                "SUPPORT2 and MIMIC use different "
                "horizons and clinical populations; "
                "effect sizes are not pooled."
            ),
            (
                "SUPPORT2 DNR, disease, and timing "
                "actions are descriptive partitions "
                "of hospital discharge."
            ),
            (
                "MIMIC subgroup contrasts are "
                "descriptive calibration-audit "
                "comparisons, not causal effects."
            ),
            (
                "Post-selected disease and service "
                "extremes are descriptive, while ICU, "
                "era, hour, and weekpart contrasts "
                "were prespecified."
            ),
            (
                "Institution heterogeneity is "
                "unavailable in both datasets."
            ),
        ],
    }

    (
        output_dir
        / "phase5h_cross_dataset_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5h_cross_dataset_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5H-G cross-dataset freeze"
    )

    print(
        "\nClaim matrix"
    )

    print(
        claim_matrix.to_string(
            index=False
        )
    )

    print(
        "\nSUPPORT2 H2 evidence"
    )

    print(
        support2_h2.to_string(
            index=False
        )
    )

    print(
        "\nMIMIC prespecified contrasts"
    )

    print(
        mimic_primary_contrasts.to_string(
            index=False
        )
    )

    print(
        "\nFreeze validation"
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
