"""Freeze cross-dataset evidence for revised H5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SUPPORT2_STATUS = (
    "phase5i_support2_partition_"
    "sensitivity_benchmarked"
)

EXPECTED_MIMIC_STATUS = (
    "phase5d_exact_tipping_and_"
    "sampling_uncertainty_validated"
)


SUPPORT2_PARTITION_VIEWS = [
    "dnr_policy_partition",
    "disease_partition",
    "hospital_timing_partition",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze cross-dataset localized "
            "sensitivity evidence for H5."
        )
    )

    parser.add_argument(
        "--support2-dir",
        default=(
            "artifacts/"
            "phase5i_support2_partition_sensitivity"
        ),
    )

    parser.add_argument(
        "--mimic-dir",
        default=(
            "artifacts/"
            "phase5d_bootstrap_tipping_points"
        ),
    )

    parser.add_argument(
        "--phase5h-dir",
        default=(
            "artifacts/"
            "phase5h_cross_dataset_"
            "frozen_for_manuscript"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5i_cross_dataset_"
            "h5_frozen_for_manuscript"
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


def interval_summary(
    values: pd.Series,
) -> dict[str, float | int]:
    finite = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if finite.empty:
        return {
            "q025": np.nan,
            "median": np.nan,
            "q975": np.nan,
            "n_valid": 0,
        }

    return {
        "q025": float(
            finite.quantile(
                0.025
            )
        ),
        "median": float(
            finite.quantile(
                0.50
            )
        ),
        "q975": float(
            finite.quantile(
                0.975
            )
        ),
        "n_valid": int(
            len(
                finite
            )
        ),
    }


def find_bootstrap_file(
    directory: Path,
    candidates: list[str],
) -> Path:
    for filename in candidates:
        path = directory / filename

        if path.exists():
            return path

    raise FileNotFoundError(
        "None of the expected bootstrap files "
        f"exist in {directory}: {candidates}"
    )


def summarize_support2(
    directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    point_actions = pd.read_csv(
        directory
        / "exact_action_tipping_points.csv"
    )

    point_global = pd.read_csv(
        directory
        / "exact_global_tipping_points.csv"
    )

    bootstrap_global = pd.read_csv(
        directory
        / "bootstrap_global_replicates.csv.gz"
    )

    require_columns(
        point_actions,
        [
            "view",
            "action",
            "distance_oracle_to_tipping",
            "exact_integrated_sab_tipping_delta",
            "eligible_for_inference",
        ],
        label="SUPPORT2 action table",
    )

    require_columns(
        point_global,
        [
            "view",
            "common_oracle_matching_delta",
            "common_signed_distortion_tipping_delta",
            "minimum_oracle_to_tipping_distance",
            "action_tipping_delta_range",
            "closest_action",
        ],
        label="SUPPORT2 global table",
    )

    require_columns(
        bootstrap_global,
        [
            "bootstrap_id",
            "view",
            "common_oracle_matching_delta",
            "common_signed_distortion_tipping_delta",
            "minimum_oracle_to_tipping_distance",
            "action_tipping_delta_range",
        ],
        label="SUPPORT2 bootstrap global table",
    )

    rows = []

    for view in [
        "original_boundary_action",
        *SUPPORT2_PARTITION_VIEWS,
    ]:
        point = require_single_row(
            point_global,
            point_global[
                "view"
            ].eq(
                view
            ),
            label=(
                f"SUPPORT2 {view}"
            ),
        )

        boot = bootstrap_global.loc[
            bootstrap_global[
                "view"
            ].eq(
                view
            )
        ].copy()

        boot[
            "common_global_margin"
        ] = (
            boot[
                "common_signed_distortion_tipping_delta"
            ]
            - boot[
                "common_oracle_matching_delta"
            ]
        )

        boot[
            "localized_margin_advantage"
        ] = (
            boot[
                "common_global_margin"
            ]
            - boot[
                "minimum_oracle_to_tipping_distance"
            ]
        )

        common_margin = float(
            point[
                "common_signed_distortion_tipping_delta"
            ]
            - point[
                "common_oracle_matching_delta"
            ]
        )

        minimum_local_margin = float(
            point[
                "minimum_oracle_to_tipping_distance"
            ]
        )

        localized_advantage = float(
            common_margin
            - minimum_local_margin
        )

        relative_reduction = (
            float(
                localized_advantage
                / common_margin
            )
            if common_margin > 0
            else np.nan
        )

        advantage_interval = interval_summary(
            boot[
                "localized_margin_advantage"
            ]
        )

        range_interval = interval_summary(
            boot[
                "action_tipping_delta_range"
            ]
        )

        rows.append(
            {
                "dataset": "SUPPORT2",
                "view": view,
                "evidence_type": (
                    "native single boundary action"
                    if view
                    == "original_boundary_action"
                    else (
                        "descriptive discharge-linked "
                        "partition"
                    )
                ),
                "common_global_margin": (
                    common_margin
                ),
                "minimum_local_margin": (
                    minimum_local_margin
                ),
                "closest_local_action": str(
                    point[
                        "closest_action"
                    ]
                ),
                "localized_margin_advantage": (
                    localized_advantage
                ),
                "relative_margin_reduction": (
                    relative_reduction
                ),
                "advantage_q025": (
                    advantage_interval[
                        "q025"
                    ]
                ),
                "advantage_median": (
                    advantage_interval[
                        "median"
                    ]
                ),
                "advantage_q975": (
                    advantage_interval[
                        "q975"
                    ]
                ),
                "advantage_n_valid": (
                    advantage_interval[
                        "n_valid"
                    ]
                ),
                "action_tipping_delta_range": float(
                    point[
                        "action_tipping_delta_range"
                    ]
                ),
                "range_q025": (
                    range_interval[
                        "q025"
                    ]
                ),
                "range_median": (
                    range_interval[
                        "median"
                    ]
                ),
                "range_q975": (
                    range_interval[
                        "q975"
                    ]
                ),
                "range_n_valid": (
                    range_interval[
                        "n_valid"
                    ]
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return (
        summary,
        point_actions,
    )


def summarize_mimic(
    directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    point_actions = pd.read_csv(
        directory
        / "exact_action_tipping_points.csv"
    )

    point_global = pd.read_csv(
        directory
        / "exact_global_tipping_points.csv"
    )

    action_bootstrap_path = (
        find_bootstrap_file(
            directory,
            [
                "bootstrap_action_replicates.csv.gz",
                "bootstrap_action_replicates.csv",
            ],
        )
    )

    global_bootstrap_path = (
        find_bootstrap_file(
            directory,
            [
                "bootstrap_global_replicates.csv.gz",
                "bootstrap_global_replicates.csv",
            ],
        )
    )

    bootstrap_actions = pd.read_csv(
        action_bootstrap_path
    )

    bootstrap_global = pd.read_csv(
        global_bootstrap_path
    )

    require_columns(
        point_actions,
        [
            "action",
            "distance_oracle_to_tipping",
            "exact_integrated_sab_tipping_delta",
        ],
        label="MIMIC action table",
    )

    require_columns(
        point_global,
        [
            "common_oracle_matching_delta",
            "common_signed_distortion_tipping_delta",
        ],
        label="MIMIC global table",
    )

    require_columns(
        bootstrap_actions,
        [
            "bootstrap_id",
            "action",
            "distance_oracle_to_tipping",
            "exact_integrated_sab_tipping_delta",
        ],
        label="MIMIC bootstrap actions",
    )

    require_columns(
        bootstrap_global,
        [
            "bootstrap_id",
            "common_oracle_matching_delta",
            "common_signed_distortion_tipping_delta",
        ],
        label="MIMIC bootstrap global",
    )

    global_point = point_global.iloc[
        0
    ]

    common_margin = float(
        global_point[
            "common_signed_distortion_tipping_delta"
        ]
        - global_point[
            "common_oracle_matching_delta"
        ]
    )

    finite_actions = point_actions.loc[
        point_actions[
            "distance_oracle_to_tipping"
        ].notna()
    ]

    closest_index = finite_actions[
        "distance_oracle_to_tipping"
    ].idxmin()

    closest_action = str(
        finite_actions.loc[
            closest_index,
            "action",
        ]
    )

    minimum_local_margin = float(
        finite_actions.loc[
            closest_index,
            "distance_oracle_to_tipping",
        ]
    )

    localized_advantage = float(
        common_margin
        - minimum_local_margin
    )

    tipping_range = float(
        finite_actions[
            "exact_integrated_sab_tipping_delta"
        ].max()
        - finite_actions[
            "exact_integrated_sab_tipping_delta"
        ].min()
    )

    minimum_by_bootstrap = (
        bootstrap_actions.groupby(
            "bootstrap_id"
        )[
            "distance_oracle_to_tipping"
        ]
        .min()
        .rename(
            "minimum_local_margin"
        )
        .reset_index()
    )

    range_by_bootstrap = (
        bootstrap_actions.groupby(
            "bootstrap_id"
        )[
            "exact_integrated_sab_tipping_delta"
        ]
        .agg(
            lambda values: (
                values.max()
                - values.min()
            )
        )
        .rename(
            "action_tipping_delta_range"
        )
        .reset_index()
    )

    global_boot = bootstrap_global.copy()

    global_boot[
        "common_global_margin"
    ] = (
        global_boot[
            "common_signed_distortion_tipping_delta"
        ]
        - global_boot[
            "common_oracle_matching_delta"
        ]
    )

    combined = (
        global_boot[
            [
                "bootstrap_id",
                "common_global_margin",
            ]
        ]
        .merge(
            minimum_by_bootstrap,
            on="bootstrap_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            range_by_bootstrap,
            on="bootstrap_id",
            how="inner",
            validate="one_to_one",
        )
    )

    combined[
        "localized_margin_advantage"
    ] = (
        combined[
            "common_global_margin"
        ]
        - combined[
            "minimum_local_margin"
        ]
    )

    advantage_interval = interval_summary(
        combined[
            "localized_margin_advantage"
        ]
    )

    range_interval = interval_summary(
        combined[
            "action_tipping_delta_range"
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "dataset": "MIMIC-IV",
                "view": (
                    "native_discharge_actions"
                ),
                "evidence_type": (
                    "native multi-action evidence"
                ),
                "common_global_margin": (
                    common_margin
                ),
                "minimum_local_margin": (
                    minimum_local_margin
                ),
                "closest_local_action": (
                    closest_action
                ),
                "localized_margin_advantage": (
                    localized_advantage
                ),
                "relative_margin_reduction": float(
                    localized_advantage
                    / common_margin
                ),
                "advantage_q025": (
                    advantage_interval[
                        "q025"
                    ]
                ),
                "advantage_median": (
                    advantage_interval[
                        "median"
                    ]
                ),
                "advantage_q975": (
                    advantage_interval[
                        "q975"
                    ]
                ),
                "advantage_n_valid": (
                    advantage_interval[
                        "n_valid"
                    ]
                ),
                "action_tipping_delta_range": (
                    tipping_range
                ),
                "range_q025": (
                    range_interval[
                        "q025"
                    ]
                ),
                "range_median": (
                    range_interval[
                        "median"
                    ]
                ),
                "range_q975": (
                    range_interval[
                        "q975"
                    ]
                ),
                "range_n_valid": (
                    range_interval[
                        "n_valid"
                    ]
                ),
            }
        ]
    )

    return (
        summary,
        point_actions,
    )


def build_validations(
    *,
    support2_status: str,
    mimic_status: str,
    phase5h_status: str,
    support2_summary: pd.DataFrame,
    mimic_summary: pd.DataFrame,
) -> pd.DataFrame:
    original = require_single_row(
        support2_summary,
        support2_summary[
            "view"
        ].eq(
            "original_boundary_action"
        ),
        label="SUPPORT2 structural control",
    )

    partitions = support2_summary.loc[
        support2_summary[
            "view"
        ].isin(
            SUPPORT2_PARTITION_VIEWS
        )
    ]

    mimic = mimic_summary.iloc[
        0
    ]

    support2_positive_advantages = int(
        (
            partitions[
                "advantage_q025"
            ]
            > 0
        ).sum()
    )

    support2_positive_ranges = int(
        (
            partitions[
                "range_q025"
            ]
            > 0
        ).sum()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "SUPPORT2 Phase 5I benchmark passed"
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
                    "MIMIC Phase 5D sensitivity "
                    "benchmark passed"
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
                    "Phase 5H H1 H2 H4 package "
                    "already frozen"
                ),
                "value": phase5h_status,
                "threshold": (
                    "phase5h_cross_dataset_h1_h2_h4_"
                    "frozen_for_manuscript"
                ),
                "comparison": "==",
                "passed": (
                    phase5h_status
                    == (
                        "phase5h_cross_dataset_h1_h2_h4_"
                        "frozen_for_manuscript"
                    )
                ),
            },
            {
                "check": (
                    "SUPPORT2 single-action control "
                    "has zero localized advantage"
                ),
                "value": abs(
                    float(
                        original[
                            "localized_margin_advantage"
                        ]
                    )
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": bool(
                    abs(
                        float(
                            original[
                                "localized_margin_advantage"
                            ]
                        )
                    )
                    < 1e-10
                ),
            },
            {
                "check": (
                    "at least one SUPPORT2 partition "
                    "has positive localized-advantage "
                    "bootstrap lower bound"
                ),
                "value": (
                    support2_positive_advantages
                ),
                "threshold": 1,
                "comparison": ">=",
                "passed": (
                    support2_positive_advantages
                    >= 1
                ),
            },
            {
                "check": (
                    "at least one SUPPORT2 partition "
                    "has positive tipping-range "
                    "bootstrap lower bound"
                ),
                "value": (
                    support2_positive_ranges
                ),
                "threshold": 1,
                "comparison": ">=",
                "passed": (
                    support2_positive_ranges
                    >= 1
                ),
            },
            {
                "check": (
                    "MIMIC native actions have "
                    "positive localized-advantage "
                    "bootstrap lower bound"
                ),
                "value": float(
                    mimic[
                        "advantage_q025"
                    ]
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    float(
                        mimic[
                            "advantage_q025"
                        ]
                    )
                    > 0
                ),
            },
            {
                "check": (
                    "MIMIC native actions have "
                    "positive tipping-range bootstrap "
                    "lower bound"
                ),
                "value": float(
                    mimic[
                        "range_q025"
                    ]
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    float(
                        mimic[
                            "range_q025"
                        ]
                    )
                    > 0
                ),
            },
            {
                "check": (
                    "localized analysis reduces "
                    "minimum robustness margin by "
                    "at least 20 percent in both "
                    "clinical datasets"
                ),
                "value": (
                    f"SUPPORT2="
                    f"{float(partitions['relative_margin_reduction'].max()):.6f};"
                    f"MIMIC="
                    f"{float(mimic['relative_margin_reduction']):.6f}"
                ),
                "threshold": (
                    "both >= 0.20"
                ),
                "comparison": ">=",
                "passed": bool(
                    float(
                        partitions[
                            "relative_margin_reduction"
                        ].max()
                    )
                    >= 0.20
                    and float(
                        mimic[
                            "relative_margin_reduction"
                        ]
                    )
                    >= 0.20
                ),
            },
            {
                "check": (
                    "SUPPORT2 evidence remains "
                    "labeled descriptive"
                ),
                "value": int(
                    partitions[
                        "evidence_type"
                    ]
                    .str.contains(
                        "descriptive",
                        case=False,
                        na=False,
                    )
                    .all()
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": bool(
                    partitions[
                        "evidence_type"
                    ]
                    .str.contains(
                        "descriptive",
                        case=False,
                        na=False,
                    )
                    .all()
                ),
            },
        ]
    )


def build_claim_matrix(
    support2_summary: pd.DataFrame,
    mimic_summary: pd.DataFrame,
) -> pd.DataFrame:
    partitions = support2_summary.loc[
        support2_summary[
            "view"
        ].isin(
            SUPPORT2_PARTITION_VIEWS
        )
    ]

    strongest_support2 = partitions.loc[
        partitions[
            "localized_margin_advantage"
        ].idxmax()
    ]

    mimic = mimic_summary.iloc[
        0
    ]

    return pd.DataFrame(
        [
            {
                "claim": "H5",
                "claim_text": (
                    "Localized action- or stratum-"
                    "specific sensitivity analysis "
                    "reveals heterogeneous robustness "
                    "that can be obscured by a common "
                    "global perturbation."
                ),
                "support2_evidence": (
                    "Descriptive discharge-linked "
                    f"partitions reduced the closest "
                    f"robustness margin by up to "
                    f"{100.0 * float(strongest_support2['relative_margin_reduction']):.1f}%; "
                    f"the largest tipping-threshold "
                    f"range was "
                    f"{float(partitions['action_tipping_delta_range'].max()):.3f}."
                ),
                "mimic_evidence": (
                    "Native discharge actions reduced "
                    f"the closest robustness margin by "
                    f"{100.0 * float(mimic['relative_margin_reduction']):.1f}%; "
                    f"{mimic['closest_local_action']} "
                    f"was closest to tipping, and the "
                    f"action tipping range was "
                    f"{float(mimic['action_tipping_delta_range']):.3f}."
                ),
                "final_status": (
                    "strong bounded cross-dataset support"
                ),
                "manuscript_role": (
                    "primary methodological claim"
                ),
                "boundary": (
                    "MIMIC provides native multi-action "
                    "evidence. SUPPORT2 provides "
                    "descriptive partition-level "
                    "corroboration because its native "
                    "boundary contains one action."
                ),
            }
        ]
    )


def main() -> None:
    args = parse_arguments()

    support2_dir = Path(
        args.support2_dir
    )

    mimic_dir = Path(
        args.mimic_dir
    )

    phase5h_dir = Path(
        args.phase5h_dir
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
            "phase5i_support2_partition_"
            "sensitivity_status.txt"
        )
    )

    mimic_status = read_status(
        mimic_dir
        / "phase5d_tipping_status.txt"
    )

    phase5h_status = read_status(
        phase5h_dir
        / (
            "phase5h_cross_dataset_"
            "freeze_status.txt"
        )
    )

    (
        support2_summary,
        support2_actions,
    ) = summarize_support2(
        support2_dir
    )

    (
        mimic_summary,
        mimic_actions,
    ) = summarize_mimic(
        mimic_dir
    )

    validations = build_validations(
        support2_status=support2_status,
        mimic_status=mimic_status,
        phase5h_status=phase5h_status,
        support2_summary=(
            support2_summary
        ),
        mimic_summary=mimic_summary,
    )

    claim_matrix = build_claim_matrix(
        support2_summary,
        mimic_summary,
    )

    status = (
        "phase5i_cross_dataset_h5_"
        "frozen_for_manuscript"
        if validations[
            "passed"
        ].all()
        else (
            "phase5i_cross_dataset_h5_"
            "freeze_requires_review"
        )
    )

    support2_summary.to_csv(
        output_dir
        / "support2_localized_sensitivity_summary.csv",
        index=False,
    )

    mimic_summary.to_csv(
        output_dir
        / "mimic_localized_sensitivity_summary.csv",
        index=False,
    )

    support2_actions.to_csv(
        output_dir
        / "support2_action_sensitivity_evidence.csv",
        index=False,
    )

    mimic_actions.to_csv(
        output_dir
        / "mimic_action_sensitivity_evidence.csv",
        index=False,
    )

    claim_matrix.to_csv(
        output_dir
        / "h5_cross_dataset_claim.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "freeze_validation_checks.csv",
        index=False,
    )

    manifest = {
        "phase": "5I-C",
        "status": status,
        "claim_frozen": "H5",
        "claim_text": (
            claim_matrix[
                "claim_text"
            ].iloc[
                0
            ]
        ),
        "final_status": (
            claim_matrix[
                "final_status"
            ].iloc[
                0
            ]
        ),
        "support2_role": (
            "Descriptive partition-level "
            "corroboration."
        ),
        "mimic_role": (
            "Direct native multi-action evidence."
        ),
        "interpretation_boundaries": [
            (
                "Sensitivity deltas are odds-scale "
                "perturbations and should not be "
                "interpreted as causal effects."
            ),
            (
                "SUPPORT2 has only one native "
                "hospital-boundary action."
            ),
            (
                "DNR, disease, and timing categories "
                "are descriptive partitions of "
                "SUPPORT2 discharge."
            ),
            (
                "The claim concerns information gained "
                "from localized sensitivity analysis, "
                "not universal superiority of every "
                "localized estimate."
            ),
        ],
    }

    (
        output_dir
        / "phase5i_cross_dataset_h5_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5i_cross_dataset_h5_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5I-C cross-dataset H5 freeze"
    )

    print(
        "\nSUPPORT2 localized sensitivity"
    )

    print(
        support2_summary.to_string(
            index=False
        )
    )

    print(
        "\nMIMIC localized sensitivity"
    )

    print(
        mimic_summary.to_string(
            index=False
        )
    )

    print(
        "\nH5 claim"
    )

    print(
        claim_matrix.to_string(
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
