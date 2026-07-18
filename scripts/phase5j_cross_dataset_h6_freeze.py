"""Freeze cross-dataset observation-policy evidence for H6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SUPPORT2_STATUS = (
    "phase5j_support2_policy_shift_benchmarked"
)

EXPECTED_MIMIC_STATUS = (
    "phase5j_mimic_policy_shift_benchmarked"
)

EXPECTED_SYNTHETIC_STATUS = (
    "phase5e_frozen_for_manuscript"
)

EXPECTED_H5_STATUS = (
    "phase5i_cross_dataset_h5_"
    "frozen_for_manuscript"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze cross-dataset H6 "
            "observation-policy evidence."
        )
    )

    parser.add_argument(
        "--support2-dir",
        default=(
            "artifacts/"
            "phase5j_support2_policy_shift"
        ),
    )

    parser.add_argument(
        "--mimic-dir",
        default=(
            "artifacts/"
            "phase5j_mimic_policy_shift"
        ),
    )

    parser.add_argument(
        "--synthetic-dir",
        default=(
            "artifacts/"
            "phase5e_frozen_for_manuscript"
        ),
    )

    parser.add_argument(
        "--h5-dir",
        default=(
            "artifacts/"
            "phase5i_cross_dataset_h5_"
            "frozen_for_manuscript"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5j_cross_dataset_h6_"
            "frozen_for_manuscript"
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


def interval_row(
    data: pd.DataFrame,
    *,
    policy: str,
    metric: str,
) -> pd.Series:
    return require_single_row(
        data,
        data[
            "policy"
        ].eq(
            policy
        )
        & data[
            "metric"
        ].eq(
            metric
        ),
        label=(
            f"{policy}, {metric}"
        ),
    )


def summarize_dataset(
    *,
    dataset: str,
    horizon_days: int,
    directory: Path,
) -> dict[str, object]:
    point = pd.read_csv(
        directory
        / "policy_scalar_summary.csv"
    )

    contrasts = pd.read_csv(
        directory
        / "policy_contrast_intervals.csv"
    )

    baseline = require_single_row(
        point,
        point[
            "policy"
        ].eq(
            "baseline"
        ),
        label=(
            f"{dataset} baseline"
        ),
    )

    aggressive = require_single_row(
        point,
        point[
            "policy"
        ].eq(
            "aggressive_early"
        ),
        label=(
            f"{dataset} aggressive"
        ),
    )

    conservative = require_single_row(
        point,
        point[
            "policy"
        ].eq(
            "conservative_late"
        ),
        label=(
            f"{dataset} conservative"
        ),
    )

    aggressive_ici = interval_row(
        contrasts,
        policy="aggressive_early",
        metric=(
            "restricted_ici_difference"
        ),
    )

    aggressive_observation = interval_row(
        contrasts,
        policy="aggressive_early",
        metric=(
            "observed_fraction_difference"
        ),
    )

    conservative_ici = interval_row(
        contrasts,
        policy="conservative_late",
        metric=(
            "restricted_ici_difference"
        ),
    )

    reference_ici_range = float(
        point[
            "reference_ici"
        ].max()
        - point[
            "reference_ici"
        ].min()
    )

    reference_event_rate_range = float(
        point[
            "reference_event_rate"
        ].max()
        - point[
            "reference_event_rate"
        ].min()
    )

    restricted_ici_range = float(
        point[
            "restricted_ici"
        ].max()
        - point[
            "restricted_ici"
        ].min()
    )

    observation_fraction_range = float(
        point[
            "observed_fraction"
        ].max()
        - point[
            "observed_fraction"
        ].min()
    )

    return {
        "dataset": dataset,
        "horizon_days": horizon_days,
        "n": int(
            baseline[
                "n"
            ]
        ),
        "reference_event_rate": float(
            baseline[
                "reference_event_rate"
            ]
        ),
        "reference_ici": float(
            baseline[
                "reference_ici"
            ]
        ),
        "reference_ici_range": (
            reference_ici_range
        ),
        "reference_event_rate_range": (
            reference_event_rate_range
        ),
        "observation_fraction_range": (
            observation_fraction_range
        ),
        "restricted_ici_range": (
            restricted_ici_range
        ),
        "baseline_observed_fraction": float(
            baseline[
                "observed_fraction"
            ]
        ),
        "aggressive_observed_fraction": float(
            aggressive[
                "observed_fraction"
            ]
        ),
        "conservative_observed_fraction": float(
            conservative[
                "observed_fraction"
            ]
        ),
        "baseline_restricted_ici": float(
            baseline[
                "restricted_ici"
            ]
        ),
        "aggressive_restricted_ici": float(
            aggressive[
                "restricted_ici"
            ]
        ),
        "conservative_restricted_ici": float(
            conservative[
                "restricted_ici"
            ]
        ),
        "aggressive_ici_difference": float(
            aggressive[
                "restricted_ici"
            ]
            - baseline[
                "restricted_ici"
            ]
        ),
        "aggressive_ici_q025": float(
            aggressive_ici[
                "q025"
            ]
        ),
        "aggressive_ici_median": float(
            aggressive_ici[
                "median"
            ]
        ),
        "aggressive_ici_q975": float(
            aggressive_ici[
                "q975"
            ]
        ),
        "aggressive_observation_q975": float(
            aggressive_observation[
                "q975"
            ]
        ),
        "conservative_ici_difference": float(
            conservative[
                "restricted_ici"
            ]
            - baseline[
                "restricted_ici"
            ]
        ),
        "conservative_ici_q025": float(
            conservative_ici[
                "q025"
            ]
        ),
        "conservative_ici_q975": float(
            conservative_ici[
                "q975"
            ]
        ),
        "bootstrap_n_valid": int(
            aggressive_ici[
                "n_valid"
            ]
        ),
    }


def build_validations(
    *,
    support2_status: str,
    mimic_status: str,
    synthetic_status: str,
    h5_status: str,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    support2 = evidence.loc[
        evidence[
            "dataset"
        ].eq(
            "SUPPORT2"
        )
    ].iloc[
        0
    ]

    mimic = evidence.loc[
        evidence[
            "dataset"
        ].eq(
            "MIMIC-IV"
        )
    ].iloc[
        0
    ]

    return pd.DataFrame(
        [
            {
                "check": (
                    "SUPPORT2 policy-shift "
                    "benchmark passed"
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
                    "MIMIC policy-shift "
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
                    "synthetic H6 mechanism "
                    "already frozen"
                ),
                "value": synthetic_status,
                "threshold": (
                    EXPECTED_SYNTHETIC_STATUS
                ),
                "comparison": "==",
                "passed": (
                    synthetic_status
                    == EXPECTED_SYNTHETIC_STATUS
                ),
            },
            {
                "check": (
                    "H5 cross-dataset package "
                    "already frozen"
                ),
                "value": h5_status,
                "threshold": (
                    EXPECTED_H5_STATUS
                ),
                "comparison": "==",
                "passed": (
                    h5_status
                    == EXPECTED_H5_STATUS
                ),
            },
            {
                "check": (
                    "reference ICI invariant "
                    "in both datasets"
                ),
                "value": (
                    f"SUPPORT2="
                    f"{float(support2['reference_ici_range']):.3e};"
                    f"MIMIC="
                    f"{float(mimic['reference_ici_range']):.3e}"
                ),
                "threshold": (
                    "both < 1e-12"
                ),
                "comparison": "<",
                "passed": bool(
                    float(
                        support2[
                            "reference_ici_range"
                        ]
                    )
                    < 1e-12
                    and float(
                        mimic[
                            "reference_ici_range"
                        ]
                    )
                    < 1e-12
                ),
            },
            {
                "check": (
                    "reference event rate invariant "
                    "in both datasets"
                ),
                "value": (
                    f"SUPPORT2="
                    f"{float(support2['reference_event_rate_range']):.3e};"
                    f"MIMIC="
                    f"{float(mimic['reference_event_rate_range']):.3e}"
                ),
                "threshold": (
                    "both < 1e-12"
                ),
                "comparison": "<",
                "passed": bool(
                    float(
                        support2[
                            "reference_event_rate_range"
                        ]
                    )
                    < 1e-12
                    and float(
                        mimic[
                            "reference_event_rate_range"
                        ]
                    )
                    < 1e-12
                ),
            },
            {
                "check": (
                    "restricted ICI range exceeds "
                    "0.03 in both datasets"
                ),
                "value": (
                    f"SUPPORT2="
                    f"{float(support2['restricted_ici_range']):.6f};"
                    f"MIMIC="
                    f"{float(mimic['restricted_ici_range']):.6f}"
                ),
                "threshold": (
                    "both > 0.03"
                ),
                "comparison": ">",
                "passed": bool(
                    float(
                        support2[
                            "restricted_ici_range"
                        ]
                    )
                    > 0.03
                    and float(
                        mimic[
                            "restricted_ici_range"
                        ]
                    )
                    > 0.03
                ),
            },
            {
                "check": (
                    "aggressive policy increases "
                    "restricted ICI in both datasets"
                ),
                "value": (
                    f"SUPPORT2 q025="
                    f"{float(support2['aggressive_ici_q025']):.6f};"
                    f"MIMIC q025="
                    f"{float(mimic['aggressive_ici_q025']):.6f}"
                ),
                "threshold": (
                    "both q025 > 0"
                ),
                "comparison": ">",
                "passed": bool(
                    float(
                        support2[
                            "aggressive_ici_q025"
                        ]
                    )
                    > 0
                    and float(
                        mimic[
                            "aggressive_ici_q025"
                        ]
                    )
                    > 0
                ),
            },
            {
                "check": (
                    "aggressive policy lowers "
                    "observation fraction "
                    "in both datasets"
                ),
                "value": (
                    f"SUPPORT2 q975="
                    f"{float(support2['aggressive_observation_q975']):.6f};"
                    f"MIMIC q975="
                    f"{float(mimic['aggressive_observation_q975']):.6f}"
                ),
                "threshold": (
                    "both q975 < 0"
                ),
                "comparison": "<",
                "passed": bool(
                    float(
                        support2[
                            "aggressive_observation_q975"
                        ]
                    )
                    < 0
                    and float(
                        mimic[
                            "aggressive_observation_q975"
                        ]
                    )
                    < 0
                ),
            },
            {
                "check": (
                    "all decisive intervals use "
                    "1000 valid paired bootstraps"
                ),
                "value": (
                    f"SUPPORT2="
                    f"{int(support2['bootstrap_n_valid'])};"
                    f"MIMIC="
                    f"{int(mimic['bootstrap_n_valid'])}"
                ),
                "threshold": (
                    "both == 1000"
                ),
                "comparison": "==",
                "passed": bool(
                    int(
                        support2[
                            "bootstrap_n_valid"
                        ]
                    )
                    == 1000
                    and int(
                        mimic[
                            "bootstrap_n_valid"
                        ]
                    )
                    == 1000
                ),
            },
            {
                "check": (
                    "SUPPORT2 conservative-policy "
                    "null is retained"
                ),
                "value": (
                    f"["
                    f"{float(support2['conservative_ici_q025']):.6f},"
                    f"{float(support2['conservative_ici_q975']):.6f}"
                    f"]"
                ),
                "threshold": (
                    "interval contains zero"
                ),
                "comparison": "contains",
                "passed": bool(
                    float(
                        support2[
                            "conservative_ici_q025"
                        ]
                    )
                    < 0
                    and float(
                        support2[
                            "conservative_ici_q975"
                        ]
                    )
                    > 0
                ),
            },
            {
                "check": (
                    "MIMIC conservative policy "
                    "decreases restricted ICI"
                ),
                "value": float(
                    mimic[
                        "conservative_ici_q975"
                    ]
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": bool(
                    float(
                        mimic[
                            "conservative_ici_q975"
                        ]
                    )
                    < 0
                ),
            },
        ]
    )


def build_claim(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    support2 = evidence.loc[
        evidence[
            "dataset"
        ].eq(
            "SUPPORT2"
        )
    ].iloc[
        0
    ]

    mimic = evidence.loc[
        evidence[
            "dataset"
        ].eq(
            "MIMIC-IV"
        )
    ].iloc[
        0
    ]

    return pd.DataFrame(
        [
            {
                "claim": "H6",
                "claim_text": (
                    "Changes in observation policy "
                    "can generate substantial apparent "
                    "calibration drift despite an "
                    "unchanged predictor, population, "
                    "and complete-outcome target."
                ),
                "support2_evidence": (
                    "Restricted ICI ranged by "
                    f"{float(support2['restricted_ici_range']):.3f}; "
                    "the aggressive policy increased "
                    "restricted ICI versus baseline by "
                    f"{float(support2['aggressive_ici_difference']):.3f}, "
                    "with paired 95% CI "
                    f"[{float(support2['aggressive_ici_q025']):.3f}, "
                    f"{float(support2['aggressive_ici_q975']):.3f}]."
                ),
                "mimic_evidence": (
                    "Restricted ICI ranged by "
                    f"{float(mimic['restricted_ici_range']):.3f}; "
                    "the aggressive policy increased "
                    "restricted ICI versus baseline by "
                    f"{float(mimic['aggressive_ici_difference']):.3f}, "
                    "with paired 95% CI "
                    f"[{float(mimic['aggressive_ici_q025']):.3f}, "
                    f"{float(mimic['aggressive_ici_q975']):.3f}]."
                ),
                "synthetic_evidence": (
                    "The frozen Phase 5E controlled "
                    "simulation establishes the same "
                    "mechanism under known fixed "
                    "data-generating conditions."
                ),
                "final_status": (
                    "strong bounded cross-dataset support"
                ),
                "manuscript_role": (
                    "primary methodological claim"
                ),
                "boundary": (
                    "The real-data analyses are "
                    "controlled policy emulations, "
                    "not causal or historical estimates "
                    "of actual policy changes."
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

    synthetic_dir = Path(
        args.synthetic_dir
    )

    h5_dir = Path(
        args.h5_dir
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
            "phase5j_support2_policy_shift_"
            "status.txt"
        )
    )

    mimic_status = read_status(
        mimic_dir
        / (
            "phase5j_mimic_policy_shift_"
            "status.txt"
        )
    )

    synthetic_status = read_status(
        synthetic_dir
        / "phase5e_freeze_status.txt"
    )

    h5_status = read_status(
        h5_dir
        / (
            "phase5i_cross_dataset_h5_"
            "status.txt"
        )
    )

    support2 = summarize_dataset(
        dataset="SUPPORT2",
        horizon_days=60,
        directory=support2_dir,
    )

    mimic = summarize_dataset(
        dataset="MIMIC-IV",
        horizon_days=30,
        directory=mimic_dir,
    )

    evidence = pd.DataFrame(
        [
            support2,
            mimic,
        ]
    )

    validations = build_validations(
        support2_status=(
            support2_status
        ),
        mimic_status=mimic_status,
        synthetic_status=(
            synthetic_status
        ),
        h5_status=h5_status,
        evidence=evidence,
    )

    claim = build_claim(
        evidence
    )

    status = (
        "phase5j_cross_dataset_h6_"
        "frozen_for_manuscript"
        if validations[
            "passed"
        ].all()
        else (
            "phase5j_cross_dataset_h6_"
            "freeze_requires_review"
        )
    )

    evidence.to_csv(
        output_dir
        / "h6_cross_dataset_evidence.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "freeze_validation_checks.csv",
        index=False,
    )

    claim.to_csv(
        output_dir
        / "h6_cross_dataset_claim.csv",
        index=False,
    )

    manifest = {
        "phase": "5J-D",
        "status": status,
        "claim_frozen": "H6",
        "claim_text": (
            claim[
                "claim_text"
            ].iloc[
                0
            ]
        ),
        "final_status": (
            claim[
                "final_status"
            ].iloc[
                0
            ]
        ),
        "evidence_layers": {
            "synthetic": (
                "Frozen controlled mechanism "
                "demonstration."
            ),
            "SUPPORT2": (
                "Controlled real-data-grounded "
                "60-day policy emulation."
            ),
            "MIMIC-IV": (
                "Controlled real-data-grounded "
                "30-day policy emulation."
            ),
        },
        "dimension_specific_results": {
            "support2_conservative_policy": (
                "Point estimate decreased restricted "
                "ICI, but its paired interval included "
                "zero; retained as an inconclusive "
                "policy-specific contrast."
            ),
            "mimic_conservative_policy": (
                "Significantly decreased restricted "
                "ICI versus baseline."
            ),
            "aggressive_policy": (
                "Significantly increased restricted "
                "ICI in both datasets."
            ),
        },
        "interpretation_boundaries": [
            (
                "Predictions and linked outcomes were "
                "held fixed within each dataset."
            ),
            (
                "Only the emulated observation and "
                "hospital-exit mechanism changed."
            ),
            (
                "The experiments do not estimate "
                "causal effects of actual clinical "
                "policies."
            ),
            (
                "SUPPORT2 and MIMIC use different "
                "outcome horizons and are not pooled."
            ),
            (
                "Policy-specific null findings are "
                "retained."
            ),
        ],
    }

    (
        output_dir
        / "phase5j_cross_dataset_h6_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5j_cross_dataset_h6_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5J-D cross-dataset H6 freeze"
    )

    print(
        "\nCross-dataset evidence"
    )

    print(
        evidence.to_string(
            index=False
        )
    )

    print(
        "\nH6 claim"
    )

    print(
        claim.to_string(
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
