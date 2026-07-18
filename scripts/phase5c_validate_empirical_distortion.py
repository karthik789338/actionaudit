"""Validate the Phase 5C empirical MIMIC-IV decomposition.

The empirical hypothesis differs from the simulation hypothesis.

MIMIC-IV is considered to support unidirectional action distortion
when:
    1. The exact decomposition reconstructs the restricted curve.
    2. Restricted calibration is materially worse than reference
       calibration.
    3. Action SABs consistently point in the same direction.
    4. Pointwise cancellation is substantively negligible.
    5. The pattern persists in the ICU sensitivity cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate empirical unidirectional action "
            "distortion in Phase 5C."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--expected-bootstrap-replicates",
        type=int,
        required=True,
    )

    return parser.parse_args()


def read_scalar_summary(
    path: Path,
) -> dict[str, float]:
    frame = pd.read_csv(
        path
    )

    return {
        str(row[
            "metric"
        ]): float(
            row[
                "value"
            ]
        )
        for _, row in frame.iterrows()
    }


def read_scalar_intervals(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path
    )

    return frame.set_index(
        "metric"
    )


def interval_value(
    intervals: pd.DataFrame,
    metric: str,
    column: str,
) -> float:
    if metric not in intervals.index:
        raise KeyError(
            f"Missing scalar interval metric: {metric}"
        )

    return float(
        intervals.loc[
            metric,
            column,
        ]
    )


def read_action_sab(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path
    )

    return (
        frame.loc[
            frame[
                "metric"
            ].eq(
                "sab"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def eligible_bin_count(
    path: Path,
) -> tuple[
    int,
    int,
]:
    frame = pd.read_csv(
        path
    )

    eligible = (
        frame[
            "eligible"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "true"
        )
    )

    return (
        int(
            eligible.sum()
        ),
        int(
            len(
                frame
            )
        ),
    )


def cohort_evidence(
    cohort_dir: Path,
) -> dict[str, object]:
    scalars = read_scalar_summary(
        cohort_dir
        / "scalar_summary.csv"
    )

    intervals = read_scalar_intervals(
        cohort_dir
        / "scalar_bootstrap_intervals.csv"
    )

    action_sab = read_action_sab(
        cohort_dir
        / "action_bootstrap_intervals.csv"
    )

    eligible_bins, total_bins = (
        eligible_bin_count(
            cohort_dir
            / "calibration_bin_summary.csv"
        )
    )

    positive_actions = (
        action_sab.loc[
            action_sab[
                "bootstrap_q025"
            ]
            > 0,
            "action",
        ]
        .astype(str)
        .tolist()
    )

    negative_actions = (
        action_sab.loc[
            action_sab[
                "bootstrap_q975"
            ]
            < 0,
            "action",
        ]
        .astype(str)
        .tolist()
    )

    uncertain_actions = (
        action_sab.loc[
            ~action_sab[
                "action"
            ].isin(
                positive_actions
                + negative_actions
            ),
            "action",
        ]
        .astype(str)
        .tolist()
    )

    minimum_bootstraps = int(
        min(
            intervals[
                "n_valid_bootstrap"
            ].min(),
            action_sab[
                "n_valid_bootstrap"
            ].min(),
        )
    )

    return {
        "scalars": scalars,
        "intervals": intervals,
        "action_sab": action_sab,
        "positive_actions": (
            positive_actions
        ),
        "negative_actions": (
            negative_actions
        ),
        "uncertain_actions": (
            uncertain_actions
        ),
        "eligible_bins": (
            eligible_bins
        ),
        "total_bins": (
            total_bins
        ),
        "minimum_valid_bootstraps": (
            minimum_bootstraps
        ),
    }


def make_check(
    *,
    check: str,
    value: object,
    threshold: object,
    comparison: str,
    passed: bool,
) -> dict[str, object]:
    return {
        "check": check,
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(
            passed
        ),
    }


def main() -> None:
    args = parse_arguments()

    input_dir = Path(
        args.input_dir
    )

    primary_dir = (
        input_dir
        / "primary"
    )

    icu_dir = (
        input_dir
        / "icu_sensitivity"
    )

    primary = cohort_evidence(
        primary_dir
    )

    icu = cohort_evidence(
        icu_dir
    )

    primary_scalars = primary[
        "scalars"
    ]

    primary_intervals = primary[
        "intervals"
    ]

    icu_scalars = icu[
        "scalars"
    ]

    icu_intervals = icu[
        "intervals"
    ]

    primary_delta_ici_lower = interval_value(
        primary_intervals,
        (
            "delta_ici_restricted_"
            "minus_reference"
        ),
        "bootstrap_q025",
    )

    primary_signed_gap_lower = interval_value(
        primary_intervals,
        "standardized_signed_gap",
        "bootstrap_q025",
    )

    primary_cancellation_upper = interval_value(
        primary_intervals,
        "pointwise_cancellation_ratio",
        "bootstrap_q975",
    )

    primary_curve_aci_upper = interval_value(
        primary_intervals,
        "curve_aci",
        "bootstrap_q975",
    )

    icu_cancellation_upper = interval_value(
        icu_intervals,
        "pointwise_cancellation_ratio",
        "bootstrap_q975",
    )

    icu_signed_gap_lower = interval_value(
        icu_intervals,
        "standardized_signed_gap",
        "bootstrap_q025",
    )

    checks = pd.DataFrame(
        [
            make_check(
                check=(
                    "primary decomposition residual below 1e-10"
                ),
                value=primary_scalars[
                    "max_decomposition_residual"
                ],
                threshold=1e-10,
                comparison="<",
                passed=(
                    primary_scalars[
                        "max_decomposition_residual"
                    ]
                    < 1e-10
                ),
            ),
            make_check(
                check=(
                    "all primary bins are estimable"
                ),
                value=primary[
                    "eligible_bins"
                ],
                threshold=primary[
                    "total_bins"
                ],
                comparison="==",
                passed=(
                    primary[
                        "eligible_bins"
                    ]
                    == primary[
                        "total_bins"
                    ]
                ),
            ),
            make_check(
                check=(
                    "all ICU bins are estimable"
                ),
                value=icu[
                    "eligible_bins"
                ],
                threshold=icu[
                    "total_bins"
                ],
                comparison="==",
                passed=(
                    icu[
                        "eligible_bins"
                    ]
                    == icu[
                        "total_bins"
                    ]
                ),
            ),
            make_check(
                check=(
                    "all requested primary bootstraps are valid"
                ),
                value=primary[
                    "minimum_valid_bootstraps"
                ],
                threshold=(
                    args.expected_bootstrap_replicates
                ),
                comparison="==",
                passed=(
                    primary[
                        "minimum_valid_bootstraps"
                    ]
                    == args.expected_bootstrap_replicates
                ),
            ),
            make_check(
                check=(
                    "all requested ICU bootstraps are valid"
                ),
                value=icu[
                    "minimum_valid_bootstraps"
                ],
                threshold=(
                    args.expected_bootstrap_replicates
                ),
                comparison="==",
                passed=(
                    icu[
                        "minimum_valid_bootstraps"
                    ]
                    == args.expected_bootstrap_replicates
                ),
            ),
            make_check(
                check=(
                    "reference calibration ICI below 0.01"
                ),
                value=primary_scalars[
                    "reference_ici"
                ],
                threshold=0.01,
                comparison="<",
                passed=(
                    primary_scalars[
                        "reference_ici"
                    ]
                    < 0.01
                ),
            ),
            make_check(
                check=(
                    "restricted-minus-reference ICI "
                    "is materially positive"
                ),
                value=(
                    primary_delta_ici_lower
                ),
                threshold=0.10,
                comparison=">",
                passed=(
                    primary_delta_ici_lower
                    > 0.10
                ),
            ),
            make_check(
                check=(
                    "primary standardized distortion "
                    "is materially positive"
                ),
                value=(
                    primary_signed_gap_lower
                ),
                threshold=0.10,
                comparison=">",
                passed=(
                    primary_signed_gap_lower
                    > 0.10
                ),
            ),
            make_check(
                check=(
                    "all primary action SAB intervals "
                    "are positive"
                ),
                value=len(
                    primary[
                        "positive_actions"
                    ]
                ),
                threshold=len(
                    primary[
                        "action_sab"
                    ]
                ),
                comparison="==",
                passed=(
                    len(
                        primary[
                            "positive_actions"
                        ]
                    )
                    == len(
                        primary[
                            "action_sab"
                        ]
                    )
                ),
            ),
            make_check(
                check=(
                    "no primary action SAB interval "
                    "is negative"
                ),
                value=len(
                    primary[
                        "negative_actions"
                    ]
                ),
                threshold=0,
                comparison="==",
                passed=(
                    len(
                        primary[
                            "negative_actions"
                        ]
                    )
                    == 0
                ),
            ),
            make_check(
                check=(
                    "primary cancellation ratio remains "
                    "below one percent"
                ),
                value=(
                    primary_cancellation_upper
                ),
                threshold=0.01,
                comparison="<",
                passed=(
                    primary_cancellation_upper
                    < 0.01
                ),
            ),
            make_check(
                check=(
                    "primary curve ACI remains below "
                    "one percent"
                ),
                value=(
                    primary_curve_aci_upper
                ),
                threshold=0.01,
                comparison="<",
                passed=(
                    primary_curve_aci_upper
                    < 0.01
                ),
            ),
            make_check(
                check=(
                    "ICU standardized distortion "
                    "is materially positive"
                ),
                value=(
                    icu_signed_gap_lower
                ),
                threshold=0.10,
                comparison=">",
                passed=(
                    icu_signed_gap_lower
                    > 0.10
                ),
            ),
            make_check(
                check=(
                    "ICU cancellation ratio remains "
                    "below one percent"
                ),
                value=(
                    icu_cancellation_upper
                ),
                threshold=0.01,
                comparison="<",
                passed=(
                    icu_cancellation_upper
                    < 0.01
                ),
            ),
            make_check(
                check=(
                    "all ICU action SAB intervals "
                    "are positive"
                ),
                value=len(
                    icu[
                        "positive_actions"
                    ]
                ),
                threshold=len(
                    icu[
                        "action_sab"
                    ]
                ),
                comparison="==",
                passed=(
                    len(
                        icu[
                            "positive_actions"
                        ]
                    )
                    == len(
                        icu[
                            "action_sab"
                        ]
                    )
                ),
            ),
        ]
    )

    status = (
        "phase5c_empirical_unidirectional_action_distortion_validated"
        if checks[
            "passed"
        ].all()
        else (
            "phase5c_empirical_distortion_"
            "validation_requires_review"
        )
    )

    evidence = {
        "primary_reference_ici": (
            primary_scalars[
                "reference_ici"
            ]
        ),
        "primary_restricted_ici": (
            primary_scalars[
                "restricted_complete_case_ici"
            ]
        ),
        "primary_standardized_signed_gap": (
            primary_scalars[
                "standardized_signed_gap"
            ]
        ),
        "primary_pointwise_cancellation_ratio": (
            primary_scalars[
                "pointwise_cancellation_ratio"
            ]
        ),
        "primary_curve_aci": (
            primary_scalars[
                "curve_aci"
            ]
        ),
        "primary_positive_sab_actions": (
            primary[
                "positive_actions"
            ]
        ),
        "primary_negative_sab_actions": (
            primary[
                "negative_actions"
            ]
        ),
        "primary_uncertain_sab_actions": (
            primary[
                "uncertain_actions"
            ]
        ),
        "icu_standardized_signed_gap": (
            icu_scalars[
                "standardized_signed_gap"
            ]
        ),
        "icu_pointwise_cancellation_ratio": (
            icu_scalars[
                "pointwise_cancellation_ratio"
            ]
        ),
        "icu_positive_sab_actions": (
            icu[
                "positive_actions"
            ]
        ),
        "interpretation": (
            "Hospital-boundary outcome restriction creates "
            "large, unidirectional action-attributed "
            "calibration distortion. Empirical cancellation "
            "is negligible in both the full inpatient and "
            "ICU-associated test cohorts."
        ),
        "status": status,
    }

    checks.to_csv(
        input_dir
        / "empirical_distortion_validation_checks.csv",
        index=False,
    )

    (
        input_dir
        / "empirical_distortion_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        input_dir
        / "phase5c_empirical_distortion_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5C empirical distortion validation"
    )

    print(
        "\nValidation checks"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        "\nEvidence"
    )

    print(
        json.dumps(
            evidence,
            indent=2,
        )
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
