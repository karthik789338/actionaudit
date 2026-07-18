"""Phase 5C-F: score-binning robustness for empirical distortion.

Re-evaluates the frozen MIMIC-IV test predictions using 3, 4,
and 5 reference-score quantile bins.

The same patient bootstrap sample is reused across bin counts
within each bootstrap replication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase5c_empirical_action_cancellation import (
    ACTION_ORDER,
    calculate_audit,
    load_predictions,
    make_config,
)


SCALAR_METRICS = [
    "reference_ici",
    "restricted_complete_case_ici",
    "delta_ici_restricted_minus_reference",
    "standardized_signed_gap",
    "gross_action_debt",
    "net_curve_distortion",
    "pointwise_cancelled_debt",
    "pointwise_cancellation_ratio",
    "curve_aci",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether Phase 5C empirical distortion "
            "is robust to score-bin resolution."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5c_binning_robustness"
        ),
    )

    parser.add_argument(
        "--bins",
        default="3,4,5",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


def parse_bins(
    text: str,
) -> list[int]:
    bins = sorted(
        {
            int(
                value.strip()
            )
            for value in text.split(",")
            if value.strip()
        }
    )

    if not bins:
        raise ValueError(
            "At least one bin count is required"
        )

    if any(
        value < 2
        for value in bins
    ):
        raise ValueError(
            "Every bin count must be at least two"
        )

    return bins


def point_summary_row(
    *,
    cohort: str,
    n_bins: int,
    result,
    action_summary: pd.DataFrame,
    scalars: dict[str, float],
) -> dict[str, object]:
    bin_summary = result.bin_summary

    eligible = bin_summary.loc[
        bin_summary[
            "eligible"
        ]
    ]

    return {
        "cohort": cohort,
        "n_bins": n_bins,
        "n_evaluation_patients": int(
            result.scalar_summary[
                "n"
            ]
        ),
        "eligible_bins": int(
            bin_summary[
                "eligible"
            ].sum()
        ),
        "total_bins": int(
            len(
                bin_summary
            )
        ),
        "minimum_bin_n": int(
            bin_summary[
                "n"
            ].min()
        ),
        "minimum_observed_n": int(
            bin_summary[
                "n_observed"
            ].min()
        ),
        "minimum_observed_fraction": float(
            bin_summary[
                "observed_fraction"
            ].min()
        ),
        "maximum_decomposition_residual": float(
            result.max_decomposition_residual
        ),
        "reference_ici": float(
            scalars[
                "reference_ici"
            ]
        ),
        "restricted_ici": float(
            scalars[
                "restricted_complete_case_ici"
            ]
        ),
        "delta_ici": float(
            scalars[
                "delta_ici_restricted_minus_reference"
            ]
        ),
        "standardized_signed_gap": float(
            scalars[
                "standardized_signed_gap"
            ]
        ),
        "gross_action_debt": float(
            scalars[
                "gross_action_debt"
            ]
        ),
        "net_curve_distortion": float(
            scalars[
                "net_curve_distortion"
            ]
        ),
        "pointwise_cancellation_ratio": float(
            scalars[
                "pointwise_cancellation_ratio"
            ]
        ),
        "curve_aci": float(
            scalars[
                "curve_aci"
            ]
        ),
        "positive_sab_actions": int(
            action_summary[
                "sab"
            ].gt(
                0
            ).sum()
        ),
        "negative_sab_actions": int(
            action_summary[
                "sab"
            ].lt(
                0
            ).sum()
        ),
        "minimum_sab": float(
            action_summary[
                "sab"
            ].min()
        ),
    }


def run_point_analyses(
    cohorts: dict[str, pd.DataFrame],
    *,
    bin_counts: list[int],
    horizon: int,
    min_bin_size: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    point_rows: list[
        dict[str, object]
    ] = []

    action_rows: list[
        dict[str, object]
    ] = []

    for cohort_name, frame in cohorts.items():
        for n_bins in bin_counts:
            config = make_config(
                horizon=horizon,
                n_bins=n_bins,
                min_bin_size=min_bin_size,
            )

            (
                result,
                action_summary,
                _,
                scalars,
            ) = calculate_audit(
                frame,
                config=config,
            )

            point_rows.append(
                point_summary_row(
                    cohort=cohort_name,
                    n_bins=n_bins,
                    result=result,
                    action_summary=(
                        action_summary
                    ),
                    scalars=scalars,
                )
            )

            for _, row in action_summary.iterrows():
                action_rows.append(
                    {
                        "cohort": (
                            cohort_name
                        ),
                        "n_bins": n_bins,
                        "action": row[
                            "action"
                        ],
                        "sab": float(
                            row[
                                "sab"
                            ]
                        ),
                        "aacd": float(
                            row[
                                "aacd"
                            ]
                        ),
                        "n_hidden": int(
                            row[
                                "n_hidden"
                            ]
                        ),
                    }
                )

    return (
        pd.DataFrame(
            point_rows
        ),
        pd.DataFrame(
            action_rows
        ),
    )


def run_common_bootstrap(
    cohorts: dict[str, pd.DataFrame],
    *,
    bin_counts: list[int],
    horizon: int,
    min_bin_size: int,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    scalar_rows: list[
        dict[str, object]
    ] = []

    action_rows: list[
        dict[str, object]
    ] = []

    invalid_rows: list[
        dict[str, object]
    ] = []

    for cohort_index, (
        cohort_name,
        frame,
    ) in enumerate(
        cohorts.items()
    ):
        rng = np.random.default_rng(
            seed
            + 100000
            * cohort_index
        )

        n = len(
            frame
        )

        invalid_counts = {
            n_bins: 0
            for n_bins in bin_counts
        }

        for bootstrap_id in range(
            replicates
        ):
            indices = rng.integers(
                0,
                n,
                size=n,
            )

            sample = (
                frame.iloc[
                    indices
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            sample[
                "patient_id"
            ] = np.arange(
                n,
                dtype=int,
            )

            for n_bins in bin_counts:
                config = make_config(
                    horizon=horizon,
                    n_bins=n_bins,
                    min_bin_size=(
                        min_bin_size
                    ),
                )

                try:
                    (
                        result,
                        action_summary,
                        _,
                        scalars,
                    ) = calculate_audit(
                        sample,
                        config=config,
                    )
                except (
                    ValueError,
                    RuntimeError,
                    FloatingPointError,
                ):
                    invalid_counts[
                        n_bins
                    ] += 1

                    continue

                scalar_rows.append(
                    {
                        "cohort": (
                            cohort_name
                        ),
                        "n_bins": n_bins,
                        "bootstrap_id": (
                            bootstrap_id
                        ),
                        "minimum_observed_n": int(
                            result.bin_summary[
                                "n_observed"
                            ].min()
                        ),
                        "maximum_decomposition_residual": float(
                            result.max_decomposition_residual
                        ),
                        **{
                            metric: float(
                                scalars[
                                    metric
                                ]
                            )
                            for metric in SCALAR_METRICS
                        },
                    }
                )

                for _, row in action_summary.iterrows():
                    action_rows.append(
                        {
                            "cohort": (
                                cohort_name
                            ),
                            "n_bins": (
                                n_bins
                            ),
                            "bootstrap_id": (
                                bootstrap_id
                            ),
                            "action": row[
                                "action"
                            ],
                            "sab": float(
                                row[
                                    "sab"
                                ]
                            ),
                            "aacd": float(
                                row[
                                    "aacd"
                                ]
                            ),
                        }
                    )

            if (
                bootstrap_id + 1
            ) % 100 == 0:
                print(
                    f"  {cohort_name}: completed "
                    f"{bootstrap_id + 1}/"
                    f"{replicates}"
                )

        for n_bins, count in (
            invalid_counts.items()
        ):
            invalid_rows.append(
                {
                    "cohort": (
                        cohort_name
                    ),
                    "n_bins": (
                        n_bins
                    ),
                    "invalid_bootstraps": (
                        count
                    ),
                    "requested_bootstraps": (
                        replicates
                    ),
                }
            )

    return (
        pd.DataFrame(
            scalar_rows
        ),
        pd.DataFrame(
            action_rows
        ),
        pd.DataFrame(
            invalid_rows
        ),
    )


def percentile_interval(
    values: pd.Series,
) -> dict[str, float | int]:
    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    return {
        "q025": float(
            clean.quantile(
                0.025
            )
        ),
        "median": float(
            clean.quantile(
                0.50
            )
        ),
        "q975": float(
            clean.quantile(
                0.975
            )
        ),
        "n_valid": int(
            len(
                clean
            )
        ),
    }


def scalar_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    grouped = bootstrap.groupby(
        [
            "cohort",
            "n_bins",
        ],
        sort=True,
    )

    metrics = (
        SCALAR_METRICS
        + [
            "minimum_observed_n",
            "maximum_decomposition_residual",
        ]
    )

    for (
        cohort,
        n_bins,
    ), group in grouped:
        for metric in metrics:
            rows.append(
                {
                    "cohort": cohort,
                    "n_bins": n_bins,
                    "metric": metric,
                    **percentile_interval(
                        group[
                            metric
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def action_interval_table(
    *,
    point_actions: pd.DataFrame,
    bootstrap_actions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    grouped = bootstrap_actions.groupby(
        [
            "cohort",
            "n_bins",
            "action",
        ],
        sort=True,
    )

    for (
        cohort,
        n_bins,
        action,
    ), group in grouped:
        point = point_actions.loc[
            point_actions[
                "cohort"
            ].eq(
                cohort
            )
            & point_actions[
                "n_bins"
            ].eq(
                n_bins
            )
            & point_actions[
                "action"
            ].eq(
                action
            )
        ]

        rows.append(
            {
                "cohort": cohort,
                "n_bins": n_bins,
                "action": action,
                "estimate": float(
                    point.iloc[
                        0
                    ][
                        "sab"
                    ]
                ),
                **percentile_interval(
                    group[
                        "sab"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def scalar_bound(
    intervals: pd.DataFrame,
    *,
    cohort: str,
    metric: str,
    column: str,
    aggregation: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "cohort"
        ].eq(
            cohort
        )
        & intervals[
            "metric"
        ].eq(
            metric
        ),
        column,
    ]

    if aggregation == "min":
        return float(
            selected.min()
        )

    if aggregation == "max":
        return float(
            selected.max()
        )

    raise ValueError(
        f"Unknown aggregation: {aggregation}"
    )


def validate_robustness(
    *,
    point_summary: pd.DataFrame,
    scalar_intervals: pd.DataFrame,
    action_intervals: pd.DataFrame,
    invalid_summary: pd.DataFrame,
    replicates: int,
) -> tuple[
    pd.DataFrame,
    str,
    dict[str, object],
]:
    primary = point_summary.loc[
        point_summary[
            "cohort"
        ].eq(
            "primary"
        )
    ]

    icu = point_summary.loc[
        point_summary[
            "cohort"
        ].eq(
            "icu_sensitivity"
        )
    ]

    minimum_observed = int(
        point_summary[
            "minimum_observed_n"
        ].min()
    )

    maximum_residual = float(
        point_summary[
            "maximum_decomposition_residual"
        ].max()
    )

    all_action_positive = bool(
        (
            action_intervals[
                "q025"
            ]
            > 0
        ).all()
    )

    minimum_primary_delta_ici = (
        scalar_bound(
            scalar_intervals,
            cohort="primary",
            metric=(
                "delta_ici_restricted_minus_reference"
            ),
            column="q025",
            aggregation="min",
        )
    )

    minimum_primary_signed_gap = (
        scalar_bound(
            scalar_intervals,
            cohort="primary",
            metric="standardized_signed_gap",
            column="q025",
            aggregation="min",
        )
    )

    maximum_primary_cancellation = (
        scalar_bound(
            scalar_intervals,
            cohort="primary",
            metric=(
                "pointwise_cancellation_ratio"
            ),
            column="q975",
            aggregation="max",
        )
    )

    maximum_icu_cancellation = (
        scalar_bound(
            scalar_intervals,
            cohort="icu_sensitivity",
            metric=(
                "pointwise_cancellation_ratio"
            ),
            column="q975",
            aggregation="max",
        )
    )

    minimum_icu_signed_gap = (
        scalar_bound(
            scalar_intervals,
            cohort="icu_sensitivity",
            metric="standardized_signed_gap",
            column="q025",
            aggregation="min",
        )
    )

    invalid_total = int(
        invalid_summary[
            "invalid_bootstraps"
        ].sum()
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "all cohort and bin configurations completed"
                ),
                "value": len(
                    point_summary
                ),
                "threshold": 6,
                "comparison": "==",
                "passed": (
                    len(
                        point_summary
                    )
                    == 6
                ),
            },
            {
                "check": (
                    "all decomposition residuals below 1e-10"
                ),
                "value": maximum_residual,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    maximum_residual
                    < 1e-10
                ),
            },
            {
                "check": (
                    "each score bin has at least 30 "
                    "outcome-observed patients"
                ),
                "value": minimum_observed,
                "threshold": 30,
                "comparison": ">=",
                "passed": (
                    minimum_observed
                    >= 30
                ),
            },
            {
                "check": (
                    "all requested bootstraps are valid"
                ),
                "value": invalid_total,
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    invalid_total
                    == 0
                    and (
                        invalid_summary[
                            "requested_bootstraps"
                        ]
                        == replicates
                    ).all()
                ),
            },
            {
                "check": (
                    "all action SAB intervals remain positive"
                ),
                "value": int(
                    (
                        action_intervals[
                            "q025"
                        ]
                        > 0
                    ).sum()
                ),
                "threshold": len(
                    action_intervals
                ),
                "comparison": "==",
                "passed": (
                    all_action_positive
                ),
            },
            {
                "check": (
                    "primary restricted-minus-reference ICI "
                    "remains above 0.10"
                ),
                "value": (
                    minimum_primary_delta_ici
                ),
                "threshold": 0.10,
                "comparison": ">",
                "passed": (
                    minimum_primary_delta_ici
                    > 0.10
                ),
            },
            {
                "check": (
                    "primary standardized distortion "
                    "remains above 0.10"
                ),
                "value": (
                    minimum_primary_signed_gap
                ),
                "threshold": 0.10,
                "comparison": ">",
                "passed": (
                    minimum_primary_signed_gap
                    > 0.10
                ),
            },
            {
                "check": (
                    "primary cancellation remains below one percent"
                ),
                "value": (
                    maximum_primary_cancellation
                ),
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    maximum_primary_cancellation
                    < 0.01
                ),
            },
            {
                "check": (
                    "ICU standardized distortion "
                    "remains above 0.10"
                ),
                "value": (
                    minimum_icu_signed_gap
                ),
                "threshold": 0.10,
                "comparison": ">",
                "passed": (
                    minimum_icu_signed_gap
                    > 0.10
                ),
            },
            {
                "check": (
                    "ICU cancellation remains below one percent"
                ),
                "value": (
                    maximum_icu_cancellation
                ),
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    maximum_icu_cancellation
                    < 0.01
                ),
            },
            {
                "check": (
                    "primary reference ICI remains below 0.02"
                ),
                "value": float(
                    primary[
                        "reference_ici"
                    ].max()
                ),
                "threshold": 0.02,
                "comparison": "<",
                "passed": (
                    primary[
                        "reference_ici"
                    ].max()
                    < 0.02
                ),
            },
            {
                "check": (
                    "all tested configurations retain "
                    "unidirectional point estimates"
                ),
                "value": int(
                    (
                        point_summary[
                            "negative_sab_actions"
                        ]
                        == 0
                    ).sum()
                ),
                "threshold": len(
                    point_summary
                ),
                "comparison": "==",
                "passed": bool(
                    (
                        point_summary[
                            "negative_sab_actions"
                        ]
                        == 0
                    ).all()
                ),
            },
        ]
    )

    status = (
        "phase5c_empirical_distortion_robust_to_binning"
        if checks[
            "passed"
        ].all()
        else (
            "phase5c_binning_robustness_"
            "requires_review"
        )
    )

    evidence = {
        "tested_bin_counts": sorted(
            point_summary[
                "n_bins"
            ].unique()
            .astype(int)
            .tolist()
        ),
        "minimum_outcome_observed_n_per_bin": (
            minimum_observed
        ),
        "minimum_primary_delta_ici_q025": (
            minimum_primary_delta_ici
        ),
        "minimum_primary_signed_gap_q025": (
            minimum_primary_signed_gap
        ),
        "maximum_primary_cancellation_q975": (
            maximum_primary_cancellation
        ),
        "minimum_icu_signed_gap_q025": (
            minimum_icu_signed_gap
        ),
        "maximum_icu_cancellation_q975": (
            maximum_icu_cancellation
        ),
        "all_action_sab_intervals_positive": (
            all_action_positive
        ),
        "status": status,
    }

    return (
        checks,
        status,
        evidence,
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

    bin_counts = parse_bins(
        args.bins
    )

    frame = load_predictions(
        input_path
    )

    cohorts = {
        "primary": frame.copy(),
        "icu_sensitivity": (
            frame.loc[
                frame[
                    "has_icu_stay"
                ].eq(
                    1
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        ),
    }

    print(
        "Phase 5C binning robustness"
    )

    print(
        f"  bin counts={bin_counts}"
    )

    print(
        f"  bootstrap replicates="
        f"{args.bootstrap_replicates}"
    )

    (
        point_summary,
        point_actions,
    ) = run_point_analyses(
        cohorts,
        bin_counts=bin_counts,
        horizon=args.horizon,
        min_bin_size=args.min_bin_size,
    )

    (
        bootstrap_scalars,
        bootstrap_actions,
        invalid_summary,
    ) = run_common_bootstrap(
        cohorts,
        bin_counts=bin_counts,
        horizon=args.horizon,
        min_bin_size=args.min_bin_size,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
    )

    scalar_intervals = (
        scalar_interval_table(
            bootstrap_scalars
        )
    )

    action_intervals = (
        action_interval_table(
            point_actions=point_actions,
            bootstrap_actions=(
                bootstrap_actions
            ),
        )
    )

    (
        checks,
        preliminary_status,
        evidence,
    ) = validate_robustness(
        point_summary=(
            point_summary
        ),
        scalar_intervals=(
            scalar_intervals
        ),
        action_intervals=(
            action_intervals
        ),
        invalid_summary=(
            invalid_summary
        ),
        replicates=(
            args.bootstrap_replicates
        ),
    )

    status = (
        "phase5c_binning_robustness_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    point_summary.to_csv(
        output_dir
        / "point_estimate_summary.csv",
        index=False,
    )

    point_actions.to_csv(
        output_dir
        / "point_action_summary.csv",
        index=False,
    )

    bootstrap_scalars.to_csv(
        output_dir
        / "bootstrap_scalar_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_actions.to_csv(
        output_dir
        / "bootstrap_action_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    scalar_intervals.to_csv(
        output_dir
        / "scalar_bootstrap_intervals.csv",
        index=False,
    )

    action_intervals.to_csv(
        output_dir
        / "action_bootstrap_intervals.csv",
        index=False,
    )

    invalid_summary.to_csv(
        output_dir
        / "invalid_bootstrap_summary.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    metadata = {
        "phase": "5C-F",
        "experiment": (
            "Empirical action-distortion robustness "
            "to reference-score bin resolution"
        ),
        "input": str(
            input_path
        ),
        "bin_counts": (
            bin_counts
        ),
        "common_bootstrap_samples_across_bin_counts": True,
        "bootstrap_replicates": int(
            args.bootstrap_replicates
        ),
        "seed": int(
            args.seed
        ),
        "validation_evidence": (
            evidence
        ),
        "preliminary_status": (
            preliminary_status
            if args.smoke
            else None
        ),
        "smoke": bool(
            args.smoke
        ),
        "status": status,
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
        / "phase5c_binning_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nPoint-estimate summary"
    )

    print(
        point_summary.to_string(
            index=False
        )
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
        f"\nPreliminary status: "
        f"{preliminary_status}"
    )

    print(
        f"Saved status: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
