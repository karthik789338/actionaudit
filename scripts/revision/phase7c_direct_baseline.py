from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from actionaudit.discrete_survival import (
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
)
from actionaudit.km import (
    evaluate_binned_estimators,
    reference_quantile_edges,
)
from actionaudit.simulation import (
    scenario_config,
    simulate_cohort,
)


CONDITIONS = (
    "baseline",
    "observation_only",
    "outcome_only",
    "both",
)

METRICS = (
    "reference_event_rate",
    "observed_fraction",
    "exit_before_horizon_fraction",
    "reference_ici",
    "complete_case_ici",
    "kaplan_meier_ici",
    "oracle_or_ici",
    "oracle_ipcw_ici",
    "oracle_or_mae",
    "oracle_ipcw_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7C: direct comparison of a simple "
            "calibration/censoring monitor with the "
            "linked-versus-restricted ActionAudit view."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "revision_kdd2027/"
            "phase7c_direct_baseline"
        ),
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--event-intercept-shift",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--clip",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def build_configs(
    args: argparse.Namespace,
    seed: int,
):
    baseline = scenario_config(
        "observed_dependent",
        n=args.n,
        horizon=args.horizon,
        seed=seed,
    )

    observation_only = replace(
        baseline,
        scenario="phase7c_observation_only",
        censor_intercept=-3.55,
        censor_x1=-1.00,
        censor_x2=1.25,
    )

    outcome_only = replace(
        baseline,
        scenario="phase7c_outcome_only",
        event_intercept=(
            baseline.event_intercept
            + args.event_intercept_shift
        ),
    )

    both = replace(
        baseline,
        scenario="phase7c_both",
        event_intercept=(
            baseline.event_intercept
            + args.event_intercept_shift
        ),
        censor_intercept=-3.55,
        censor_x1=-1.00,
        censor_x2=1.25,
    )

    return {
        "baseline": baseline,
        "observation_only": observation_only,
        "outcome_only": outcome_only,
        "both": both,
    }


def evaluate_condition(
    *,
    name: str,
    output,
    baseline_prediction: np.ndarray,
    bin_edges: np.ndarray,
    args: argparse.Namespace,
    replication: int,
) -> dict[str, object]:

    frame = output.frame.copy()

    # Freeze the deployed predictor in every condition.
    frame["prediction"] = baseline_prediction

    binned = evaluate_binned_estimators(
        frame,
        horizon=args.horizon,
        bin_edges=bin_edges,
    )

    nuisance_frame = (
        construct_survival_nuisance_predictions(
            frame,
            event_hazards=(
                output.true_event_hazards
            ),
            censor_hazards=(
                output.true_censor_hazards
            ),
            horizon=args.horizon,
            censor_survival_clip=args.clip,
        )
    )

    nuisance_result = (
        evaluate_survival_estimators(
            nuisance_frame,
            horizon=args.horizon,
            bin_edges=bin_edges,
            censor_survival_clip=args.clip,
        )
    )

    b = binned.scalar_summary
    n = nuisance_result.scalar_summary

    return {
        "replication": replication,
        "condition": name,

        "reference_event_rate": float(
            frame[
                "outcome"
            ].mean()
        ),

        "observed_fraction": float(
            frame[
                "observed"
            ].mean()
        ),

        "exit_before_horizon_fraction": float(
            (
                frame[
                    "latent_censor_time"
                ]
                <= args.horizon
            ).mean()
        ),

        "reference_ici": float(
            b[
                "reference_ici"
            ]
        ),

        "complete_case_ici": float(
            b[
                "complete_case_ici"
            ]
        ),

        "kaplan_meier_ici": float(
            b[
                "kaplan_meier_ici"
            ]
        ),

        "oracle_or_ici": float(
            n[
                "survival_or_ici"
            ]
        ),

        "oracle_ipcw_ici": float(
            n[
                "ipcw_event_ici"
            ]
        ),

        "oracle_or_mae": float(
            n[
                "survival_or_mae_to_reference"
            ]
        ),

        "oracle_ipcw_mae": float(
            n[
                "ipcw_event_mae_to_reference"
            ]
        ),
    }


def run_replication(
    args: argparse.Namespace,
    replication: int,
):
    seed = (
        args.seed
        + 1000
        * replication
    )

    configs = build_configs(
        args,
        seed,
    )

    outputs = {
        name: simulate_cohort(
            config
        )
        for name, config
        in configs.items()
    }

    baseline = (
        outputs[
            "baseline"
        ].frame
    )

    baseline_prediction = (
        baseline[
            "prediction"
        ]
        .to_numpy(
            dtype=float
        )
        .copy()
    )

    bin_edges = (
        reference_quantile_edges(
            baseline_prediction,
            args.n_bins,
        )
    )

    observation_only = (
        outputs[
            "observation_only"
        ].frame
    )

    outcome_only = (
        outputs[
            "outcome_only"
        ].frame
    )

    both = (
        outputs[
            "both"
        ].frame
    )

    checks = {
        "covariates_equal_all_conditions": all(
            np.array_equal(
                baseline[
                    [
                        "x1",
                        "x2",
                    ]
                ].to_numpy(),
                outputs[
                    name
                ].frame[
                    [
                        "x1",
                        "x2",
                    ]
                ].to_numpy(),
            )
            for name in CONDITIONS
        ),

        "observation_only_outcomes_equal": (
            np.array_equal(
                baseline[
                    "outcome"
                ].to_numpy(),
                observation_only[
                    "outcome"
                ].to_numpy(),
            )
        ),

        "outcome_only_censor_times_equal": (
            np.array_equal(
                baseline[
                    "latent_censor_time"
                ].to_numpy(),
                outcome_only[
                    "latent_censor_time"
                ].to_numpy(),
            )
        ),

        "both_uses_same_covariates": (
            np.array_equal(
                baseline[
                    [
                        "x1",
                        "x2",
                    ]
                ].to_numpy(),
                both[
                    [
                        "x1",
                        "x2",
                    ]
                ].to_numpy(),
            )
        ),
    }

    rows = []

    for name in CONDITIONS:
        rows.append(
            evaluate_condition(
                name=name,
                output=outputs[
                    name
                ],
                baseline_prediction=(
                    baseline_prediction
                ),
                bin_edges=bin_edges,
                args=args,
                replication=(
                    replication
                ),
            )
        )

    values = pd.DataFrame(
        rows
    )

    baseline_row = (
        values.loc[
            values[
                "condition"
            ].eq(
                "baseline"
            )
        ]
        .iloc[0]
    )

    contrasts = []

    for _, row in (
        values.loc[
            ~values[
                "condition"
            ].eq(
                "baseline"
            )
        ].iterrows()
    ):
        contrast = {
            "replication": (
                replication
            ),
            "condition": (
                row[
                    "condition"
                ]
            ),
        }

        for metric in METRICS:
            contrast[
                f"delta_{metric}"
            ] = float(
                row[
                    metric
                ]
                - baseline_row[
                    metric
                ]
            )

        contrasts.append(
            contrast
        )

    return (
        values,
        pd.DataFrame(
            contrasts
        ),
        checks,
    )


def summarize_contrasts(
    contrasts: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    delta_columns = [
        column
        for column
        in contrasts.columns
        if column.startswith(
            "delta_"
        )
    ]

    for condition, group in (
        contrasts.groupby(
            "condition",
            sort=False,
        )
    ):
        row = {
            "condition": condition,
            "replicates": len(
                group
            ),
        }

        for column in delta_columns:
            values = group[
                column
            ]

            row[
                f"{column}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{column}_q025"
            ] = float(
                values.quantile(
                    0.025
                )
            )

            row[
                f"{column}_q975"
            ] = float(
                values.quantile(
                    0.975
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_direct_comparison(
    contrasts: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    monitor_columns = {
        "restricted_complete_case": (
            "delta_complete_case_ici"
        ),
        "kaplan_meier": (
            "delta_kaplan_meier_ici"
        ),
        "oracle_outcome_regression": (
            "delta_oracle_or_ici"
        ),
        "oracle_ipcw": (
            "delta_oracle_ipcw_ici"
        ),
        "linked_reference": (
            "delta_reference_ici"
        ),
    }

    for condition, group in (
        contrasts.groupby(
            "condition",
            sort=False,
        )
    ):
        for (
            monitor,
            column,
        ) in monitor_columns.items():

            values = (
                group[
                    column
                ]
                .abs()
            )

            rows.append(
                {
                    "condition": (
                        condition
                    ),
                    "monitor": (
                        monitor
                    ),
                    "mean_absolute_ici_change": float(
                        values.mean()
                    ),
                    "q025_absolute_ici_change": float(
                        values.quantile(
                            0.025
                        )
                    ),
                    "q975_absolute_ici_change": float(
                        values.quantile(
                            0.975
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_figure(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:

    conditions = [
        "observation_only",
        "outcome_only",
        "both",
    ]

    monitors = [
        "restricted_complete_case",
        "kaplan_meier",
        "oracle_outcome_regression",
        "oracle_ipcw",
        "linked_reference",
    ]

    x = np.arange(
        len(
            conditions
        ),
        dtype=float,
    )

    width = 0.15

    figure, axis = (
        plt.subplots(
            figsize=(
                10,
                5.5,
            )
        )
    )

    for index, monitor in (
        enumerate(
            monitors
        )
    ):
        subset = (
            comparison.loc[
                comparison[
                    "monitor"
                ].eq(
                    monitor
                )
            ]
            .set_index(
                "condition"
            )
        )

        values = np.array(
            [
                subset.loc[
                    condition,
                    "mean_absolute_ici_change",
                ]
                for condition
                in conditions
            ]
        )

        axis.bar(
            x
            + (
                index
                - 2
            )
            * width,
            values,
            width=width,
            label=monitor,
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        conditions
    )

    axis.set_ylabel(
        "Mean absolute ICI change "
        "from baseline"
    )

    axis.set_title(
        "Simple monitoring versus "
        "linked-target auditing"
    )

    axis.legend(
        fontsize=8
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "direct_baseline_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_values = []
    all_contrasts = []
    invariant_checks = []

    print(
        "="
        * 100
    )

    print(
        "ACTIONAUDIT PHASE 7C: "
        "DIRECT SIMPLE-BASELINE COMPARISON"
    )

    print(
        "="
        * 100
    )

    print(
        f"Replicates: "
        f"{args.replicates}"
    )

    print(
        f"N per condition: "
        f"{args.n}"
    )

    for replication in range(
        args.replicates
    ):
        (
            values,
            contrasts,
            checks,
        ) = run_replication(
            args,
            replication,
        )

        all_values.append(
            values
        )

        all_contrasts.append(
            contrasts
        )

        invariant_checks.append(
            checks
        )

        if (
            (
                replication
                + 1
            )
            % 10
            == 0
            or replication
            == 0
        ):
            print(
                f"Completed "
                f"{replication + 1}/"
                f"{args.replicates}"
            )

    values = pd.concat(
        all_values,
        ignore_index=True,
    )

    contrasts = pd.concat(
        all_contrasts,
        ignore_index=True,
    )

    summary = (
        summarize_contrasts(
            contrasts
        )
    )

    comparison = (
        build_direct_comparison(
            contrasts
        )
    )

    values.to_csv(
        output_dir
        / "condition_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    contrasts.to_csv(
        output_dir
        / "paired_contrasts.csv.gz",
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        output_dir
        / "contrast_summary.csv",
        index=False,
    )

    comparison.to_csv(
        output_dir
        / "direct_baseline_comparison.csv",
        index=False,
    )

    invariants = pd.DataFrame(
        invariant_checks
    )

    numeric_finite = bool(
        values
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .notna()
        .all()
        .all()
    )

    observation_reference_max = float(
        contrasts.loc[
            contrasts[
                "condition"
            ].eq(
                "observation_only"
            ),
            "delta_reference_ici",
        ]
        .abs()
        .max()
    )

    outcome_exit_max = float(
        contrasts.loc[
            contrasts[
                "condition"
            ].eq(
                "outcome_only"
            ),
            "delta_exit_before_horizon_fraction",
        ]
        .abs()
        .max()
    )

    validation = pd.DataFrame(
        [
            {
                "check": (
                    "expected_condition_rows"
                ),
                "value": len(
                    values
                ),
                "expected": (
                    args.replicates
                    * len(
                        CONDITIONS
                    )
                ),
                "passed": (
                    len(
                        values
                    )
                    == args.replicates
                    * len(
                        CONDITIONS
                    )
                ),
            },
            {
                "check": (
                    "expected_contrast_rows"
                ),
                "value": len(
                    contrasts
                ),
                "expected": (
                    args.replicates
                    * 3
                ),
                "passed": (
                    len(
                        contrasts
                    )
                    == args.replicates
                    * 3
                ),
            },
            {
                "check": (
                    "all_controlled_invariance_checks"
                ),
                "value": bool(
                    invariants
                    .all()
                    .all()
                ),
                "expected": True,
                "passed": bool(
                    invariants
                    .all()
                    .all()
                ),
            },
            {
                "check": (
                    "observation_only_reference_ici_invariant"
                ),
                "value": (
                    observation_reference_max
                ),
                "expected": 1e-12,
                "passed": (
                    observation_reference_max
                    <= 1e-12
                ),
            },
            {
                "check": (
                    "outcome_only_exit_process_invariant"
                ),
                "value": (
                    outcome_exit_max
                ),
                "expected": 1e-12,
                "passed": (
                    outcome_exit_max
                    <= 1e-12
                ),
            },
            {
                "check": (
                    "all_numeric_outputs_finite"
                ),
                "value": (
                    numeric_finite
                ),
                "expected": True,
                "passed": (
                    numeric_finite
                ),
            },
        ]
    )

    validation.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    make_figure(
        comparison,
        output_dir,
    )

    passed = bool(
        validation[
            "passed"
        ].all()
    )

    status = (
        "phase7c_direct_baseline_passed"
        if passed
        else
        "phase7c_direct_baseline_failed"
    )

    (
        output_dir
        / "phase7c_status.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    display = summary[
        [
            "condition",
            "delta_reference_ici_mean",
            "delta_complete_case_ici_mean",
            "delta_kaplan_meier_ici_mean",
            "delta_oracle_or_ici_mean",
            "delta_oracle_ipcw_ici_mean",
            (
                "delta_exit_before_"
                "horizon_fraction_mean"
            ),
        ]
    ]

    print(
        "\n===== "
        "PAIRED SOURCE-ATTRIBUTION "
        "CONTRASTS ====="
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n===== "
        "DIRECT BASELINE COMPARISON "
        "====="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\n===== VALIDATION ====="
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        f"\nSTATUS: {status}"
    )

    print(
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
