"""Phase 5B: continuous latent event-censoring dependence sweep.

The verified Phase 3D simulator and nuisance estimators are reused.
Only the latent-U coefficient in the censoring hazard is varied.

At gamma:
    gamma < 0: event and censoring latent effects oppose each other.
    gamma = 0: no residual event-censoring dependence through U.
    gamma > 0: event and censoring latent effects point together.

Common random numbers are used across gamma values within each
replication.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from actionaudit.discrete_survival import (
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
    predict_hazard_grid,
)
from actionaudit.km import reference_quantile_edges
from actionaudit.simulation import (
    scenario_config,
    simulate_cohort,
)

from phase3d_learned_nuisance_simulation import (
    ORACLE_FEATURES,
    WORKING_FEATURES,
    fit_hazard_model,
    scalar_value,
)


DEFAULT_GAMMAS = (
    "-1.50,-1.25,-1.00,-0.90,-0.75,-0.50,-0.25,"
    "0.00,0.25,0.50,0.75,0.90,1.00,1.25,1.50"
)

PRACTICAL_MAE_METRICS = [
    "complete_case_mae",
    "kaplan_meier_mae",
    "learned_working_or_mae",
    "learned_working_ipcw_mae",
]

PRACTICAL_SIGNED_METRICS = [
    "complete_case_signed_gap",
    "kaplan_meier_signed_gap",
    "learned_working_or_signed_gap",
    "learned_working_ipcw_signed_gap",
]

SUMMARY_METRICS = [
    "reference_event_rate",
    "observed_fraction",
    "hidden_event_rate",
    "latent_risk_correlation",
    "complete_case_mae",
    "complete_case_signed_gap",
    "kaplan_meier_mae",
    "kaplan_meier_signed_gap",
    "true_oracle_or_mae",
    "true_oracle_or_signed_gap",
    "true_oracle_ipcw_mae",
    "true_oracle_ipcw_signed_gap",
    "learned_oracle_or_mae",
    "learned_oracle_or_signed_gap",
    "learned_oracle_ipcw_mae",
    "learned_oracle_ipcw_signed_gap",
    "learned_working_or_mae",
    "learned_working_or_signed_gap",
    "learned_working_ipcw_mae",
    "learned_working_ipcw_signed_gap",
    "learned_oracle_ipcw_ess",
    "learned_working_ipcw_ess",
    "learned_oracle_fraction_clipped",
    "learned_working_fraction_clipped",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously vary latent event-censoring dependence "
            "and estimate calibration-audit crossover curves."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5b_continuous_dependence_sweep"
        ),
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--development-n",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--test-n",
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
        default=5,
    )

    parser.add_argument(
        "--gammas",
        default=DEFAULT_GAMMAS,
    )

    parser.add_argument(
        "--clips",
        default="0.01,0.05,0.10",
    )

    parser.add_argument(
        "--primary-clip",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--logistic-c",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Mark the run as preliminary regardless "
            "of its validation results."
        ),
    )

    return parser.parse_args()


def parse_float_list(
    text: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> list[float]:
    values = sorted(
        {
            float(value.strip())
            for value in text.split(",")
            if value.strip()
        }
    )

    if not values:
        raise ValueError(
            "At least one numeric value is required"
        )

    if (
        lower is not None
        and any(
            value <= lower
            for value in values
        )
    ):
        raise ValueError(
            f"All values must exceed {lower}"
        )

    if (
        upper is not None
        and any(
            value >= upper
            for value in values
        )
    ):
        raise ValueError(
            f"All values must be below {upper}"
        )

    return values


def risk_correlation(
    event_hazards: np.ndarray,
    censor_hazards: np.ndarray,
) -> float:
    """Correlation between true cumulative event and censoring risks."""

    event_risk = (
        1.0
        - np.prod(
            1.0 - event_hazards,
            axis=1,
        )
    )

    censor_risk = (
        1.0
        - np.prod(
            1.0 - censor_hazards,
            axis=1,
        )
    )

    if (
        np.std(event_risk) <= 0
        or np.std(censor_risk) <= 0
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            event_risk,
            censor_risk,
        )[0, 1]
    )


def evaluate_gamma_replication(
    *,
    gamma: float,
    replication: int,
    development_n: int,
    test_n: int,
    horizon: int,
    n_bins: int,
    clips: list[float],
    logistic_c: float,
    seed: int,
) -> list[dict[str, object]]:
    """Fit all nuisance models for one gamma and replication."""

    # The seed deliberately excludes gamma. Therefore all gamma
    # values within a replication share the same X1, X2, U,
    # event uniforms, and censoring uniforms.
    base_seed = (
        seed
        + 10 * replication
    )

    development_base = scenario_config(
        "residual_same_direction",
        n=development_n,
        horizon=horizon,
        seed=base_seed + 1,
    )

    test_base = scenario_config(
        "residual_same_direction",
        n=test_n,
        horizon=horizon,
        seed=base_seed + 2,
    )

    development_config = replace(
        development_base,
        scenario=(
            f"continuous_gamma_{gamma:+.2f}"
        ),
        censor_u=gamma,
    )

    test_config = replace(
        test_base,
        scenario=(
            f"continuous_gamma_{gamma:+.2f}"
        ),
        censor_u=gamma,
    )

    development_output = simulate_cohort(
        development_config
    )

    test_output = simulate_cohort(
        test_config
    )

    development_frame = (
        development_output.frame
    )

    test_frame = (
        test_output.frame
    )

    working_event_model = fit_hazard_model(
        development_frame,
        WORKING_FEATURES,
        horizon=horizon,
        target="event",
        c_value=logistic_c,
        seed=base_seed + 101,
    )

    working_censor_model = fit_hazard_model(
        development_frame,
        WORKING_FEATURES,
        horizon=horizon,
        target="censor",
        c_value=logistic_c,
        seed=base_seed + 102,
    )

    oracle_event_model = fit_hazard_model(
        development_frame,
        ORACLE_FEATURES,
        horizon=horizon,
        target="event",
        c_value=logistic_c,
        seed=base_seed + 103,
    )

    oracle_censor_model = fit_hazard_model(
        development_frame,
        ORACLE_FEATURES,
        horizon=horizon,
        target="censor",
        c_value=logistic_c,
        seed=base_seed + 104,
    )

    working_event_hazards = predict_hazard_grid(
        working_event_model,
        test_frame,
        WORKING_FEATURES,
        horizon=horizon,
    )

    working_censor_hazards = predict_hazard_grid(
        working_censor_model,
        test_frame,
        WORKING_FEATURES,
        horizon=horizon,
    )

    learned_oracle_event_hazards = (
        predict_hazard_grid(
            oracle_event_model,
            test_frame,
            ORACLE_FEATURES,
            horizon=horizon,
        )
    )

    learned_oracle_censor_hazards = (
        predict_hazard_grid(
            oracle_censor_model,
            test_frame,
            ORACLE_FEATURES,
            horizon=horizon,
        )
    )

    bin_edges = reference_quantile_edges(
        test_frame[
            "prediction"
        ].to_numpy(),
        n_bins,
    )

    latent_correlation = risk_correlation(
        test_output.true_event_hazards,
        test_output.true_censor_hazards,
    )

    reference_event_rate = float(
        test_frame[
            "outcome"
        ].mean()
    )

    observed_fraction = float(
        test_frame[
            "observed"
        ].mean()
    )

    hidden_event_rate = float(
        test_frame[
            "hidden_event"
        ].mean()
    )

    rows: list[
        dict[str, object]
    ] = []

    for clip in clips:
        true_oracle_frame = (
            construct_survival_nuisance_predictions(
                test_frame,
                event_hazards=(
                    test_output.true_event_hazards
                ),
                censor_hazards=(
                    test_output.true_censor_hazards
                ),
                horizon=horizon,
                censor_survival_clip=clip,
            )
        )

        true_oracle_result = (
            evaluate_survival_estimators(
                true_oracle_frame,
                horizon=horizon,
                bin_edges=bin_edges,
                censor_survival_clip=clip,
            )
        )

        learned_oracle_frame = (
            construct_survival_nuisance_predictions(
                test_frame,
                event_hazards=(
                    learned_oracle_event_hazards
                ),
                censor_hazards=(
                    learned_oracle_censor_hazards
                ),
                horizon=horizon,
                censor_survival_clip=clip,
            )
        )

        learned_oracle_result = (
            evaluate_survival_estimators(
                learned_oracle_frame,
                horizon=horizon,
                bin_edges=bin_edges,
                censor_survival_clip=clip,
            )
        )

        learned_working_frame = (
            construct_survival_nuisance_predictions(
                test_frame,
                event_hazards=(
                    working_event_hazards
                ),
                censor_hazards=(
                    working_censor_hazards
                ),
                horizon=horizon,
                censor_survival_clip=clip,
            )
        )

        learned_working_result = (
            evaluate_survival_estimators(
                learned_working_frame,
                horizon=horizon,
                bin_edges=bin_edges,
                censor_survival_clip=clip,
            )
        )

        true_scalar = (
            true_oracle_result.scalar_summary
        )

        oracle_scalar = (
            learned_oracle_result.scalar_summary
        )

        working_scalar = (
            learned_working_result.scalar_summary
        )

        rows.append(
            {
                "gamma": gamma,
                "replication": replication,
                "development_seed": (
                    base_seed + 1
                ),
                "test_seed": (
                    base_seed + 2
                ),
                "clip": clip,
                "development_n": development_n,
                "test_n": test_n,
                "horizon": horizon,
                "n_bins": n_bins,
                "event_u": float(
                    test_config.event_u
                ),
                "censor_u": float(
                    test_config.censor_u
                ),
                "reference_event_rate": (
                    reference_event_rate
                ),
                "observed_fraction": (
                    observed_fraction
                ),
                "hidden_event_rate": (
                    hidden_event_rate
                ),
                "latent_risk_correlation": (
                    latent_correlation
                ),
                "complete_case_mae": scalar_value(
                    true_scalar,
                    "complete_case_mae_to_reference",
                ),
                "complete_case_signed_gap": (
                    scalar_value(
                        true_scalar,
                        "complete_case_signed_gap",
                    )
                ),
                "kaplan_meier_mae": scalar_value(
                    true_scalar,
                    "kaplan_meier_mae_to_reference",
                ),
                "kaplan_meier_signed_gap": (
                    scalar_value(
                        true_scalar,
                        "kaplan_meier_signed_gap",
                    )
                ),
                "true_oracle_or_mae": scalar_value(
                    true_scalar,
                    "survival_or_mae_to_reference",
                ),
                "true_oracle_or_signed_gap": (
                    scalar_value(
                        true_scalar,
                        "survival_or_signed_gap",
                    )
                ),
                "true_oracle_ipcw_mae": scalar_value(
                    true_scalar,
                    "ipcw_event_mae_to_reference",
                ),
                "true_oracle_ipcw_signed_gap": (
                    scalar_value(
                        true_scalar,
                        "ipcw_event_signed_gap",
                    )
                ),
                "learned_oracle_or_mae": (
                    scalar_value(
                        oracle_scalar,
                        "survival_or_mae_to_reference",
                    )
                ),
                "learned_oracle_or_signed_gap": (
                    scalar_value(
                        oracle_scalar,
                        "survival_or_signed_gap",
                    )
                ),
                "learned_oracle_ipcw_mae": (
                    scalar_value(
                        oracle_scalar,
                        "ipcw_event_mae_to_reference",
                    )
                ),
                "learned_oracle_ipcw_signed_gap": (
                    scalar_value(
                        oracle_scalar,
                        "ipcw_event_signed_gap",
                    )
                ),
                "learned_working_or_mae": (
                    scalar_value(
                        working_scalar,
                        "survival_or_mae_to_reference",
                    )
                ),
                "learned_working_or_signed_gap": (
                    scalar_value(
                        working_scalar,
                        "survival_or_signed_gap",
                    )
                ),
                "learned_working_ipcw_mae": (
                    scalar_value(
                        working_scalar,
                        "ipcw_event_mae_to_reference",
                    )
                ),
                "learned_working_ipcw_signed_gap": (
                    scalar_value(
                        working_scalar,
                        "ipcw_event_signed_gap",
                    )
                ),
                "learned_oracle_ipcw_ess": float(
                    learned_oracle_result.diagnostics[
                        "event_weight_effective_sample_size"
                    ]
                ),
                "learned_working_ipcw_ess": float(
                    learned_working_result.diagnostics[
                        "event_weight_effective_sample_size"
                    ]
                ),
                "learned_oracle_fraction_clipped": float(
                    learned_oracle_result.diagnostics[
                        "fraction_event_g_clipped"
                    ]
                ),
                "learned_working_fraction_clipped": float(
                    learned_working_result.diagnostics[
                        "fraction_event_g_clipped"
                    ]
                ),
            }
        )

    return rows


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Produce long-form Monte Carlo summaries."""

    rows: list[
        dict[str, object]
    ] = []

    grouped = results.groupby(
        [
            "gamma",
            "clip",
        ],
        sort=True,
    )

    for (
        gamma,
        clip,
    ), group in grouped:
        for metric in SUMMARY_METRICS:
            values = (
                group[
                    metric
                ]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            rows.append(
                {
                    "gamma": gamma,
                    "clip": clip,
                    "metric": metric,
                    "mean": float(
                        values.mean()
                    ),
                    "standard_deviation": float(
                        values.std(
                            ddof=1
                        )
                    ),
                    "q025": float(
                        values.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        values.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        values.quantile(
                            0.975
                        )
                    ),
                    "n_valid_replications": int(
                        len(values)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def create_winner_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Identify the lowest-MAE practical estimator at each point."""

    practical = summary.loc[
        summary[
            "metric"
        ].isin(
            PRACTICAL_MAE_METRICS
        ),
        [
            "gamma",
            "clip",
            "metric",
            "mean",
        ],
    ]

    wide = practical.pivot(
        index=[
            "gamma",
            "clip",
        ],
        columns="metric",
        values="mean",
    ).reset_index()

    wide.columns.name = None

    wide[
        "best_practical_estimator"
    ] = wide[
        PRACTICAL_MAE_METRICS
    ].idxmin(
        axis=1
    )

    wide[
        "best_practical_mae"
    ] = wide[
        PRACTICAL_MAE_METRICS
    ].min(
        axis=1
    )

    sorted_values = np.sort(
        wide[
            PRACTICAL_MAE_METRICS
        ].to_numpy(),
        axis=1,
    )

    wide[
        "runner_up_mae"
    ] = sorted_values[
        :,
        1,
    ]

    wide[
        "winner_margin"
    ] = (
        wide[
            "runner_up_mae"
        ]
        - wide[
            "best_practical_mae"
        ]
    )

    return wide


def create_anchor_table(
    summary: pd.DataFrame,
    *,
    primary_clip: float,
) -> pd.DataFrame:
    """Extract the original opposite, zero and same-direction anchors."""

    anchor_values = [
        -0.90,
        0.00,
        0.90,
    ]

    metrics = (
        PRACTICAL_MAE_METRICS
        + PRACTICAL_SIGNED_METRICS
        + [
            "learned_oracle_or_mae",
            "latent_risk_correlation",
            "observed_fraction",
            "hidden_event_rate",
        ]
    )

    anchor = summary.loc[
        np.isclose(
            summary[
                "clip"
            ],
            primary_clip,
        )
        & summary[
            "gamma"
        ].isin(
            anchor_values
        )
        & summary[
            "metric"
        ].isin(
            metrics
        )
    ].copy()

    return anchor.sort_values(
        [
            "gamma",
            "metric",
        ]
    )


def validate_sweep(
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    winners: pd.DataFrame,
    gammas: list[float],
    clips: list[float],
    replicates: int,
    primary_clip: float,
) -> tuple[
    pd.DataFrame,
    str,
    dict[str, object],
]:
    """Validate simulator behavior and estimator crossovers."""

    expected_rows = (
        len(gammas)
        * len(clips)
        * replicates
    )

    primary_summary = summary.loc[
        np.isclose(
            summary[
                "clip"
            ],
            primary_clip,
        )
    ]

    primary_winners = winners.loc[
        np.isclose(
            winners[
                "clip"
            ],
            primary_clip,
        )
    ]

    oracle_or = primary_summary.loc[
        primary_summary[
            "metric"
        ].eq(
            "learned_oracle_or_mae"
        )
    ]

    maximum_oracle_or_mean = float(
        oracle_or[
            "mean"
        ].max()
    )

    unique_winners = sorted(
        primary_winners[
            "best_practical_estimator"
        ].unique()
        .tolist()
    )

    working_ranges: dict[
        str,
        float
    ] = {}

    for metric in PRACTICAL_MAE_METRICS:
        metric_rows = primary_summary.loc[
            primary_summary[
                "metric"
            ].eq(
                metric
            )
        ]

        working_ranges[
            metric
        ] = float(
            metric_rows[
                "mean"
            ].max()
            - metric_rows[
                "mean"
            ].min()
        )

    maximum_practical_range = max(
        working_ranges.values()
    )

    signed_ranges: dict[
        str,
        float
    ] = {}

    for metric in PRACTICAL_SIGNED_METRICS:
        metric_rows = primary_summary.loc[
            primary_summary[
                "metric"
            ].eq(
                metric
            )
        ]

        signed_ranges[
            metric
        ] = float(
            metric_rows[
                "mean"
            ].max()
            - metric_rows[
                "mean"
            ].min()
        )

    maximum_signed_range = max(
        signed_ranges.values()
    )

    required_anchors_present = all(
        any(
            np.isclose(
                gamma,
                anchor,
            )
            for gamma in gammas
        )
        for anchor in (
            -0.90,
            0.00,
            0.90,
        )
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "all requested replication rows completed"
                ),
                "value": len(
                    results
                ),
                "threshold": expected_rows,
                "comparison": "==",
                "passed": (
                    len(results)
                    == expected_rows
                ),
            },
            {
                "check": (
                    "original dependence anchors are included"
                ),
                "value": (
                    required_anchors_present
                ),
                "threshold": True,
                "comparison": "==",
                "passed": (
                    required_anchors_present
                ),
            },
            {
                "check": (
                    "learned oracle OR remains accurate"
                ),
                "value": (
                    maximum_oracle_or_mean
                ),
                "threshold": 0.04,
                "comparison": "<",
                "passed": (
                    maximum_oracle_or_mean
                    < 0.04
                ),
            },
            {
                "check": (
                    "practical winner changes across gamma"
                ),
                "value": len(
                    unique_winners
                ),
                "threshold": 2,
                "comparison": ">=",
                "passed": (
                    len(
                        unique_winners
                    )
                    >= 2
                ),
            },
            {
                "check": (
                    "practical MAE varies materially across gamma"
                ),
                "value": (
                    maximum_practical_range
                ),
                "threshold": 0.03,
                "comparison": ">",
                "passed": (
                    maximum_practical_range
                    > 0.03
                ),
            },
            {
                "check": (
                    "signed distortion changes materially "
                    "across gamma"
                ),
                "value": (
                    maximum_signed_range
                ),
                "threshold": 0.05,
                "comparison": ">",
                "passed": (
                    maximum_signed_range
                    > 0.05
                ),
            },
        ]
    )

    status = (
        "phase5b_continuous_estimator_crossover_supported"
        if checks[
            "passed"
        ].all()
        else "phase5b_requires_review"
    )

    evidence = {
        "primary_clip": (
            primary_clip
        ),
        "maximum_learned_oracle_or_mean_mae": (
            maximum_oracle_or_mean
        ),
        "unique_practical_winners": (
            unique_winners
        ),
        "practical_mae_ranges": (
            working_ranges
        ),
        "signed_gap_ranges": (
            signed_ranges
        ),
        "maximum_practical_mae_range": (
            maximum_practical_range
        ),
        "maximum_signed_gap_range": (
            maximum_signed_range
        ),
        "status": status,
    }

    return (
        checks,
        status,
        evidence,
    )


def plot_metric_curves(
    *,
    summary: pd.DataFrame,
    metrics: list[str],
    primary_clip: float,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    """Plot Monte Carlo means and central replication ranges."""

    selected = summary.loc[
        np.isclose(
            summary[
                "clip"
            ],
            primary_clip,
        )
        & summary[
            "metric"
        ].isin(
            metrics
        )
    ]

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    for metric in metrics:
        current = (
            selected.loc[
                selected[
                    "metric"
                ].eq(
                    metric
                )
            ]
            .sort_values(
                "gamma"
            )
        )

        x = current[
            "gamma"
        ].to_numpy(
            dtype=float
        )

        mean = current[
            "mean"
        ].to_numpy(
            dtype=float
        )

        lower = current[
            "q025"
        ].to_numpy(
            dtype=float
        )

        upper = current[
            "q975"
        ].to_numpy(
            dtype=float
        )

        line = axis.plot(
            x,
            mean,
            marker="o",
            label=metric,
        )[0]

        axis.fill_between(
            x,
            lower,
            upper,
            alpha=0.12,
            color=line.get_color(),
        )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    axis.axvline(
        -0.90,
        linestyle=":",
        linewidth=1,
    )

    axis.axvline(
        0.90,
        linestyle=":",
        linewidth=1,
    )

    axis.set_xlabel(
        "Latent censoring coefficient γ"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.legend(
        fontsize=8
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_arguments()

    gammas = parse_float_list(
        args.gammas
    )

    clips = parse_float_list(
        args.clips,
        lower=0.0,
        upper=1.0,
    )

    if not any(
        np.isclose(
            clip,
            args.primary_clip,
        )
        for clip in clips
    ):
        raise ValueError(
            "primary-clip must be present in --clips"
        )

    if args.replicates < 1:
        raise ValueError(
            "replicates must be positive"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[
        dict[str, object]
    ] = []

    print(
        "Phase 5B continuous dependence sweep"
    )

    print(
        f"  gammas={len(gammas)}"
    )

    print(
        f"  clips={clips}"
    )

    print(
        f"  replications={args.replicates}"
    )

    for replication in range(
        args.replicates
    ):
        for gamma in gammas:
            rows.extend(
                evaluate_gamma_replication(
                    gamma=gamma,
                    replication=replication,
                    development_n=(
                        args.development_n
                    ),
                    test_n=args.test_n,
                    horizon=args.horizon,
                    n_bins=args.n_bins,
                    clips=clips,
                    logistic_c=(
                        args.logistic_c
                    ),
                    seed=args.seed,
                )
            )

        if (
            replication + 1
        ) % 5 == 0:
            print(
                f"Completed "
                f"{replication + 1}/"
                f"{args.replicates}"
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        output_dir
        / "replication_results.csv",
        index=False,
    )

    summary = summarize_results(
        results
    )

    summary.to_csv(
        output_dir
        / "monte_carlo_summary_long.csv",
        index=False,
    )

    winners = create_winner_table(
        summary
    )

    winners.to_csv(
        output_dir
        / "practical_estimator_winners.csv",
        index=False,
    )

    anchor_table = create_anchor_table(
        summary,
        primary_clip=args.primary_clip,
    )

    anchor_table.to_csv(
        output_dir
        / "anchor_comparison.csv",
        index=False,
    )

    (
        checks,
        status,
        evidence,
    ) = validate_sweep(
        results=results,
        summary=summary,
        winners=winners,
        gammas=gammas,
        clips=clips,
        replicates=args.replicates,
        primary_clip=args.primary_clip,
    )

    preliminary_status = status

    if args.smoke:
        status = (
            "phase5b_smoke_completed_not_decisive"
        )

        evidence[
            "preliminary_smoke_status"
        ] = preliminary_status

        evidence[
            "status"
        ] = status

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    (
        output_dir
        / "phase5b_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    metadata = {
        "phase": "5B",
        "experiment": (
            "Continuous latent event-censoring "
            "dependence sweep"
        ),
        "gamma_definition": (
            "Coefficient of latent U in the daily "
            "censoring hazard"
        ),
        "event_u_fixed": 0.85,
        "censor_x1_fixed": -0.75,
        "censor_x2_fixed": 1.00,
        "censor_intercept_fixed": -4.10,
        "censor_time_slope_fixed": 0.012,
        "gammas": gammas,
        "clips": clips,
        "primary_clip": (
            args.primary_clip
        ),
        "replicates": (
            args.replicates
        ),
        "development_n": (
            args.development_n
        ),
        "test_n": args.test_n,
        "horizon": args.horizon,
        "n_bins": args.n_bins,
        "logistic_c": (
            args.logistic_c
        ),
        "common_random_numbers_across_gamma": True,
        "working_features": (
            WORKING_FEATURES
        ),
        "oracle_features": (
            ORACLE_FEATURES
        ),
        "smoke": bool(
            args.smoke
        ),
        "validation_evidence": (
            evidence
        ),
        "status": status,
        "seed": args.seed,
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
        },
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

    plot_metric_curves(
        summary=summary,
        metrics=(
            PRACTICAL_MAE_METRICS
        ),
        primary_clip=args.primary_clip,
        output_path=(
            output_dir
            / "practical_mae_crossover.png"
        ),
        ylabel=(
            "MAE to full-follow-up "
            "calibration curve"
        ),
        title=(
            "Practical estimator performance "
            "across latent dependence"
        ),
    )

    plot_metric_curves(
        summary=summary,
        metrics=(
            PRACTICAL_SIGNED_METRICS
        ),
        primary_clip=args.primary_clip,
        output_path=(
            output_dir
            / "practical_signed_gap_sweep.png"
        ),
        ylabel="Signed calibration gap",
        title=(
            "Direction of calibration distortion "
            "across latent dependence"
        ),
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
        "\nPrimary-clip practical winners"
    )

    print(
        winners.loc[
            np.isclose(
                winners[
                    "clip"
                ],
                args.primary_clip,
            ),
            [
                "gamma",
                "best_practical_estimator",
                "best_practical_mae",
                "runner_up_mae",
                "winner_margin",
            ],
        ].to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
