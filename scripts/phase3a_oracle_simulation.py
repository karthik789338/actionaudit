"""Phase 3A: controlled oracle censoring simulations."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from actionaudit.discrete_survival import (
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
)
from actionaudit.km import (
    reference_quantile_edges,
)
from actionaudit.simulation import (
    scenario_config,
    simulate_cohort,
)


SCENARIOS = [
    "independent",
    "observed_dependent",
    "residual_same_direction",
    "residual_opposite_direction",
    "positivity_stress",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark calibration estimators under known "
            "censoring mechanisms."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase3a_oracle_simulation",
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
        default=5,
    )

    parser.add_argument(
        "--clips",
        default="0.01,0.05,0.10",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def parse_clips(
    text: str,
) -> list[float]:
    clips = [
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    ]

    if not clips:
        raise ValueError(
            "At least one clipping threshold is required"
        )

    for clip in clips:
        if not 0 < clip < 1:
            raise ValueError(
                "Every clipping threshold must lie in (0, 1)"
            )

    return sorted(
        set(clips)
    )


def evaluate_one_replication(
    *,
    scenario: str,
    replication: int,
    n: int,
    horizon: int,
    n_bins: int,
    clips: list[float],
    seed: int,
) -> list[dict[str, object]]:
    """Run one known-data-generating-process replication."""

    scenario_seed = (
        seed
        + 100000 * SCENARIOS.index(
            scenario
        )
        + replication
    )

    config = scenario_config(
        scenario,
        n=n,
        horizon=horizon,
        seed=scenario_seed,
    )

    simulated = simulate_cohort(
        config
    )

    frame = simulated.frame

    bin_edges = reference_quantile_edges(
        frame[
            "prediction"
        ].to_numpy(),
        n_bins,
    )

    observed_fraction = float(
        frame["observed"].mean()
    )

    reference_event_rate = float(
        frame["outcome"].mean()
    )

    hidden_event_rate = (
        float(
            frame.loc[
                frame[
                    "observed"
                ].eq(0),
                "outcome",
            ].mean()
        )
        if frame[
            "observed"
        ].eq(0).any()
        else float("nan")
    )

    rows: list[dict[str, object]] = []

    for clip in clips:
        full_oracle_frame = (
            construct_survival_nuisance_predictions(
                frame,
                event_hazards=(
                    simulated.true_event_hazards
                ),
                censor_hazards=(
                    simulated.true_censor_hazards
                ),
                horizon=horizon,
                censor_survival_clip=clip,
            )
        )

        full_oracle_result = (
            evaluate_survival_estimators(
                full_oracle_frame,
                horizon=horizon,
                bin_edges=bin_edges,
                censor_survival_clip=clip,
            )
        )

        working_frame = (
            construct_survival_nuisance_predictions(
                frame,
                event_hazards=(
                    simulated.working_event_hazards
                ),
                censor_hazards=(
                    simulated.working_censor_hazards
                ),
                horizon=horizon,
                censor_survival_clip=clip,
            )
        )

        working_result = (
            evaluate_survival_estimators(
                working_frame,
                horizon=horizon,
                bin_edges=bin_edges,
                censor_survival_clip=clip,
            )
        )

        full_scalar = (
            full_oracle_result.scalar_summary
        )

        working_scalar = (
            working_result.scalar_summary
        )

        rows.append(
            {
                "scenario": scenario,
                "replication": replication,
                "seed": scenario_seed,
                "clip": clip,
                "n": n,
                "horizon": horizon,
                "n_bins": n_bins,
                "reference_event_rate": (
                    reference_event_rate
                ),
                "observed_fraction": (
                    observed_fraction
                ),
                "hidden_event_rate": (
                    hidden_event_rate
                ),
                "complete_case_mae": float(
                    full_scalar[
                        "complete_case_mae_to_reference"
                    ]
                ),
                "kaplan_meier_mae": float(
                    full_scalar[
                        "kaplan_meier_mae_to_reference"
                    ]
                ),
                "full_oracle_or_mae": float(
                    full_scalar[
                        "survival_or_mae_to_reference"
                    ]
                ),
                "full_oracle_ipcw_mae": float(
                    full_scalar[
                        "ipcw_event_mae_to_reference"
                    ]
                ),
                "working_or_mae": float(
                    working_scalar[
                        "survival_or_mae_to_reference"
                    ]
                ),
                "working_ipcw_mae": float(
                    working_scalar[
                        "ipcw_event_mae_to_reference"
                    ]
                ),
                "complete_case_signed_gap": float(
                    full_scalar[
                        "complete_case_signed_gap"
                    ]
                ),
                "kaplan_meier_signed_gap": float(
                    full_scalar[
                        "kaplan_meier_signed_gap"
                    ]
                ),
                "full_oracle_or_signed_gap": float(
                    full_scalar[
                        "survival_or_signed_gap"
                    ]
                ),
                "full_oracle_ipcw_signed_gap": float(
                    full_scalar[
                        "ipcw_event_signed_gap"
                    ]
                ),
                "working_or_signed_gap": float(
                    working_scalar[
                        "survival_or_signed_gap"
                    ]
                ),
                "working_ipcw_signed_gap": float(
                    working_scalar[
                        "ipcw_event_signed_gap"
                    ]
                ),
                "working_or_minus_oracle_or": float(
                    working_scalar[
                        "survival_or_mae_to_reference"
                    ]
                    - full_scalar[
                        "survival_or_mae_to_reference"
                    ]
                ),
                "working_ipcw_minus_oracle_ipcw": float(
                    working_scalar[
                        "ipcw_event_mae_to_reference"
                    ]
                    - full_scalar[
                        "ipcw_event_mae_to_reference"
                    ]
                ),
                "full_oracle_fraction_events_clipped": float(
                    full_oracle_result.diagnostics[
                        "fraction_event_g_clipped"
                    ]
                ),
                "working_fraction_events_clipped": float(
                    working_result.diagnostics[
                        "fraction_event_g_clipped"
                    ]
                ),
                "full_oracle_ipcw_ess": float(
                    full_oracle_result.diagnostics[
                        "event_weight_effective_sample_size"
                    ]
                ),
                "working_ipcw_ess": float(
                    working_result.diagnostics[
                        "event_weight_effective_sample_size"
                    ]
                ),
            }
        )

    return rows


def summarize_results(
    replications: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Monte Carlo distributions by scenario and clipping."""

    metric_columns = [
        "reference_event_rate",
        "observed_fraction",
        "hidden_event_rate",
        "complete_case_mae",
        "kaplan_meier_mae",
        "full_oracle_or_mae",
        "full_oracle_ipcw_mae",
        "working_or_mae",
        "working_ipcw_mae",
        "complete_case_signed_gap",
        "kaplan_meier_signed_gap",
        "full_oracle_or_signed_gap",
        "full_oracle_ipcw_signed_gap",
        "working_or_signed_gap",
        "working_ipcw_signed_gap",
        "working_or_minus_oracle_or",
        "working_ipcw_minus_oracle_ipcw",
        "full_oracle_fraction_events_clipped",
        "working_fraction_events_clipped",
        "full_oracle_ipcw_ess",
        "working_ipcw_ess",
    ]

    rows: list[dict[str, object]] = []

    for (
        scenario,
        clip,
    ), group in replications.groupby(
        [
            "scenario",
            "clip",
        ],
        sort=True,
    ):
        for metric in metric_columns:
            values = (
                group[metric]
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
                    "scenario": scenario,
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
                    "n_replications": int(
                        len(values)
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_wide_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact table containing mean estimator errors."""

    selected_metrics = [
        "observed_fraction",
        "complete_case_mae",
        "kaplan_meier_mae",
        "full_oracle_or_mae",
        "full_oracle_ipcw_mae",
        "working_or_mae",
        "working_ipcw_mae",
        "working_ipcw_ess",
    ]

    selected = summary.loc[
        summary[
            "metric"
        ].isin(
            selected_metrics
        )
    ]

    wide = (
        selected.pivot_table(
            index=[
                "scenario",
                "clip",
            ],
            columns="metric",
            values="mean",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    return wide


def main() -> None:
    args = parse_arguments()

    if args.replicates < 1:
        raise ValueError(
            "replicates must be at least 1"
        )

    clips = parse_clips(
        args.clips
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        print(
            f"Running scenario: {scenario}"
        )

        for replication in range(
            args.replicates
        ):
            rows.extend(
                evaluate_one_replication(
                    scenario=scenario,
                    replication=replication,
                    n=args.n,
                    horizon=args.horizon,
                    n_bins=args.n_bins,
                    clips=clips,
                    seed=args.seed,
                )
            )

            if (
                replication + 1
            ) % 10 == 0:
                print(
                    f"  completed "
                    f"{replication + 1}/"
                    f"{args.replicates}"
                )

    replications = pd.DataFrame(
        rows
    )

    summary = summarize_results(
        replications
    )

    wide_summary = create_wide_summary(
        summary
    )

    replications.to_csv(
        output_dir
        / "replication_results.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "monte_carlo_summary_long.csv",
        index=False,
    )

    wide_summary.to_csv(
        output_dir
        / "monte_carlo_summary_wide.csv",
        index=False,
    )

    metadata = {
        "phase": "3A",
        "design": (
            "controlled discrete-time survival and "
            "discharge-censoring simulation"
        ),
        "scenarios": SCENARIOS,
        "replicates": int(
            args.replicates
        ),
        "n_per_replication": int(
            args.n
        ),
        "horizon": int(
            args.horizon
        ),
        "n_bins": int(
            args.n_bins
        ),
        "clipping_thresholds": clips,
        "prediction_model": (
            "fixed score omitting X2 and latent U"
        ),
        "working_nuisance_models": (
            "correct observed-covariate formulas but "
            "omit latent U"
        ),
        "event_censor_tie_rule": (
            "event occurs before censoring on tied day"
        ),
        "random_seed": int(
            args.seed
        ),
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

    print(
        "\nPhase 3A complete"
    )

    print(
        "\nMean Monte Carlo results"
    )

    print(
        wide_summary.to_string(
            index=False
        )
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
