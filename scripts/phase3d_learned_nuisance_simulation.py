"""Phase 3D: learned pooled-logistic nuisance-model simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from actionaudit.discrete_survival import (
    TIME_FEATURES,
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
    expand_person_period,
    predict_hazard_grid,
)
from actionaudit.km import reference_quantile_edges
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

WORKING_FEATURES = [
    "x1",
    "x2",
]

ORACLE_FEATURES = [
    "x1",
    "x2",
    "latent_u",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate learned event and censoring nuisance models "
            "under controlled censoring mechanisms."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase3d_learned_nuisance",
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
        "--clips",
        default="0.01,0.05,0.10",
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

    return parser.parse_args()


def parse_clips(
    text: str,
) -> list[float]:
    clips = sorted(
        {
            float(value.strip())
            for value in text.split(",")
            if value.strip()
        }
    )

    if not clips:
        raise ValueError(
            "At least one clipping threshold is required"
        )

    if any(
        not 0 < clip < 1
        for clip in clips
    ):
        raise ValueError(
            "Clipping thresholds must lie in (0, 1)"
        )

    return clips


def build_hazard_model(
    *,
    c_value: float,
    seed: int,
) -> Pipeline:
    """Build a fixed pooled-logistic hazard model."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_hazard_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    horizon: int,
    target: str,
    c_value: float,
    seed: int,
) -> Pipeline:
    """Fit one discrete-time event or censoring hazard model."""

    person_period = expand_person_period(
        frame,
        feature_columns,
        horizon=horizon,
        target=target,
    )

    model_columns = (
        feature_columns
        + TIME_FEATURES
    )

    outcome = (
        person_period[
            "hazard_target"
        ]
        .astype(int)
    )

    if outcome.nunique() != 2:
        raise RuntimeError(
            f"{target} hazard outcome does not contain both classes"
        )

    model = build_hazard_model(
        c_value=c_value,
        seed=seed,
    )

    model.fit(
        person_period[
            model_columns
        ],
        outcome,
    )

    return model


def scalar_value(
    summary: pd.Series,
    metric: str,
) -> float:
    """Read one scalar estimator metric."""

    return float(
        summary[metric]
    )


def evaluate_replication(
    *,
    scenario: str,
    replication: int,
    development_n: int,
    test_n: int,
    horizon: int,
    n_bins: int,
    clips: list[float],
    logistic_c: float,
    seed: int,
) -> list[dict[str, object]]:
    """Fit nuisance models and evaluate one independent test cohort."""

    scenario_index = SCENARIOS.index(
        scenario
    )

    base_seed = (
        seed
        + 100000 * scenario_index
        + 10 * replication
    )

    development_config = scenario_config(
        scenario,
        n=development_n,
        horizon=horizon,
        seed=base_seed + 1,
    )

    test_config = scenario_config(
        scenario,
        n=test_n,
        horizon=horizon,
        seed=base_seed + 2,
    )

    development = simulate_cohort(
        development_config
    )

    test_output = simulate_cohort(
        test_config
    )

    development_frame = (
        development.frame
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

    learned_oracle_event_hazards = predict_hazard_grid(
        oracle_event_model,
        test_frame,
        ORACLE_FEATURES,
        horizon=horizon,
    )

    learned_oracle_censor_hazards = predict_hazard_grid(
        oracle_censor_model,
        test_frame,
        ORACLE_FEATURES,
        horizon=horizon,
    )

    bin_edges = reference_quantile_edges(
        test_frame[
            "prediction"
        ].to_numpy(),
        n_bins,
    )

    rows: list[dict[str, object]] = []

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
                "scenario": scenario,
                "replication": replication,
                "development_seed": base_seed + 1,
                "test_seed": base_seed + 2,
                "clip": clip,
                "development_n": development_n,
                "test_n": test_n,
                "horizon": horizon,
                "n_bins": n_bins,
                "reference_event_rate": float(
                    test_frame[
                        "outcome"
                    ].mean()
                ),
                "observed_fraction": float(
                    test_frame[
                        "observed"
                    ].mean()
                ),
                "complete_case_mae": scalar_value(
                    true_scalar,
                    "complete_case_mae_to_reference",
                ),
                "complete_case_signed_gap": scalar_value(
                    true_scalar,
                    "complete_case_signed_gap",
                ),
                "kaplan_meier_mae": scalar_value(
                    true_scalar,
                    "kaplan_meier_mae_to_reference",
                ),
                "kaplan_meier_signed_gap": scalar_value(
                    true_scalar,
                    "kaplan_meier_signed_gap",
                ),
                "true_oracle_or_mae": scalar_value(
                    true_scalar,
                    "survival_or_mae_to_reference",
                ),
                "true_oracle_or_signed_gap": scalar_value(
                    true_scalar,
                    "survival_or_signed_gap",
                ),
                "true_oracle_ipcw_mae": scalar_value(
                    true_scalar,
                    "ipcw_event_mae_to_reference",
                ),
                "true_oracle_ipcw_signed_gap": scalar_value(
                    true_scalar,
                    "ipcw_event_signed_gap",
                ),
                "learned_oracle_or_mae": scalar_value(
                    oracle_scalar,
                    "survival_or_mae_to_reference",
                ),
                "learned_oracle_or_signed_gap": scalar_value(
                    oracle_scalar,
                    "survival_or_signed_gap",
                ),
                "learned_oracle_ipcw_mae": scalar_value(
                    oracle_scalar,
                    "ipcw_event_mae_to_reference",
                ),
                "learned_oracle_ipcw_signed_gap": scalar_value(
                    oracle_scalar,
                    "ipcw_event_signed_gap",
                ),
                "learned_working_or_mae": scalar_value(
                    working_scalar,
                    "survival_or_mae_to_reference",
                ),
                "learned_working_or_signed_gap": scalar_value(
                    working_scalar,
                    "survival_or_signed_gap",
                ),
                "learned_working_ipcw_mae": scalar_value(
                    working_scalar,
                    "ipcw_event_mae_to_reference",
                ),
                "learned_working_ipcw_signed_gap": scalar_value(
                    working_scalar,
                    "ipcw_event_signed_gap",
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
    """Create long Monte Carlo summaries."""

    metric_columns = [
        column
        for column in results.columns
        if (
            column.endswith("_mae")
            or column.endswith(
                "_signed_gap"
            )
            or column.endswith("_ess")
            or column.endswith(
                "_fraction_clipped"
            )
            or column in {
                "reference_event_rate",
                "observed_fraction",
            }
        )
    ]

    rows: list[dict[str, object]] = []

    for (
        scenario,
        clip,
    ), group in results.groupby(
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
                    "clip": float(
                        clip
                    ),
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
    """Create a compact table of mean estimator errors."""

    selected_metrics = [
        "observed_fraction",
        "complete_case_mae",
        "kaplan_meier_mae",
        "true_oracle_or_mae",
        "true_oracle_ipcw_mae",
        "learned_oracle_or_mae",
        "learned_oracle_ipcw_mae",
        "learned_working_or_mae",
        "learned_working_ipcw_mae",
        "learned_oracle_ipcw_ess",
        "learned_working_ipcw_ess",
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
            "replicates must be positive"
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
                evaluate_replication(
                    scenario=scenario,
                    replication=replication,
                    development_n=(
                        args.development_n
                    ),
                    test_n=args.test_n,
                    horizon=args.horizon,
                    n_bins=args.n_bins,
                    clips=clips,
                    logistic_c=args.logistic_c,
                    seed=args.seed,
                )
            )

            if (
                replication + 1
            ) % 5 == 0:
                print(
                    f"  completed "
                    f"{replication + 1}/"
                    f"{args.replicates}"
                )

    replication_results = pd.DataFrame(
        rows
    )

    summary = summarize_results(
        replication_results
    )

    wide_summary = create_wide_summary(
        summary
    )

    replication_results.to_csv(
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
        "phase": "3D",
        "design": (
            "learned pooled-logistic event and censoring "
            "nuisance models fitted on an independent "
            "development cohort"
        ),
        "scenarios": SCENARIOS,
        "replicates": int(
            args.replicates
        ),
        "development_n": int(
            args.development_n
        ),
        "test_n": int(
            args.test_n
        ),
        "horizon": int(
            args.horizon
        ),
        "n_bins": int(
            args.n_bins
        ),
        "clips": clips,
        "working_features": (
            WORKING_FEATURES
        ),
        "oracle_features": (
            ORACLE_FEATURES
        ),
        "logistic_c": float(
            args.logistic_c
        ),
        "test_data_used_for_nuisance_fitting": False,
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
        "\nPhase 3D complete"
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
