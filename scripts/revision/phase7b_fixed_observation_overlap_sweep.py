from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow importing the existing phase scripts.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from phase3d_learned_nuisance_simulation import (
    WORKING_FEATURES,
    fit_hazard_model,
)

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


DEFAULT_STRENGTHS = (
    "0.0,0.5,1.0,1.5,2.0,2.5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7B: controlled overlap sweep "
            "at approximately fixed outcome-observation rate."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "revision_kdd2027/"
            "phase7b_fixed_observation_overlap"
        ),
    )

    parser.add_argument(
        "--strengths",
        default=DEFAULT_STRENGTHS,
    )

    parser.add_argument(
        "--target-observed-fraction",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--calibration-n",
        type=int,
        default=20000,
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
        "--clip",
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

    return parser.parse_args()


def parse_strengths(
    text: str,
) -> list[float]:
    values = sorted(
        {
            float(x.strip())
            for x in text.split(",")
            if x.strip()
        }
    )

    if not values:
        raise ValueError(
            "At least one overlap strength is required"
        )

    if any(
        value < 0
        for value in values
    ):
        raise ValueError(
            "Overlap strengths must be nonnegative"
        )

    return values


def make_config(
    *,
    strength: float,
    intercept: float,
    n: int,
    horizon: int,
    seed: int,
):
    """
    Keep the event process fixed.

    Increase strength of observed-covariate selection
    into censoring while tuning the intercept separately
    so the marginal observed fraction remains similar.
    """

    base = scenario_config(
        "observed_dependent",
        n=n,
        horizon=horizon,
        seed=seed,
    )

    return replace(
        base,
        scenario="fixed_observation_overlap_sweep",
        event_u=0.0,
        censor_intercept=float(
            intercept
        ),
        censor_x1=float(
            -0.75 * strength
        ),
        censor_x2=float(
            1.00 * strength
        ),
        censor_u=0.0,
    )


def observed_fraction_for_intercept(
    *,
    strength: float,
    intercept: float,
    n: int,
    horizon: int,
    seed: int,
) -> float:
    config = make_config(
        strength=strength,
        intercept=intercept,
        n=n,
        horizon=horizon,
        seed=seed,
    )

    output = simulate_cohort(
        config
    )

    return float(
        output.frame[
            "observed"
        ].mean()
    )


def calibrate_intercept(
    *,
    strength: float,
    target: float,
    n: int,
    horizon: int,
    seed: int,
) -> tuple[
    float,
    float,
]:
    """
    Tune censoring intercept using a separate
    calibration cohort.

    Larger intercept -> more censoring ->
    smaller observed fraction.
    """

    low = -10.0
    high = 2.0

    low_fraction = (
        observed_fraction_for_intercept(
            strength=strength,
            intercept=low,
            n=n,
            horizon=horizon,
            seed=seed,
        )
    )

    high_fraction = (
        observed_fraction_for_intercept(
            strength=strength,
            intercept=high,
            n=n,
            horizon=horizon,
            seed=seed,
        )
    )

    if not (
        low_fraction
        >= target
        >= high_fraction
    ):
        raise RuntimeError(
            "Target observed fraction is not "
            "bracketed for strength "
            f"{strength}: "
            f"low={low_fraction:.4f}, "
            f"target={target:.4f}, "
            f"high={high_fraction:.4f}"
        )

    best_intercept = None
    best_fraction = None
    best_error = np.inf

    for _ in range(
        28
    ):
        middle = (
            low
            + high
        ) / 2.0

        fraction = (
            observed_fraction_for_intercept(
                strength=strength,
                intercept=middle,
                n=n,
                horizon=horizon,
                seed=seed,
            )
        )

        error = abs(
            fraction
            - target
        )

        if error < best_error:
            best_error = error
            best_intercept = middle
            best_fraction = fraction

        if fraction > target:
            # Too many outcomes are visible.
            # Increase censoring.
            low = middle
        else:
            # Too few outcomes are visible.
            # Reduce censoring.
            high = middle

    assert best_intercept is not None
    assert best_fraction is not None

    return (
        float(
            best_intercept
        ),
        float(
            best_fraction
        ),
    )


def scalar(
    result,
    metric: str,
) -> float:
    return float(
        result.scalar_summary[
            metric
        ]
    )


def diagnostic(
    result,
    metric: str,
) -> float:
    return float(
        result.diagnostics[
            metric
        ]
    )


def run_replication(
    *,
    strength: float,
    intercept: float,
    replication: int,
    development_n: int,
    test_n: int,
    horizon: int,
    n_bins: int,
    clip: float,
    logistic_c: float,
    seed: int,
) -> dict[str, object]:

    # Same seeds across strengths within a replication.
    # This gives common random numbers and holds the
    # event-generating process fixed.
    base_seed = (
        seed
        + 100 * replication
    )

    development_config = (
        make_config(
            strength=strength,
            intercept=intercept,
            n=development_n,
            horizon=horizon,
            seed=base_seed + 1,
        )
    )

    test_config = make_config(
        strength=strength,
        intercept=intercept,
        n=test_n,
        horizon=horizon,
        seed=base_seed + 2,
    )

    development_output = (
        simulate_cohort(
            development_config
        )
    )

    test_output = (
        simulate_cohort(
            test_config
        )
    )

    development = (
        development_output.frame
    )

    test = test_output.frame

    event_model = fit_hazard_model(
        development,
        WORKING_FEATURES,
        horizon=horizon,
        target="event",
        c_value=logistic_c,
        seed=base_seed + 101,
    )

    censor_model = fit_hazard_model(
        development,
        WORKING_FEATURES,
        horizon=horizon,
        target="censor",
        c_value=logistic_c,
        seed=base_seed + 102,
    )

    learned_event_hazards = (
        predict_hazard_grid(
            event_model,
            test,
            WORKING_FEATURES,
            horizon=horizon,
        )
    )

    learned_censor_hazards = (
        predict_hazard_grid(
            censor_model,
            test,
            WORKING_FEATURES,
            horizon=horizon,
        )
    )

    bin_edges = (
        reference_quantile_edges(
            test[
                "prediction"
            ].to_numpy(),
            n_bins,
        )
    )

    # -------------------------------------------------
    # True nuisance functions.
    # -------------------------------------------------

    true_frame = (
        construct_survival_nuisance_predictions(
            test,
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

    true_result = (
        evaluate_survival_estimators(
            true_frame,
            horizon=horizon,
            bin_edges=bin_edges,
            censor_survival_clip=clip,
        )
    )

    # -------------------------------------------------
    # Learned, correctly specified observed-covariate
    # nuisance family.
    # -------------------------------------------------

    learned_frame = (
        construct_survival_nuisance_predictions(
            test,
            event_hazards=(
                learned_event_hazards
            ),
            censor_hazards=(
                learned_censor_hazards
            ),
            horizon=horizon,
            censor_survival_clip=clip,
        )
    )

    learned_result = (
        evaluate_survival_estimators(
            learned_frame,
            horizon=horizon,
            bin_edges=bin_edges,
            censor_survival_clip=clip,
        )
    )

    # Probability of remaining uncensored through
    # the complete horizon under the true censor model.
    censor_survival_horizon = (
        np.prod(
            1.0
            - test_output.true_censor_hazards,
            axis=1,
        )
    )

    return {
        "strength": strength,
        "replication": replication,
        "censor_intercept": intercept,
        "development_n": development_n,
        "test_n": test_n,
        "reference_event_rate": float(
            test[
                "outcome"
            ].mean()
        ),
        "observed_fraction": float(
            test[
                "observed"
            ].mean()
        ),

        # Direct support geometry.
        "g_tau_min": float(
            np.min(
                censor_survival_horizon
            )
        ),
        "g_tau_q01": float(
            np.quantile(
                censor_survival_horizon,
                0.01,
            )
        ),
        "g_tau_q05": float(
            np.quantile(
                censor_survival_horizon,
                0.05,
            )
        ),
        "g_tau_median": float(
            np.median(
                censor_survival_horizon
            )
        ),
        "g_tau_q95": float(
            np.quantile(
                censor_survival_horizon,
                0.95,
            )
        ),
        "fraction_g_tau_lt_0p01": float(
            np.mean(
                censor_survival_horizon
                < 0.01
            )
        ),
        "fraction_g_tau_lt_0p05": float(
            np.mean(
                censor_survival_horizon
                < 0.05
            )
        ),
        "fraction_g_tau_lt_0p10": float(
            np.mean(
                censor_survival_horizon
                < 0.10
            )
        ),

        # Methods that do not use estimated nuisances.
        "complete_case_mae": scalar(
            true_result,
            "complete_case_mae_to_reference",
        ),
        "kaplan_meier_mae": scalar(
            true_result,
            "kaplan_meier_mae_to_reference",
        ),

        # True nuisance functions:
        # separates support/finite-sample limitations
        # from nuisance misspecification.
        "true_oracle_or_mae": scalar(
            true_result,
            "survival_or_mae_to_reference",
        ),
        "true_oracle_ipcw_mae": scalar(
            true_result,
            "ipcw_event_mae_to_reference",
        ),
        "true_event_g_q01": diagnostic(
            true_result,
            "event_g_q01",
        ),
        "true_event_g_q05": diagnostic(
            true_result,
            "event_g_q05",
        ),
        "true_event_g_median": diagnostic(
            true_result,
            "event_g_median",
        ),
        "true_ipcw_ess": diagnostic(
            true_result,
            "event_weight_effective_sample_size",
        ),
        "true_fraction_event_g_clipped": diagnostic(
            true_result,
            "fraction_event_g_clipped",
        ),
        "true_maximum_event_weight": diagnostic(
            true_result,
            "maximum_event_weight",
        ),

        # Learned nuisance models.
        "learned_or_mae": scalar(
            learned_result,
            "survival_or_mae_to_reference",
        ),
        "learned_ipcw_mae": scalar(
            learned_result,
            "ipcw_event_mae_to_reference",
        ),
        "learned_ipcw_ess": diagnostic(
            learned_result,
            "event_weight_effective_sample_size",
        ),
        "learned_fraction_event_g_clipped": diagnostic(
            learned_result,
            "fraction_event_g_clipped",
        ),
        "learned_maximum_event_weight": diagnostic(
            learned_result,
            "maximum_event_weight",
        ),
    }


SUMMARY_METRICS = [
    "reference_event_rate",
    "observed_fraction",
    "g_tau_q01",
    "g_tau_q05",
    "g_tau_median",
    "fraction_g_tau_lt_0p01",
    "fraction_g_tau_lt_0p05",
    "fraction_g_tau_lt_0p10",
    "complete_case_mae",
    "kaplan_meier_mae",
    "true_oracle_or_mae",
    "true_oracle_ipcw_mae",
    "true_event_g_q05",
    "true_ipcw_ess",
    "true_fraction_event_g_clipped",
    "true_maximum_event_weight",
    "learned_or_mae",
    "learned_ipcw_mae",
    "learned_ipcw_ess",
    "learned_fraction_event_g_clipped",
    "learned_maximum_event_weight",
]


def summarize(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for strength, group in (
        results.groupby(
            "strength",
            sort=True,
        )
    ):
        row = {
            "strength": strength,
            "replicates": len(
                group
            ),
            "censor_intercept": float(
                group[
                    "censor_intercept"
                ].iloc[0]
            ),
        }

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

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_q025"
            ] = float(
                values.quantile(
                    0.025
                )
            )

            row[
                f"{metric}_q975"
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


def make_plots(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    x = summary[
        "strength"
    ]

    figure = plt.figure(
        figsize=(8, 5)
    )

    for metric, label in [
        (
            "complete_case_mae_mean",
            "Complete case",
        ),
        (
            "kaplan_meier_mae_mean",
            "Kaplan-Meier",
        ),
        (
            "true_oracle_ipcw_mae_mean",
            "IPCW, true censor model",
        ),
        (
            "learned_ipcw_mae_mean",
            "IPCW, learned censor model",
        ),
        (
            "learned_or_mae_mean",
            "Outcome regression, learned",
        ),
    ]:
        plt.plot(
            x,
            summary[
                metric
            ],
            marker="o",
            label=label,
        )

    plt.xlabel(
        "Censor-selection strength"
    )

    plt.ylabel(
        "MAE to linked calibration curve"
    )

    plt.title(
        "Adjustment error as overlap weakens "
        "at fixed observation rate"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        output_dir
        / "estimator_error_vs_overlap_strength.png",
        dpi=220,
    )

    plt.close(
        figure
    )

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        summary[
            "g_tau_q05_mean"
        ],
        marker="o",
        label=(
            "5th percentile censor survival"
        ),
    )

    plt.plot(
        x,
        summary[
            "g_tau_median_mean"
        ],
        marker="o",
        label=(
            "Median censor survival"
        ),
    )

    plt.xlabel(
        "Censor-selection strength"
    )

    plt.ylabel(
        "Probability of remaining uncensored "
        "through horizon"
    )

    plt.title(
        "Support weakens while marginal "
        "observation rate is held fixed"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        output_dir
        / "support_vs_overlap_strength.png",
        dpi=220,
    )

    plt.close(
        figure
    )


def validate(
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    strengths: list[float],
    replicates: int,
    target_observed_fraction: float,
) -> pd.DataFrame:

    expected_rows = (
        len(
            strengths
        )
        * replicates
    )

    event_ranges = (
        results.groupby(
            "replication"
        )[
            "reference_event_rate"
        ]
        .agg(
            lambda x: (
                float(
                    x.max()
                    - x.min()
                )
            )
        )
    )

    observed_error = (
        summary[
            "observed_fraction_mean"
        ]
        - target_observed_fraction
    ).abs()

    all_numeric = (
        results.select_dtypes(
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

    rows = [
        {
            "check": (
                "expected_replication_rows"
            ),
            "value": len(
                results
            ),
            "threshold": expected_rows,
            "passed": (
                len(
                    results
                )
                == expected_rows
            ),
        },
        {
            "check": (
                "reference_event_process_invariant"
            ),
            "value": float(
                event_ranges.max()
            ),
            "threshold": 1e-12,
            "passed": bool(
                event_ranges.max()
                <= 1e-12
            ),
        },
        {
            "check": (
                "mean_observed_fraction_within_0p03"
            ),
            "value": float(
                observed_error.max()
            ),
            "threshold": 0.03,
            "passed": bool(
                observed_error.max()
                <= 0.03
            ),
        },
        {
            "check": (
                "all_numeric_outputs_finite"
            ),
            "value": bool(
                all_numeric
            ),
            "threshold": True,
            "passed": bool(
                all_numeric
            ),
        },
        {
            "check": (
                "support_q05_first_minus_last"
            ),
            "value": float(
                summary[
                    "g_tau_q05_mean"
                ].iloc[0]
                - summary[
                    "g_tau_q05_mean"
                ].iloc[-1]
            ),
            "threshold": (
                "diagnostic"
            ),
            "passed": True,
        },
        {
            "check": (
                "true_ipcw_mae_last_minus_first"
            ),
            "value": float(
                summary[
                    "true_oracle_ipcw_mae_mean"
                ].iloc[-1]
                - summary[
                    "true_oracle_ipcw_mae_mean"
                ].iloc[0]
            ),
            "threshold": (
                "diagnostic"
            ),
            "passed": True,
        },
        {
            "check": (
                "learned_ipcw_mae_last_minus_first"
            ),
            "value": float(
                summary[
                    "learned_ipcw_mae_mean"
                ].iloc[-1]
                - summary[
                    "learned_ipcw_mae_mean"
                ].iloc[0]
            ),
            "threshold": (
                "diagnostic"
            ),
            "passed": True,
        },
    ]

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    strengths = parse_strengths(
        args.strengths
    )

    if not (
        0
        < args.target_observed_fraction
        < 1
    ):
        raise ValueError(
            "Target observed fraction must lie in (0,1)"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "ACTIONAUDIT PHASE 7B: "
        "FIXED-OBSERVATION OVERLAP SWEEP"
    )

    print(
        "=" * 100
    )

    print(
        f"Strengths: {strengths}"
    )

    print(
        "Target observed fraction: "
        f"{args.target_observed_fraction}"
    )

    print(
        f"Replicates: {args.replicates}"
    )

    # -------------------------------------------------
    # Calibrate one censor intercept per strength
    # using an independent synthetic cohort.
    # -------------------------------------------------

    calibration_rows = []

    calibration_seed = (
        args.seed
        + 9_000_000
    )

    intercepts = {}

    print(
        "\n===== INTERCEPT CALIBRATION ====="
    )

    for strength in strengths:
        (
            intercept,
            achieved,
        ) = calibrate_intercept(
            strength=strength,
            target=(
                args.target_observed_fraction
            ),
            n=args.calibration_n,
            horizon=args.horizon,
            seed=calibration_seed,
        )

        intercepts[
            strength
        ] = intercept

        calibration_rows.append(
            {
                "strength": strength,
                "censor_intercept": intercept,
                "target_observed_fraction": (
                    args.target_observed_fraction
                ),
                "calibration_observed_fraction": (
                    achieved
                ),
                "absolute_error": abs(
                    achieved
                    - args.target_observed_fraction
                ),
            }
        )

        print(
            f"strength={strength:4.2f} "
            f"intercept={intercept:8.4f} "
            f"observed={achieved:.4f}"
        )

    calibration = pd.DataFrame(
        calibration_rows
    )

    calibration.to_csv(
        output_dir
        / "intercept_calibration.csv",
        index=False,
    )

    # -------------------------------------------------
    # Monte Carlo experiment.
    # -------------------------------------------------

    rows = []

    print(
        "\n===== MONTE CARLO SWEEP ====="
    )

    for replication in range(
        args.replicates
    ):
        for strength in strengths:
            rows.append(
                run_replication(
                    strength=strength,
                    intercept=(
                        intercepts[
                            strength
                        ]
                    ),
                    replication=replication,
                    development_n=(
                        args.development_n
                    ),
                    test_n=args.test_n,
                    horizon=args.horizon,
                    n_bins=args.n_bins,
                    clip=args.clip,
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
        / "replication_results.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = summarize(
        results
    )

    summary.to_csv(
        output_dir
        / "overlap_sweep_summary.csv",
        index=False,
    )

    checks = validate(
        results=results,
        summary=summary,
        strengths=strengths,
        replicates=args.replicates,
        target_observed_fraction=(
            args.target_observed_fraction
        ),
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    hard_checks = checks.loc[
        checks[
            "threshold"
        ].astype(
            str
        ).ne(
            "diagnostic"
        )
    ]

    passed = bool(
        hard_checks[
            "passed"
        ].all()
    )

    status = (
        "phase7b_fixed_observation_overlap_passed"
        if passed
        else
        "phase7b_fixed_observation_overlap_failed"
    )

    (
        output_dir
        / "phase7b_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    make_plots(
        summary,
        output_dir,
    )

    display_columns = [
        "strength",
        "censor_intercept",
        "observed_fraction_mean",
        "g_tau_q05_mean",
        "fraction_g_tau_lt_0p05_mean",
        "true_ipcw_ess_mean",
        "complete_case_mae_mean",
        "kaplan_meier_mae_mean",
        "true_oracle_ipcw_mae_mean",
        "learned_ipcw_mae_mean",
        "learned_or_mae_mean",
    ]

    print(
        "\n===== PRIMARY RESULTS ====="
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n===== VALIDATION ====="
    )

    print(
        checks.to_string(
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
