from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Existing ActionAudit infrastructure
# ---------------------------------------------------------------------

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

from actionaudit.simulation import simulate_cohort

# Reuse the exact Phase 7B DGP.
from phase7b_fixed_observation_overlap_sweep import (
    make_config,
)


DEFAULT_CLIPS = "0.005,0.01,0.025,0.05,0.10,0.20"

REFERENCE_CLIP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7F-A: survival-IPCW clipping benchmark "
            "under controlled deterioration of overlap."
        )
    )

    parser.add_argument(
        "--phase7b-dir",
        default=(
            "revision_kdd2027/"
            "phase7b_fixed_observation_overlap_full"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "revision_kdd2027/"
            "phase7f_overlap_remedy_benchmark"
        ),
    )

    parser.add_argument(
        "--clips",
        default=DEFAULT_CLIPS,
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
        "--logistic-c",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--target-observed-fraction",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def parse_clips(text: str) -> list[float]:
    values = sorted(
        {
            float(x.strip())
            for x in text.split(",")
            if x.strip()
        }
    )

    if not values:
        raise ValueError(
            "At least one clipping threshold is required."
        )

    if any(
        not 0 < value < 1
        for value in values
    ):
        raise ValueError(
            "Every clipping threshold must lie in (0, 1)."
        )

    if not any(
        np.isclose(
            value,
            REFERENCE_CLIP,
        )
        for value in values
    ):
        raise ValueError(
            "The clipping grid must contain the Phase 7B "
            "reference threshold 0.05."
        )

    return values


def scalar_value(
    result,
    metric: str,
) -> float:
    if metric not in result.scalar_summary.index:
        raise KeyError(
            f"Missing scalar metric: {metric}. "
            f"Available metrics: "
            f"{list(result.scalar_summary.index)}"
        )

    return float(
        result.scalar_summary[
            metric
        ]
    )


def diagnostic_value(
    result,
    metric: str,
) -> float:
    if metric not in result.diagnostics:
        raise KeyError(
            f"Missing diagnostic metric: {metric}. "
            f"Available diagnostics: "
            f"{list(result.diagnostics.keys())}"
        )

    return float(
        result.diagnostics[
            metric
        ]
    )


def load_phase7b_design(
    phase7b_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    intercept_path = (
        phase7b_dir
        / "intercept_calibration.csv"
    )

    replication_path = (
        phase7b_dir
        / "replication_results.csv.gz"
    )

    if not intercept_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 7B intercept file: "
            f"{intercept_path}"
        )

    if not replication_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 7B replication file: "
            f"{replication_path}"
        )

    intercepts = pd.read_csv(
        intercept_path
    )

    phase7b_replications = pd.read_csv(
        replication_path
    )

    required_intercept = {
        "strength",
        "censor_intercept",
        "target_observed_fraction",
    }

    missing = (
        required_intercept
        - set(intercepts.columns)
    )

    if missing:
        raise RuntimeError(
            "Phase 7B intercept file is missing: "
            f"{sorted(missing)}"
        )

    required_replication = {
        "strength",
        "replication",
        "complete_case_mae",
        "kaplan_meier_mae",
        "true_oracle_or_mae",
        "true_oracle_ipcw_mae",
        "true_ipcw_ess",
        "true_fraction_event_g_clipped",
        "true_maximum_event_weight",
        "learned_or_mae",
        "learned_ipcw_mae",
        "learned_ipcw_ess",
        "learned_fraction_event_g_clipped",
        "learned_maximum_event_weight",
    }

    missing = (
        required_replication
        - set(
            phase7b_replications.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Phase 7B replication file is missing: "
            f"{sorted(missing)}"
        )

    intercepts = (
        intercepts
        .sort_values("strength")
        .reset_index(drop=True)
    )

    return (
        intercepts,
        phase7b_replications,
    )


def fit_learned_nuisance(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    logistic_c: float,
    base_seed: int,
):
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

    event_hazards = (
        predict_hazard_grid(
            event_model,
            test,
            WORKING_FEATURES,
            horizon=horizon,
        )
    )

    censor_hazards = (
        predict_hazard_grid(
            censor_model,
            test,
            WORKING_FEATURES,
            horizon=horizon,
        )
    )

    return (
        event_hazards,
        censor_hazards,
    )


def evaluate_at_clip(
    *,
    test: pd.DataFrame,
    true_event_hazards: np.ndarray,
    true_censor_hazards: np.ndarray,
    learned_event_hazards: np.ndarray,
    learned_censor_hazards: np.ndarray,
    horizon: int,
    bin_edges: np.ndarray,
    clip: float,
) -> dict[str, float]:

    # -------------------------------------------------------------
    # True/oracle nuisance functions.
    # -------------------------------------------------------------

    true_frame = (
        construct_survival_nuisance_predictions(
            test,
            event_hazards=(
                true_event_hazards
            ),
            censor_hazards=(
                true_censor_hazards
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

    # -------------------------------------------------------------
    # Learned correctly specified observed-covariate nuisance family.
    # -------------------------------------------------------------

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

    return {
        # Baselines. These should not change with clipping.
        "complete_case_mae": scalar_value(
            true_result,
            "complete_case_mae_to_reference",
        ),
        "complete_case_signed_gap": (
            scalar_value(
                true_result,
                "complete_case_signed_gap",
            )
        ),
        "kaplan_meier_mae": scalar_value(
            true_result,
            "kaplan_meier_mae_to_reference",
        ),
        "kaplan_meier_signed_gap": (
            scalar_value(
                true_result,
                "kaplan_meier_signed_gap",
            )
        ),

        # Oracle outcome regression should also be clip-invariant.
        "true_oracle_or_mae": scalar_value(
            true_result,
            "survival_or_mae_to_reference",
        ),
        "true_oracle_or_signed_gap": (
            scalar_value(
                true_result,
                "survival_or_signed_gap",
            )
        ),

        # Oracle IPCW.
        "true_ipcw_mae": scalar_value(
            true_result,
            "ipcw_event_mae_to_reference",
        ),
        "true_ipcw_signed_gap": (
            scalar_value(
                true_result,
                "ipcw_event_signed_gap",
            )
        ),
        "true_ipcw_ess": diagnostic_value(
            true_result,
            "event_weight_effective_sample_size",
        ),
        "true_fraction_event_g_clipped": (
            diagnostic_value(
                true_result,
                "fraction_event_g_clipped",
            )
        ),
        "true_maximum_event_weight": (
            diagnostic_value(
                true_result,
                "maximum_event_weight",
            )
        ),
        "true_event_g_q01": diagnostic_value(
            true_result,
            "event_g_q01",
        ),
        "true_event_g_q05": diagnostic_value(
            true_result,
            "event_g_q05",
        ),
        "true_event_g_median": diagnostic_value(
            true_result,
            "event_g_median",
        ),

        # Learned outcome regression.
        "learned_or_mae": scalar_value(
            learned_result,
            "survival_or_mae_to_reference",
        ),
        "learned_or_signed_gap": (
            scalar_value(
                learned_result,
                "survival_or_signed_gap",
            )
        ),

        # Learned IPCW.
        "learned_ipcw_mae": scalar_value(
            learned_result,
            "ipcw_event_mae_to_reference",
        ),
        "learned_ipcw_signed_gap": (
            scalar_value(
                learned_result,
                "ipcw_event_signed_gap",
            )
        ),
        "learned_ipcw_ess": diagnostic_value(
            learned_result,
            "event_weight_effective_sample_size",
        ),
        "learned_fraction_event_g_clipped": (
            diagnostic_value(
                learned_result,
                "fraction_event_g_clipped",
            )
        ),
        "learned_maximum_event_weight": (
            diagnostic_value(
                learned_result,
                "maximum_event_weight",
            )
        ),
        "learned_event_g_q01": diagnostic_value(
            learned_result,
            "event_g_q01",
        ),
        "learned_event_g_q05": diagnostic_value(
            learned_result,
            "event_g_q05",
        ),
        "learned_event_g_median": diagnostic_value(
            learned_result,
            "event_g_median",
        ),
    }


def run_replication(
    *,
    strength: float,
    intercept: float,
    replication: int,
    clips: list[float],
    development_n: int,
    test_n: int,
    horizon: int,
    n_bins: int,
    logistic_c: float,
    seed: int,
) -> list[dict[str, object]]:
    """
    Reproduce Phase 7B's common-random-number construction,
    fit nuisance models once, then evaluate every clipping rule
    on the exact same simulated patients and nuisance predictions.
    """

    base_seed = (
        seed
        + 100 * replication
    )

    development_config = make_config(
        strength=strength,
        intercept=intercept,
        n=development_n,
        horizon=horizon,
        seed=base_seed + 1,
    )

    test_config = make_config(
        strength=strength,
        intercept=intercept,
        n=test_n,
        horizon=horizon,
        seed=base_seed + 2,
    )

    development_output = simulate_cohort(
        development_config
    )

    test_output = simulate_cohort(
        test_config
    )

    development = (
        development_output.frame
    )

    test = (
        test_output.frame
    )

    (
        learned_event_hazards,
        learned_censor_hazards,
    ) = fit_learned_nuisance(
        development,
        test,
        horizon=horizon,
        logistic_c=logistic_c,
        base_seed=base_seed,
    )

    bin_edges = reference_quantile_edges(
        test[
            "prediction"
        ].to_numpy(),
        n_bins,
    )

    # True probability of remaining uncensored through horizon.
    g_tau = np.prod(
        1.0
        - test_output.true_censor_hazards,
        axis=1,
    )

    common = {
        "strength": float(
            strength
        ),
        "replication": int(
            replication
        ),
        "censor_intercept": float(
            intercept
        ),
        "development_n": int(
            development_n
        ),
        "test_n": int(
            test_n
        ),
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
        "g_tau_min": float(
            np.min(
                g_tau
            )
        ),
        "g_tau_q01": float(
            np.quantile(
                g_tau,
                0.01,
            )
        ),
        "g_tau_q05": float(
            np.quantile(
                g_tau,
                0.05,
            )
        ),
        "g_tau_median": float(
            np.median(
                g_tau
            )
        ),
        "fraction_g_tau_lt_0p01": float(
            np.mean(
                g_tau < 0.01
            )
        ),
        "fraction_g_tau_lt_0p05": float(
            np.mean(
                g_tau < 0.05
            )
        ),
        "fraction_g_tau_lt_0p10": float(
            np.mean(
                g_tau < 0.10
            )
        ),
    }

    rows = []

    for clip in clips:
        metrics = evaluate_at_clip(
            test=test,
            true_event_hazards=(
                test_output.true_event_hazards
            ),
            true_censor_hazards=(
                test_output.true_censor_hazards
            ),
            learned_event_hazards=(
                learned_event_hazards
            ),
            learned_censor_hazards=(
                learned_censor_hazards
            ),
            horizon=horizon,
            bin_edges=bin_edges,
            clip=clip,
        )

        rows.append(
            {
                **common,
                "clip": float(
                    clip
                ),
                **metrics,
            }
        )

    return rows


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
    "true_ipcw_mae",
    "true_ipcw_signed_gap",
    "true_ipcw_ess",
    "true_fraction_event_g_clipped",
    "true_maximum_event_weight",
    "learned_or_mae",
    "learned_ipcw_mae",
    "learned_ipcw_signed_gap",
    "learned_ipcw_ess",
    "learned_fraction_event_g_clipped",
    "learned_maximum_event_weight",
]


def summarize(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = results.groupby(
        [
            "strength",
            "clip",
        ],
        sort=True,
    )

    for (
        strength,
        clip,
    ), group in grouped:

        row = {
            "strength": float(
                strength
            ),
            "clip": float(
                clip
            ),
            "replicates": int(
                len(group)
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
                .astype(float)
            )

            if values.empty:
                row[
                    f"{metric}_mean"
                ] = np.nan
                row[
                    f"{metric}_sd"
                ] = np.nan
                row[
                    f"{metric}_q025"
                ] = np.nan
                row[
                    f"{metric}_median"
                ] = np.nan
                row[
                    f"{metric}_q975"
                ] = np.nan
                continue

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_sd"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

            row[
                f"{metric}_q025"
            ] = float(
                values.quantile(
                    0.025
                )
            )

            row[
                f"{metric}_median"
            ] = float(
                values.quantile(
                    0.50
                )
            )

            row[
                f"{metric}_q975"
            ] = float(
                values.quantile(
                    0.975
                )
            )

        true_gap = (
            group[
                "true_ipcw_signed_gap"
            ]
            .astype(float)
            .to_numpy()
        )

        learned_gap = (
            group[
                "learned_ipcw_signed_gap"
            ]
            .astype(float)
            .to_numpy()
        )

        row[
            "true_ipcw_signed_gap_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    np.square(
                        true_gap
                    )
                )
            )
        )

        row[
            "learned_ipcw_signed_gap_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    np.square(
                        learned_gap
                    )
                )
            )
        )

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------
    # Compare each remedy with the frozen Phase 7B clip=0.05.
    # Positive "improvement" means lower MAE than clip=0.05.
    # -------------------------------------------------------------

    reference = (
        summary.loc[
            np.isclose(
                summary["clip"],
                REFERENCE_CLIP,
            ),
            [
                "strength",
                "true_ipcw_mae_mean",
                "learned_ipcw_mae_mean",
            ],
        ]
        .rename(
            columns={
                "true_ipcw_mae_mean": (
                    "true_ipcw_mae_clip005"
                ),
                "learned_ipcw_mae_mean": (
                    "learned_ipcw_mae_clip005"
                ),
            }
        )
    )

    summary = summary.merge(
        reference,
        on="strength",
        how="left",
        validate="many_to_one",
    )

    summary[
        "true_ipcw_improvement_vs_clip005"
    ] = (
        summary[
            "true_ipcw_mae_clip005"
        ]
        - summary[
            "true_ipcw_mae_mean"
        ]
    )

    summary[
        "learned_ipcw_improvement_vs_clip005"
    ] = (
        summary[
            "learned_ipcw_mae_clip005"
        ]
        - summary[
            "learned_ipcw_mae_mean"
        ]
    )

    summary[
        "learned_ipcw_improvement_vs_km"
    ] = (
        summary[
            "kaplan_meier_mae_mean"
        ]
        - summary[
            "learned_ipcw_mae_mean"
        ]
    )

    summary[
        "learned_ipcw_improvement_vs_complete_case"
    ] = (
        summary[
            "complete_case_mae_mean"
        ]
        - summary[
            "learned_ipcw_mae_mean"
        ]
    )

    return summary


def reproduce_phase7b_reference(
    results: pd.DataFrame,
    phase7b: pd.DataFrame,
    replicates: int,
) -> pd.DataFrame:
    """
    Verify that Phase 7F at clip=0.05 exactly reproduces
    the corresponding frozen Phase 7B outputs.

    Metric names differ slightly between the two phases,
    so each comparison pair is explicitly renamed before
    merging rather than relying on pandas suffix behavior.
    """

    current = results.loc[
        np.isclose(
            results["clip"],
            REFERENCE_CLIP,
        )
        & results["replication"].lt(
            replicates
        )
    ].copy()

    previous = phase7b.loc[
        phase7b[
            "replication"
        ].lt(
            replicates
        )
    ].copy()

    compare_columns = {
        "complete_case_mae": (
            "complete_case_mae"
        ),
        "kaplan_meier_mae": (
            "kaplan_meier_mae"
        ),
        "true_oracle_or_mae": (
            "true_oracle_or_mae"
        ),
        "true_ipcw_mae": (
            "true_oracle_ipcw_mae"
        ),
        "true_ipcw_ess": (
            "true_ipcw_ess"
        ),
        "true_fraction_event_g_clipped": (
            "true_fraction_event_g_clipped"
        ),
        "true_maximum_event_weight": (
            "true_maximum_event_weight"
        ),
        "learned_or_mae": (
            "learned_or_mae"
        ),
        "learned_ipcw_mae": (
            "learned_ipcw_mae"
        ),
        "learned_ipcw_ess": (
            "learned_ipcw_ess"
        ),
        "learned_fraction_event_g_clipped": (
            "learned_fraction_event_g_clipped"
        ),
        "learned_maximum_event_weight": (
            "learned_maximum_event_weight"
        ),
    }

    keys = [
        "strength",
        "replication",
    ]

    rows = []

    for (
        phase7f_name,
        phase7b_name,
    ) in compare_columns.items():

        left = (
            current[
                keys
                + [
                    phase7f_name
                ]
            ]
            .rename(
                columns={
                    phase7f_name:
                    "phase7f_value"
                }
            )
        )

        right = (
            previous[
                keys
                + [
                    phase7b_name
                ]
            ]
            .rename(
                columns={
                    phase7b_name:
                    "phase7b_value"
                }
            )
        )

        merged = left.merge(
            right,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        difference = (
            merged[
                "phase7f_value"
            ].astype(float)
            - merged[
                "phase7b_value"
            ].astype(float)
        ).abs()

        rows.append(
            {
                "metric_phase7f":
                    phase7f_name,
                "metric_phase7b":
                    phase7b_name,
                "n_compared":
                    int(len(merged)),
                "maximum_absolute_difference":
                    float(
                        difference.max()
                    ),
                "mean_absolute_difference":
                    float(
                        difference.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def clipping_monotonicity(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        strength,
        replication,
    ), group in results.groupby(
        [
            "strength",
            "replication",
        ],
        sort=True,
    ):
        group = group.sort_values(
            "clip"
        )

        for prefix in [
            "true",
            "learned",
        ]:
            max_weight = group[
                f"{prefix}_maximum_event_weight"
            ].to_numpy(
                dtype=float
            )

            fraction = group[
                (
                    f"{prefix}_fraction_"
                    "event_g_clipped"
                )
            ].to_numpy(
                dtype=float
            )

            ess = group[
                f"{prefix}_ipcw_ess"
            ].to_numpy(
                dtype=float
            )

            rows.append(
                {
                    "strength": strength,
                    "replication": replication,
                    "nuisance": prefix,
                    "max_weight_nonincreasing": bool(
                        np.all(
                            np.diff(
                                max_weight
                            )
                            <= 1e-10
                        )
                    ),
                    "fraction_clipped_nondecreasing": bool(
                        np.all(
                            np.diff(
                                fraction
                            )
                            >= -1e-10
                        )
                    ),
                    # Diagnostic only: ESS should generally rise
                    # with stronger clipping, but it is not used
                    # as a validity gate.
                    "ess_nondecreasing": bool(
                        np.all(
                            np.diff(
                                ess
                            )
                            >= -1e-10
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def validate(
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    reproduction: pd.DataFrame,
    monotonicity: pd.DataFrame,
    intercepts: pd.DataFrame,
    clips: list[float],
    replicates: int,
    target_observed_fraction: float,
) -> pd.DataFrame:

    checks = []

    expected_rows = (
        len(intercepts)
        * replicates
        * len(clips)
    )

    checks.append(
        {
            "check": (
                "expected_remedy_rows"
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
        }
    )

    # Event process must remain invariant across overlap strengths
    # within each common-random-number replication.
    event_ranges = (
        results.loc[
            np.isclose(
                results["clip"],
                REFERENCE_CLIP,
            )
        ]
        .groupby(
            "replication"
        )[
            "reference_event_rate"
        ]
        .agg(
            lambda values: (
                float(
                    values.max()
                    - values.min()
                )
            )
        )
    )

    max_event_range = float(
        event_ranges.max()
    )

    checks.append(
        {
            "check": (
                "reference_event_process_invariant"
            ),
            "value": max_event_range,
            "threshold": 1e-12,
            "comparison": "<=",
            "passed": (
                max_event_range
                <= 1e-12
            ),
        }
    )

    reference_clip_rows = (
        results.loc[
            np.isclose(
                results["clip"],
                REFERENCE_CLIP,
            )
        ]
    )

    observed_means = (
        reference_clip_rows
        .groupby(
            "strength"
        )[
            "observed_fraction"
        ]
        .mean()
    )

    observed_error = (
        observed_means
        - target_observed_fraction
    ).abs()

    checks.append(
        {
            "check": (
                "mean_observed_fraction_within_0p03"
            ),
            "value": float(
                observed_error.max()
            ),
            "threshold": 0.03,
            "comparison": "<=",
            "passed": bool(
                observed_error.max()
                <= 0.03
            ),
        }
    )

    numeric = results.select_dtypes(
        include=[
            np.number
        ]
    )

    all_finite = bool(
        np.isfinite(
            numeric.to_numpy(
                dtype=float
            )
        ).all()
    )

    checks.append(
        {
            "check": (
                "all_replication_numeric_outputs_finite"
            ),
            "value": all_finite,
            "threshold": True,
            "comparison": "==",
            "passed": all_finite,
        }
    )

    # Exact reproduction of Phase 7B at its frozen clipping rule.
    max_reproduction_difference = float(
        reproduction[
            "maximum_absolute_difference"
        ].max()
    )

    checks.append(
        {
            "check": (
                "clip0p05_exactly_reproduces_phase7b"
            ),
            "value": max_reproduction_difference,
            "threshold": 1e-10,
            "comparison": "<=",
            "passed": (
                max_reproduction_difference
                <= 1e-10
            ),
        }
    )

    max_weight_ok = bool(
        monotonicity[
            "max_weight_nonincreasing"
        ].all()
    )

    checks.append(
        {
            "check": (
                "maximum_weight_nonincreasing_with_clip"
            ),
            "value": max_weight_ok,
            "threshold": True,
            "comparison": "==",
            "passed": max_weight_ok,
        }
    )

    fraction_ok = bool(
        monotonicity[
            "fraction_clipped_nondecreasing"
        ].all()
    )

    checks.append(
        {
            "check": (
                "fraction_clipped_nondecreasing_with_clip"
            ),
            "value": fraction_ok,
            "threshold": True,
            "comparison": "==",
            "passed": fraction_ok,
        }
    )

    summary_finite = bool(
        np.isfinite(
            summary.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
    )

    checks.append(
        {
            "check": (
                "all_summary_numeric_outputs_finite"
            ),
            "value": summary_finite,
            "threshold": True,
            "comparison": "==",
            "passed": summary_finite,
        }
    )

    # Diagnostic only. We do NOT require clipping to improve accuracy.
    ess_monotone_fraction = float(
        monotonicity[
            "ess_nondecreasing"
        ].mean()
    )

    checks.append(
        {
            "check": (
                "ess_nondecreasing_fraction_diagnostic"
            ),
            "value": ess_monotone_fraction,
            "threshold": "diagnostic_only",
            "comparison": "report",
            "passed": True,
        }
    )

    return pd.DataFrame(
        checks
    )


def make_error_plot(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    learned: bool,
) -> None:

    metric = (
        "learned_ipcw_mae_mean"
        if learned
        else
        "true_ipcw_mae_mean"
    )

    title = (
        "Learned survival IPCW: "
        "clipping under weakening overlap"
        if learned
        else
        "Oracle survival IPCW: "
        "clipping under weakening overlap"
    )

    figure = plt.figure(
        figsize=(8, 5)
    )

    for clip in sorted(
        summary[
            "clip"
        ].unique()
    ):
        part = (
            summary.loc[
                np.isclose(
                    summary["clip"],
                    clip,
                )
            ]
            .sort_values(
                "strength"
            )
        )

        plt.plot(
            part[
                "strength"
            ],
            part[
                metric
            ],
            marker="o",
            label=f"clip={clip:g}",
        )

    plt.xlabel(
        "Censor-selection strength"
    )

    plt.ylabel(
        "MAE to linked calibration curve"
    )

    plt.title(
        title
    )

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    filename = (
        "learned_ipcw_error_vs_clip.png"
        if learned
        else
        "oracle_ipcw_error_vs_clip.png"
    )

    figure.savefig(
        output_dir
        / filename,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def make_ess_plot(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:

    figure = plt.figure(
        figsize=(8, 5)
    )

    for clip in sorted(
        summary[
            "clip"
        ].unique()
    ):
        part = (
            summary.loc[
                np.isclose(
                    summary["clip"],
                    clip,
                )
            ]
            .sort_values(
                "strength"
            )
        )

        plt.plot(
            part[
                "strength"
            ],
            part[
                "learned_ipcw_ess_mean"
            ],
            marker="o",
            label=f"clip={clip:g}",
        )

    plt.xlabel(
        "Censor-selection strength"
    )

    plt.ylabel(
        "Effective sample size"
    )

    plt.title(
        "Learned IPCW effective sample size "
        "under clipping"
    )

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    figure.savefig(
        output_dir
        / "learned_ipcw_ess_vs_clip.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_args()

    clips = parse_clips(
        args.clips
    )

    phase7b_dir = Path(
        args.phase7b_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        intercepts,
        phase7b_replications,
    ) = load_phase7b_design(
        phase7b_dir
    )

    if args.replicates < 1:
        raise ValueError(
            "Replicates must be positive."
        )

    maximum_available_replication = int(
        phase7b_replications[
            "replication"
        ].max()
    ) + 1

    if (
        args.replicates
        > maximum_available_replication
    ):
        raise ValueError(
            "Phase 7F should not exceed the frozen "
            "Phase 7B replication count. Requested "
            f"{args.replicates}; available "
            f"{maximum_available_replication}."
        )

    print("=" * 88)
    print(
        "ACTIONAUDIT PHASE 7F-A: "
        "LIMITED-OVERLAP REMEDY BENCHMARK"
    )
    print("=" * 88)

    print(
        "Strengths:",
        intercepts[
            "strength"
        ].tolist(),
    )

    print(
        "Clipping thresholds:",
        clips,
    )

    print(
        "Replicates:",
        args.replicates,
    )

    print(
        "Frozen Phase 7B reference clip:",
        REFERENCE_CLIP,
    )

    rows = []

    total = (
        len(intercepts)
        * args.replicates
    )

    counter = 0

    for _, design in (
        intercepts.iterrows()
    ):
        strength = float(
            design[
                "strength"
            ]
        )

        intercept = float(
            design[
                "censor_intercept"
            ]
        )

        for replication in range(
            args.replicates
        ):
            counter += 1

            print(
                f"[{counter:4d}/{total:4d}] "
                f"strength={strength:g} "
                f"replication={replication}"
            )

            replication_rows = (
                run_replication(
                    strength=strength,
                    intercept=intercept,
                    replication=replication,
                    clips=clips,
                    development_n=(
                        args.development_n
                    ),
                    test_n=args.test_n,
                    horizon=args.horizon,
                    n_bins=args.n_bins,
                    logistic_c=(
                        args.logistic_c
                    ),
                    seed=args.seed,
                )
            )

            rows.extend(
                replication_rows
            )

    results = pd.DataFrame(
        rows
    )

    summary = summarize(
        results
    )

    # Save expensive simulation outputs before
    # post-processing and validation.
    results.to_csv(
        output_dir
        / "remedy_replication_results.csv.gz",
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        output_dir
        / "remedy_summary.csv",
        index=False,
    )

    reproduction = (
        reproduce_phase7b_reference(
            results,
            phase7b_replications,
            args.replicates,
        )
    )

    monotonicity = (
        clipping_monotonicity(
            results
        )
    )

    checks = validate(
        results=results,
        summary=summary,
        reproduction=reproduction,
        monotonicity=monotonicity,
        intercepts=intercepts,
        clips=clips,
        replicates=args.replicates,
        target_observed_fraction=(
            args.target_observed_fraction
        ),
    )

    passed = bool(
        checks[
            "passed"
        ].all()
    )

    results.to_csv(
        output_dir
        / "remedy_replication_results.csv.gz",
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        output_dir
        / "remedy_summary.csv",
        index=False,
    )

    reproduction.to_csv(
        output_dir
        / "clip005_phase7b_reproduction.csv",
        index=False,
    )

    monotonicity.to_csv(
        output_dir
        / "clipping_monotonicity.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    make_error_plot(
        summary,
        output_dir,
        learned=True,
    )

    make_error_plot(
        summary,
        output_dir,
        learned=False,
    )

    make_ess_plot(
        summary,
        output_dir,
    )

    design_text = f"""Phase 7F-A design

Purpose:
Test practical weight-truncation remedies as overlap weakens
while preserving the exact controlled Phase 7B data-generating
process.

Phase 7B source:
{phase7b_dir}

Strengths:
{intercepts["strength"].tolist()}

Clipping thresholds:
{clips}

Reference clipping threshold:
{REFERENCE_CLIP}

Replicates:
{args.replicates}

Development n:
{args.development_n}

Test n:
{args.test_n}

Horizon:
{args.horizon}

Calibration bins:
{args.n_bins}

Seed:
{args.seed}

Interpretation:
This is a survival-risk-set IPCW remedy experiment. It does not
substitute a binary label-observability propensity model for the
survival censoring process.

The clipping threshold lower-bounds estimated censoring survival
probabilities, thereby upper-bounding inverse censoring weights.
Stronger clipping can improve variance and effective sample size
but can increase bias. No validation criterion requires a remedy
to improve estimation error.

Oracle nuisance results use the true simulated event and censoring
hazards. Learned nuisance results use the same correctly specified
observed-covariate pooled-logistic nuisance family as Phase 7B.

The clip=0.05 condition must reproduce Phase 7B numerically.
"""

    (
        output_dir
        / "phase7f_design.txt"
    ).write_text(
        design_text
    )

    status = (
        "phase7f_overlap_remedy_benchmark_passed"
        if passed
        else
        "phase7f_overlap_remedy_benchmark_failed"
    )

    (
        output_dir
        / "phase7f_status.txt"
    ).write_text(
        status + "\n"
    )

    print("\n" + "=" * 88)
    print("PHASE 7F-A SUMMARY")
    print("=" * 88)

    display_columns = [
        "strength",
        "clip",
        "replicates",
        "observed_fraction_mean",
        "g_tau_q05_mean",
        "true_ipcw_mae_mean",
        "true_ipcw_ess_mean",
        "true_fraction_event_g_clipped_mean",
        "true_maximum_event_weight_mean",
        "learned_ipcw_mae_mean",
        "learned_ipcw_ess_mean",
        "learned_fraction_event_g_clipped_mean",
        "learned_maximum_event_weight_mean",
        "learned_ipcw_improvement_vs_clip005",
        "learned_ipcw_improvement_vs_km",
    ]

    print(
        "\nRemedy summary"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nPhase 7B reproduction"
    )

    print(
        reproduction.to_string(
            index=False
        )
    )

    print(
        "\nValidation"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:",
        status,
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
