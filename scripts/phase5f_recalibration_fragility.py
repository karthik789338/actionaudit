"""Phase 5F: restricted-outcome recalibration can target the wrong curve.

A fixed predictor and event process are shared across observation
policies. Recalibration is fitted only on policy-observed outcomes
in a calibration split and evaluated on a separate test split.

The experiment asks whether standard recalibration:

  * improves calibration on the restricted observed subset;
  * worsens calibration for the complete reference population;
  * fails to transfer across observation policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from phase5e_observation_policy_shift import (
    ObservationPolicy,
    apply_observation_policy,
    binary_log_loss,
    fit_calibration_intercept_slope,
    generate_fixed_population,
    policy_definitions,
)


POLICY_ORDER = [
    "conservative_late",
    "baseline",
    "aggressive_early",
    "risk_neutral",
    "high_risk_exit",
]


METHOD_ORDER = [
    "identity",
    "intercept_only",
    "logistic",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test fragility of recalibration fitted "
            "under restricted follow-up."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5f_recalibration_fragility"
        ),
    )

    parser.add_argument(
        "--n-patients",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.50,
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
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


def selected_policies() -> list[ObservationPolicy]:
    available = {
        policy.name: policy
        for policy in policy_definitions()
    }

    return [
        available[name]
        for name in POLICY_ORDER
    ]


def assign_split(
    population: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not (
        0.20
        <= calibration_fraction
        <= 0.80
    ):
        raise ValueError(
            "calibration-fraction must be "
            "between 0.20 and 0.80"
        )

    rng = np.random.default_rng(
        seed + 101
    )

    order = rng.permutation(
        len(
            population
        )
    )

    n_calibration = int(
        round(
            calibration_fraction
            * len(
                population
            )
        )
    )

    split = np.full(
        len(
            population
        ),
        "test",
        dtype=object,
    )

    split[
        order[
            :n_calibration
        ]
    ] = "calibration"

    result = population.copy()

    result[
        "split"
    ] = split

    return result


def fit_intercept_only(
    outcome: np.ndarray,
    prediction: np.ndarray,
    *,
    lower: float = -12.0,
    upper: float = 12.0,
) -> float:
    y = np.asarray(
        outcome,
        dtype=float,
    )

    score = logit(
        np.clip(
            np.asarray(
                prediction,
                dtype=float,
            ),
            1e-8,
            1.0 - 1e-8,
        )
    )

    target = float(
        y.mean()
    )

    left = float(
        lower
    )

    right = float(
        upper
    )

    for _ in range(
        200
    ):
        midpoint = (
            left
            + right
        ) / 2.0

        fitted_mean = float(
            np.mean(
                expit(
                    score
                    + midpoint
                )
            )
        )

        if abs(
            fitted_mean
            - target
        ) < 1e-12:
            return float(
                midpoint
            )

        if fitted_mean < target:
            left = midpoint
        else:
            right = midpoint

    return float(
        (
            left
            + right
        )
        / 2.0
    )


def fit_recalibration(
    outcome: np.ndarray,
    prediction: np.ndarray,
    method: str,
) -> tuple[float, float]:
    if method == "identity":
        return (
            0.0,
            1.0,
        )

    if len(
        np.unique(
            outcome
        )
    ) < 2:
        raise ValueError(
            "Recalibration sample has only one class"
        )

    if method == "intercept_only":
        return (
            fit_intercept_only(
                outcome,
                prediction,
            ),
            1.0,
        )

    if method == "logistic":
        return fit_calibration_intercept_slope(
            outcome,
            prediction,
        )

    raise ValueError(
        f"Unknown recalibration method: {method}"
    )


def apply_recalibration(
    prediction: np.ndarray,
    *,
    intercept: float,
    slope: float,
) -> np.ndarray:
    score = logit(
        np.clip(
            np.asarray(
                prediction,
                dtype=float,
            ),
            1e-8,
            1.0 - 1e-8,
        )
    )

    return expit(
        intercept
        + slope * score
    )


def evaluation_metrics(
    frame: pd.DataFrame,
    recalibrated_prediction: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, float]:
    if len(
        frame
    ) == 0:
        raise ValueError(
            "Cannot evaluate an empty sample"
        )

    outcome = frame[
        "outcome"
    ].to_numpy(
        dtype=float
    )

    prediction = np.clip(
        np.asarray(
            recalibrated_prediction,
            dtype=float,
        ),
        1e-8,
        1.0 - 1e-8,
    )

    bin_rows = []

    for risk_bin in range(
        n_bins
    ):
        mask = frame[
            "risk_bin"
        ].to_numpy(
            dtype=int
        ) == risk_bin

        if not mask.any():
            continue

        bin_rows.append(
            {
                "weight": float(
                    mask.mean()
                ),
                "outcome_mean": float(
                    outcome[
                        mask
                    ].mean()
                ),
                "prediction_mean": float(
                    prediction[
                        mask
                    ].mean()
                ),
            }
        )

    bins = pd.DataFrame(
        bin_rows
    )

    ici = float(
        np.sum(
            bins[
                "weight"
            ]
            * np.abs(
                bins[
                    "outcome_mean"
                ]
                - bins[
                    "prediction_mean"
                ]
            )
        )
    )

    intercept, slope = (
        fit_calibration_intercept_slope(
            outcome,
            prediction,
        )
    )

    return {
        "n_evaluated": float(
            len(
                frame
            )
        ),
        "event_rate": float(
            outcome.mean()
        ),
        "mean_prediction": float(
            prediction.mean()
        ),
        "ici": ici,
        "signed_calibration_gap": float(
            np.mean(
                outcome
                - prediction
            )
        ),
        "brier": float(
            np.mean(
                (
                    outcome
                    - prediction
                )
                ** 2
            )
        ),
        "log_loss": binary_log_loss(
            outcome,
            prediction,
        ),
        "calibration_intercept": (
            intercept
        ),
        "calibration_slope": (
            slope
        ),
    }


def prepare_policy_frames(
    population: pd.DataFrame,
    *,
    policies: list[ObservationPolicy],
    horizon: int,
) -> dict[str, pd.DataFrame]:
    split_lookup = population[
        [
            "patient_id",
            "split",
        ]
    ]

    frames = {}

    for policy in policies:
        frame = apply_observation_policy(
            population,
            policy,
            horizon=horizon,
        )

        frame = frame.merge(
            split_lookup,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )

        frames[
            policy.name
        ] = frame

    return frames


def fit_policy_models(
    policy_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for policy_name in POLICY_ORDER:
        frame = policy_frames[
            policy_name
        ]

        calibration = frame.loc[
            frame[
                "split"
            ].eq(
                "calibration"
            )
            & frame[
                "observed"
            ].eq(
                1
            )
        ]

        for method in METHOD_ORDER:
            intercept, slope = (
                fit_recalibration(
                    calibration[
                        "outcome"
                    ].to_numpy(
                        dtype=float
                    ),
                    calibration[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    ),
                    method,
                )
            )

            rows.append(
                {
                    "fit_policy": (
                        policy_name
                    ),
                    "method": method,
                    "n_calibration_observed": int(
                        len(
                            calibration
                        )
                    ),
                    "calibration_event_rate": float(
                        calibration[
                            "outcome"
                        ].mean()
                    ),
                    "intercept": (
                        intercept
                    ),
                    "slope": (
                        slope
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def point_evaluation(
    policy_frames: dict[str, pd.DataFrame],
    parameters: pd.DataFrame,
    *,
    n_bins: int,
) -> pd.DataFrame:
    rows = []

    reference_test = (
        policy_frames[
            "baseline"
        ]
        .loc[
            lambda data: (
                data[
                    "split"
                ].eq(
                    "test"
                )
            )
        ]
        .copy()
    )

    for _, parameter in (
        parameters.iterrows()
    ):
        fit_policy = str(
            parameter[
                "fit_policy"
            ]
        )

        method = str(
            parameter[
                "method"
            ]
        )

        intercept = float(
            parameter[
                "intercept"
            ]
        )

        slope = float(
            parameter[
                "slope"
            ]
        )

        reference_prediction = (
            apply_recalibration(
                reference_test[
                    "prediction"
                ].to_numpy(
                    dtype=float
                ),
                intercept=intercept,
                slope=slope,
            )
        )

        rows.append(
            {
                "fit_policy": (
                    fit_policy
                ),
                "method": method,
                "evaluation_target": (
                    "reference_full"
                ),
                "evaluation_policy": (
                    "full_population"
                ),
                "intercept": intercept,
                "slope": slope,
                **evaluation_metrics(
                    reference_test,
                    reference_prediction,
                    n_bins=n_bins,
                ),
            }
        )

        for evaluation_policy in (
            POLICY_ORDER
        ):
            evaluation = (
                policy_frames[
                    evaluation_policy
                ]
                .loc[
                    lambda data: (
                        data[
                            "split"
                        ].eq(
                            "test"
                        )
                        & data[
                            "observed"
                        ].eq(
                            1
                        )
                    )
                ]
                .copy()
            )

            restricted_prediction = (
                apply_recalibration(
                    evaluation[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    ),
                    intercept=intercept,
                    slope=slope,
                )
            )

            rows.append(
                {
                    "fit_policy": (
                        fit_policy
                    ),
                    "method": method,
                    "evaluation_target": (
                        "restricted"
                    ),
                    "evaluation_policy": (
                        evaluation_policy
                    ),
                    "intercept": intercept,
                    "slope": slope,
                    **evaluation_metrics(
                        evaluation,
                        restricted_prediction,
                        n_bins=n_bins,
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def metric_lookup(
    results: pd.DataFrame,
    *,
    fit_policy: str,
    method: str,
    evaluation_target: str,
    evaluation_policy: str,
    metric: str,
) -> float:
    selected = results.loc[
        results[
            "fit_policy"
        ].eq(
            fit_policy
        )
        & results[
            "method"
        ].eq(
            method
        )
        & results[
            "evaluation_target"
        ].eq(
            evaluation_target
        )
        & results[
            "evaluation_policy"
        ].eq(
            evaluation_policy
        ),
        metric,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Metric lookup failed: "
            f"{fit_policy}, {method}, "
            f"{evaluation_target}, "
            f"{evaluation_policy}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


CONTRAST_METRICS = [
    "ici",
    "brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
]


def point_contrasts(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for policy in POLICY_ORDER:
        for method in [
            "intercept_only",
            "logistic",
        ]:
            row = {
                "fit_policy": policy,
                "method": method,
            }

            for metric in CONTRAST_METRICS:
                restricted_method = metric_lookup(
                    results,
                    fit_policy=policy,
                    method=method,
                    evaluation_target="restricted",
                    evaluation_policy=policy,
                    metric=metric,
                )

                restricted_identity = metric_lookup(
                    results,
                    fit_policy=policy,
                    method="identity",
                    evaluation_target="restricted",
                    evaluation_policy=policy,
                    metric=metric,
                )

                reference_method = metric_lookup(
                    results,
                    fit_policy=policy,
                    method=method,
                    evaluation_target="reference_full",
                    evaluation_policy="full_population",
                    metric=metric,
                )

                reference_identity = metric_lookup(
                    results,
                    fit_policy=policy,
                    method="identity",
                    evaluation_target="reference_full",
                    evaluation_policy="full_population",
                    metric=metric,
                )

                row[
                    f"restricted_{metric}_change"
                ] = (
                    restricted_method
                    - restricted_identity
                )

                row[
                    f"reference_{metric}_change"
                ] = (
                    reference_method
                    - reference_identity
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def bootstrap_once(
    policy_frames: dict[str, pd.DataFrame],
    *,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    n_bins: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    calibration_frames = {}

    test_frames = {}

    for policy in POLICY_ORDER:
        calibration_base = (
            policy_frames[
                policy
            ]
            .loc[
                lambda data: (
                    data[
                        "split"
                    ].eq(
                        "calibration"
                    )
                )
            ]
            .reset_index(
                drop=True
            )
        )

        test_base = (
            policy_frames[
                policy
            ]
            .loc[
                lambda data: (
                    data[
                        "split"
                    ].eq(
                        "test"
                    )
                )
            ]
            .reset_index(
                drop=True
            )
        )

        calibration_frames[
            policy
        ] = calibration_base.iloc[
            calibration_indices
        ].reset_index(
            drop=True
        )

        test_frames[
            policy
        ] = test_base.iloc[
            test_indices
        ].reset_index(
            drop=True
        )

    parameters = {}

    for policy in POLICY_ORDER:
        observed_calibration = (
            calibration_frames[
                policy
            ]
            .loc[
                lambda data: (
                    data[
                        "observed"
                    ].eq(
                        1
                    )
                )
            ]
        )

        for method in [
            "intercept_only",
            "logistic",
        ]:
            parameters[
                (
                    policy,
                    method,
                )
            ] = fit_recalibration(
                observed_calibration[
                    "outcome"
                ].to_numpy(
                    dtype=float
                ),
                observed_calibration[
                    "prediction"
                ].to_numpy(
                    dtype=float
                ),
                method,
            )

    reference_test = test_frames[
        "baseline"
    ]

    identity_reference = (
        evaluation_metrics(
            reference_test,
            reference_test[
                "prediction"
            ].to_numpy(
                dtype=float
            ),
            n_bins=n_bins,
        )
    )

    identity_restricted = {}

    for policy in POLICY_ORDER:
        observed_test = test_frames[
            policy
        ].loc[
            lambda data: (
                data[
                    "observed"
                ].eq(
                    1
                )
            )
        ]

        identity_restricted[
            policy
        ] = evaluation_metrics(
            observed_test,
            observed_test[
                "prediction"
            ].to_numpy(
                dtype=float
            ),
            n_bins=n_bins,
        )

    contrast_rows = []

    fitted_metrics = {}

    for policy in POLICY_ORDER:
        observed_test = test_frames[
            policy
        ].loc[
            lambda data: (
                data[
                    "observed"
                ].eq(
                    1
                )
            )
        ]

        for method in [
            "intercept_only",
            "logistic",
        ]:
            intercept, slope = parameters[
                (
                    policy,
                    method,
                )
            ]

            reference_prediction = (
                apply_recalibration(
                    reference_test[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    ),
                    intercept=intercept,
                    slope=slope,
                )
            )

            restricted_prediction = (
                apply_recalibration(
                    observed_test[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    ),
                    intercept=intercept,
                    slope=slope,
                )
            )

            reference_metrics = (
                evaluation_metrics(
                    reference_test,
                    reference_prediction,
                    n_bins=n_bins,
                )
            )

            restricted_metrics = (
                evaluation_metrics(
                    observed_test,
                    restricted_prediction,
                    n_bins=n_bins,
                )
            )

            fitted_metrics[
                (
                    policy,
                    method,
                )
            ] = {
                "reference": (
                    reference_metrics
                ),
                "restricted": (
                    restricted_metrics
                ),
            }

            row = {
                "fit_policy": policy,
                "method": method,
                "intercept": intercept,
                "slope": slope,
            }

            for metric in (
                CONTRAST_METRICS
            ):
                row[
                    f"restricted_{metric}_change"
                ] = (
                    restricted_metrics[
                        metric
                    ]
                    - identity_restricted[
                        policy
                    ][
                        metric
                    ]
                )

                row[
                    f"reference_{metric}_change"
                ] = (
                    reference_metrics[
                        metric
                    ]
                    - identity_reference[
                        metric
                    ]
                )

            contrast_rows.append(
                row
            )

    policy_comparison_rows = []

    for method in [
        "intercept_only",
        "logistic",
    ]:
        aggressive_harm = (
            fitted_metrics[
                (
                    "aggressive_early",
                    method,
                )
            ][
                "reference"
            ][
                "ici"
            ]
            - identity_reference[
                "ici"
            ]
        )

        baseline_harm = (
            fitted_metrics[
                (
                    "baseline",
                    method,
                )
            ][
                "reference"
            ][
                "ici"
            ]
            - identity_reference[
                "ici"
            ]
        )

        conservative_harm = (
            fitted_metrics[
                (
                    "conservative_late",
                    method,
                )
            ][
                "reference"
            ][
                "ici"
            ]
            - identity_reference[
                "ici"
            ]
        )

        policy_comparison_rows.append(
            {
                "method": method,
                "metric": (
                    "aggressive_minus_baseline_"
                    "reference_ici_harm"
                ),
                "value": (
                    aggressive_harm
                    - baseline_harm
                ),
            }
        )

        policy_comparison_rows.append(
            {
                "method": method,
                "metric": (
                    "baseline_minus_conservative_"
                    "reference_ici_harm"
                ),
                "value": (
                    baseline_harm
                    - conservative_harm
                ),
            }
        )

    # Cross-policy transfer: compare a baseline-fitted model
    # against a policy-specific fit on the same restricted target.
    baseline_intercept, baseline_slope = (
        parameters[
            (
                "baseline",
                "logistic",
            )
        ]
    )

    for evaluation_policy in [
        "aggressive_early",
        "risk_neutral",
        "high_risk_exit",
        "conservative_late",
    ]:
        observed_test = test_frames[
            evaluation_policy
        ].loc[
            lambda data: (
                data[
                    "observed"
                ].eq(
                    1
                )
            )
        ]

        baseline_prediction = (
            apply_recalibration(
                observed_test[
                    "prediction"
                ].to_numpy(
                    dtype=float
                ),
                intercept=(
                    baseline_intercept
                ),
                slope=baseline_slope,
            )
        )

        own_intercept, own_slope = (
            parameters[
                (
                    evaluation_policy,
                    "logistic",
                )
            ]
        )

        own_prediction = (
            apply_recalibration(
                observed_test[
                    "prediction"
                ].to_numpy(
                    dtype=float
                ),
                intercept=own_intercept,
                slope=own_slope,
            )
        )

        baseline_metrics = (
            evaluation_metrics(
                observed_test,
                baseline_prediction,
                n_bins=n_bins,
            )
        )

        own_metrics = evaluation_metrics(
            observed_test,
            own_prediction,
            n_bins=n_bins,
        )

        for metric in [
            "ici",
            "brier",
            "log_loss",
        ]:
            policy_comparison_rows.append(
                {
                    "method": "logistic",
                    "metric": (
                        "baseline_fit_minus_own_fit_"
                        f"{evaluation_policy}_"
                        f"restricted_{metric}"
                    ),
                    "value": (
                        baseline_metrics[
                            metric
                        ]
                        - own_metrics[
                            metric
                        ]
                    ),
                }
            )

    return (
        contrast_rows,
        policy_comparison_rows,
    )


def run_bootstrap(
    policy_frames: dict[str, pd.DataFrame],
    *,
    replicates: int,
    seed: int,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    baseline = policy_frames[
        "baseline"
    ]

    n_calibration = int(
        baseline[
            "split"
        ].eq(
            "calibration"
        ).sum()
    )

    n_test = int(
        baseline[
            "split"
        ].eq(
            "test"
        ).sum()
    )

    contrast_rows = []

    policy_rows = []

    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        calibration_indices = rng.integers(
            0,
            n_calibration,
            size=n_calibration,
        )

        test_indices = rng.integers(
            0,
            n_test,
            size=n_test,
        )

        try:
            (
                replicate_contrasts,
                replicate_policy_rows,
            ) = bootstrap_once(
                policy_frames,
                calibration_indices=(
                    calibration_indices
                ),
                test_indices=(
                    test_indices
                ),
                n_bins=n_bins,
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            invalid += 1
            continue

        for row in replicate_contrasts:
            contrast_rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    **row,
                }
            )

        for row in replicate_policy_rows:
            policy_rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    **row,
                }
            )

        if (
            bootstrap_id + 1
        ) % 100 == 0:
            print(
                f"  completed "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

    return (
        pd.DataFrame(
            contrast_rows
        ),
        pd.DataFrame(
            policy_rows
        ),
        invalid,
    )


def summarize_interval(
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


def contrast_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    value_columns = [
        column
        for column in bootstrap.columns
        if column
        not in {
            "bootstrap_id",
            "fit_policy",
            "method",
        }
    ]

    rows = []

    for (
        fit_policy,
        method,
    ), group in bootstrap.groupby(
        [
            "fit_policy",
            "method",
        ],
        sort=True,
    ):
        for metric in value_columns:
            rows.append(
                {
                    "fit_policy": (
                        fit_policy
                    ),
                    "method": method,
                    "metric": metric,
                    **summarize_interval(
                        group[
                            metric
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def policy_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        method,
        metric,
    ), group in bootstrap.groupby(
        [
            "method",
            "metric",
        ],
        sort=True,
    ):
        rows.append(
            {
                "method": method,
                "metric": metric,
                **summarize_interval(
                    group[
                        "value"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def interval_value(
    intervals: pd.DataFrame,
    *,
    fit_policy: str,
    method: str,
    metric: str,
    column: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "fit_policy"
        ].eq(
            fit_policy
        )
        & intervals[
            "method"
        ].eq(
            method
        )
        & intervals[
            "metric"
        ].eq(
            metric
        ),
        column,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Interval lookup failed for "
            f"{fit_policy}, {method}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def policy_interval_value(
    intervals: pd.DataFrame,
    *,
    method: str,
    metric: str,
    column: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "method"
        ].eq(
            method
        )
        & intervals[
            "metric"
        ].eq(
            metric
        ),
        column,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Policy interval lookup failed for "
            f"{method}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def build_checks(
    *,
    parameters: pd.DataFrame,
    contrast_intervals: pd.DataFrame,
    policy_intervals: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
) -> pd.DataFrame:
    aggressive_restricted_ici_q975 = (
        interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "restricted_ici_change"
            ),
            column="q975",
        )
    )

    aggressive_reference_ici_q025 = (
        interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "reference_ici_change"
            ),
            column="q025",
        )
    )

    aggressive_restricted_logloss_q975 = (
        interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "restricted_log_loss_change"
            ),
            column="q975",
        )
    )

    aggressive_reference_logloss_q025 = (
        interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "reference_log_loss_change"
            ),
            column="q025",
        )
    )

    policy_gradient_q025 = (
        policy_interval_value(
            policy_intervals,
            method="logistic",
            metric=(
                "aggressive_minus_baseline_"
                "reference_ici_harm"
            ),
            column="q025",
        )
    )

    cross_policy_q025 = (
        policy_interval_value(
            policy_intervals,
            method="logistic",
            metric=(
                "baseline_fit_minus_own_fit_"
                "aggressive_early_restricted_ici"
            ),
            column="q025",
        )
    )

    logistic_parameters = (
        parameters.loc[
            parameters[
                "method"
            ].eq(
                "logistic"
            )
        ]
        .set_index(
            "fit_policy"
        )
    )

    slope_range = float(
        logistic_parameters[
            "slope"
        ].max()
        - logistic_parameters[
            "slope"
        ].min()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "aggressive logistic recalibration "
                    "improves restricted ICI"
                ),
                "value": (
                    aggressive_restricted_ici_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    aggressive_restricted_ici_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "aggressive logistic recalibration "
                    "worsens reference ICI"
                ),
                "value": (
                    aggressive_reference_ici_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    aggressive_reference_ici_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "aggressive logistic recalibration "
                    "improves restricted log loss"
                ),
                "value": (
                    aggressive_restricted_logloss_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    aggressive_restricted_logloss_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "aggressive logistic recalibration "
                    "worsens reference log loss"
                ),
                "value": (
                    aggressive_reference_logloss_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    aggressive_reference_logloss_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "reference harm increases under "
                    "aggressive policy"
                ),
                "value": (
                    policy_gradient_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    policy_gradient_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "baseline recalibration transfers "
                    "poorly to aggressive policy"
                ),
                "value": (
                    cross_policy_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    cross_policy_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "recalibration slopes differ materially "
                    "across policies"
                ),
                "value": slope_range,
                "threshold": 0.30,
                "comparison": ">",
                "passed": (
                    slope_range
                    > 0.30
                ),
            },
            {
                "check": (
                    "all requested paired bootstraps valid"
                ),
                "value": (
                    requested_bootstraps
                    - invalid_bootstraps
                ),
                "threshold": (
                    requested_bootstraps
                ),
                "comparison": "==",
                "passed": (
                    invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    population = generate_fixed_population(
        n_patients=args.n_patients,
        horizon=args.horizon,
        seed=args.seed,
        n_bins=args.n_bins,
    )

    population = assign_split(
        population,
        calibration_fraction=(
            args.calibration_fraction
        ),
        seed=args.seed,
    )

    policies = selected_policies()

    policy_frames = prepare_policy_frames(
        population,
        policies=policies,
        horizon=args.horizon,
    )

    parameters = fit_policy_models(
        policy_frames
    )

    evaluation = point_evaluation(
        policy_frames,
        parameters,
        n_bins=args.n_bins,
    )

    contrasts = point_contrasts(
        evaluation
    )

    print(
        "Phase 5F recalibration fragility"
    )

    print(
        f"  fixed patients="
        f"{args.n_patients:,}"
    )

    print(
        f"  calibration patients="
        f"{population['split'].eq('calibration').sum():,}"
    )

    print(
        f"  test patients="
        f"{population['split'].eq('test').sum():,}"
    )

    print(
        f"  bootstrap replicates="
        f"{args.bootstrap_replicates}"
    )

    print(
        "\nRunning paired calibration/test bootstrap..."
    )

    (
        bootstrap_contrasts,
        bootstrap_policy,
        invalid_bootstraps,
    ) = run_bootstrap(
        policy_frames,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
        n_bins=args.n_bins,
    )

    contrast_intervals = (
        contrast_interval_table(
            bootstrap_contrasts
        )
    )

    policy_intervals = (
        policy_interval_table(
            bootstrap_policy
        )
    )

    checks = build_checks(
        parameters=parameters,
        contrast_intervals=(
            contrast_intervals
        ),
        policy_intervals=(
            policy_intervals
        ),
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
    )

    preliminary_status = (
        "phase5f_recalibration_assumption_fragility_supported"
        if checks[
            "passed"
        ].all()
        else (
            "phase5f_recalibration_fragility_"
            "requires_review"
        )
    )

    status = (
        "phase5f_recalibration_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    parameters.to_csv(
        output_dir
        / "recalibration_parameters.csv",
        index=False,
    )

    evaluation.to_csv(
        output_dir
        / "evaluation_summary.csv",
        index=False,
    )

    contrasts.to_csv(
        output_dir
        / "point_contrasts.csv",
        index=False,
    )

    bootstrap_contrasts.to_csv(
        output_dir
        / "bootstrap_contrast_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_policy.to_csv(
        output_dir
        / "bootstrap_policy_comparisons.csv.gz",
        index=False,
        compression="gzip",
    )

    contrast_intervals.to_csv(
        output_dir
        / "contrast_bootstrap_intervals.csv",
        index=False,
    )

    policy_intervals.to_csv(
        output_dir
        / "policy_comparison_intervals.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5F",
        "hypothesis": (
            "Standard recalibration may be "
            "assumption-fragile under restricted "
            "follow-up."
        ),
        "n_patients": (
            args.n_patients
        ),
        "calibration_fraction": (
            args.calibration_fraction
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstraps": (
            invalid_bootstraps
        ),
        "recalibration_methods": (
            METHOD_ORDER
        ),
        "fit_target": (
            "Policy-observed outcomes in the "
            "calibration split only."
        ),
        "evaluation_targets": [
            (
                "Policy-observed outcomes in an "
                "independent test split."
            ),
            (
                "Complete reference outcomes in the "
                "same independent test split."
            ),
            (
                "Restricted outcomes under different "
                "observation policies."
            ),
        ],
        "interpretation": (
            "Restricted-outcome recalibration may "
            "optimize a policy-selected calibration "
            "curve rather than the unchanged full-"
            "population target."
        ),
        "preliminary_status": (
            preliminary_status
            if args.smoke
            else None
        ),
        "status": status,
    }

    (
        output_dir
        / "recalibration_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5f_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nRecalibration parameters"
    )

    print(
        parameters.to_string(
            index=False
        )
    )

    print(
        "\nPoint contrasts"
    )

    print(
        contrasts.to_string(
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
