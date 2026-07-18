"""Phase 5E: observation-policy shift can mimic model drift.

A single frozen synthetic population is generated with fixed:

  * patient covariates;
  * risk predictions;
  * binary outcomes;
  * latent event times;
  * common random numbers.

Only the observation/discharge policy changes.

The experiment tests whether restricted calibration changes even
though reference calibration, predictions, and event outcomes are
identical across policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


ACTION_ORDER = [
    "home",
    "home_health",
    "skilled_nursing",
    "rehabilitation",
    "hospice",
    "long_term_or_assisted",
    "acute_hospital_transfer",
    "other_exit",
]


BASE_ACTION_PROBABILITIES = np.asarray(
    [
        0.45,
        0.18,
        0.12,
        0.06,
        0.03,
        0.04,
        0.04,
        0.08,
    ],
    dtype=float,
)


ACTION_RISK_LOADINGS = np.asarray(
    [
        -0.80,
        -0.20,
        0.45,
        0.30,
        1.50,
        0.60,
        1.00,
        0.10,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ObservationPolicy:
    name: str
    exit_intercept: float
    risk_selection: float
    timing_power: float
    action_shifts: dict[str, float]
    description: str


def policy_definitions() -> list[ObservationPolicy]:
    return [
        ObservationPolicy(
            name="conservative_late",
            exit_intercept=-1.80,
            risk_selection=0.80,
            timing_power=0.70,
            action_shifts={},
            description=(
                "Lower discharge probability and later exits."
            ),
        ),
        ObservationPolicy(
            name="baseline",
            exit_intercept=-0.20,
            risk_selection=1.00,
            timing_power=1.20,
            action_shifts={},
            description=(
                "Reference observation policy."
            ),
        ),
        ObservationPolicy(
            name="aggressive_early",
            exit_intercept=1.50,
            risk_selection=1.20,
            timing_power=2.00,
            action_shifts={
                "home": 0.30,
                "home_health": 0.15,
            },
            description=(
                "Higher discharge probability and earlier exits."
            ),
        ),
        ObservationPolicy(
            name="risk_neutral",
            exit_intercept=0.50,
            risk_selection=0.00,
            timing_power=1.50,
            action_shifts={},
            description=(
                "Exit probability does not depend on model risk."
            ),
        ),
        ObservationPolicy(
            name="high_risk_exit",
            exit_intercept=0.20,
            risk_selection=-1.00,
            timing_power=1.50,
            action_shifts={
                "acute_hospital_transfer": 0.80,
                "hospice": 0.30,
            },
            description=(
                "Higher-risk patients are more likely to exit."
            ),
        ),
        ObservationPolicy(
            name="baseline_transfer_mix",
            exit_intercept=-0.20,
            risk_selection=1.00,
            timing_power=1.20,
            action_shifts={
                "acute_hospital_transfer": 1.50,
                "hospice": 0.60,
                "home": -0.80,
            },
            description=(
                "Baseline observability with a transfer-heavy "
                "action-label mix."
            ),
        ),
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether observation-policy shift "
            "can mimic model drift."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5e_observation_policy_shift"
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
        default=5,
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


def sha256_arrays(
    *arrays: np.ndarray,
) -> str:
    digest = hashlib.sha256()

    for array in arrays:
        contiguous = np.ascontiguousarray(
            array
        )

        digest.update(
            contiguous.view(
                np.uint8
            )
        )

    return digest.hexdigest()


def generate_fixed_population(
    *,
    n_patients: int,
    horizon: int,
    seed: int,
    n_bins: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        seed
    )

    x1 = rng.normal(
        size=n_patients
    )

    x2 = rng.normal(
        size=n_patients
    )

    x3 = rng.binomial(
        1,
        0.35,
        size=n_patients,
    )

    latent_logit = (
        -2.80
        + 0.90 * x1
        + 0.45 * x2
        + 0.60 * x3
        - 0.20 * x1 * x2
    )

    prediction = expit(
        latent_logit
    )

    outcome = rng.binomial(
        1,
        prediction,
    ).astype(
        int
    )

    event_uniform = rng.uniform(
        size=n_patients
    )

    event_time_if_event = (
        1.0
        + (
            horizon
            - 1.0
        )
        * np.power(
            event_uniform,
            1.30
            + 2.00
            * prediction,
        )
    )

    event_time = np.where(
        outcome == 1,
        event_time_if_event,
        np.inf,
    )

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        prediction,
        quantiles,
    )

    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    risk_bin = np.digitize(
        prediction,
        edges[
            1:-1
        ],
        right=True,
    )

    frame = pd.DataFrame(
        {
            "patient_id": np.arange(
                n_patients,
                dtype=int,
            ),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "latent_logit": (
                latent_logit
            ),
            "prediction": prediction,
            "outcome": outcome,
            "event_time": event_time,
            "risk_bin": risk_bin,
            "u_exit": rng.uniform(
                size=n_patients
            ),
            "u_time": rng.uniform(
                size=n_patients
            ),
            "u_action": rng.uniform(
                size=n_patients
            ),
        }
    )

    return frame


def action_shift_vector(
    policy: ObservationPolicy,
) -> np.ndarray:
    shifts = np.zeros(
        len(
            ACTION_ORDER
        ),
        dtype=float,
    )

    for action, value in (
        policy.action_shifts.items()
    ):
        if action not in ACTION_ORDER:
            raise ValueError(
                f"Unknown action shift: {action}"
            )

        shifts[
            ACTION_ORDER.index(
                action
            )
        ] = float(
            value
        )

    return shifts


def assign_actions(
    population: pd.DataFrame,
    policy: ObservationPolicy,
) -> np.ndarray:
    standardized_risk = (
        population[
            "latent_logit"
        ].to_numpy(
            dtype=float
        )
        - population[
            "latent_logit"
        ].mean()
    ) / population[
        "latent_logit"
    ].std(
        ddof=0
    )

    base_logits = np.log(
        BASE_ACTION_PROBABILITIES
    )

    logits = (
        base_logits[
            None,
            :
        ]
        + standardized_risk[
            :,
            None,
        ]
        * ACTION_RISK_LOADINGS[
            None,
            :
        ]
        + action_shift_vector(
            policy
        )[
            None,
            :
        ]
    )

    logits = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    probabilities = np.exp(
        logits
    )

    probabilities = (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    cumulative = np.cumsum(
        probabilities,
        axis=1,
    )

    action_index = (
        population[
            "u_action"
        ]
        .to_numpy(
            dtype=float
        )[
            :,
            None,
        ]
        > cumulative
    ).sum(
        axis=1
    )

    action_index = np.minimum(
        action_index,
        len(
            ACTION_ORDER
        )
        - 1,
    )

    return np.asarray(
        ACTION_ORDER,
        dtype=object,
    )[
        action_index
    ]


def apply_observation_policy(
    population: pd.DataFrame,
    policy: ObservationPolicy,
    *,
    horizon: int,
) -> pd.DataFrame:
    standardized_risk = (
        population[
            "latent_logit"
        ].to_numpy(
            dtype=float
        )
        - population[
            "latent_logit"
        ].mean()
    ) / population[
        "latent_logit"
    ].std(
        ddof=0
    )

    exit_probability = expit(
        policy.exit_intercept
        - policy.risk_selection
        * standardized_risk
    )

    planned_exit = (
        population[
            "u_exit"
        ].to_numpy(
            dtype=float
        )
        < exit_probability
    )

    exit_time = np.where(
        planned_exit,
        1.0
        + (
            horizon
            - 2.0
        )
        * np.power(
            population[
                "u_time"
            ].to_numpy(
                dtype=float
            ),
            policy.timing_power,
        ),
        horizon
        + 1.0,
    )

    event_time = population[
        "event_time"
    ].to_numpy(
        dtype=float
    )

    observed = (
        (
            event_time
            <= exit_time
        )
        | (
            exit_time
            >= horizon
        )
    ).astype(
        int
    )

    action = assign_actions(
        population,
        policy,
    )

    result = population[
        [
            "patient_id",
            "prediction",
            "outcome",
            "event_time",
            "risk_bin",
        ]
    ].copy()

    result[
        "policy"
    ] = policy.name

    result[
        "exit_probability"
    ] = exit_probability

    result[
        "exit_time"
    ] = exit_time

    result[
        "planned_exit"
    ] = planned_exit.astype(
        int
    )

    result[
        "observed"
    ] = observed

    result[
        "action"
    ] = action

    return result


def binary_log_loss(
    outcome: np.ndarray,
    prediction: np.ndarray,
) -> float:
    probability = np.clip(
        prediction,
        1e-12,
        1.0 - 1e-12,
    )

    return float(
        -np.mean(
            outcome
            * np.log(
                probability
            )
            + (
                1
                - outcome
            )
            * np.log(
                1.0
                - probability
            )
        )
    )


def fit_calibration_intercept_slope(
    outcome: np.ndarray,
    prediction: np.ndarray,
) -> tuple[
    float,
    float,
]:
    y = np.asarray(
        outcome,
        dtype=float,
    )

    p = np.clip(
        np.asarray(
            prediction,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    if (
        len(
            np.unique(
                y
            )
        )
        < 2
    ):
        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )

    score = logit(
        p
    )

    design = np.column_stack(
        [
            np.ones(
                len(
                    y
                )
            ),
            score,
        ]
    )

    coefficients = np.asarray(
        [
            0.0,
            1.0,
        ],
        dtype=float,
    )

    ridge = np.diag(
        [
            1e-8,
            1e-8,
        ]
    )

    for _ in range(
        100
    ):
        fitted = expit(
            design
            @ coefficients
        )

        weights = np.clip(
            fitted
            * (
                1.0
                - fitted
            ),
            1e-10,
            None,
        )

        gradient = (
            design.T
            @ (
                y
                - fitted
            )
        )

        information = (
            design.T
            @ (
                design
                * weights[
                    :,
                    None,
                ]
            )
            + ridge
        )

        step = np.linalg.solve(
            information,
            gradient,
        )

        coefficients = (
            coefficients
            + step
        )

        if np.max(
            np.abs(
                step
            )
        ) < 1e-9:
            break

    return (
        float(
            coefficients[
                0
            ]
        ),
        float(
            coefficients[
                1
            ]
        ),
    )


def audit_policy(
    frame: pd.DataFrame,
    *,
    n_bins: int,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
]:
    n_total = len(
        frame
    )

    bin_rows: list[
        dict[str, object]
    ] = []

    action_rows: list[
        dict[str, object]
    ] = []

    for risk_bin in range(
        n_bins
    ):
        in_bin = frame[
            "risk_bin"
        ].eq(
            risk_bin
        )

        group = frame.loc[
            in_bin
        ]

        observed_group = group.loc[
            group[
                "observed"
            ].eq(
                1
            )
        ]

        if observed_group.empty:
            raise ValueError(
                f"No observed outcomes in bin {risk_bin}"
            )

        n_bin = len(
            group
        )

        weight = (
            n_bin
            / n_total
        )

        mean_prediction = float(
            group[
                "prediction"
            ].mean()
        )

        mu_full = float(
            group[
                "outcome"
            ].mean()
        )

        mu_observed = float(
            observed_group[
                "outcome"
            ].mean()
        )

        contribution_sum = 0.0
        absolute_sum = 0.0

        for action in ACTION_ORDER:
            hidden_action = group.loc[
                group[
                    "observed"
                ].eq(
                    0
                )
                & group[
                    "action"
                ].eq(
                    action
                )
            ]

            n_action = len(
                hidden_action
            )

            pi_action = (
                n_action
                / n_bin
            )

            if n_action == 0:
                mu_action = np.nan
                contribution = 0.0
            else:
                mu_action = float(
                    hidden_action[
                        "outcome"
                    ].mean()
                )

                contribution = float(
                    pi_action
                    * (
                        mu_observed
                        - mu_action
                    )
                )

            contribution_sum += (
                contribution
            )

            absolute_sum += abs(
                contribution
            )

            action_rows.append(
                {
                    "policy": frame[
                        "policy"
                    ].iloc[
                        0
                    ],
                    "risk_bin": (
                        risk_bin
                    ),
                    "action": action,
                    "n_action": (
                        n_action
                    ),
                    "pi_action": (
                        pi_action
                    ),
                    "mu_observed": (
                        mu_observed
                    ),
                    "mu_action": (
                        mu_action
                    ),
                    "contribution": (
                        contribution
                    ),
                    "reference_weight": (
                        weight
                    ),
                    "weighted_contribution": (
                        weight
                        * contribution
                    ),
                    "weighted_absolute_contribution": (
                        weight
                        * abs(
                            contribution
                        )
                    ),
                }
            )

        difference = (
            mu_observed
            - mu_full
        )

        bin_rows.append(
            {
                "policy": frame[
                    "policy"
                ].iloc[
                    0
                ],
                "risk_bin": (
                    risk_bin
                ),
                "n_bin": (
                    n_bin
                ),
                "n_observed": int(
                    len(
                        observed_group
                    )
                ),
                "observed_fraction": float(
                    len(
                        observed_group
                    )
                    / n_bin
                ),
                "reference_weight": (
                    weight
                ),
                "mean_prediction": (
                    mean_prediction
                ),
                "mu_full": (
                    mu_full
                ),
                "mu_observed": (
                    mu_observed
                ),
                "observed_minus_full": (
                    difference
                ),
                "contribution_sum": (
                    contribution_sum
                ),
                "absolute_contribution_sum": (
                    absolute_sum
                ),
                "pointwise_aci": float(
                    1.0
                    - abs(
                        contribution_sum
                    )
                    / (
                        absolute_sum
                        + 1e-12
                    )
                ),
                "decomposition_residual": float(
                    difference
                    - contribution_sum
                ),
            }
        )

    bins = pd.DataFrame(
        bin_rows
    )

    action_bins = pd.DataFrame(
        action_rows
    )

    action_summary = (
        action_bins.groupby(
            [
                "policy",
                "action",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            sab=(
                "weighted_contribution",
                "sum",
            ),
            aacd=(
                "weighted_absolute_contribution",
                "sum",
            ),
            n_hidden=(
                "n_action",
                "sum",
            ),
        )
    )

    weights = bins[
        "reference_weight"
    ].to_numpy(
        dtype=float
    )

    full_curve = bins[
        "mu_full"
    ].to_numpy(
        dtype=float
    )

    restricted_curve = bins[
        "mu_observed"
    ].to_numpy(
        dtype=float
    )

    prediction_curve = bins[
        "mean_prediction"
    ].to_numpy(
        dtype=float
    )

    observed_rows = frame.loc[
        frame[
            "observed"
        ].eq(
            1
        )
    ]

    reference_intercept, reference_slope = (
        fit_calibration_intercept_slope(
            frame[
                "outcome"
            ].to_numpy(
                dtype=float
            ),
            frame[
                "prediction"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    restricted_intercept, restricted_slope = (
        fit_calibration_intercept_slope(
            observed_rows[
                "outcome"
            ].to_numpy(
                dtype=float
            ),
            observed_rows[
                "prediction"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    gross_action_debt = float(
        action_summary[
            "aacd"
        ].sum()
    )

    net_curve_distortion = float(
        np.sum(
            weights
            * np.abs(
                restricted_curve
                - full_curve
            )
        )
    )

    scalar = {
        "n": float(
            len(
                frame
            )
        ),
        "reference_event_rate": float(
            frame[
                "outcome"
            ].mean()
        ),
        "mean_prediction": float(
            frame[
                "prediction"
            ].mean()
        ),
        "observed_fraction": float(
            frame[
                "observed"
            ].mean()
        ),
        "crude_observed_event_rate": float(
            observed_rows[
                "outcome"
            ].mean()
        ),
        "standardized_observed_event_rate": float(
            np.sum(
                weights
                * restricted_curve
            )
        ),
        "reference_ici": float(
            np.sum(
                weights
                * np.abs(
                    full_curve
                    - prediction_curve
                )
            )
        ),
        "restricted_ici": float(
            np.sum(
                weights
                * np.abs(
                    restricted_curve
                    - prediction_curve
                )
            )
        ),
        "delta_ici": float(
            np.sum(
                weights
                * np.abs(
                    restricted_curve
                    - prediction_curve
                )
            )
            - np.sum(
                weights
                * np.abs(
                    full_curve
                    - prediction_curve
                )
            )
        ),
        "signed_distortion": float(
            np.sum(
                weights
                * (
                    restricted_curve
                    - full_curve
                )
            )
        ),
        "net_curve_distortion": (
            net_curve_distortion
        ),
        "gross_action_debt": (
            gross_action_debt
        ),
        "pointwise_cancellation_ratio": float(
            1.0
            - net_curve_distortion
            / (
                gross_action_debt
                + 1e-12
            )
        ),
        "reference_brier": float(
            np.mean(
                (
                    frame[
                        "outcome"
                    ].to_numpy(
                        dtype=float
                    )
                    - frame[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    )
                )
                ** 2
            )
        ),
        "restricted_brier": float(
            np.mean(
                (
                    observed_rows[
                        "outcome"
                    ].to_numpy(
                        dtype=float
                    )
                    - observed_rows[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    )
                )
                ** 2
            )
        ),
        "reference_log_loss": binary_log_loss(
            frame[
                "outcome"
            ].to_numpy(
                dtype=float
            ),
            frame[
                "prediction"
            ].to_numpy(
                dtype=float
            ),
        ),
        "restricted_log_loss": binary_log_loss(
            observed_rows[
                "outcome"
            ].to_numpy(
                dtype=float
            ),
            observed_rows[
                "prediction"
            ].to_numpy(
                dtype=float
            ),
        ),
        "reference_calibration_intercept": (
            reference_intercept
        ),
        "reference_calibration_slope": (
            reference_slope
        ),
        "restricted_calibration_intercept": (
            restricted_intercept
        ),
        "restricted_calibration_slope": (
            restricted_slope
        ),
        "maximum_decomposition_residual": float(
            bins[
                "decomposition_residual"
            ].abs().max()
        ),
    }

    return (
        scalar,
        action_summary,
        bins,
    )


def point_analysis(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    scalar_rows = []
    action_frames = []
    bin_frames = []

    for policy_name, frame in (
        policy_frames.items()
    ):
        (
            scalar,
            action_summary,
            bins,
        ) = audit_policy(
            frame,
            n_bins=n_bins,
        )

        scalar_rows.append(
            {
                "policy": (
                    policy_name
                ),
                **scalar,
            }
        )

        action_frames.append(
            action_summary
        )

        bin_frames.append(
            bins
        )

    return (
        pd.DataFrame(
            scalar_rows
        ),
        pd.concat(
            action_frames,
            ignore_index=True,
        ),
        pd.concat(
            bin_frames,
            ignore_index=True,
        ),
    )


BOOTSTRAP_SCALAR_METRICS = [
    "observed_fraction",
    "crude_observed_event_rate",
    "standardized_observed_event_rate",
    "restricted_ici",
    "delta_ici",
    "signed_distortion",
    "net_curve_distortion",
    "gross_action_debt",
    "restricted_brier",
    "restricted_log_loss",
    "restricted_calibration_intercept",
    "restricted_calibration_slope",
]


def run_paired_bootstrap(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    n_bins: int,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    policy_names = list(
        policy_frames
    )

    n = len(
        next(
            iter(
                policy_frames.values()
            )
        )
    )

    scalar_rows = []
    action_rows = []

    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        replicate_scalars = {}

        try:
            for policy_name in policy_names:
                sample = (
                    policy_frames[
                        policy_name
                    ]
                    .iloc[
                        indices
                    ]
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )

                (
                    scalar,
                    action_summary,
                    _,
                ) = audit_policy(
                    sample,
                    n_bins=n_bins,
                )

                replicate_scalars[
                    policy_name
                ] = scalar

                for _, row in (
                    action_summary.iterrows()
                ):
                    action_rows.append(
                        {
                            "bootstrap_id": (
                                bootstrap_id
                            ),
                            "policy": (
                                policy_name
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

            baseline = replicate_scalars[
                "baseline"
            ]

            for policy_name, scalar in (
                replicate_scalars.items()
            ):
                row = {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    "policy": (
                        policy_name
                    ),
                }

                for metric in (
                    BOOTSTRAP_SCALAR_METRICS
                ):
                    row[
                        metric
                    ] = scalar[
                        metric
                    ]

                    row[
                        f"{metric}_minus_baseline"
                    ] = (
                        scalar[
                            metric
                        ]
                        - baseline[
                            metric
                        ]
                    )

                scalar_rows.append(
                    row
                )

        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            invalid += 1

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
            scalar_rows
        ),
        pd.DataFrame(
            action_rows
        ),
        invalid,
    )


def interval_summary(
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
    rows = []

    metrics = (
        BOOTSTRAP_SCALAR_METRICS
        + [
            f"{metric}_minus_baseline"
            for metric in (
                BOOTSTRAP_SCALAR_METRICS
            )
        ]
    )

    for policy, group in (
        bootstrap.groupby(
            "policy",
            sort=True,
        )
    ):
        for metric in metrics:
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    **interval_summary(
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
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        policy,
        action,
    ), group in bootstrap.groupby(
        [
            "policy",
            "action",
        ],
        sort=True,
    ):
        for metric in [
            "sab",
            "aacd",
        ]:
            rows.append(
                {
                    "policy": policy,
                    "action": action,
                    "metric": metric,
                    **interval_summary(
                        group[
                            metric
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
    policy: str,
    metric: str,
    column: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "policy"
        ].eq(
            policy
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
            f"{policy}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def build_validation_checks(
    *,
    point_scalars: pd.DataFrame,
    point_actions: pd.DataFrame,
    scalar_intervals: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
) -> pd.DataFrame:
    indexed = point_scalars.set_index(
        "policy"
    )

    reference_ici_range = float(
        point_scalars[
            "reference_ici"
        ].max()
        - point_scalars[
            "reference_ici"
        ].min()
    )

    reference_event_rate_range = float(
        point_scalars[
            "reference_event_rate"
        ].max()
        - point_scalars[
            "reference_event_rate"
        ].min()
    )

    observed_fraction_range = float(
        point_scalars[
            "observed_fraction"
        ].max()
        - point_scalars[
            "observed_fraction"
        ].min()
    )

    restricted_ici_range = float(
        point_scalars[
            "restricted_ici"
        ].max()
        - point_scalars[
            "restricted_ici"
        ].min()
    )

    aggressive_difference_lower = (
        interval_value(
            scalar_intervals,
            policy="aggressive_early",
            metric=(
                "restricted_ici_minus_baseline"
            ),
            column="q025",
        )
    )

    conservative_difference_upper = (
        interval_value(
            scalar_intervals,
            policy="conservative_late",
            metric=(
                "restricted_ici_minus_baseline"
            ),
            column="q975",
        )
    )

    baseline = indexed.loc[
        "baseline"
    ]

    transfer_mix = indexed.loc[
        "baseline_transfer_mix"
    ]

    global_mix_difference = max(
        abs(
            baseline[
                "observed_fraction"
            ]
            - transfer_mix[
                "observed_fraction"
            ]
        ),
        abs(
            baseline[
                "restricted_ici"
            ]
            - transfer_mix[
                "restricted_ici"
            ]
        ),
        abs(
            baseline[
                "signed_distortion"
            ]
            - transfer_mix[
                "signed_distortion"
            ]
        ),
    )

    baseline_actions = (
        point_actions.loc[
            point_actions[
                "policy"
            ].eq(
                "baseline"
            )
        ]
        .set_index(
            "action"
        )
        .loc[
            ACTION_ORDER,
            "sab",
        ]
    )

    transfer_actions = (
        point_actions.loc[
            point_actions[
                "policy"
            ].eq(
                "baseline_transfer_mix"
            )
        ]
        .set_index(
            "action"
        )
        .loc[
            ACTION_ORDER,
            "sab",
        ]
    )

    action_attribution_l1 = float(
        (
            baseline_actions
            - transfer_actions
        )
        .abs()
        .sum()
    )

    maximum_residual = float(
        point_scalars[
            "maximum_decomposition_residual"
        ].max()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "reference ICI invariant across policies"
                ),
                "value": reference_ici_range,
                "threshold": 1e-12,
                "comparison": "<",
                "passed": (
                    reference_ici_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "reference event rate invariant "
                    "across policies"
                ),
                "value": (
                    reference_event_rate_range
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": (
                    reference_event_rate_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "observation fraction changes materially"
                ),
                "value": (
                    observed_fraction_range
                ),
                "threshold": 0.30,
                "comparison": ">",
                "passed": (
                    observed_fraction_range
                    > 0.30
                ),
            },
            {
                "check": (
                    "restricted ICI changes materially"
                ),
                "value": (
                    restricted_ici_range
                ),
                "threshold": 0.05,
                "comparison": ">",
                "passed": (
                    restricted_ici_range
                    > 0.05
                ),
            },
            {
                "check": (
                    "aggressive policy worsens restricted "
                    "ICI versus baseline"
                ),
                "value": (
                    aggressive_difference_lower
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    aggressive_difference_lower
                    > 0.0
                ),
            },
            {
                "check": (
                    "conservative policy improves restricted "
                    "ICI versus baseline"
                ),
                "value": (
                    conservative_difference_upper
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    conservative_difference_upper
                    < 0.0
                ),
            },
            {
                "check": (
                    "action-mix policy preserves global audit"
                ),
                "value": (
                    global_mix_difference
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": (
                    global_mix_difference
                    < 1e-12
                ),
            },
            {
                "check": (
                    "action-mix policy changes attribution"
                ),
                "value": (
                    action_attribution_l1
                ),
                "threshold": 0.01,
                "comparison": ">",
                "passed": (
                    action_attribution_l1
                    > 0.01
                ),
            },
            {
                "check": (
                    "all decomposition identities exact"
                ),
                "value": (
                    maximum_residual
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    maximum_residual
                    < 1e-10
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

    policies = policy_definitions()

    population = generate_fixed_population(
        n_patients=args.n_patients,
        horizon=args.horizon,
        seed=args.seed,
        n_bins=args.n_bins,
    )

    fixed_population_hash = sha256_arrays(
        population[
            "prediction"
        ].to_numpy(),
        population[
            "outcome"
        ].to_numpy(),
        population[
            "event_time"
        ].to_numpy(),
    )

    policy_frames = {
        policy.name: apply_observation_policy(
            population,
            policy,
            horizon=args.horizon,
        )
        for policy in policies
    }

    print(
        "Phase 5E observation-policy shift"
    )

    print(
        f"  fixed patients="
        f"{args.n_patients:,}"
    )

    print(
        f"  policies="
        f"{len(policies)}"
    )

    print(
        "  paired bootstrap replicates="
        f"{args.bootstrap_replicates}"
    )

    (
        point_scalars,
        point_actions,
        point_bins,
    ) = point_analysis(
        policy_frames,
        n_bins=args.n_bins,
    )

    print(
        "\nRunning paired bootstrap..."
    )

    (
        bootstrap_scalars,
        bootstrap_actions,
        invalid_bootstraps,
    ) = run_paired_bootstrap(
        policy_frames,
        n_bins=args.n_bins,
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
            bootstrap_actions
        )
    )

    checks = build_validation_checks(
        point_scalars=point_scalars,
        point_actions=point_actions,
        scalar_intervals=(
            scalar_intervals
        ),
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
    )

    preliminary_status = (
        "phase5e_observation_policy_shift_supported"
        if checks[
            "passed"
        ].all()
        else (
            "phase5e_observation_policy_shift_"
            "requires_review"
        )
    )

    status = (
        "phase5e_policy_shift_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    policy_table = pd.DataFrame(
        [
            {
                "policy": policy.name,
                "exit_intercept": (
                    policy.exit_intercept
                ),
                "risk_selection": (
                    policy.risk_selection
                ),
                "timing_power": (
                    policy.timing_power
                ),
                "action_shifts": json.dumps(
                    policy.action_shifts,
                    sort_keys=True,
                ),
                "description": (
                    policy.description
                ),
            }
            for policy in policies
        ]
    )

    population[
        [
            "patient_id",
            "prediction",
            "outcome",
            "event_time",
            "risk_bin",
        ]
    ].to_csv(
        output_dir
        / "fixed_population.csv.gz",
        index=False,
        compression="gzip",
    )

    policy_table.to_csv(
        output_dir
        / "policy_definitions.csv",
        index=False,
    )

    point_scalars.to_csv(
        output_dir
        / "policy_scalar_summary.csv",
        index=False,
    )

    point_actions.to_csv(
        output_dir
        / "policy_action_summary.csv",
        index=False,
    )

    point_bins.to_csv(
        output_dir
        / "policy_bin_summary.csv",
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

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5E",
        "hypothesis": (
            "Observation-policy shift can mimic "
            "model drift."
        ),
        "fixed_population_hash": (
            fixed_population_hash
        ),
        "n_patients": (
            args.n_patients
        ),
        "horizon": (
            args.horizon
        ),
        "n_bins": (
            args.n_bins
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstraps": (
            invalid_bootstraps
        ),
        "fixed_components": [
            "patient covariates",
            "predictions",
            "outcomes",
            "latent event times",
            "common random numbers",
        ],
        "shifted_components": [
            "exit probability",
            "risk-based discharge selection",
            "exit timing",
            "action mixture",
        ],
        "interpretation": (
            "Differences in restricted calibration "
            "are induced only by observation-policy "
            "changes because predictions and the event "
            "process are identical across policies."
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
        / "policy_shift_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5e_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "policy",
        "reference_event_rate",
        "mean_prediction",
        "observed_fraction",
        "crude_observed_event_rate",
        "reference_ici",
        "restricted_ici",
        "delta_ici",
        "signed_distortion",
        "net_curve_distortion",
        "reference_calibration_intercept",
        "reference_calibration_slope",
        "restricted_calibration_intercept",
        "restricted_calibration_slope",
    ]

    print(
        "\nPolicy summary"
    )

    print(
        point_scalars[
            display_columns
        ].to_string(
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
