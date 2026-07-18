"""Phase 5J-B: SUPPORT2 observation-policy-shift emulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


EXPECTED_INPUT_STATUS = (
    "phase5h_support2_definitions_frozen"
)

EXPECTED_N = 1821
HORIZON = 60.0
EPSILON = 1e-10


@dataclass(frozen=True)
class ObservationPolicy:
    name: str
    exit_intercept: float
    risk_selection: float
    timing_power: float
    description: str


def policy_definitions() -> list[ObservationPolicy]:
    return [
        ObservationPolicy(
            name="conservative_late",
            exit_intercept=-1.8,
            risk_selection=0.8,
            timing_power=0.7,
            description=(
                "Lower discharge probability "
                "and later exits."
            ),
        ),
        ObservationPolicy(
            name="baseline",
            exit_intercept=-0.2,
            risk_selection=1.0,
            timing_power=1.2,
            description=(
                "Reference observation policy."
            ),
        ),
        ObservationPolicy(
            name="aggressive_early",
            exit_intercept=1.5,
            risk_selection=1.2,
            timing_power=2.0,
            description=(
                "Higher discharge probability "
                "and earlier exits."
            ),
        ),
        ObservationPolicy(
            name="risk_neutral",
            exit_intercept=0.5,
            risk_selection=0.0,
            timing_power=1.5,
            description=(
                "Exit probability does not "
                "depend on model risk."
            ),
        ),
        ObservationPolicy(
            name="high_risk_exit",
            exit_intercept=0.2,
            risk_selection=-1.0,
            timing_power=1.5,
            description=(
                "Higher-risk patients are "
                "more likely to exit."
            ),
        ),
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled SUPPORT2 "
            "observation-policy-shift emulation."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/"
            "phase5h_support2_definitions/"
            "support2_harmonized_test.csv.gz"
        ),
    )

    parser.add_argument(
        "--input-status",
        default=(
            "artifacts/"
            "phase5h_support2_definitions/"
            "phase5h_support2_definition_status.txt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5j_support2_policy_shift"
        ),
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

    return parser.parse_args()


def require_columns(
    data: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def fixed_hash(
    frame: pd.DataFrame,
) -> str:
    columns = [
        "patient_id",
        "prediction",
        "outcome",
        "latent_event_time",
        "u_exit",
        "u_time",
    ]

    hashed = pd.util.hash_pandas_object(
        frame[
            columns
        ],
        index=False,
    ).to_numpy()

    return hashlib.sha256(
        hashed.tobytes()
    ).hexdigest()


def prepare_fixed_population(
    data: pd.DataFrame,
    *,
    n_bins: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    str,
]:
    require_columns(
        data,
        [
            "patient_id",
            "prediction",
            "outcome",
            "observed",
            "exit_time",
        ],
    )

    if data[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate SUPPORT2 patient IDs"
        )

    frame = data.copy()

    frame[
        "prediction"
    ] = pd.to_numeric(
        frame[
            "prediction"
        ],
        errors="raise",
    ).clip(
        EPSILON,
        1.0 - EPSILON,
    )

    frame[
        "outcome"
    ] = pd.to_numeric(
        frame[
            "outcome"
        ],
        errors="raise",
    ).astype(
        int
    )

    frame[
        "observed"
    ] = pd.to_numeric(
        frame[
            "observed"
        ],
        errors="raise",
    ).astype(
        int
    )

    frame[
        "exit_time"
    ] = pd.to_numeric(
        frame[
            "exit_time"
        ],
        errors="coerce",
    ).fillna(
        HORIZON
    ).clip(
        lower=0.0,
        upper=HORIZON,
    )

    prediction = frame[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    mean = float(
        prediction.mean()
    )

    standard_deviation = float(
        prediction.std(
            ddof=0
        )
    )

    if standard_deviation <= 0:
        raise ValueError(
            "Prediction standard deviation is zero"
        )

    frame[
        "standardized_risk"
    ] = (
        prediction
        - mean
    ) / standard_deviation

    edges = np.quantile(
        prediction,
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        ),
    )

    if len(
        np.unique(
            edges
        )
    ) != n_bins + 1:
        raise ValueError(
            "Prediction distribution does not "
            "support requested risk bins"
        )

    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    frame[
        "risk_bin"
    ] = pd.cut(
        prediction,
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(
        int
    )

    rng = np.random.default_rng(
        seed
    )

    frame[
        "u_exit"
    ] = rng.uniform(
        size=len(
            frame
        )
    )

    frame[
        "u_time"
    ] = rng.uniform(
        size=len(
            frame
        )
    )

    hidden_event_uniform = rng.uniform(
        size=len(
            frame
        )
    )

    latent_event_time = np.full(
        len(
            frame
        ),
        np.inf,
        dtype=float,
    )

    observed_death = (
        frame[
            "outcome"
        ].eq(
            1
        )
        & frame[
            "observed"
        ].eq(
            1
        )
    )

    hidden_death = (
        frame[
            "outcome"
        ].eq(
            1
        )
        & frame[
            "observed"
        ].eq(
            0
        )
    )

    latent_event_time[
        observed_death.to_numpy()
    ] = frame.loc[
        observed_death,
        "exit_time",
    ].to_numpy(
        dtype=float
    )

    hidden_exit = frame.loc[
        hidden_death,
        "exit_time",
    ].to_numpy(
        dtype=float
    ).clip(
        0.0,
        HORIZON - 1e-8,
    )

    latent_event_time[
        hidden_death.to_numpy()
    ] = (
        hidden_exit
        + (
            HORIZON
            - hidden_exit
        )
        * hidden_event_uniform[
            hidden_death.to_numpy()
        ]
    )

    frame[
        "latent_event_time"
    ] = latent_event_time

    return (
        frame,
        edges,
        fixed_hash(
            frame
        ),
    )


def apply_policy(
    fixed: pd.DataFrame,
    policy: ObservationPolicy,
) -> pd.DataFrame:
    frame = fixed.copy()

    exit_probability = expit(
        policy.exit_intercept
        - policy.risk_selection
        * frame[
            "standardized_risk"
        ].to_numpy(
            dtype=float
        )
    )

    exits_before_horizon = (
        frame[
            "u_exit"
        ].to_numpy(
            dtype=float
        )
        < exit_probability
    )

    emulated_exit_time = np.where(
        exits_before_horizon,
        HORIZON
        * np.power(
            frame[
                "u_time"
            ].to_numpy(
                dtype=float
            ),
            policy.timing_power,
        ),
        HORIZON,
    )

    event_before_exit = (
        frame[
            "latent_event_time"
        ].to_numpy(
            dtype=float
        )
        <= emulated_exit_time
    )

    emulated_observed = (
        (
            ~exits_before_horizon
        )
        | event_before_exit
    ).astype(
        int
    )

    frame[
        "policy"
    ] = policy.name

    frame[
        "exit_probability"
    ] = exit_probability

    frame[
        "emulated_exit_time"
    ] = emulated_exit_time

    frame[
        "emulated_exit"
    ] = exits_before_horizon.astype(
        int
    )

    frame[
        "emulated_observed"
    ] = emulated_observed

    frame[
        "emulated_restricted_event"
    ] = (
        frame[
            "outcome"
        ]
        * frame[
            "emulated_observed"
        ]
    )

    return frame


def binary_log_loss(
    outcome: np.ndarray,
    prediction: np.ndarray,
) -> float:
    probability = np.clip(
        prediction,
        EPSILON,
        1.0 - EPSILON,
    )

    return float(
        -np.mean(
            outcome
            * np.log(
                probability
            )
            + (
                1.0
                - outcome
            )
            * np.log(
                1.0
                - probability
            )
        )
    )


def audit_policy(
    frame: pd.DataFrame,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
]:
    total_n = len(
        frame
    )

    bin_rows = []

    reference_ici = 0.0
    restricted_ici = 0.0
    signed_distortion = 0.0
    standardized_observed_rate = 0.0

    for risk_bin in sorted(
        frame[
            "risk_bin"
        ].unique()
    ):
        group = frame.loc[
            frame[
                "risk_bin"
            ].eq(
                risk_bin
            )
        ]

        observed_group = group.loc[
            group[
                "emulated_observed"
            ].eq(
                1
            )
        ]

        if observed_group.empty:
            raise ValueError(
                f"No observed outcomes in "
                f"risk bin {risk_bin}"
            )

        n = int(
            len(
                group
            )
        )

        weight = float(
            n
            / total_n
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

        reference_ici += (
            weight
            * abs(
                mu_full
                - mean_prediction
            )
        )

        restricted_ici += (
            weight
            * abs(
                mu_observed
                - mean_prediction
            )
        )

        signed_distortion += (
            weight
            * (
                mu_observed
                - mu_full
            )
        )

        standardized_observed_rate += (
            weight
            * mu_observed
        )

        bin_rows.append(
            {
                "policy": frame[
                    "policy"
                ].iloc[
                    0
                ],
                "risk_bin": int(
                    risk_bin
                ),
                "n": n,
                "n_observed": int(
                    len(
                        observed_group
                    )
                ),
                "reference_weight": (
                    weight
                ),
                "mean_prediction": (
                    mean_prediction
                ),
                "mu_full": mu_full,
                "mu_observed": (
                    mu_observed
                ),
                "observed_minus_full": (
                    mu_observed
                    - mu_full
                ),
            }
        )

    observed_rows = frame.loc[
        frame[
            "emulated_observed"
        ].eq(
            1
        )
    ]

    outcome = frame[
        "outcome"
    ].to_numpy(
        dtype=float
    )

    prediction = frame[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    observed_outcome = observed_rows[
        "outcome"
    ].to_numpy(
        dtype=float
    )

    observed_prediction = observed_rows[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    scalar = {
        "policy": frame[
            "policy"
        ].iloc[
            0
        ],
        "n": int(
            total_n
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
                "emulated_observed"
            ].mean()
        ),
        "crude_observed_event_rate": float(
            observed_rows[
                "outcome"
            ].mean()
        ),
        "standardized_observed_event_rate": float(
            standardized_observed_rate
        ),
        "reference_ici": float(
            reference_ici
        ),
        "restricted_ici": float(
            restricted_ici
        ),
        "delta_ici": float(
            restricted_ici
            - reference_ici
        ),
        "signed_distortion": float(
            signed_distortion
        ),
        "reference_brier": float(
            np.mean(
                (
                    outcome
                    - prediction
                )
                ** 2
            )
        ),
        "restricted_brier": float(
            np.mean(
                (
                    observed_outcome
                    - observed_prediction
                )
                ** 2
            )
        ),
        "reference_log_loss": (
            binary_log_loss(
                outcome,
                prediction,
            )
        ),
        "restricted_log_loss": (
            binary_log_loss(
                observed_outcome,
                observed_prediction,
            )
        ),
        "fixed_population_hash": (
            fixed_hash(
                frame
            )
        ),
    }

    return (
        scalar,
        pd.DataFrame(
            bin_rows
        ),
    )


SCALAR_METRICS = [
    "reference_event_rate",
    "mean_prediction",
    "observed_fraction",
    "crude_observed_event_rate",
    "standardized_observed_event_rate",
    "reference_ici",
    "restricted_ici",
    "delta_ici",
    "signed_distortion",
    "reference_brier",
    "restricted_brier",
    "reference_log_loss",
    "restricted_log_loss",
]


def point_analysis(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    scalar_rows = []
    bin_tables = []

    for policy_name, frame in (
        policy_frames.items()
    ):
        scalar, bins = audit_policy(
            frame
        )

        if scalar[
            "policy"
        ] != policy_name:
            raise ValueError(
                "Policy-name mismatch"
            )

        scalar_rows.append(
            scalar
        )

        bin_tables.append(
            bins
        )

    return (
        pd.DataFrame(
            scalar_rows
        ),
        pd.concat(
            bin_tables,
            ignore_index=True,
        ),
    )


def run_bootstrap(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    first_frame = next(
        iter(
            policy_frames.values()
        )
    )

    rows = []
    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        indices = rng.integers(
            0,
            len(
                first_frame
            ),
            size=len(
                first_frame
            ),
        )

        replicate_rows = []

        try:
            for policy_name, frame in (
                policy_frames.items()
            ):
                sample = (
                    frame.iloc[
                        indices
                    ]
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )

                scalar, _ = audit_policy(
                    sample
                )

                scalar[
                    "bootstrap_id"
                ] = bootstrap_id

                replicate_rows.append(
                    scalar
                )
        except ValueError:
            invalid += 1
            continue

        rows.extend(
            replicate_rows
        )

        if (
            bootstrap_id
            + 1
        ) % 100 == 0:
            print(
                f"  completed "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

    if not rows:
        raise RuntimeError(
            "No valid bootstrap "
            "replications completed"
        )

    return (
        pd.DataFrame(
            rows
        ),
        invalid,
    )


def scalar_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for policy, group in bootstrap.groupby(
        "policy",
        sort=True,
    ):
        for metric in SCALAR_METRICS:
            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            ).dropna()

            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
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
                    "n_valid": int(
                        len(
                            values
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def point_contrasts(
    point: pd.DataFrame,
) -> pd.DataFrame:
    baseline = point.loc[
        point[
            "policy"
        ].eq(
            "baseline"
        )
    ].iloc[
        0
    ]

    rows = []

    for _, policy in point.iterrows():
        if policy[
            "policy"
        ] == "baseline":
            continue

        rows.append(
            {
                "policy": policy[
                    "policy"
                ],
                "comparison": (
                    "policy_minus_baseline"
                ),
                "observed_fraction_difference": float(
                    policy[
                        "observed_fraction"
                    ]
                    - baseline[
                        "observed_fraction"
                    ]
                ),
                "restricted_ici_difference": float(
                    policy[
                        "restricted_ici"
                    ]
                    - baseline[
                        "restricted_ici"
                    ]
                ),
                "signed_distortion_difference": float(
                    policy[
                        "signed_distortion"
                    ]
                    - baseline[
                        "signed_distortion"
                    ]
                ),
                "restricted_brier_difference": float(
                    policy[
                        "restricted_brier"
                    ]
                    - baseline[
                        "restricted_brier"
                    ]
                ),
                "restricted_log_loss_difference": float(
                    policy[
                        "restricted_log_loss"
                    ]
                    - baseline[
                        "restricted_log_loss"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def contrast_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "observed_fraction",
        "restricted_ici",
        "signed_distortion",
        "restricted_brier",
        "restricted_log_loss",
    ]

    policies = [
        policy
        for policy in sorted(
            bootstrap[
                "policy"
            ].unique()
        )
        if policy != "baseline"
    ]

    rows = []

    for policy in policies:
        for metric in metrics:
            selected = bootstrap.loc[
                bootstrap[
                    "policy"
                ].isin(
                    [
                        "baseline",
                        policy,
                    ]
                ),
                [
                    "bootstrap_id",
                    "policy",
                    metric,
                ],
            ]

            pivot = selected.pivot(
                index="bootstrap_id",
                columns="policy",
                values=metric,
            ).dropna()

            difference = (
                pivot[
                    policy
                ]
                - pivot[
                    "baseline"
                ]
            )

            rows.append(
                {
                    "policy": policy,
                    "comparison": (
                        "policy_minus_baseline"
                    ),
                    "metric": (
                        f"{metric}_difference"
                    ),
                    "q025": float(
                        difference.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        difference.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        difference.quantile(
                            0.975
                        )
                    ),
                    "n_valid": int(
                        len(
                            difference
                        )
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
    quantile: str,
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
        )
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            f"Missing interval: {policy}, {metric}"
        )

    return float(
        selected[
            quantile
        ].iloc[
            0
        ]
    )


def build_validation_checks(
    *,
    fixed: pd.DataFrame,
    input_status: str,
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    fixed_columns = [
        "patient_id",
        "prediction",
        "outcome",
        "latent_event_time",
        "risk_bin",
    ]

    baseline = policy_frames[
        "baseline"
    ][
        fixed_columns
    ].reset_index(
        drop=True
    )

    fixed_components_identical = all(
        baseline.equals(
            frame[
                fixed_columns
            ].reset_index(
                drop=True
            )
        )
        for frame in policy_frames.values()
    )

    observed_patterns = {
        tuple(
            frame[
                "emulated_observed"
            ].to_numpy(
                dtype=int
            )
        )
        for frame in policy_frames.values()
    }

    reference_ici_range = float(
        point[
            "reference_ici"
        ].max()
        - point[
            "reference_ici"
        ].min()
    )

    reference_event_rate_range = float(
        point[
            "reference_event_rate"
        ].max()
        - point[
            "reference_event_rate"
        ].min()
    )

    observed_fraction_range = float(
        point[
            "observed_fraction"
        ].max()
        - point[
            "observed_fraction"
        ].min()
    )

    expected_rows = (
        requested_bootstraps
        * len(
            policy_frames
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "SUPPORT2 definitions already frozen"
                ),
                "value": input_status,
                "threshold": (
                    EXPECTED_INPUT_STATUS
                ),
                "comparison": "==",
                "passed": (
                    input_status
                    == EXPECTED_INPUT_STATUS
                ),
            },
            {
                "check": (
                    "fixed cohort contains 1821 "
                    "unique patients"
                ),
                "value": len(
                    fixed
                ),
                "threshold": EXPECTED_N,
                "comparison": "==",
                "passed": bool(
                    len(
                        fixed
                    )
                    == EXPECTED_N
                    and not fixed[
                        "patient_id"
                    ].duplicated().any()
                ),
            },
            {
                "check": (
                    "five frozen observation "
                    "policies evaluated"
                ),
                "value": int(
                    point[
                        "policy"
                    ].nunique()
                ),
                "threshold": 5,
                "comparison": "==",
                "passed": (
                    point[
                        "policy"
                    ].nunique()
                    == 5
                ),
            },
            {
                "check": (
                    "predictions outcomes and latent "
                    "event process identical "
                    "across policies"
                ),
                "value": int(
                    fixed_components_identical
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    fixed_components_identical
                ),
            },
            {
                "check": (
                    "policies generate distinct "
                    "observation patterns"
                ),
                "value": len(
                    observed_patterns
                ),
                "threshold": 5,
                "comparison": "==",
                "passed": (
                    len(
                        observed_patterns
                    )
                    == 5
                ),
            },
            {
                "check": (
                    "reference ICI invariant "
                    "across policies"
                ),
                "value": (
                    reference_ici_range
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": bool(
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
                "passed": bool(
                    reference_event_rate_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "observation policies materially "
                    "change observed fraction"
                ),
                "value": (
                    observed_fraction_range
                ),
                "threshold": 0.20,
                "comparison": ">",
                "passed": bool(
                    observed_fraction_range
                    > 0.20
                ),
            },
            {
                "check": (
                    "all requested paired "
                    "bootstrap rows completed"
                ),
                "value": len(
                    bootstrap
                ),
                "threshold": expected_rows,
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap
                    )
                    == expected_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def build_h6_evidence_checks(
    point: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    restricted_ici_range = float(
        point[
            "restricted_ici"
        ].max()
        - point[
            "restricted_ici"
        ].min()
    )

    aggressive_q025 = interval_value(
        contrasts,
        policy="aggressive_early",
        metric=(
            "restricted_ici_difference"
        ),
        quantile="q025",
    )

    conservative_q975 = interval_value(
        contrasts,
        policy="conservative_late",
        metric=(
            "restricted_ici_difference"
        ),
        quantile="q975",
    )

    aggressive_observation_q975 = (
        interval_value(
            contrasts,
            policy="aggressive_early",
            metric=(
                "observed_fraction_difference"
            ),
            quantile="q975",
        )
    )

    conservative_observation_q025 = (
        interval_value(
            contrasts,
            policy="conservative_late",
            metric=(
                "observed_fraction_difference"
            ),
            quantile="q025",
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "restricted ICI changes "
                    "materially across policies"
                ),
                "value": (
                    restricted_ici_range
                ),
                "threshold": 0.03,
                "comparison": ">",
                "passed": bool(
                    restricted_ici_range
                    > 0.03
                ),
            },
            {
                "check": (
                    "aggressive policy increases "
                    "restricted ICI versus baseline"
                ),
                "value": (
                    aggressive_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    aggressive_q025
                    > 0
                ),
            },
            {
                "check": (
                    "conservative policy decreases "
                    "restricted ICI versus baseline"
                ),
                "value": (
                    conservative_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": bool(
                    conservative_q975
                    < 0
                ),
            },
            {
                "check": (
                    "aggressive policy lowers "
                    "observation fraction"
                ),
                "value": (
                    aggressive_observation_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": bool(
                    aggressive_observation_q975
                    < 0
                ),
            },
            {
                "check": (
                    "conservative policy raises "
                    "observation fraction"
                ),
                "value": (
                    conservative_observation_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    conservative_observation_q025
                    > 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    input_path = Path(
        args.input
    )

    status_path = Path(
        args.input_status
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_status = status_path.read_text(
        encoding="utf-8"
    ).strip()

    raw = pd.read_csv(
        input_path,
        low_memory=False,
    )

    (
        fixed,
        bin_edges,
        population_hash,
    ) = prepare_fixed_population(
        raw,
        n_bins=args.n_bins,
        seed=args.seed,
    )

    policies = policy_definitions()

    policy_frames = {
        policy.name: apply_policy(
            fixed,
            policy,
        )
        for policy in policies
    }

    (
        point,
        bin_summary,
    ) = point_analysis(
        policy_frames
    )

    point_contrast_table = (
        point_contrasts(
            point
        )
    )

    print(
        "Phase 5J-B SUPPORT2 "
        "observation-policy shift"
    )

    print(
        "\nRunning paired patient bootstrap..."
    )

    (
        bootstrap,
        invalid_bootstraps,
    ) = run_bootstrap(
        policy_frames,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed + 1,
    )

    scalar_intervals = (
        scalar_interval_table(
            bootstrap
        )
    )

    contrast_intervals = (
        contrast_interval_table(
            bootstrap
        )
    )

    validations = build_validation_checks(
        fixed=fixed,
        input_status=input_status,
        point=point,
        bootstrap=bootstrap,
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
        policy_frames=policy_frames,
    )

    evidence_checks = (
        build_h6_evidence_checks(
            point,
            contrast_intervals,
        )
    )

    status = (
        "phase5j_support2_policy_shift_benchmarked"
        if validations[
            "passed"
        ].all()
        else (
            "phase5j_support2_policy_shift_"
            "requires_review"
        )
    )

    pd.DataFrame(
        [
            asdict(
                policy
            )
            for policy in policies
        ]
    ).to_csv(
        output_dir
        / "policy_definitions.csv",
        index=False,
    )

    point.to_csv(
        output_dir
        / "policy_scalar_summary.csv",
        index=False,
    )

    bin_summary.to_csv(
        output_dir
        / "policy_bin_summary.csv",
        index=False,
    )

    point_contrast_table.to_csv(
        output_dir
        / "point_policy_contrasts.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_scalar_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    scalar_intervals.to_csv(
        output_dir
        / "scalar_bootstrap_intervals.csv",
        index=False,
    )

    contrast_intervals.to_csv(
        output_dir
        / "policy_contrast_intervals.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence_checks.to_csv(
        output_dir
        / "h6_evidence_checks.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "edge_index": np.arange(
                len(
                    bin_edges
                )
            ),
            "edge": bin_edges,
        }
    ).to_csv(
        output_dir
        / "frozen_risk_bin_edges.csv",
        index=False,
    )

    evidence = {
        "phase": "5J-B",
        "status": status,
        "dataset": "SUPPORT2",
        "horizon_days": HORIZON,
        "fixed_population_hash": (
            population_hash
        ),
        "n_patients": int(
            len(
                fixed
            )
        ),
        "n_bins": args.n_bins,
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstraps": (
            invalid_bootstraps
        ),
        "fixed_components": [
            "patient identities",
            "predictions",
            "60-day outcomes",
            "latent event-time realization",
            "common random numbers",
        ],
        "shifted_components": [
            "hospital-exit probability",
            "risk-based exit selection",
            "hospital-exit timing",
            "emulated observation indicator",
        ],
        "latent_event_time_rule": (
            "Observed hospital deaths retain their "
            "recorded exit time. Hidden deaths are "
            "assigned a fixed time between observed "
            "hospital exit and day 60. Non-events "
            "have no event before day 60."
        ),
        "policy_source": (
            "Uses the same policy coefficients as "
            "the frozen Phase 5E MIMIC-compatible "
            "synthetic demonstration."
        ),
        "interpretation": (
            "This is a controlled real-data-grounded "
            "policy emulation. It is not evidence of "
            "a historical or causal SUPPORT2 policy "
            "change."
        ),
        "h6_evidence_checks_passed": int(
            evidence_checks[
                "passed"
            ].sum()
        ),
        "h6_evidence_checks_total": int(
            len(
                evidence_checks
            )
        ),
    }

    (
        output_dir
        / "support2_policy_shift_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5j_support2_policy_shift_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nPolicy summary"
    )

    print(
        point.to_string(
            index=False
        )
    )

    print(
        "\nPolicy contrasts"
    )

    print(
        point_contrast_table.to_string(
            index=False
        )
    )

    print(
        "\nStructural validation"
    )

    print(
        validations.to_string(
            index=False
        )
    )

    print(
        "\nH6 evidence checks"
    )

    print(
        evidence_checks.to_string(
            index=False
        )
    )

    print(
        "\nInvalid bootstrap replications: "
        f"{invalid_bootstraps}"
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
