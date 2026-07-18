"""Phase 5J-E: real-data-grounded recalibration fragility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import phase5j_support2_policy_shift as policy_core


EPSILON = 1e-10

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

CONTRAST_METRICS = [
    "restricted_ici_change",
    "reference_ici_change",
    "restricted_brier_change",
    "reference_brier_change",
    "restricted_log_loss_change",
    "reference_log_loss_change",
]


DATASET_CONFIG = {
    "support2": {
        "display_name": "SUPPORT2",
        "expected_status": (
            "phase5h_support2_definitions_frozen"
        ),
        "expected_n": 1821,
        "horizon": 60.0,
        "default_input": (
            "artifacts/"
            "phase5h_support2_definitions/"
            "support2_harmonized_test.csv.gz"
        ),
        "default_status": (
            "artifacts/"
            "phase5h_support2_definitions/"
            "phase5h_support2_definition_status.txt"
        ),
    },
    "mimic": {
        "display_name": "MIMIC-IV",
        "expected_status": (
            "phase5c_frozen_for_manuscript"
        ),
        "expected_n": 31012,
        "horizon": 30.0,
        "default_input": (
            "artifacts/"
            "phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
        "default_status": (
            "artifacts/"
            "phase5c_frozen_for_manuscript/"
            "phase5c_freeze_status.txt"
        ),
    },
}


EXPECTED_H6_STATUS = (
    "phase5j_cross_dataset_h6_"
    "frozen_for_manuscript"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recalibration fitted on "
            "policy-selected outcomes against "
            "restricted and linked reference targets."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "support2",
            "mimic",
        ],
    )

    parser.add_argument(
        "--input",
        default=None,
    )

    parser.add_argument(
        "--input-status",
        default=None,
    )

    parser.add_argument(
        "--h6-status",
        default=(
            "artifacts/"
            "phase5j_cross_dataset_h6_"
            "frozen_for_manuscript/"
            "phase5j_cross_dataset_h6_status.txt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--minimum-observed-split-n",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def read_status(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing status file: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


def prepare_dataset(
    *,
    dataset: str,
    raw: pd.DataFrame,
    n_bins: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    str,
]:
    configuration = DATASET_CONFIG[
        dataset
    ]

    policy_core.HORIZON = float(
        configuration[
            "horizon"
        ]
    )

    if dataset == "support2":
        return (
            policy_core.prepare_fixed_population(
                raw,
                n_bins=n_bins,
                seed=seed,
            )
        )

    # Importing the MIMIC adapter installs its
    # MIMIC-specific hash and 30-day horizon.
    import phase5j_mimic_policy_shift as mimic_adapter

    policy_core.HORIZON = float(
        configuration[
            "horizon"
        ]
    )

    return (
        mimic_adapter.prepare_fixed_population(
            raw,
            n_bins=n_bins,
            seed=seed,
        )
    )


def assign_analysis_split(
    fixed: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not (
        0.0
        < calibration_fraction
        < 1.0
    ):
        raise ValueError(
            "Calibration fraction must be "
            "strictly between zero and one"
        )

    result = fixed.copy()

    split = np.full(
        len(
            result
        ),
        "test",
        dtype=object,
    )

    rng = np.random.default_rng(
        seed
    )

    outcome = result[
        "outcome"
    ].to_numpy(
        dtype=int
    )

    for outcome_value in [
        0,
        1,
    ]:
        indices = np.flatnonzero(
            outcome
            == outcome_value
        )

        rng.shuffle(
            indices
        )

        n_calibration = int(
            round(
                calibration_fraction
                * len(
                    indices
                )
            )
        )

        n_calibration = min(
            max(
                n_calibration,
                1,
            ),
            len(
                indices
            )
            - 1,
        )

        split[
            indices[
                :n_calibration
            ]
        ] = "calibration"

    result[
        "analysis_split"
    ] = split

    return result


def prediction_score(
    prediction: np.ndarray,
) -> np.ndarray:
    return logit(
        np.clip(
            prediction,
            EPSILON,
            1.0 - EPSILON,
        )
    )


def fit_intercept_only(
    score: np.ndarray,
    outcome: np.ndarray,
) -> tuple[
    float,
    float,
]:
    intercept = 0.0

    for _ in range(
        100
    ):
        probability = expit(
            score
            + intercept
        )

        gradient = float(
            np.sum(
                outcome
                - probability
            )
        )

        information = float(
            np.sum(
                probability
                * (
                    1.0
                    - probability
                )
            )
        )

        if information <= 1e-12:
            raise ValueError(
                "Intercept-only information "
                "matrix is singular"
            )

        step = gradient / information

        intercept += float(
            np.clip(
                step,
                -5.0,
                5.0,
            )
        )

        if abs(
            step
        ) < 1e-10:
            break

    if not np.isfinite(
        intercept
    ):
        raise ValueError(
            "Non-finite intercept-only fit"
        )

    return (
        float(
            intercept
        ),
        1.0,
    )


def fit_logistic_recalibration(
    score: np.ndarray,
    outcome: np.ndarray,
) -> tuple[
    float,
    float,
]:
    if len(
        np.unique(
            outcome
        )
    ) != 2:
        raise ValueError(
            "Logistic recalibration requires "
            "both outcome classes"
        )

    beta = np.array(
        [
            0.0,
            1.0,
        ],
        dtype=float,
    )

    ridge = 1e-7

    for _ in range(
        100
    ):
        linear_predictor = (
            beta[
                0
            ]
            + beta[
                1
            ]
            * score
        )

        probability = expit(
            linear_predictor
        )

        residual = (
            outcome
            - probability
        )

        weight = (
            probability
            * (
                1.0
                - probability
            )
        )

        gradient = np.array(
            [
                residual.sum(),
                np.dot(
                    score,
                    residual,
                ),
            ],
            dtype=float,
        )

        information = np.array(
            [
                [
                    weight.sum(),
                    np.dot(
                        weight,
                        score,
                    ),
                ],
                [
                    np.dot(
                        weight,
                        score,
                    ),
                    np.dot(
                        weight,
                        score
                        * score,
                    ),
                ],
            ],
            dtype=float,
        )

        information += (
            ridge
            * np.eye(
                2
            )
        )

        try:
            step = np.linalg.solve(
                information,
                gradient,
            )
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(
                information,
                gradient,
                rcond=None,
            )[
                0
            ]

        step = np.clip(
            step,
            -5.0,
            5.0,
        )

        beta += step

        beta[
            0
        ] = np.clip(
            beta[
                0
            ],
            -20.0,
            20.0,
        )

        beta[
            1
        ] = np.clip(
            beta[
                1
            ],
            -10.0,
            10.0,
        )

        if float(
            np.max(
                np.abs(
                    step
                )
            )
        ) < 1e-9:
            break

    if not np.isfinite(
        beta
    ).all():
        raise ValueError(
            "Non-finite logistic "
            "recalibration fit"
        )

    return (
        float(
            beta[
                0
            ]
        ),
        float(
            beta[
                1
            ]
        ),
    )


def fit_recalibration(
    frame: pd.DataFrame,
    *,
    method: str,
) -> tuple[
    float,
    float,
]:
    prediction = frame[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    outcome = frame[
        "outcome"
    ].to_numpy(
        dtype=float
    )

    score = prediction_score(
        prediction
    )

    if method == "identity":
        return (
            0.0,
            1.0,
        )

    if method == "intercept_only":
        return fit_intercept_only(
            score,
            outcome,
        )

    if method == "logistic":
        return (
            fit_logistic_recalibration(
                score,
                outcome,
            )
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def apply_recalibration(
    prediction: np.ndarray,
    *,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return expit(
        intercept
        + slope
        * prediction_score(
            prediction
        )
    )


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


def evaluation_metrics(
    frame: pd.DataFrame,
    recalibrated_prediction: np.ndarray,
) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError(
            "Cannot evaluate an empty frame"
        )

    work = frame[
        [
            "outcome",
            "risk_bin",
        ]
    ].copy()

    work[
        "_prediction"
    ] = recalibrated_prediction

    ici = 0.0

    for _, group in work.groupby(
        "risk_bin",
        sort=True,
    ):
        weight = float(
            len(
                group
            )
            / len(
                work
            )
        )

        ici += (
            weight
            * abs(
                float(
                    group[
                        "outcome"
                    ].mean()
                )
                - float(
                    group[
                        "_prediction"
                    ].mean()
                )
            )
        )

    outcome = work[
        "outcome"
    ].to_numpy(
        dtype=float
    )

    prediction = work[
        "_prediction"
    ].to_numpy(
        dtype=float
    )

    return {
        "n_evaluated": int(
            len(
                work
            )
        ),
        "event_rate": float(
            outcome.mean()
        ),
        "mean_prediction": float(
            prediction.mean()
        ),
        "ici": float(
            ici
        ),
        "signed_calibration_gap": float(
            outcome.mean()
            - prediction.mean()
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
    }


def fit_policy_models(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    calibration_positions: np.ndarray,
    minimum_observed_n: int,
) -> pd.DataFrame:
    rows = []

    for policy_name in POLICY_ORDER:
        calibration = policy_frames[
            policy_name
        ].iloc[
            calibration_positions
        ]

        observed_calibration = (
            calibration.loc[
                calibration[
                    "emulated_observed"
                ].eq(
                    1
                )
            ]
        )

        if len(
            observed_calibration
        ) < minimum_observed_n:
            raise ValueError(
                f"{policy_name} has only "
                f"{len(observed_calibration)} "
                "observed calibration rows"
            )

        if observed_calibration[
            "outcome"
        ].nunique() != 2:
            raise ValueError(
                f"{policy_name} calibration "
                "subset lacks both classes"
            )

        for method in METHOD_ORDER:
            (
                intercept,
                slope,
            ) = fit_recalibration(
                observed_calibration,
                method=method,
            )

            rows.append(
                {
                    "fit_policy": (
                        policy_name
                    ),
                    "method": method,
                    "n_calibration_observed": int(
                        len(
                            observed_calibration
                        )
                    ),
                    "calibration_event_rate": float(
                        observed_calibration[
                            "outcome"
                        ].mean()
                    ),
                    "intercept": intercept,
                    "slope": slope,
                }
            )

    return pd.DataFrame(
        rows
    )


def parameter_lookup(
    parameters: pd.DataFrame,
    *,
    policy: str,
    method: str,
) -> pd.Series:
    selected = parameters.loc[
        parameters[
            "fit_policy"
        ].eq(
            policy
        )
        & parameters[
            "method"
        ].eq(
            method
        )
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            f"Missing parameters for "
            f"{policy}, {method}"
        )

    return selected.iloc[
        0
    ]


def evaluate_policy_models(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    parameters: pd.DataFrame,
    *,
    test_positions: np.ndarray,
    minimum_observed_n: int,
) -> pd.DataFrame:
    reference_test = (
        policy_frames[
            "baseline"
        ]
        .iloc[
            test_positions
        ]
        .copy()
    )

    rows = []

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
                ),
            }
        )

        restricted_test = (
            policy_frames[
                fit_policy
            ]
            .iloc[
                test_positions
            ]
        )

        restricted_test = (
            restricted_test.loc[
                restricted_test[
                    "emulated_observed"
                ].eq(
                    1
                )
            ]
            .copy()
        )

        if len(
            restricted_test
        ) < minimum_observed_n:
            raise ValueError(
                f"{fit_policy} has only "
                f"{len(restricted_test)} "
                "observed test rows"
            )

        restricted_prediction = (
            apply_recalibration(
                restricted_test[
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
                    fit_policy
                ),
                "intercept": intercept,
                "slope": slope,
                **evaluation_metrics(
                    restricted_test,
                    restricted_prediction,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def metric_lookup(
    evaluation: pd.DataFrame,
    *,
    fit_policy: str,
    method: str,
    evaluation_target: str,
    metric: str,
) -> float:
    selected = evaluation.loc[
        evaluation[
            "fit_policy"
        ].eq(
            fit_policy
        )
        & evaluation[
            "method"
        ].eq(
            method
        )
        & evaluation[
            "evaluation_target"
        ].eq(
            evaluation_target
        )
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Missing evaluation row: "
            f"{fit_policy}, {method}, "
            f"{evaluation_target}"
        )

    return float(
        selected[
            metric
        ].iloc[
            0
        ]
    )


def build_point_contrasts(
    parameters: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for policy in POLICY_ORDER:
        identity_restricted = {
            metric: metric_lookup(
                evaluation,
                fit_policy=policy,
                method="identity",
                evaluation_target="restricted",
                metric=metric,
            )
            for metric in [
                "ici",
                "brier",
                "log_loss",
            ]
        }

        identity_reference = {
            metric: metric_lookup(
                evaluation,
                fit_policy=policy,
                method="identity",
                evaluation_target="reference_full",
                metric=metric,
            )
            for metric in [
                "ici",
                "brier",
                "log_loss",
            ]
        }

        for method in [
            "intercept_only",
            "logistic",
        ]:
            parameter = parameter_lookup(
                parameters,
                policy=policy,
                method=method,
            )

            row = {
                "fit_policy": policy,
                "method": method,
                "intercept": float(
                    parameter[
                        "intercept"
                    ]
                ),
                "slope": float(
                    parameter[
                        "slope"
                    ]
                ),
            }

            for metric in [
                "ici",
                "brier",
                "log_loss",
            ]:
                row[
                    f"restricted_{metric}_change"
                ] = (
                    metric_lookup(
                        evaluation,
                        fit_policy=policy,
                        method=method,
                        evaluation_target=(
                            "restricted"
                        ),
                        metric=metric,
                    )
                    - identity_restricted[
                        metric
                    ]
                )

                row[
                    f"reference_{metric}_change"
                ] = (
                    metric_lookup(
                        evaluation,
                        fit_policy=policy,
                        method=method,
                        evaluation_target=(
                            "reference_full"
                        ),
                        metric=metric,
                    )
                    - identity_reference[
                        metric
                    ]
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def aggressive_transfer_penalty(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    parameters: pd.DataFrame,
    *,
    test_positions: np.ndarray,
    method: str,
) -> float:
    aggressive_test = (
        policy_frames[
            "aggressive_early"
        ]
        .iloc[
            test_positions
        ]
    )

    aggressive_test = (
        aggressive_test.loc[
            aggressive_test[
                "emulated_observed"
            ].eq(
                1
            )
        ]
        .copy()
    )

    baseline_parameter = parameter_lookup(
        parameters,
        policy="baseline",
        method=method,
    )

    aggressive_parameter = parameter_lookup(
        parameters,
        policy="aggressive_early",
        method=method,
    )

    baseline_prediction = apply_recalibration(
        aggressive_test[
            "prediction"
        ].to_numpy(
            dtype=float
        ),
        intercept=float(
            baseline_parameter[
                "intercept"
            ]
        ),
        slope=float(
            baseline_parameter[
                "slope"
            ]
        ),
    )

    aggressive_prediction = apply_recalibration(
        aggressive_test[
            "prediction"
        ].to_numpy(
            dtype=float
        ),
        intercept=float(
            aggressive_parameter[
                "intercept"
            ]
        ),
        slope=float(
            aggressive_parameter[
                "slope"
            ]
        ),
    )

    baseline_ici = float(
        evaluation_metrics(
            aggressive_test,
            baseline_prediction,
        )[
            "ici"
        ]
    )

    aggressive_ici = float(
        evaluation_metrics(
            aggressive_test,
            aggressive_prediction,
        )[
            "ici"
        ]
    )

    return float(
        baseline_ici
        - aggressive_ici
    )


def policy_comparison_table(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    parameters: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    test_positions: np.ndarray,
) -> pd.DataFrame:
    logistic_rows = contrasts.loc[
        contrasts[
            "method"
        ].eq(
            "logistic"
        )
    ]

    aggressive_reference_harm = float(
        logistic_rows.loc[
            logistic_rows[
                "fit_policy"
            ].eq(
                "aggressive_early"
            ),
            "reference_ici_change",
        ].iloc[
            0
        ]
    )

    conservative_reference_harm = float(
        logistic_rows.loc[
            logistic_rows[
                "fit_policy"
            ].eq(
                "conservative_late"
            ),
            "reference_ici_change",
        ].iloc[
            0
        ]
    )

    logistic_parameters = parameters.loc[
        parameters[
            "method"
        ].eq(
            "logistic"
        )
    ]

    slope_range = float(
        logistic_parameters[
            "slope"
        ].max()
        - logistic_parameters[
            "slope"
        ].min()
    )

    rows = [
        {
            "comparison": (
                "aggressive_minus_conservative_"
                "reference_ici_harm"
            ),
            "method": "logistic",
            "metric": (
                "reference_ici_harm_difference"
            ),
            "value": float(
                aggressive_reference_harm
                - conservative_reference_harm
            ),
        },
        {
            "comparison": (
                "baseline_fit_on_aggressive_"
                "minus_aggressive_fit"
            ),
            "method": "logistic",
            "metric": (
                "aggressive_restricted_ici_"
                "transfer_penalty"
            ),
            "value": (
                aggressive_transfer_penalty(
                    policy_frames,
                    parameters,
                    test_positions=(
                        test_positions
                    ),
                    method="logistic",
                )
            ),
        },
        {
            "comparison": (
                "across_policy_logistic_"
                "parameter_variation"
            ),
            "method": "logistic",
            "metric": "slope_range",
            "value": slope_range,
        },
    ]

    return pd.DataFrame(
        rows
    )


def run_single_analysis(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    calibration_positions: np.ndarray,
    test_positions: np.ndarray,
    minimum_observed_n: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    parameters = fit_policy_models(
        policy_frames,
        calibration_positions=(
            calibration_positions
        ),
        minimum_observed_n=(
            minimum_observed_n
        ),
    )

    evaluation = evaluate_policy_models(
        policy_frames,
        parameters,
        test_positions=test_positions,
        minimum_observed_n=(
            minimum_observed_n
        ),
    )

    contrasts = build_point_contrasts(
        parameters,
        evaluation,
    )

    comparisons = policy_comparison_table(
        policy_frames,
        parameters,
        contrasts,
        test_positions=test_positions,
    )

    return (
        parameters,
        evaluation,
        contrasts,
        comparisons,
    )


def run_bootstrap(
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    calibration_positions: np.ndarray,
    test_positions: np.ndarray,
    replicates: int,
    seed: int,
    minimum_observed_n: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    contrast_rows = []
    comparison_rows = []
    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        calibration_sample = rng.choice(
            calibration_positions,
            size=len(
                calibration_positions
            ),
            replace=True,
        )

        test_sample = rng.choice(
            test_positions,
            size=len(
                test_positions
            ),
            replace=True,
        )

        try:
            (
                _,
                _,
                contrasts,
                comparisons,
            ) = run_single_analysis(
                policy_frames,
                calibration_positions=(
                    calibration_sample
                ),
                test_positions=(
                    test_sample
                ),
                minimum_observed_n=(
                    minimum_observed_n
                ),
            )
        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            invalid += 1
            continue

        for _, row in contrasts.iterrows():
            for metric in [
                "intercept",
                "slope",
                *CONTRAST_METRICS,
            ]:
                contrast_rows.append(
                    {
                        "bootstrap_id": (
                            bootstrap_id
                        ),
                        "fit_policy": row[
                            "fit_policy"
                        ],
                        "method": row[
                            "method"
                        ],
                        "metric": metric,
                        "value": float(
                            row[
                                metric
                            ]
                        ),
                    }
                )

        for _, row in comparisons.iterrows():
            comparison_rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    "comparison": row[
                        "comparison"
                    ],
                    "method": row[
                        "method"
                    ],
                    "metric": row[
                        "metric"
                    ],
                    "value": float(
                        row[
                            "value"
                        ]
                    ),
                }
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

    if not contrast_rows:
        raise RuntimeError(
            "No valid bootstrap "
            "replications completed"
        )

    return (
        pd.DataFrame(
            contrast_rows
        ),
        pd.DataFrame(
            comparison_rows
        ),
        invalid,
    )


def contrast_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = bootstrap.groupby(
        [
            "fit_policy",
            "method",
            "metric",
        ],
        sort=True,
    )

    for (
        fit_policy,
        method,
        metric,
    ), group in grouped:
        values = group[
            "value"
        ].dropna()

        rows.append(
            {
                "fit_policy": fit_policy,
                "method": method,
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


def comparison_interval_table(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = bootstrap.groupby(
        [
            "comparison",
            "method",
            "metric",
        ],
        sort=True,
    )

    for (
        comparison,
        method,
        metric,
    ), group in grouped:
        values = group[
            "value"
        ].dropna()

        rows.append(
            {
                "comparison": comparison,
                "method": method,
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


def contrast_interval_value(
    intervals: pd.DataFrame,
    *,
    fit_policy: str,
    method: str,
    metric: str,
    quantile: str,
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
        )
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Missing contrast interval: "
            f"{fit_policy}, {method}, {metric}"
        )

    return float(
        selected[
            quantile
        ].iloc[
            0
        ]
    )


def comparison_interval_value(
    intervals: pd.DataFrame,
    *,
    comparison: str,
    metric: str,
    quantile: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "comparison"
        ].eq(
            comparison
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
            "Missing comparison interval: "
            f"{comparison}, {metric}"
        )

    return float(
        selected[
            quantile
        ].iloc[
            0
        ]
    )


def build_h7_core_checks(
    contrast_intervals: pd.DataFrame,
) -> pd.DataFrame:
    restricted_ici_q975 = (
        contrast_interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "restricted_ici_change"
            ),
            quantile="q975",
        )
    )

    reference_ici_q025 = (
        contrast_interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "reference_ici_change"
            ),
            quantile="q025",
        )
    )

    restricted_logloss_q975 = (
        contrast_interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "restricted_log_loss_change"
            ),
            quantile="q975",
        )
    )

    reference_logloss_q025 = (
        contrast_interval_value(
            contrast_intervals,
            fit_policy="aggressive_early",
            method="logistic",
            metric=(
                "reference_log_loss_change"
            ),
            quantile="q025",
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "aggressive logistic "
                    "recalibration improves "
                    "restricted ICI"
                ),
                "value": (
                    restricted_ici_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": bool(
                    restricted_ici_q975
                    < 0
                ),
            },
            {
                "check": (
                    "aggressive logistic "
                    "recalibration worsens "
                    "reference ICI"
                ),
                "value": (
                    reference_ici_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    reference_ici_q025
                    > 0
                ),
            },
            {
                "check": (
                    "aggressive logistic "
                    "recalibration improves "
                    "restricted log loss"
                ),
                "value": (
                    restricted_logloss_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": bool(
                    restricted_logloss_q975
                    < 0
                ),
            },
            {
                "check": (
                    "aggressive logistic "
                    "recalibration worsens "
                    "reference log loss"
                ),
                "value": (
                    reference_logloss_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    reference_logloss_q025
                    > 0
                ),
            },
        ]
    )


def build_secondary_checks(
    contrast_intervals: pd.DataFrame,
    comparison_intervals: pd.DataFrame,
) -> pd.DataFrame:
    transfer_q025 = (
        comparison_interval_value(
            comparison_intervals,
            comparison=(
                "baseline_fit_on_aggressive_"
                "minus_aggressive_fit"
            ),
            metric=(
                "aggressive_restricted_ici_"
                "transfer_penalty"
            ),
            quantile="q025",
        )
    )

    slope_range_q025 = (
        comparison_interval_value(
            comparison_intervals,
            comparison=(
                "across_policy_logistic_"
                "parameter_variation"
            ),
            metric="slope_range",
            quantile="q025",
        )
    )

    reference_gradient_q025 = (
        comparison_interval_value(
            comparison_intervals,
            comparison=(
                "aggressive_minus_conservative_"
                "reference_ici_harm"
            ),
            metric=(
                "reference_ici_harm_difference"
            ),
            quantile="q025",
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "baseline recalibration "
                    "transfers poorly to the "
                    "aggressive policy"
                ),
                "value": transfer_q025,
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    transfer_q025
                    > 0
                ),
            },
            {
                "check": (
                    "logistic recalibration slopes "
                    "vary across policies"
                ),
                "value": slope_range_q025,
                "threshold": 0.10,
                "comparison": ">",
                "passed": bool(
                    slope_range_q025
                    > 0.10
                ),
            },
            {
                "check": (
                    "reference harm is greater "
                    "under aggressive than "
                    "conservative policy"
                ),
                "value": (
                    reference_gradient_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    reference_gradient_q025
                    > 0
                ),
            },
        ]
    )


def build_structural_checks(
    *,
    dataset: str,
    fixed: pd.DataFrame,
    input_status: str,
    h6_status: str,
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
    parameters: pd.DataFrame,
    bootstrap_contrasts: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
    minimum_observed_n: int,
) -> pd.DataFrame:
    configuration = DATASET_CONFIG[
        dataset
    ]

    calibration_counts = []

    test_counts = []

    for policy in POLICY_ORDER:
        frame = policy_frames[
            policy
        ]

        calibration_counts.append(
            int(
                frame.loc[
                    frame[
                        "analysis_split"
                    ].eq(
                        "calibration"
                    )
                    & frame[
                        "emulated_observed"
                    ].eq(
                        1
                    )
                ].shape[
                    0
                ]
            )
        )

        test_counts.append(
            int(
                frame.loc[
                    frame[
                        "analysis_split"
                    ].eq(
                        "test"
                    )
                    & frame[
                        "emulated_observed"
                    ].eq(
                        1
                    )
                ].shape[
                    0
                ]
            )
        )

    expected_bootstrap_rows = (
        requested_bootstraps
        * len(
            POLICY_ORDER
        )
        * 2
        * (
            2
            + len(
                CONTRAST_METRICS
            )
        )
    )

    identity = parameters.loc[
        parameters[
            "method"
        ].eq(
            "identity"
        )
    ]

    return pd.DataFrame(
        [
            {
                "check": (
                    "dataset benchmark already frozen"
                ),
                "value": input_status,
                "threshold": (
                    configuration[
                        "expected_status"
                    ]
                ),
                "comparison": "==",
                "passed": (
                    input_status
                    == configuration[
                        "expected_status"
                    ]
                ),
            },
            {
                "check": (
                    "cross-dataset H6 package "
                    "already frozen"
                ),
                "value": h6_status,
                "threshold": (
                    EXPECTED_H6_STATUS
                ),
                "comparison": "==",
                "passed": (
                    h6_status
                    == EXPECTED_H6_STATUS
                ),
            },
            {
                "check": (
                    "fixed population has expected size"
                ),
                "value": len(
                    fixed
                ),
                "threshold": int(
                    configuration[
                        "expected_n"
                    ]
                ),
                "comparison": "==",
                "passed": (
                    len(
                        fixed
                    )
                    == int(
                        configuration[
                            "expected_n"
                        ]
                    )
                ),
            },
            {
                "check": (
                    "all five policies and three "
                    "methods fitted"
                ),
                "value": len(
                    parameters
                ),
                "threshold": 15,
                "comparison": "==",
                "passed": (
                    len(
                        parameters
                    )
                    == 15
                ),
            },
            {
                "check": (
                    "identity parameters remain "
                    "zero intercept and unit slope"
                ),
                "value": float(
                    np.max(
                        np.abs(
                            identity[
                                "intercept"
                            ]
                        )
                    )
                    + np.max(
                        np.abs(
                            identity[
                                "slope"
                            ]
                            - 1.0
                        )
                    )
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": bool(
                    np.allclose(
                        identity[
                            "intercept"
                        ],
                        0.0,
                    )
                    and np.allclose(
                        identity[
                            "slope"
                        ],
                        1.0,
                    )
                ),
            },
            {
                "check": (
                    "every policy has sufficient "
                    "observed calibration rows"
                ),
                "value": min(
                    calibration_counts
                ),
                "threshold": (
                    minimum_observed_n
                ),
                "comparison": ">=",
                "passed": bool(
                    min(
                        calibration_counts
                    )
                    >= minimum_observed_n
                ),
            },
            {
                "check": (
                    "every policy has sufficient "
                    "observed test rows"
                ),
                "value": min(
                    test_counts
                ),
                "threshold": (
                    minimum_observed_n
                ),
                "comparison": ">=",
                "passed": bool(
                    min(
                        test_counts
                    )
                    >= minimum_observed_n
                ),
            },
            {
                "check": (
                    "all requested paired "
                    "bootstrap contrast rows completed"
                ),
                "value": len(
                    bootstrap_contrasts
                ),
                "threshold": (
                    expected_bootstrap_rows
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap_contrasts
                    )
                    == expected_bootstrap_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    configuration = DATASET_CONFIG[
        args.dataset
    ]

    input_path = Path(
        args.input
        or configuration[
            "default_input"
        ]
    )

    input_status_path = Path(
        args.input_status
        or configuration[
            "default_status"
        ]
    )

    h6_status_path = Path(
        args.h6_status
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_status = read_status(
        input_status_path
    )

    h6_status = read_status(
        h6_status_path
    )

    raw = pd.read_csv(
        input_path,
        low_memory=False,
    )

    (
        fixed,
        risk_bin_edges,
        population_hash,
    ) = prepare_dataset(
        dataset=args.dataset,
        raw=raw,
        n_bins=args.n_bins,
        seed=args.seed,
    )

    fixed = assign_analysis_split(
        fixed,
        calibration_fraction=(
            args.calibration_fraction
        ),
        seed=args.seed + 11,
    )

    policy_frames = {
        policy.name: (
            policy_core.apply_policy(
                fixed,
                policy,
            )
        )
        for policy in (
            policy_core.policy_definitions()
        )
    }

    calibration_positions = np.flatnonzero(
        fixed[
            "analysis_split"
        ].eq(
            "calibration"
        ).to_numpy()
    )

    test_positions = np.flatnonzero(
        fixed[
            "analysis_split"
        ].eq(
            "test"
        ).to_numpy()
    )

    (
        parameters,
        evaluation,
        point_contrasts,
        point_comparisons,
    ) = run_single_analysis(
        policy_frames,
        calibration_positions=(
            calibration_positions
        ),
        test_positions=test_positions,
        minimum_observed_n=(
            args.minimum_observed_split_n
        ),
    )

    print(
        "Phase 5J-E real-data "
        "recalibration fragility"
    )

    print(
        f"\nDataset: "
        f"{configuration['display_name']}"
    )

    print(
        "\nRunning paired recalibration bootstrap..."
    )

    (
        bootstrap_contrasts,
        bootstrap_comparisons,
        invalid_bootstraps,
    ) = run_bootstrap(
        policy_frames,
        calibration_positions=(
            calibration_positions
        ),
        test_positions=test_positions,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed + 29,
        minimum_observed_n=(
            args.minimum_observed_split_n
        ),
    )

    contrast_intervals = (
        contrast_interval_table(
            bootstrap_contrasts
        )
    )

    comparison_intervals = (
        comparison_interval_table(
            bootstrap_comparisons
        )
    )

    structural_checks = (
        build_structural_checks(
            dataset=args.dataset,
            fixed=fixed,
            input_status=input_status,
            h6_status=h6_status,
            policy_frames=policy_frames,
            parameters=parameters,
            bootstrap_contrasts=(
                bootstrap_contrasts
            ),
            invalid_bootstraps=(
                invalid_bootstraps
            ),
            requested_bootstraps=(
                args.bootstrap_replicates
            ),
            minimum_observed_n=(
                args.minimum_observed_split_n
            ),
        )
    )

    core_checks = build_h7_core_checks(
        contrast_intervals
    )

    secondary_checks = (
        build_secondary_checks(
            contrast_intervals,
            comparison_intervals,
        )
    )

    status = (
        f"phase5j_{args.dataset}_"
        "recalibration_benchmarked"
        if structural_checks[
            "passed"
        ].all()
        else (
            f"phase5j_{args.dataset}_"
            "recalibration_requires_review"
        )
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

    point_contrasts.to_csv(
        output_dir
        / "point_contrasts.csv",
        index=False,
    )

    point_comparisons.to_csv(
        output_dir
        / "point_policy_comparisons.csv",
        index=False,
    )

    bootstrap_contrasts.to_csv(
        output_dir
        / "bootstrap_contrasts.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_comparisons.to_csv(
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

    comparison_intervals.to_csv(
        output_dir
        / "policy_comparison_intervals.csv",
        index=False,
    )

    structural_checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    core_checks.to_csv(
        output_dir
        / "h7_core_decision_checks.csv",
        index=False,
    )

    secondary_checks.to_csv(
        output_dir
        / "h7_secondary_checks.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "edge_index": np.arange(
                len(
                    risk_bin_edges
                )
            ),
            "edge": risk_bin_edges,
        }
    ).to_csv(
        output_dir
        / "frozen_risk_bin_edges.csv",
        index=False,
    )

    evidence = {
        "phase": "5J-E",
        "status": status,
        "dataset": (
            configuration[
                "display_name"
            ]
        ),
        "horizon_days": float(
            configuration[
                "horizon"
            ]
        ),
        "fixed_population_hash": (
            population_hash
        ),
        "n": int(
            len(
                fixed
            )
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
        "methods": METHOD_ORDER,
        "fit_target": (
            "Policy-observed outcomes in the "
            "calibration split."
        ),
        "evaluation_targets": [
            (
                "Policy-observed outcomes in the "
                "independent test split."
            ),
            (
                "Linked complete outcomes in the "
                "same independent test split."
            ),
        ],
        "core_h7_supported": bool(
            core_checks[
                "passed"
            ].all()
        ),
        "core_h7_checks_passed": int(
            core_checks[
                "passed"
            ].sum()
        ),
        "core_h7_checks_total": int(
            len(
                core_checks
            )
        ),
        "interpretation": (
            "A controlled real-data-grounded "
            "recalibration emulation. It evaluates "
            "target mismatch under policy-selected "
            "outcome observation and is not a causal "
            "estimate of an actual clinical policy."
        ),
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
        / (
            f"phase5j_{args.dataset}_"
            "recalibration_status.txt"
        )
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
        point_contrasts.to_string(
            index=False
        )
    )

    print(
        "\nStructural validation"
    )

    print(
        structural_checks.to_string(
            index=False
        )
    )

    print(
        "\nCore H7 decision checks"
    )

    print(
        core_checks.to_string(
            index=False
        )
    )

    print(
        "\nSecondary checks"
    )

    print(
        secondary_checks.to_string(
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
