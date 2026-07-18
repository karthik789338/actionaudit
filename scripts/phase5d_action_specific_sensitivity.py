"""Phase 5D-A: action-specific pattern-mixture sensitivity.

For hidden outcomes after action a:

    logit E[Y | P, R=0, A=a] = logit(P) + delta_a

The actual post-exit outcomes are used only for retrospective
validation and for locating the oracle-matching delta values.

No prediction model is refitted and no Phase 5C artifact is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from actionaudit.km import reference_quantile_edges


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute action-specific pattern-mixture "
            "sensitivity surfaces and tipping points."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--phase5c-dir",
        default=(
            "artifacts/"
            "phase5c_empirical_action_decomposition/"
            "primary"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5d_action_specific_sensitivity"
        ),
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--delta-min",
        type=float,
        default=-2.0,
    )

    parser.add_argument(
        "--delta-max",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--delta-step",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--practical-distortion-threshold",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--reference-ici-threshold",
        type=float,
        default=0.02,
    )

    return parser.parse_args()


def build_delta_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    if step <= 0:
        raise ValueError(
            "delta-step must be positive"
        )

    if minimum >= maximum:
        raise ValueError(
            "delta-min must be below delta-max"
        )

    count = int(
        round(
            (
                maximum
                - minimum
            )
            / step
        )
    )

    grid = np.linspace(
        minimum,
        maximum,
        count + 1,
    )

    if not np.isclose(
        grid,
        0.0,
        atol=1e-12,
    ).any():
        grid = np.sort(
            np.append(
                grid,
                0.0,
            )
        )

    return np.round(
        grid,
        10,
    )


def load_test_predictions(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen predictions: {path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "subject_id",
        "hadm_id",
        "prediction",
        "outcome",
        "observed",
        "analysis_action",
    }

    missing = sorted(
        required.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    frame[
        "prediction"
    ] = pd.to_numeric(
        frame[
            "prediction"
        ],
        errors="raise",
    ).clip(
        1e-6,
        1.0 - 1e-6,
    )

    frame[
        "outcome"
    ] = frame[
        "outcome"
    ].astype(
        int
    )

    frame[
        "observed"
    ] = frame[
        "observed"
    ].astype(
        int
    )

    frame[
        "analysis_action"
    ] = frame[
        "analysis_action"
    ].astype(
        str
    )

    hidden_actions = set(
        frame.loc[
            frame[
                "observed"
            ].eq(
                0
            ),
            "analysis_action",
        ]
    )

    missing_actions = sorted(
        hidden_actions.difference(
            ACTION_ORDER
        )
    )

    if missing_actions:
        raise ValueError(
            "Unrecognized hidden actions: "
            f"{missing_actions}"
        )

    if frame[
        "subject_id"
    ].duplicated().any():
        raise ValueError(
            "Expected one test row per patient"
        )

    return frame


class SensitivityEngine:
    """Preaggregated exact sensitivity calculations."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        n_bins: int,
    ) -> None:
        self.frame = frame.copy()
        self.n_bins = int(
            n_bins
        )

        self.bin_edges = reference_quantile_edges(
            self.frame[
                "prediction"
            ].to_numpy(),
            self.n_bins,
        )

        self.frame[
            "risk_bin"
        ] = np.digitize(
            self.frame[
                "prediction"
            ].to_numpy(),
            self.bin_edges[
                1:-1
            ],
            right=True,
        )

        self.n_total = int(
            len(
                self.frame
            )
        )

        self.action_to_index = {
            action: index
            for index, action in enumerate(
                ACTION_ORDER
            )
        }

        self.cells: dict[
            tuple[int, str],
            dict[str, object],
        ] = {}

        self.bin_stats: dict[
            int,
            dict[str, float | int],
        ] = {}

        for risk_bin in range(
            self.n_bins
        ):
            group = self.frame.loc[
                self.frame[
                    "risk_bin"
                ].eq(
                    risk_bin
                )
            ]

            observed = group.loc[
                group[
                    "observed"
                ].eq(
                    1
                )
            ]

            if observed.empty:
                raise ValueError(
                    f"Risk bin {risk_bin} has "
                    "no outcome-observed patients"
                )

            self.bin_stats[
                risk_bin
            ] = {
                "n_bin": int(
                    len(
                        group
                    )
                ),
                "reference_weight": float(
                    len(
                        group
                    )
                    / self.n_total
                ),
                "prediction_mean": float(
                    group[
                        "prediction"
                    ].mean()
                ),
                "n_observed": int(
                    len(
                        observed
                    )
                ),
                "observed_fraction": float(
                    len(
                        observed
                    )
                    / len(
                        group
                    )
                ),
                "mu_observed": float(
                    observed[
                        "outcome"
                    ].mean()
                ),
            }

            for action in ACTION_ORDER:
                action_group = group.loc[
                    group[
                        "observed"
                    ].eq(
                        0
                    )
                    & group[
                        "analysis_action"
                    ].eq(
                        action
                    )
                ]

                prediction = action_group[
                    "prediction"
                ].to_numpy(
                    dtype=float
                )

                self.cells[
                    (
                        risk_bin,
                        action,
                    )
                ] = {
                    "n": int(
                        len(
                            action_group
                        )
                    ),
                    "pi": float(
                        len(
                            action_group
                        )
                        / len(
                            group
                        )
                    ),
                    "logit_prediction": (
                        logit(
                            prediction
                        )
                        if len(
                            prediction
                        )
                        > 0
                        else np.empty(
                            0,
                            dtype=float,
                        )
                    ),
                    "actual_outcome": (
                        action_group[
                            "outcome"
                        ].to_numpy(
                            dtype=float
                        )
                    ),
                }

    def evaluate(
        self,
        *,
        delta_by_action: dict[str, float] | None = None,
        use_actual_hidden_outcomes: bool = False,
    ) -> tuple[
        dict[str, float],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if delta_by_action is None:
            delta_by_action = {
                action: 0.0
                for action in ACTION_ORDER
            }

        bin_rows: list[
            dict[str, object]
        ] = []

        action_bin_rows: list[
            dict[str, object]
        ] = []

        for risk_bin in range(
            self.n_bins
        ):
            stats = self.bin_stats[
                risk_bin
            ]

            mu_observed = float(
                stats[
                    "mu_observed"
                ]
            )

            mu_reference = (
                float(
                    stats[
                        "observed_fraction"
                    ]
                )
                * mu_observed
            )

            contribution_sum = 0.0
            absolute_contribution_sum = 0.0

            for action in ACTION_ORDER:
                cell = self.cells[
                    (
                        risk_bin,
                        action,
                    )
                ]

                n_action = int(
                    cell[
                        "n"
                    ]
                )

                pi_action = float(
                    cell[
                        "pi"
                    ]
                )

                if n_action == 0:
                    mu_action = np.nan
                    contribution = 0.0
                elif use_actual_hidden_outcomes:
                    mu_action = float(
                        np.mean(
                            cell[
                                "actual_outcome"
                            ]
                        )
                    )

                    contribution = float(
                        pi_action
                        * (
                            mu_observed
                            - mu_action
                        )
                    )
                else:
                    delta = float(
                        delta_by_action.get(
                            action,
                            0.0,
                        )
                    )

                    mu_action = float(
                        np.mean(
                            expit(
                                cell[
                                    "logit_prediction"
                                ]
                                + delta
                            )
                        )
                    )

                    contribution = float(
                        pi_action
                        * (
                            mu_observed
                            - mu_action
                        )
                    )

                if n_action > 0:
                    mu_reference += (
                        pi_action
                        * mu_action
                    )

                contribution_sum += contribution

                absolute_contribution_sum += abs(
                    contribution
                )

                action_bin_rows.append(
                    {
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
                            stats[
                                "reference_weight"
                            ]
                        ),
                        "weighted_contribution": (
                            contribution
                            * float(
                                stats[
                                    "reference_weight"
                                ]
                            )
                        ),
                        "weighted_absolute_contribution": (
                            abs(
                                contribution
                            )
                            * float(
                                stats[
                                    "reference_weight"
                                ]
                            )
                        ),
                    }
                )

            observed_minus_reference = float(
                mu_observed
                - mu_reference
            )

            decomposition_residual = float(
                observed_minus_reference
                - contribution_sum
            )

            pointwise_aci = float(
                1.0
                - abs(
                    contribution_sum
                )
                / (
                    absolute_contribution_sum
                    + 1e-12
                )
            )

            bin_rows.append(
                {
                    "risk_bin": (
                        risk_bin
                    ),
                    **stats,
                    "mu_reference": (
                        mu_reference
                    ),
                    "observed_minus_reference": (
                        observed_minus_reference
                    ),
                    "contribution_sum": (
                        contribution_sum
                    ),
                    "absolute_contribution_sum": (
                        absolute_contribution_sum
                    ),
                    "pointwise_aci": (
                        pointwise_aci
                    ),
                    "decomposition_residual": (
                        decomposition_residual
                    ),
                }
            )

        bin_summary = pd.DataFrame(
            bin_rows
        )

        action_bins = pd.DataFrame(
            action_bin_rows
        )

        weights = bin_summary[
            "reference_weight"
        ].to_numpy(
            dtype=float
        )

        prediction_mean = bin_summary[
            "prediction_mean"
        ].to_numpy(
            dtype=float
        )

        mu_reference = bin_summary[
            "mu_reference"
        ].to_numpy(
            dtype=float
        )

        mu_observed = bin_summary[
            "mu_observed"
        ].to_numpy(
            dtype=float
        )

        action_summary = (
            action_bins.groupby(
                "action",
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
                    mu_observed
                    - mu_reference
                )
            )
        )

        metrics = {
            "reference_ici": float(
                np.sum(
                    weights
                    * np.abs(
                        mu_reference
                        - prediction_mean
                    )
                )
            ),
            "restricted_ici": float(
                np.sum(
                    weights
                    * np.abs(
                        mu_observed
                        - prediction_mean
                    )
                )
            ),
            "delta_ici_restricted_minus_reference": float(
                np.sum(
                    weights
                    * np.abs(
                        mu_observed
                        - prediction_mean
                    )
                )
                - np.sum(
                    weights
                    * np.abs(
                        mu_reference
                        - prediction_mean
                    )
                )
            ),
            "reference_signed_calibration_gap": float(
                np.sum(
                    weights
                    * (
                        mu_reference
                        - prediction_mean
                    )
                )
            ),
            "restricted_signed_calibration_gap": float(
                np.sum(
                    weights
                    * (
                        mu_observed
                        - prediction_mean
                    )
                )
            ),
            "restricted_minus_reference_signed_gap": float(
                np.sum(
                    weights
                    * (
                        mu_observed
                        - mu_reference
                    )
                )
            ),
            "net_curve_distortion": (
                net_curve_distortion
            ),
            "gross_action_debt": (
                gross_action_debt
            ),
            "pointwise_cancelled_debt": float(
                gross_action_debt
                - net_curve_distortion
            ),
            "pointwise_cancellation_ratio": float(
                1.0
                - net_curve_distortion
                / (
                    gross_action_debt
                    + 1e-12
                )
            ),
            "reference_event_rate": float(
                np.sum(
                    weights
                    * mu_reference
                )
            ),
            "restricted_event_rate": float(
                np.sum(
                    weights
                    * mu_observed
                )
            ),
            "maximum_decomposition_residual": float(
                bin_summary[
                    "decomposition_residual"
                ].abs().max()
            ),
        }

        return (
            metrics,
            action_summary,
            bin_summary,
        )


def solve_delta_for_target_mean(
    logit_prediction: np.ndarray,
    target_mean: float,
    *,
    lower: float = -12.0,
    upper: float = 12.0,
    tolerance: float = 1e-10,
    iterations: int = 200,
) -> float:
    if len(
        logit_prediction
    ) == 0:
        return float(
            "nan"
        )

    target = float(
        np.clip(
            target_mean,
            1e-10,
            1.0 - 1e-10,
        )
    )

    def mean_at(
        delta: float,
    ) -> float:
        return float(
            np.mean(
                expit(
                    logit_prediction
                    + delta
                )
            )
        )

    lower_mean = mean_at(
        lower
    )

    upper_mean = mean_at(
        upper
    )

    if target <= lower_mean:
        return lower

    if target >= upper_mean:
        return upper

    left = lower
    right = upper

    for _ in range(
        iterations
    ):
        midpoint = (
            left
            + right
        ) / 2.0

        midpoint_mean = mean_at(
            midpoint
        )

        if abs(
            midpoint_mean
            - target
        ) <= tolerance:
            return float(
                midpoint
            )

        if midpoint_mean < target:
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


def nearest_zero_crossing(
    x: np.ndarray,
    y: np.ndarray,
) -> float | None:
    candidates: list[
        float
    ] = []

    for index in range(
        len(
            x
        )
        - 1
    ):
        x1 = float(
            x[
                index
            ]
        )

        x2 = float(
            x[
                index
                + 1
            ]
        )

        y1 = float(
            y[
                index
            ]
        )

        y2 = float(
            y[
                index
                + 1
            ]
        )

        if np.isclose(
            y1,
            0.0,
            atol=1e-12,
        ):
            candidates.append(
                x1
            )

            continue

        if y1 * y2 < 0:
            crossing = (
                x1
                + (
                    0.0
                    - y1
                )
                * (
                    x2
                    - x1
                )
                / (
                    y2
                    - y1
                )
            )

            candidates.append(
                float(
                    crossing
                )
            )

    if np.isclose(
        y[
            -1
        ],
        0.0,
        atol=1e-12,
    ):
        candidates.append(
            float(
                x[
                    -1
                ]
            )
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=abs,
    )


def nearest_delta_satisfying(
    frame: pd.DataFrame,
    condition: Callable[
        [
            pd.DataFrame,
        ],
        pd.Series,
    ],
) -> float | None:
    selected = frame.loc[
        condition(
            frame
        )
    ].copy()

    if selected.empty:
        return None

    selected[
        "absolute_delta"
    ] = selected[
        "delta"
    ].abs()

    return float(
        selected.sort_values(
            [
                "absolute_delta",
                "delta",
            ]
        ).iloc[
            0
        ][
            "delta"
        ]
    )


def action_hidden_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    hidden = frame.loc[
        frame[
            "observed"
        ].eq(
            0
        )
    ].copy()

    return (
        hidden.groupby(
            "analysis_action",
            as_index=False,
            sort=False,
        )
        .agg(
            n_hidden=(
                "subject_id",
                "size",
            ),
            actual_hidden_event_rate=(
                "outcome",
                "mean",
            ),
            mean_prediction=(
                "prediction",
                "mean",
            ),
        )
        .rename(
            columns={
                "analysis_action": (
                    "action"
                )
            }
        )
    )


def collect_action_logit_predictions(
    frame: pd.DataFrame,
    action: str,
) -> np.ndarray:
    return logit(
        frame.loc[
            frame[
                "observed"
            ].eq(
                0
            )
            & frame[
                "analysis_action"
            ].eq(
                action
            ),
            "prediction",
        ].to_numpy(
            dtype=float
        )
    )


def scenario_comparison(
    *,
    oracle_metrics: dict[str, float],
    oracle_bins: pd.DataFrame,
    scenario_name: str,
    metrics: dict[str, float],
    bins: pd.DataFrame,
    action_rate_error: float,
) -> dict[str, object]:
    bin_curve_mae = float(
        np.sum(
            oracle_bins[
                "reference_weight"
            ].to_numpy(
                dtype=float
            )
            * np.abs(
                bins[
                    "mu_reference"
                ].to_numpy(
                    dtype=float
                )
                - oracle_bins[
                    "mu_reference"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    return {
        "scenario": scenario_name,
        "reference_ici": (
            metrics[
                "reference_ici"
            ]
        ),
        "reference_ici_error": abs(
            metrics[
                "reference_ici"
            ]
            - oracle_metrics[
                "reference_ici"
            ]
        ),
        "reference_event_rate": (
            metrics[
                "reference_event_rate"
            ]
        ),
        "reference_event_rate_error": abs(
            metrics[
                "reference_event_rate"
            ]
            - oracle_metrics[
                "reference_event_rate"
            ]
        ),
        "bin_reference_curve_mae": (
            bin_curve_mae
        ),
        "weighted_action_rate_mae": (
            action_rate_error
        ),
        "restricted_minus_reference_signed_gap": (
            metrics[
                "restricted_minus_reference_signed_gap"
            ]
        ),
        "net_curve_distortion": (
            metrics[
                "net_curve_distortion"
            ]
        ),
    }


def run_sensitivity_sweeps(
    *,
    engine: SensitivityEngine,
    delta_grid: np.ndarray,
    practical_threshold: float,
    ici_threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    scenario_rows: list[
        dict[str, object]
    ] = []

    action_curve_rows: list[
        dict[str, object]
    ] = []

    action_bin_curve_rows: list[
        dict[str, object]
    ] = []

    # Common delta across every hidden action.
    for delta in delta_grid:
        delta_map = {
            action: float(
                delta
            )
            for action in ACTION_ORDER
        }

        (
            metrics,
            action_summary,
            bin_summary,
        ) = engine.evaluate(
            delta_by_action=delta_map
        )

        scenario_rows.append(
            {
                "scenario_type": (
                    "common_delta"
                ),
                "target_action": (
                    "all_actions"
                ),
                "delta": float(
                    delta
                ),
                **metrics,
            }
        )

        for _, row in action_summary.iterrows():
            action_curve_rows.append(
                {
                    "scenario_type": (
                        "common_delta"
                    ),
                    "target_action": (
                        "all_actions"
                    ),
                    "delta": float(
                        delta
                    ),
                    **row.to_dict(),
                }
            )

    # One action at a time.
    for target_action in ACTION_ORDER:
        for delta in delta_grid:
            delta_map = {
                action: 0.0
                for action in ACTION_ORDER
            }

            delta_map[
                target_action
            ] = float(
                delta
            )

            (
                metrics,
                action_summary,
                bin_summary,
            ) = engine.evaluate(
                delta_by_action=delta_map
            )

            scenario_rows.append(
                {
                    "scenario_type": (
                        "action_specific"
                    ),
                    "target_action": (
                        target_action
                    ),
                    "delta": float(
                        delta
                    ),
                    **metrics,
                }
            )

            for _, row in action_summary.iterrows():
                action_curve_rows.append(
                    {
                        "scenario_type": (
                            "action_specific"
                        ),
                        "target_action": (
                            target_action
                        ),
                        "delta": float(
                            delta
                        ),
                        **row.to_dict(),
                    }
                )

            target_bins = (
                bin_summary[
                    [
                        "risk_bin",
                    ]
                ]
                .copy()
            )

            # Recalculate the target action's contribution by bin.
            _, _, target_bin_summary = engine.evaluate(
                delta_by_action=delta_map
            )

            # Contributions are obtained from the exact identity.
            # To retain action-bin values, evaluate them directly.
            for risk_bin in range(
                engine.n_bins
            ):
                stats = engine.bin_stats[
                    risk_bin
                ]

                cell = engine.cells[
                    (
                        risk_bin,
                        target_action,
                    )
                ]

                if int(
                    cell[
                        "n"
                    ]
                ) == 0:
                    contribution = 0.0
                else:
                    mu_action = float(
                        np.mean(
                            expit(
                                cell[
                                    "logit_prediction"
                                ]
                                + float(
                                    delta
                                )
                            )
                        )
                    )

                    contribution = float(
                        cell[
                            "pi"
                        ]
                        * (
                            stats[
                                "mu_observed"
                            ]
                            - mu_action
                        )
                    )

                action_bin_curve_rows.append(
                    {
                        "target_action": (
                            target_action
                        ),
                        "risk_bin": (
                            risk_bin
                        ),
                        "delta": float(
                            delta
                        ),
                        "contribution": (
                            contribution
                        ),
                    }
                )

    scenarios = pd.DataFrame(
        scenario_rows
    )

    action_curves = pd.DataFrame(
        action_curve_rows
    )

    action_bin_curves = pd.DataFrame(
        action_bin_curve_rows
    )

    action_tipping_rows: list[
        dict[str, object]
    ] = []

    for action in ACTION_ORDER:
        scenario_curve = (
            scenarios.loc[
                scenarios[
                    "scenario_type"
                ].eq(
                    "action_specific"
                )
                & scenarios[
                    "target_action"
                ].eq(
                    action
                )
            ]
            .sort_values(
                "delta"
            )
            .reset_index(
                drop=True
            )
        )

        sab_curve = (
            action_curves.loc[
                action_curves[
                    "scenario_type"
                ].eq(
                    "action_specific"
                )
                & action_curves[
                    "target_action"
                ].eq(
                    action
                )
                & action_curves[
                    "action"
                ].eq(
                    action
                )
            ]
            .sort_values(
                "delta"
            )
            .reset_index(
                drop=True
            )
        )

        sab_zero = nearest_zero_crossing(
            sab_curve[
                "delta"
            ].to_numpy(
                dtype=float
            ),
            sab_curve[
                "sab"
            ].to_numpy(
                dtype=float
            ),
        )

        global_zero = nearest_zero_crossing(
            scenario_curve[
                "delta"
            ].to_numpy(
                dtype=float
            ),
            scenario_curve[
                "restricted_minus_reference_signed_gap"
            ].to_numpy(
                dtype=float
            ),
        )

        practical_delta = nearest_delta_satisfying(
            scenario_curve,
            lambda data: (
                data[
                    "net_curve_distortion"
                ]
                <= practical_threshold
            ),
        )

        acceptable_ici_delta = nearest_delta_satisfying(
            scenario_curve,
            lambda data: (
                data[
                    "reference_ici"
                ]
                <= ici_threshold
            ),
        )

        baseline_sab = float(
            sab_curve.loc[
                np.isclose(
                    sab_curve[
                        "delta"
                    ],
                    0.0,
                ),
                "sab",
            ].iloc[
                0
            ]
        )

        action_tipping_rows.append(
            {
                "action": action,
                "baseline_sab_at_delta_zero": (
                    baseline_sab
                ),
                "minimum_sab_in_range": float(
                    sab_curve[
                        "sab"
                    ].min()
                ),
                "maximum_sab_in_range": float(
                    sab_curve[
                        "sab"
                    ].max()
                ),
                "delta_where_action_sab_changes_sign": (
                    sab_zero
                ),
                "action_sab_sign_change_within_range": (
                    sab_zero is not None
                ),
                "delta_where_global_signed_distortion_changes_sign": (
                    global_zero
                ),
                "global_sign_change_within_range": (
                    global_zero is not None
                ),
                "nearest_delta_where_net_distortion_is_practically_small": (
                    practical_delta
                ),
                "practical_smallness_reached_within_range": (
                    practical_delta is not None
                ),
                "nearest_delta_where_reference_ici_is_acceptable": (
                    acceptable_ici_delta
                ),
                "reference_ici_acceptability_reached_within_range": (
                    acceptable_ici_delta
                    is not None
                ),
            }
        )

    action_tipping = pd.DataFrame(
        action_tipping_rows
    )

    action_bin_tipping_rows: list[
        dict[str, object]
    ] = []

    for (
        action,
        risk_bin,
    ), group in action_bin_curves.groupby(
        [
            "target_action",
            "risk_bin",
        ],
        sort=True,
    ):
        group = group.sort_values(
            "delta"
        )

        crossing = nearest_zero_crossing(
            group[
                "delta"
            ].to_numpy(
                dtype=float
            ),
            group[
                "contribution"
            ].to_numpy(
                dtype=float
            ),
        )

        action_bin_tipping_rows.append(
            {
                "action": action,
                "risk_bin": int(
                    risk_bin
                ),
                "delta_where_bin_contribution_changes_sign": (
                    crossing
                ),
                "sign_change_within_range": (
                    crossing is not None
                ),
                "minimum_contribution_in_range": float(
                    group[
                        "contribution"
                    ].min()
                ),
                "maximum_contribution_in_range": float(
                    group[
                        "contribution"
                    ].max()
                ),
            }
        )

    common_curve = (
        scenarios.loc[
            scenarios[
                "scenario_type"
            ].eq(
                "common_delta"
            )
        ]
        .sort_values(
            "delta"
        )
        .reset_index(
            drop=True
        )
    )

    common_tipping = {
        "delta_where_signed_distortion_changes_sign": (
            nearest_zero_crossing(
                common_curve[
                    "delta"
                ].to_numpy(
                    dtype=float
                ),
                common_curve[
                    "restricted_minus_reference_signed_gap"
                ].to_numpy(
                    dtype=float
                ),
            )
        ),
        "nearest_delta_where_net_distortion_is_practically_small": (
            nearest_delta_satisfying(
                common_curve,
                lambda data: (
                    data[
                        "net_curve_distortion"
                    ]
                    <= practical_threshold
                ),
            )
        ),
        "nearest_delta_where_reference_ici_is_acceptable": (
            nearest_delta_satisfying(
                common_curve,
                lambda data: (
                    data[
                        "reference_ici"
                    ]
                    <= ici_threshold
                ),
            )
        ),
        "practical_distortion_threshold": (
            practical_threshold
        ),
        "reference_ici_threshold": (
            ici_threshold
        ),
    }

    return (
        scenarios,
        action_curves,
        action_tipping,
        pd.DataFrame(
            action_bin_tipping_rows
        ),
        common_tipping,
    )


def main() -> None:
    args = parse_arguments()

    input_path = Path(
        args.input
    )

    phase5c_dir = Path(
        args.phase5c_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_test_predictions(
        input_path
    )

    delta_grid = build_delta_grid(
        args.delta_min,
        args.delta_max,
        args.delta_step,
    )

    engine = SensitivityEngine(
        frame,
        n_bins=args.n_bins,
    )

    (
        oracle_metrics,
        oracle_actions,
        oracle_bins,
    ) = engine.evaluate(
        use_actual_hidden_outcomes=True
    )

    hidden_summary = (
        action_hidden_summary(
            frame
        )
        .set_index(
            "action"
        )
        .reindex(
            ACTION_ORDER
        )
        .reset_index()
    )

    action_delta_rows: list[
        dict[str, object]
    ] = []

    action_delta_map: dict[
        str,
        float
    ] = {}

    for _, row in hidden_summary.iterrows():
        action = str(
            row[
                "action"
            ]
        )

        logits = collect_action_logit_predictions(
            frame,
            action,
        )

        oracle_delta = solve_delta_for_target_mean(
            logits,
            float(
                row[
                    "actual_hidden_event_rate"
                ]
            ),
        )

        action_delta_map[
            action
        ] = oracle_delta

        action_delta_rows.append(
            {
                **row.to_dict(),
                "oracle_matching_delta": (
                    oracle_delta
                ),
                "within_planned_delta_range": bool(
                    args.delta_min
                    <= oracle_delta
                    <= args.delta_max
                ),
                "delta_direction": (
                    "positive"
                    if oracle_delta > 0
                    else (
                        "negative"
                        if oracle_delta < 0
                        else "zero"
                    )
                ),
            }
        )

    hidden = frame.loc[
        frame[
            "observed"
        ].eq(
            0
        )
    ]

    common_delta = solve_delta_for_target_mean(
        logit(
            hidden[
                "prediction"
            ].to_numpy(
                dtype=float
            )
        ),
        float(
            hidden[
                "outcome"
            ].mean()
        ),
    )

    common_delta_map = {
        action: common_delta
        for action in ACTION_ORDER
    }

    (
        common_metrics,
        common_actions,
        common_bins,
    ) = engine.evaluate(
        delta_by_action=(
            common_delta_map
        )
    )

    (
        action_specific_metrics,
        action_specific_actions,
        action_specific_bins,
    ) = engine.evaluate(
        delta_by_action=(
            action_delta_map
        )
    )

    action_delta_table = pd.DataFrame(
        action_delta_rows
    )

    hidden_total = float(
        action_delta_table[
            "n_hidden"
        ].sum()
    )

    common_expected_rates = []

    action_specific_expected_rates = []

    for action in ACTION_ORDER:
        logits = collect_action_logit_predictions(
            frame,
            action,
        )

        common_expected_rates.append(
            float(
                np.mean(
                    expit(
                        logits
                        + common_delta
                    )
                )
            )
        )

        action_specific_expected_rates.append(
            float(
                np.mean(
                    expit(
                        logits
                        + action_delta_map[
                            action
                        ]
                    )
                )
            )
        )

    action_delta_table[
        "common_delta_expected_rate"
    ] = common_expected_rates

    action_delta_table[
        "action_specific_expected_rate"
    ] = action_specific_expected_rates

    action_delta_table[
        "common_absolute_rate_error"
    ] = (
        action_delta_table[
            "common_delta_expected_rate"
        ]
        - action_delta_table[
            "actual_hidden_event_rate"
        ]
    ).abs()

    action_delta_table[
        "action_specific_absolute_rate_error"
    ] = (
        action_delta_table[
            "action_specific_expected_rate"
        ]
        - action_delta_table[
            "actual_hidden_event_rate"
        ]
    ).abs()

    action_delta_table[
        "hidden_weight"
    ] = (
        action_delta_table[
            "n_hidden"
        ]
        / hidden_total
    )

    common_weighted_rate_mae = float(
        (
            action_delta_table[
                "hidden_weight"
            ]
            * action_delta_table[
                "common_absolute_rate_error"
            ]
        ).sum()
    )

    action_specific_weighted_rate_mae = float(
        (
            action_delta_table[
                "hidden_weight"
            ]
            * action_delta_table[
                "action_specific_absolute_rate_error"
            ]
        ).sum()
    )

    reconstruction = pd.DataFrame(
        [
            scenario_comparison(
                oracle_metrics=(
                    oracle_metrics
                ),
                oracle_bins=(
                    oracle_bins
                ),
                scenario_name=(
                    "common_oracle_matching_delta"
                ),
                metrics=common_metrics,
                bins=common_bins,
                action_rate_error=(
                    common_weighted_rate_mae
                ),
            ),
            scenario_comparison(
                oracle_metrics=(
                    oracle_metrics
                ),
                oracle_bins=(
                    oracle_bins
                ),
                scenario_name=(
                    "action_specific_oracle_matching_deltas"
                ),
                metrics=(
                    action_specific_metrics
                ),
                bins=(
                    action_specific_bins
                ),
                action_rate_error=(
                    action_specific_weighted_rate_mae
                ),
            ),
        ]
    )

    (
        sensitivity_scenarios,
        action_sab_curves,
        action_tipping,
        action_bin_tipping,
        common_tipping,
    ) = run_sensitivity_sweeps(
        engine=engine,
        delta_grid=delta_grid,
        practical_threshold=(
            args.practical_distortion_threshold
        ),
        ici_threshold=(
            args.reference_ici_threshold
        ),
    )

    frozen_scalars = pd.read_csv(
        phase5c_dir
        / "scalar_summary.csv"
    ).set_index(
        "metric"
    )[
        "value"
    ]

    frozen_actions = pd.read_csv(
        phase5c_dir
        / "action_summary.csv"
    ).set_index(
        "action"
    )

    oracle_action_index = (
        oracle_actions.set_index(
            "action"
        )
    )

    maximum_action_sab_difference = float(
        (
            oracle_action_index.loc[
                ACTION_ORDER,
                "sab",
            ]
            - frozen_actions.loc[
                ACTION_ORDER,
                "sab",
            ]
        )
        .abs()
        .max()
    )

    oracle_scalar_difference = abs(
        oracle_metrics[
            "restricted_minus_reference_signed_gap"
        ]
        - float(
            frozen_scalars.loc[
                "standardized_signed_gap"
            ]
        )
    )

    maximum_sweep_residual = float(
        sensitivity_scenarios[
            "maximum_decomposition_residual"
        ].max()
    )

    expected_scenario_count = int(
        len(
            delta_grid
        )
        * (
            1
            + len(
                ACTION_ORDER
            )
        )
    )

    mixed_oracle_directions = bool(
        (
            action_delta_table[
                "oracle_matching_delta"
            ]
            > 0
        ).any()
        and (
            action_delta_table[
                "oracle_matching_delta"
            ]
            < 0
        ).any()
    )

    common_curve_mae = float(
        reconstruction.loc[
            reconstruction[
                "scenario"
            ].eq(
                "common_oracle_matching_delta"
            ),
            "bin_reference_curve_mae",
        ].iloc[
            0
        ]
    )

    action_curve_mae = float(
        reconstruction.loc[
            reconstruction[
                "scenario"
            ].eq(
                "action_specific_oracle_matching_deltas"
            ),
            "bin_reference_curve_mae",
        ].iloc[
            0
        ]
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "oracle scalar reproduces frozen Phase 5C"
                ),
                "value": (
                    oracle_scalar_difference
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    oracle_scalar_difference
                    < 1e-10
                ),
            },
            {
                "check": (
                    "oracle action SABs reproduce frozen Phase 5C"
                ),
                "value": (
                    maximum_action_sab_difference
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    maximum_action_sab_difference
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all sensitivity scenarios completed"
                ),
                "value": len(
                    sensitivity_scenarios
                ),
                "threshold": (
                    expected_scenario_count
                ),
                "comparison": "==",
                "passed": (
                    len(
                        sensitivity_scenarios
                    )
                    == expected_scenario_count
                ),
            },
            {
                "check": (
                    "all sensitivity decompositions are exact"
                ),
                "value": (
                    maximum_sweep_residual
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    maximum_sweep_residual
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all eight oracle-matching action deltas solved"
                ),
                "value": int(
                    action_delta_table[
                        "oracle_matching_delta"
                    ].notna().sum()
                ),
                "threshold": 8,
                "comparison": "==",
                "passed": (
                    action_delta_table[
                        "oracle_matching_delta"
                    ].notna().sum()
                    == 8
                ),
            },
            {
                "check": (
                    "oracle-matching action deltas have "
                    "both directions"
                ),
                "value": (
                    "mixed"
                    if mixed_oracle_directions
                    else "single_direction"
                ),
                "threshold": "mixed",
                "comparison": "==",
                "passed": (
                    mixed_oracle_directions
                ),
            },
            {
                "check": (
                    "action-specific deltas improve "
                    "action-rate reconstruction"
                ),
                "value": (
                    action_specific_weighted_rate_mae
                ),
                "threshold": (
                    common_weighted_rate_mae
                ),
                "comparison": "<",
                "passed": (
                    action_specific_weighted_rate_mae
                    < common_weighted_rate_mae
                ),
            },
            {
                "check": (
                    "action-specific deltas improve "
                    "reference-curve reconstruction"
                ),
                "value": (
                    action_curve_mae
                ),
                "threshold": (
                    common_curve_mae
                ),
                "comparison": "<",
                "passed": (
                    action_curve_mae
                    < common_curve_mae
                ),
            },
        ]
    )

    status = (
        "phase5d_action_specific_sensitivity_informative"
        if checks[
            "passed"
        ].all()
        else (
            "phase5d_action_specific_sensitivity_"
            "completed_requires_interpretation"
        )
    )

    sensitivity_scenarios.to_csv(
        output_dir
        / "sensitivity_scenario_curves.csv.gz",
        index=False,
        compression="gzip",
    )

    action_sab_curves.to_csv(
        output_dir
        / "action_sab_curves.csv.gz",
        index=False,
        compression="gzip",
    )

    action_delta_table.to_csv(
        output_dir
        / "oracle_matching_action_deltas.csv",
        index=False,
    )

    reconstruction.to_csv(
        output_dir
        / "global_vs_action_specific_reconstruction.csv",
        index=False,
    )

    action_tipping.to_csv(
        output_dir
        / "action_tipping_points.csv",
        index=False,
    )

    action_bin_tipping.to_csv(
        output_dir
        / "action_bin_tipping_points.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "pattern_mixture_model": (
            "logit E[Y|P,R=0,A=a] = "
            "logit(P) + delta_a"
        ),
        "delta_grid": {
            "minimum": (
                args.delta_min
            ),
            "maximum": (
                args.delta_max
            ),
            "step": (
                args.delta_step
            ),
            "n_values": int(
                len(
                    delta_grid
                )
            ),
        },
        "oracle_matching_common_delta": (
            common_delta
        ),
        "oracle_matching_action_deltas": {
            row[
                "action"
            ]: row[
                "oracle_matching_delta"
            ]
            for _, row in (
                action_delta_table.iterrows()
            )
        },
        "common_weighted_action_rate_mae": (
            common_weighted_rate_mae
        ),
        "action_specific_weighted_action_rate_mae": (
            action_specific_weighted_rate_mae
        ),
        "common_reference_curve_mae": (
            common_curve_mae
        ),
        "action_specific_reference_curve_mae": (
            action_curve_mae
        ),
        "common_tipping_points": (
            common_tipping
        ),
        "practical_thresholds": {
            "net_curve_distortion": (
                args.practical_distortion_threshold
            ),
            "reference_ici": (
                args.reference_ici_threshold
            ),
        },
        "interpretation_note": (
            "Actual linked outcomes are used only to "
            "validate the sensitivity model and locate "
            "retrospective oracle-matching deltas."
        ),
        "status": status,
    }

    (
        output_dir
        / "sensitivity_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5d_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5D action-specific sensitivity"
    )

    print(
        "\nOracle-matching action deltas"
    )

    print(
        action_delta_table.to_string(
            index=False
        )
    )

    print(
        "\nGlobal versus action-specific reconstruction"
    )

    print(
        reconstruction.to_string(
            index=False
        )
    )

    print(
        "\nAction tipping points"
    )

    print(
        action_tipping.to_string(
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
        f"\nStatus: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
