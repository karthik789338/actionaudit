"""Phase 5D-B: exact tipping points and bootstrap uncertainty.

Computes:
  * retrospective oracle-matching action-specific deltas;
  * exact integrated SAB sign-reversal deltas;
  * exact bin-level contribution sign-reversal deltas;
  * common-delta global distortion tipping points;
  * patient-level nonparametric bootstrap intervals.

Actual linked outcomes are used only to anchor retrospective
oracle deltas and evaluate sampling uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from phase5d_action_specific_sensitivity import (
    ACTION_ORDER,
    SensitivityEngine,
    load_test_predictions,
    solve_delta_for_target_mean,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact Phase 5D tipping points "
            "with bootstrap intervals."
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
        "--output-dir",
        default=(
            "artifacts/"
            "phase5d_bootstrap_tipping_points"
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

    parser.add_argument(
        "--root-lower",
        type=float,
        default=-12.0,
    )

    parser.add_argument(
        "--root-upper",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--practical-distortion-threshold",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


def solve_decreasing_root(
    function: Callable[
        [
            float,
        ],
        float,
    ],
    *,
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
    iterations: int = 250,
) -> float | None:
    """Solve a monotonically decreasing function's zero."""

    lower_value = float(
        function(
            lower
        )
    )

    upper_value = float(
        function(
            upper
        )
    )

    if np.isclose(
        lower_value,
        0.0,
        atol=tolerance,
    ):
        return float(
            lower
        )

    if np.isclose(
        upper_value,
        0.0,
        atol=tolerance,
    ):
        return float(
            upper
        )

    if (
        lower_value < 0
        or upper_value > 0
    ):
        return None

    left = float(
        lower
    )

    right = float(
        upper
    )

    for _ in range(
        iterations
    ):
        midpoint = (
            left
            + right
        ) / 2.0

        value = float(
            function(
                midpoint
            )
        )

        if abs(
            value
        ) <= tolerance:
            return float(
                midpoint
            )

        if value > 0:
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


def action_sab(
    engine: SensitivityEngine,
    *,
    action: str,
    delta: float,
) -> float:
    total = 0.0

    for risk_bin in range(
        engine.n_bins
    ):
        stats = engine.bin_stats[
            risk_bin
        ]

        cell = engine.cells[
            (
                risk_bin,
                action,
            )
        ]

        if int(
            cell[
                "n"
            ]
        ) == 0:
            continue

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

        total += float(
            stats[
                "reference_weight"
            ]
        ) * contribution

    return float(
        total
    )


def common_signed_distortion(
    engine: SensitivityEngine,
    delta: float,
) -> float:
    total = 0.0

    for risk_bin in range(
        engine.n_bins
    ):
        stats = engine.bin_stats[
            risk_bin
        ]

        bin_difference = 0.0

        for action in ACTION_ORDER:
            cell = engine.cells[
                (
                    risk_bin,
                    action,
                )
            ]

            if int(
                cell[
                    "n"
                ]
            ) == 0:
                continue

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

            bin_difference += float(
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

        total += float(
            stats[
                "reference_weight"
            ]
        ) * bin_difference

    return float(
        total
    )


def common_net_distortion(
    engine: SensitivityEngine,
    delta: float,
) -> float:
    total = 0.0

    for risk_bin in range(
        engine.n_bins
    ):
        stats = engine.bin_stats[
            risk_bin
        ]

        bin_difference = 0.0

        for action in ACTION_ORDER:
            cell = engine.cells[
                (
                    risk_bin,
                    action,
                )
            ]

            if int(
                cell[
                    "n"
                ]
            ) == 0:
                continue

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

            bin_difference += float(
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

        total += float(
            stats[
                "reference_weight"
            ]
        ) * abs(
            bin_difference
        )

    return float(
        total
    )


def first_threshold_crossing(
    function: Callable[
        [
            float,
        ],
        float,
    ],
    *,
    threshold: float,
    lower: float,
    upper: float,
    grid_size: int = 2401,
) -> float | None:
    """Find the first positive-direction crossing below threshold."""

    grid = np.linspace(
        lower,
        upper,
        grid_size,
    )

    values = np.asarray(
        [
            function(
                float(
                    value
                )
            )
            - threshold
            for value in grid
        ],
        dtype=float,
    )

    if values[
        0
    ] <= 0:
        return float(
            grid[
                0
            ]
        )

    crossing_indices = np.flatnonzero(
        values <= 0
    )

    if len(
        crossing_indices
    ) == 0:
        return None

    right_index = int(
        crossing_indices[
            0
        ]
    )

    if right_index == 0:
        return float(
            grid[
                0
            ]
        )

    left = float(
        grid[
            right_index
            - 1
        ]
    )

    right = float(
        grid[
            right_index
        ]
    )

    for _ in range(
        100
    ):
        midpoint = (
            left
            + right
        ) / 2.0

        value = (
            function(
                midpoint
            )
            - threshold
        )

        if abs(
            value
        ) <= 1e-10:
            return float(
                midpoint
            )

        if value > 0:
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


def hidden_action_rows(
    frame: pd.DataFrame,
    action: str,
) -> pd.DataFrame:
    return frame.loc[
        frame[
            "observed"
        ].eq(
            0
        )
        & frame[
            "analysis_action"
        ].eq(
            action
        )
    ]


def oracle_matching_delta(
    frame: pd.DataFrame,
    action: str,
) -> float:
    group = hidden_action_rows(
        frame,
        action,
    )

    if group.empty:
        return float(
            "nan"
        )

    return solve_delta_for_target_mean(
        logit(
            group[
                "prediction"
            ].to_numpy(
                dtype=float
            )
        ),
        float(
            group[
                "outcome"
            ].mean()
        ),
    )


def common_oracle_matching_delta(
    frame: pd.DataFrame,
) -> float:
    hidden = frame.loc[
        frame[
            "observed"
        ].eq(
            0
        )
    ]

    return solve_delta_for_target_mean(
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


def exact_action_table(
    frame: pd.DataFrame,
    *,
    engine: SensitivityEngine,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for action in ACTION_ORDER:
        oracle_delta = oracle_matching_delta(
            frame,
            action,
        )

        tipping_delta = solve_decreasing_root(
            lambda delta: action_sab(
                engine,
                action=action,
                delta=delta,
            ),
            lower=lower,
            upper=upper,
        )

        sab_at_oracle = action_sab(
            engine,
            action=action,
            delta=oracle_delta,
        )

        rows.append(
            {
                "action": action,
                "oracle_matching_delta": (
                    oracle_delta
                ),
                "exact_integrated_sab_tipping_delta": (
                    tipping_delta
                ),
                "sab_at_oracle_matching_delta": (
                    sab_at_oracle
                ),
                "distance_oracle_to_tipping": (
                    tipping_delta
                    - oracle_delta
                    if tipping_delta
                    is not None
                    else np.nan
                ),
                "oracle_is_below_tipping_point": (
                    oracle_delta
                    < tipping_delta
                    if tipping_delta
                    is not None
                    else False
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def exact_bin_table(
    engine: SensitivityEngine,
    *,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for action in ACTION_ORDER:
        for risk_bin in range(
            engine.n_bins
        ):
            stats = engine.bin_stats[
                risk_bin
            ]

            cell = engine.cells[
                (
                    risk_bin,
                    action,
                )
            ]

            if int(
                cell[
                    "n"
                ]
            ) == 0:
                tipping_delta = np.nan
            else:
                tipping_delta = (
                    solve_delta_for_target_mean(
                        cell[
                            "logit_prediction"
                        ],
                        float(
                            stats[
                                "mu_observed"
                            ]
                        ),
                        lower=lower,
                        upper=upper,
                    )
                )

            rows.append(
                {
                    "action": action,
                    "risk_bin": risk_bin,
                    "n_action": int(
                        cell[
                            "n"
                        ]
                    ),
                    "mu_observed": float(
                        stats[
                            "mu_observed"
                        ]
                    ),
                    "exact_bin_tipping_delta": (
                        tipping_delta
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def calculate_point_results(
    frame: pd.DataFrame,
    *,
    n_bins: int,
    lower: float,
    upper: float,
    practical_threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | None],
]:
    engine = SensitivityEngine(
        frame,
        n_bins=n_bins,
    )

    actions = exact_action_table(
        frame,
        engine=engine,
        lower=lower,
        upper=upper,
    )

    bins = exact_bin_table(
        engine,
        lower=lower,
        upper=upper,
    )

    common_oracle = (
        common_oracle_matching_delta(
            frame
        )
    )

    common_sign_tipping = (
        solve_decreasing_root(
            lambda delta: (
                common_signed_distortion(
                    engine,
                    delta,
                )
            ),
            lower=lower,
            upper=upper,
        )
    )

    practical_tipping = (
        first_threshold_crossing(
            lambda delta: (
                common_net_distortion(
                    engine,
                    delta,
                )
            ),
            threshold=(
                practical_threshold
            ),
            lower=0.0,
            upper=upper,
        )
    )

    global_results = {
        "common_oracle_matching_delta": (
            common_oracle
        ),
        "common_signed_distortion_tipping_delta": (
            common_sign_tipping
        ),
        "common_practical_distortion_tipping_delta": (
            practical_tipping
        ),
        "net_distortion_at_common_oracle_delta": (
            common_net_distortion(
                engine,
                common_oracle,
            )
        ),
        "signed_distortion_at_common_oracle_delta": (
            common_signed_distortion(
                engine,
                common_oracle,
            )
        ),
    }

    return (
        actions,
        bins,
        global_results,
    )


def run_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bins: int,
    replicates: int,
    seed: int,
    lower: float,
    upper: float,
    practical_threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    action_rows: list[
        dict[str, object]
    ] = []

    global_rows: list[
        dict[str, object]
    ] = []

    invalid = 0

    n = len(
        frame
    )

    for bootstrap_id in range(
        replicates
    ):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = (
            frame.iloc[
                indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        sample[
            "subject_id"
        ] = np.arange(
            n,
            dtype=int,
        )

        try:
            (
                action_table,
                _,
                global_results,
            ) = calculate_point_results(
                sample,
                n_bins=n_bins,
                lower=lower,
                upper=upper,
                practical_threshold=(
                    practical_threshold
                ),
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            invalid += 1
            continue

        for _, row in (
            action_table.iterrows()
        ):
            action_rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    **row.to_dict(),
                }
            )

        global_rows.append(
            {
                "bootstrap_id": (
                    bootstrap_id
                ),
                **global_results,
            }
        )

        if (
            bootstrap_id + 1
        ) % 100 == 0:
            print(
                "  completed "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

    return (
        pd.DataFrame(
            action_rows
        ),
        pd.DataFrame(
            global_rows
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

    if clean.empty:
        return {
            "q025": np.nan,
            "median": np.nan,
            "q975": np.nan,
            "n_valid": 0,
        }

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


def action_interval_table(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "oracle_matching_delta",
        "exact_integrated_sab_tipping_delta",
        "sab_at_oracle_matching_delta",
        "distance_oracle_to_tipping",
    ]

    rows: list[
        dict[str, object]
    ] = []

    for action in ACTION_ORDER:
        point_row = point.loc[
            point[
                "action"
            ].eq(
                action
            )
        ].iloc[
            0
        ]

        bootstrap_action = (
            bootstrap.loc[
                bootstrap[
                    "action"
                ].eq(
                    action
                )
            ]
        )

        for metric in metrics:
            rows.append(
                {
                    "action": action,
                    "metric": metric,
                    "estimate": float(
                        point_row[
                            metric
                        ]
                    ),
                    **interval_summary(
                        bootstrap_action[
                            metric
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def global_interval_table(
    point: dict[str, float | None],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for metric, estimate in (
        point.items()
    ):
        rows.append(
            {
                "metric": metric,
                "estimate": (
                    float(
                        estimate
                    )
                    if estimate is not None
                    else np.nan
                ),
                **interval_summary(
                    bootstrap[
                        metric
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_arguments()

    input_path = Path(
        args.input
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

    print(
        "Phase 5D exact tipping points"
    )

    print(
        f"  test patients={len(frame):,}"
    )

    print(
        "  bootstrap replicates="
        f"{args.bootstrap_replicates}"
    )

    (
        point_actions,
        point_bins,
        point_global,
    ) = calculate_point_results(
        frame,
        n_bins=args.n_bins,
        lower=args.root_lower,
        upper=args.root_upper,
        practical_threshold=(
            args.practical_distortion_threshold
        ),
    )

    print(
        "\nRunning bootstrap..."
    )

    (
        bootstrap_actions,
        bootstrap_global,
        invalid,
    ) = run_bootstrap(
        frame,
        n_bins=args.n_bins,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
        lower=args.root_lower,
        upper=args.root_upper,
        practical_threshold=(
            args.practical_distortion_threshold
        ),
    )

    action_intervals = (
        action_interval_table(
            point_actions,
            bootstrap_actions,
        )
    )

    global_intervals = (
        global_interval_table(
            point_global,
            bootstrap_global,
        )
    )

    action_tipping_intervals = (
        action_intervals.loc[
            action_intervals[
                "metric"
            ].eq(
                "exact_integrated_sab_tipping_delta"
            )
        ]
    )

    action_oracle_intervals = (
        action_intervals.loc[
            action_intervals[
                "metric"
            ].eq(
                "oracle_matching_delta"
            )
        ]
    )

    distance_intervals = (
        action_intervals.loc[
            action_intervals[
                "metric"
            ].eq(
                "distance_oracle_to_tipping"
            )
        ]
    )

    expected_action_rows = (
        args.bootstrap_replicates
        * len(
            ACTION_ORDER
        )
    )

    all_point_roots_solved = bool(
        point_actions[
            "exact_integrated_sab_tipping_delta"
        ].notna().all()
    )

    all_oracle_below_tipping = bool(
        point_actions[
            "oracle_is_below_tipping_point"
        ].all()
    )

    all_distance_intervals_positive = bool(
        (
            distance_intervals[
                "q025"
            ]
            > 0
        ).all()
    )

    all_oracle_intervals_finite = bool(
        action_oracle_intervals[
            "q025"
        ].notna().all()
        and action_oracle_intervals[
            "q975"
        ].notna().all()
    )

    all_tipping_intervals_finite = bool(
        action_tipping_intervals[
            "q025"
        ].notna().all()
        and action_tipping_intervals[
            "q975"
        ].notna().all()
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "all exact integrated action tipping "
                    "points solved"
                ),
                "value": int(
                    point_actions[
                        "exact_integrated_sab_tipping_delta"
                    ].notna().sum()
                ),
                "threshold": 8,
                "comparison": "==",
                "passed": (
                    all_point_roots_solved
                ),
            },
            {
                "check": (
                    "common signed-distortion tipping "
                    "point solved"
                ),
                "value": (
                    point_global[
                        "common_signed_distortion_tipping_delta"
                    ]
                ),
                "threshold": "finite",
                "comparison": "==",
                "passed": (
                    point_global[
                        "common_signed_distortion_tipping_delta"
                    ]
                    is not None
                ),
            },
            {
                "check": (
                    "all requested bootstrap action "
                    "rows completed"
                ),
                "value": len(
                    bootstrap_actions
                ),
                "threshold": (
                    expected_action_rows
                ),
                "comparison": "==",
                "passed": (
                    len(
                        bootstrap_actions
                    )
                    == expected_action_rows
                    and invalid == 0
                ),
            },
            {
                "check": (
                    "all oracle-delta intervals are finite"
                ),
                "value": int(
                    action_oracle_intervals[
                        "n_valid"
                    ].min()
                ),
                "threshold": (
                    args.bootstrap_replicates
                ),
                "comparison": "==",
                "passed": (
                    all_oracle_intervals_finite
                    and action_oracle_intervals[
                        "n_valid"
                    ].min()
                    == args.bootstrap_replicates
                ),
            },
            {
                "check": (
                    "all action-tipping intervals are finite"
                ),
                "value": int(
                    action_tipping_intervals[
                        "n_valid"
                    ].min()
                ),
                "threshold": (
                    args.bootstrap_replicates
                ),
                "comparison": "==",
                "passed": (
                    all_tipping_intervals_finite
                    and action_tipping_intervals[
                        "n_valid"
                    ].min()
                    == args.bootstrap_replicates
                ),
            },
            {
                "check": (
                    "every oracle delta lies below its "
                    "integrated SAB tipping point"
                ),
                "value": int(
                    point_actions[
                        "oracle_is_below_tipping_point"
                    ].sum()
                ),
                "threshold": 8,
                "comparison": "==",
                "passed": (
                    all_oracle_below_tipping
                ),
            },
            {
                "check": (
                    "bootstrap distance from oracle to "
                    "tipping remains positive"
                ),
                "value": float(
                    distance_intervals[
                        "q025"
                    ].min()
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    all_distance_intervals_positive
                ),
            },
        ]
    )

    preliminary_status = (
        "phase5d_exact_tipping_and_sampling_uncertainty_validated"
        if checks[
            "passed"
        ].all()
        else (
            "phase5d_exact_tipping_bootstrap_"
            "requires_review"
        )
    )

    status = (
        "phase5d_tipping_bootstrap_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    point_actions.to_csv(
        output_dir
        / "exact_action_tipping_points.csv",
        index=False,
    )

    point_bins.to_csv(
        output_dir
        / "exact_bin_tipping_points.csv",
        index=False,
    )

    pd.DataFrame(
        [
            point_global
        ]
    ).to_csv(
        output_dir
        / "exact_global_tipping_points.csv",
        index=False,
    )

    bootstrap_actions.to_csv(
        output_dir
        / "bootstrap_action_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_global.to_csv(
        output_dir
        / "bootstrap_global_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    action_intervals.to_csv(
        output_dir
        / "action_bootstrap_intervals.csv",
        index=False,
    )

    global_intervals.to_csv(
        output_dir
        / "global_bootstrap_intervals.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5D-B",
        "root_search_range": [
            args.root_lower,
            args.root_upper,
        ],
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstraps": (
            invalid
        ),
        "point_global_tipping": (
            point_global
        ),
        "minimum_exact_action_tipping_delta": float(
            point_actions[
                "exact_integrated_sab_tipping_delta"
            ].min()
        ),
        "maximum_exact_action_tipping_delta": float(
            point_actions[
                "exact_integrated_sab_tipping_delta"
            ].max()
        ),
        "minimum_oracle_to_tipping_distance": float(
            point_actions[
                "distance_oracle_to_tipping"
            ].min()
        ),
        "minimum_bootstrap_oracle_to_tipping_q025": float(
            distance_intervals[
                "q025"
            ].min()
        ),
        "interpretation": (
            "Oracle-matching action-specific residual "
            "shifts remain below the exact integrated "
            "action-contribution reversal thresholds."
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
        / "tipping_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5d_tipping_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nExact action tipping points"
    )

    print(
        point_actions.to_string(
            index=False
        )
    )

    print(
        "\nExact global tipping points"
    )

    print(
        pd.DataFrame(
            [
                point_global
            ]
        ).to_string(
            index=False
        )
    )

    print(
        "\nAction bootstrap intervals"
    )

    print(
        action_intervals.to_string(
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
