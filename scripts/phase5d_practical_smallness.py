"""Phase 5D-C: minimum attainable distortion under common delta.

Quantifies whether any common residual-dependence shift can make
the restricted-versus-reference curve practically similar.

This supplements the tipping-point analysis because a common
signed distortion can cross zero through cancellation while the
absolute curve distortion remains large.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta

from phase5d_action_specific_sensitivity import (
    ACTION_ORDER,
    SensitivityEngine,
    load_test_predictions,
)

from phase5d_bootstrap_tipping_points import (
    common_net_distortion,
    common_signed_distortion,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the minimum common-delta net "
            "distortion with bootstrap uncertainty."
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
            "phase5d_practical_smallness"
        ),
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--delta-lower",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--delta-upper",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--practical-threshold",
        type=float,
        default=0.02,
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


def evaluate_common_delta(
    engine: SensitivityEngine,
    delta: float,
) -> dict[str, float]:
    delta_map = {
        action: float(
            delta
        )
        for action in ACTION_ORDER
    }

    (
        metrics,
        _,
        _,
    ) = engine.evaluate(
        delta_by_action=delta_map
    )

    return {
        "delta": float(
            delta
        ),
        "net_curve_distortion": float(
            metrics[
                "net_curve_distortion"
            ]
        ),
        "signed_distortion": float(
            metrics[
                "restricted_minus_reference_signed_gap"
            ]
        ),
        "reference_ici": float(
            metrics[
                "reference_ici"
            ]
        ),
        "pointwise_cancellation_ratio": float(
            metrics[
                "pointwise_cancellation_ratio"
            ]
        ),
    }


def optimize_minimum_distortion(
    engine: SensitivityEngine,
    *,
    lower: float,
    upper: float,
) -> dict[str, float]:
    if lower >= upper:
        raise ValueError(
            "delta-lower must be below delta-upper"
        )

    result = minimize_scalar(
        lambda delta: (
            common_net_distortion(
                engine,
                float(
                    delta
                ),
            )
        ),
        bounds=(
            lower,
            upper,
        ),
        method="bounded",
        options={
            "xatol": 1e-8,
            "maxiter": 150,
        },
    )

    candidate_deltas = [
        float(
            lower
        ),
        float(
            upper
        ),
        float(
            result.x
        ),
    ]

    # Guard against an optimizer settling in a local basin.
    grid = np.linspace(
        lower,
        upper,
        25,
    )

    grid_values = np.asarray(
        [
            common_net_distortion(
                engine,
                float(
                    delta
                ),
            )
            for delta in grid
        ]
    )

    best_grid_index = int(
        np.argmin(
            grid_values
        )
    )

    candidate_deltas.append(
        float(
            grid[
                best_grid_index
            ]
        )
    )

    if (
        best_grid_index > 0
        and best_grid_index
        < len(
            grid
        )
        - 1
    ):
        local_result = minimize_scalar(
            lambda delta: (
                common_net_distortion(
                    engine,
                    float(
                        delta
                    ),
                )
            ),
            bounds=(
                float(
                    grid[
                        best_grid_index
                        - 1
                    ]
                ),
                float(
                    grid[
                        best_grid_index
                        + 1
                    ]
                ),
            ),
            method="bounded",
            options={
                "xatol": 1e-8,
                "maxiter": 100,
            },
        )

        candidate_deltas.append(
            float(
                local_result.x
            )
        )

    evaluations = [
        evaluate_common_delta(
            engine,
            delta,
        )
        for delta in candidate_deltas
    ]

    return min(
        evaluations,
        key=lambda row: (
            row[
                "net_curve_distortion"
            ]
        ),
    )


def run_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bins: int,
    lower: float,
    upper: float,
    threshold: float,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    rows: list[
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
            engine = SensitivityEngine(
                sample,
                n_bins=n_bins,
            )

            result = optimize_minimum_distortion(
                engine,
                lower=lower,
                upper=upper,
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            invalid += 1
            continue

        rows.append(
            {
                "bootstrap_id": (
                    bootstrap_id
                ),
                **result,
                "practical_threshold": (
                    threshold
                ),
                "practical_smallness_reached": (
                    result[
                        "net_curve_distortion"
                    ]
                    <= threshold
                ),
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
            rows
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


def binomial_upper_bound(
    successes: int,
    trials: int,
) -> float:
    if trials <= 0:
        return float(
            "nan"
        )

    if successes >= trials:
        return 1.0

    return float(
        beta.ppf(
            0.975,
            successes + 1,
            trials - successes,
        )
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

    engine = SensitivityEngine(
        frame,
        n_bins=args.n_bins,
    )

    point = optimize_minimum_distortion(
        engine,
        lower=args.delta_lower,
        upper=args.delta_upper,
    )

    point[
        "practical_threshold"
    ] = args.practical_threshold

    point[
        "practical_smallness_reached"
    ] = (
        point[
            "net_curve_distortion"
        ]
        <= args.practical_threshold
    )

    print(
        "Phase 5D practical-smallness analysis"
    )

    print(
        "\nPoint estimate"
    )

    print(
        pd.DataFrame(
            [
                point
            ]
        ).to_string(
            index=False
        )
    )

    print(
        "\nRunning bootstrap..."
    )

    (
        bootstrap,
        invalid,
    ) = run_bootstrap(
        frame,
        n_bins=args.n_bins,
        lower=args.delta_lower,
        upper=args.delta_upper,
        threshold=(
            args.practical_threshold
        ),
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
    )

    metric_rows = []

    for metric in [
        "delta",
        "net_curve_distortion",
        "signed_distortion",
        "reference_ici",
        "pointwise_cancellation_ratio",
    ]:
        metric_rows.append(
            {
                "metric": metric,
                "estimate": float(
                    point[
                        metric
                    ]
                ),
                **interval_summary(
                    bootstrap[
                        metric
                    ]
                ),
            }
        )

    intervals = pd.DataFrame(
        metric_rows
    )

    valid_bootstraps = len(
        bootstrap
    )

    threshold_successes = int(
        bootstrap[
            "practical_smallness_reached"
        ].astype(bool).sum()
    )

    threshold_probability = float(
        threshold_successes
        / valid_bootstraps
    )

    probability_upper = (
        binomial_upper_bound(
            threshold_successes,
            valid_bootstraps,
        )
    )

    minimum_distortion_lower = float(
        intervals.loc[
            intervals[
                "metric"
            ].eq(
                "net_curve_distortion"
            ),
            "q025",
        ].iloc[
            0
        ]
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "point minimum distortion is finite"
                ),
                "value": (
                    point[
                        "net_curve_distortion"
                    ]
                ),
                "threshold": "finite",
                "comparison": "==",
                "passed": bool(
                    np.isfinite(
                        point[
                            "net_curve_distortion"
                        ]
                    )
                ),
            },
            {
                "check": (
                    "all requested bootstrap minima completed"
                ),
                "value": (
                    valid_bootstraps
                ),
                "threshold": (
                    args.bootstrap_replicates
                ),
                "comparison": "==",
                "passed": (
                    valid_bootstraps
                    == args.bootstrap_replicates
                    and invalid == 0
                ),
            },
            {
                "check": (
                    "point minimum remains above "
                    "practical threshold"
                ),
                "value": (
                    point[
                        "net_curve_distortion"
                    ]
                ),
                "threshold": (
                    args.practical_threshold
                ),
                "comparison": ">",
                "passed": (
                    point[
                        "net_curve_distortion"
                    ]
                    > args.practical_threshold
                ),
            },
            {
                "check": (
                    "bootstrap lower bound remains above "
                    "practical threshold"
                ),
                "value": (
                    minimum_distortion_lower
                ),
                "threshold": (
                    args.practical_threshold
                ),
                "comparison": ">",
                "passed": (
                    minimum_distortion_lower
                    > args.practical_threshold
                ),
            },
            {
                "check": (
                    "upper confidence bound on practical-"
                    "smallness probability below ten percent"
                ),
                "value": (
                    probability_upper
                ),
                "threshold": 0.10,
                "comparison": "<",
                "passed": (
                    probability_upper
                    < 0.10
                ),
            },
        ]
    )

    preliminary_status = (
        "phase5d_practical_smallness_not_reached"
        if checks[
            "passed"
        ].all()
        else (
            "phase5d_practical_smallness_"
            "requires_review"
        )
    )

    status = (
        "phase5d_practical_smallness_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    pd.DataFrame(
        [
            point
        ]
    ).to_csv(
        output_dir
        / "point_minimum_distortion.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_minimum_distortion.csv.gz",
        index=False,
        compression="gzip",
    )

    intervals.to_csv(
        output_dir
        / "bootstrap_intervals.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5D-C",
        "delta_search_range": [
            args.delta_lower,
            args.delta_upper,
        ],
        "practical_threshold": (
            args.practical_threshold
        ),
        "point_minimum": point,
        "bootstrap_threshold_successes": (
            threshold_successes
        ),
        "valid_bootstraps": (
            valid_bootstraps
        ),
        "estimated_practical_smallness_probability": (
            threshold_probability
        ),
        "practical_smallness_probability_q975": (
            probability_upper
        ),
        "minimum_distortion_bootstrap_q025": (
            minimum_distortion_lower
        ),
        "interpretation": (
            "A zero aggregate signed distortion can "
            "arise through cross-bin cancellation even "
            "when the absolute restricted-versus-reference "
            "curve difference remains substantively large."
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
        / "practical_smallness_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5d_practical_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nBootstrap intervals"
    )

    print(
        intervals.to_string(
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
        "\nPractical-smallness frequency: "
        f"{threshold_successes}/"
        f"{valid_bootstraps}"
    )

    print(
        f"Upper 95% probability bound: "
        f"{probability_upper}"
    )

    print(
        f"\nPreliminary status: "
        f"{preliminary_status}"
    )

    print(
        f"Saved status: {status}"
    )


if __name__ == "__main__":
    main()
