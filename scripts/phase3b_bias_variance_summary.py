"""Separate Monte Carlo estimator bias from sampling variability."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ESTIMATORS = {
    "complete_case": {
        "mae": "complete_case_mae",
        "signed_gap": "complete_case_signed_gap",
    },
    "kaplan_meier": {
        "mae": "kaplan_meier_mae",
        "signed_gap": "kaplan_meier_signed_gap",
    },
    "full_oracle_or": {
        "mae": "full_oracle_or_mae",
        "signed_gap": "full_oracle_or_signed_gap",
    },
    "full_oracle_ipcw": {
        "mae": "full_oracle_ipcw_mae",
        "signed_gap": "full_oracle_ipcw_signed_gap",
    },
    "working_or": {
        "mae": "working_or_mae",
        "signed_gap": "working_or_signed_gap",
    },
    "working_ipcw": {
        "mae": "working_ipcw_mae",
        "signed_gap": "working_ipcw_signed_gap",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Monte Carlo calibration bias, "
            "absolute error, and variability."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/phase3a_oracle_simulation_bias_variance/"
            "replication_results.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase3b_bias_variance",
    )

    return parser.parse_args()


def summarize_estimator(
    group: pd.DataFrame,
    *,
    scenario: str,
    clip: float,
    estimator: str,
    mae_column: str,
    signed_gap_column: str,
) -> dict[str, object]:
    """Summarize one estimator across Monte Carlo replications."""

    mae = (
        group[mae_column]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    signed_gap = (
        group[signed_gap_column]
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
        "scenario": scenario,
        "clip": float(clip),
        "estimator": estimator,
        "n_replications": int(
            min(
                len(mae),
                len(signed_gap),
            )
        ),
        "mean_mae": float(
            mae.mean()
        ),
        "sd_mae": float(
            mae.std(ddof=1)
        ),
        "median_mae": float(
            mae.median()
        ),
        "mae_q025": float(
            mae.quantile(0.025)
        ),
        "mae_q975": float(
            mae.quantile(0.975)
        ),
        "mean_signed_gap": float(
            signed_gap.mean()
        ),
        "absolute_monte_carlo_bias": float(
            abs(
                signed_gap.mean()
            )
        ),
        "sd_signed_gap": float(
            signed_gap.std(ddof=1)
        ),
        "rmse_signed_gap": float(
            np.sqrt(
                np.mean(
                    np.square(
                        signed_gap
                    )
                )
            )
        ),
        "signed_gap_q025": float(
            signed_gap.quantile(0.025)
        ),
        "signed_gap_q975": float(
            signed_gap.quantile(0.975)
        ),
        "fraction_positive_signed_gap": float(
            (
                signed_gap > 0
            ).mean()
        ),
    }


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

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing replication file: {input_path}"
        )

    replications = pd.read_csv(
        input_path
    )

    required_columns = {
        "scenario",
        "clip",
    }

    for columns in ESTIMATORS.values():
        required_columns.update(
            columns.values()
        )

    missing = sorted(
        required_columns.difference(
            replications.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing signed-gap fields: {missing}"
        )

    rows: list[dict[str, object]] = []

    for (
        scenario,
        clip,
    ), group in replications.groupby(
        [
            "scenario",
            "clip",
        ],
        sort=True,
    ):
        for estimator, columns in ESTIMATORS.items():
            rows.append(
                summarize_estimator(
                    group,
                    scenario=str(
                        scenario
                    ),
                    clip=float(
                        clip
                    ),
                    estimator=estimator,
                    mae_column=columns[
                        "mae"
                    ],
                    signed_gap_column=columns[
                        "signed_gap"
                    ],
                )
            )

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "scenario",
                "clip",
                "mean_mae",
            ]
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        output_dir
        / "bias_variance_summary.csv",
        index=False,
    )

    oracle_check = summary.loc[
        summary[
            "estimator"
        ].isin(
            [
                "full_oracle_or",
                "full_oracle_ipcw",
            ]
        )
    ].copy()

    oracle_check[
        "oracle_bias_flag"
    ] = (
        oracle_check[
            "absolute_monte_carlo_bias"
        ]
        > 0.02
    )

    oracle_check.to_csv(
        output_dir
        / "oracle_recovery_check.csv",
        index=False,
    )

    print(
        "Phase 3B bias-variance summary"
    )

    print(
        "\nAll estimators"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nOracle recovery check"
    )

    print(
        oracle_check.to_string(
            index=False
        )
    )

    print(
        f"\nSaved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
