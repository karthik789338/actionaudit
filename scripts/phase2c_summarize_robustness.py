"""Consolidate Phase 2 right-censoring robustness experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path("artifacts")
OUTPUT_DIR = ARTIFACT_ROOT / "phase2c_robustness"


def discover_runs() -> list[Path]:
    """Find completed right-censoring experiment directories."""

    runs = []

    for directory in sorted(
        ARTIFACT_ROOT.glob(
            "phase2b_right_censoring*"
        )
    ):
        if not directory.is_dir():
            continue

        required = [
            directory / "experiment_metadata.json",
            directory / "estimator_ranking.csv",
            directory / "event_weight_diagnostics.csv",
            directory / "bootstrap_scalar_replicates.csv",
        ]

        if all(path.exists() for path in required):
            runs.append(directory)

    return runs


def read_metric(
    frame: pd.DataFrame,
    metric: str,
) -> float:
    """Read one metric from a metric-value table."""

    indexed = frame.set_index("metric")["value"]

    if metric not in indexed.index:
        return float("nan")

    return float(
        indexed.loc[metric]
    )


def pairwise_bootstrap_row(
    bootstrap: pd.DataFrame,
    *,
    run_name: str,
    horizon: int,
    n_bins: int,
    clip: float,
    estimator_a: str,
    estimator_b: str,
) -> dict[str, object]:
    """Compare estimator A against estimator B using paired replicates.

    The difference is:

        MAE(A) - MAE(B)

    Negative values mean estimator A has lower error.
    """

    column_a = (
        f"{estimator_a}_mae_to_reference"
    )

    column_b = (
        f"{estimator_b}_mae_to_reference"
    )

    if (
        column_a not in bootstrap.columns
        or column_b not in bootstrap.columns
    ):
        raise ValueError(
            f"Missing bootstrap columns: "
            f"{column_a} or {column_b}"
        )

    difference = (
        bootstrap[column_a]
        - bootstrap[column_b]
    )

    difference = (
        difference
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
        "run": run_name,
        "horizon_days": horizon,
        "n_bins": n_bins,
        "censor_survival_clip": clip,
        "estimator_a": estimator_a,
        "estimator_b": estimator_b,
        "difference_definition": (
            "MAE(estimator_a) - MAE(estimator_b)"
        ),
        "estimate": float(
            difference.mean()
        ),
        "bootstrap_q025": float(
            difference.quantile(0.025)
        ),
        "bootstrap_median": float(
            difference.quantile(0.50)
        ),
        "bootstrap_q975": float(
            difference.quantile(0.975)
        ),
        "probability_a_better": float(
            (difference < 0).mean()
        ),
        "n_valid_bootstrap": int(
            len(difference)
        ),
    }


def main() -> None:
    """Create consolidated robustness and pairwise comparison tables."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = discover_runs()

    if not runs:
        raise RuntimeError(
            "No completed right-censoring runs were found"
        )

    summary_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for directory in runs:
        metadata = json.loads(
            (
                directory
                / "experiment_metadata.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        ranking = pd.read_csv(
            directory
            / "estimator_ranking.csv"
        )

        diagnostics = pd.read_csv(
            directory
            / "event_weight_diagnostics.csv"
        )

        bootstrap = pd.read_csv(
            directory
            / "bootstrap_scalar_replicates.csv"
        )

        horizon = int(
            metadata["horizon_days"]
        )

        n_bins = int(
            metadata["n_bins"]
        )

        clip = float(
            metadata["censor_survival_clip"]
        )

        ranking_index = ranking.set_index(
            "estimator"
        )

        summary_rows.append(
            {
                "run": directory.name,
                "horizon_days": horizon,
                "n_bins": n_bins,
                "censor_survival_clip": clip,
                "best_estimator": metadata[
                    "best_estimator"
                ],
                "complete_case_mae": float(
                    ranking_index.loc[
                        "complete_case",
                        "mae_to_reference",
                    ]
                ),
                "kaplan_meier_mae": float(
                    ranking_index.loc[
                        "kaplan_meier",
                        "mae_to_reference",
                    ]
                ),
                "survival_or_mae": float(
                    ranking_index.loc[
                        "survival_or",
                        "mae_to_reference",
                    ]
                ),
                "ipcw_event_mae": float(
                    ranking_index.loc[
                        "ipcw_event",
                        "mae_to_reference",
                    ]
                ),
                "ipcw_signed_gap": float(
                    ranking_index.loc[
                        "ipcw_event",
                        "signed_gap",
                    ]
                ),
                "fraction_events_clipped": read_metric(
                    diagnostics,
                    "fraction_event_g_clipped",
                ),
                "maximum_event_weight": read_metric(
                    diagnostics,
                    "maximum_event_weight",
                ),
                "event_weight_effective_sample_size": read_metric(
                    diagnostics,
                    "event_weight_effective_sample_size",
                ),
                "event_g_q01": read_metric(
                    diagnostics,
                    "event_g_q01",
                ),
                "event_g_q05": read_metric(
                    diagnostics,
                    "event_g_q05",
                ),
            }
        )

        comparisons = [
            (
                "survival_or",
                "kaplan_meier",
            ),
            (
                "ipcw_event",
                "kaplan_meier",
            ),
            (
                "ipcw_event",
                "survival_or",
            ),
        ]

        for estimator_a, estimator_b in comparisons:
            pairwise_rows.append(
                pairwise_bootstrap_row(
                    bootstrap,
                    run_name=directory.name,
                    horizon=horizon,
                    n_bins=n_bins,
                    clip=clip,
                    estimator_a=estimator_a,
                    estimator_b=estimator_b,
                )
            )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "horizon_days",
                "n_bins",
                "censor_survival_clip",
            ]
        )
        .reset_index(drop=True)
    )

    pairwise = (
        pd.DataFrame(pairwise_rows)
        .sort_values(
            [
                "horizon_days",
                "n_bins",
                "censor_survival_clip",
                "estimator_a",
                "estimator_b",
            ]
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR
        / "robustness_summary.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR
        / "paired_bootstrap_comparisons.csv",
        index=False,
    )

    print(
        "Phase 2C robustness summary"
    )

    print(
        "\nRun-level results"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPaired comparisons"
    )

    print(
        pairwise.to_string(
            index=False
        )
    )

    print(
        f"\nSaved under: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
