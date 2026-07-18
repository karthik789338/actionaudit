"""Phase 2A: complete-case versus Kaplan-Meier calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from actionaudit.km import (
    evaluate_binned_estimators,
    make_hospital_restricted_survival,
    reference_quantile_edges,
)
from actionaudit.support2 import (
    build_support2_horizon_view,
    load_support2_csv,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reference, complete-case, and Kaplan-Meier "
            "calibration on the frozen SUPPORT2 test predictions."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/support2.csv",
    )

    parser.add_argument(
        "--phase1b-dir",
        default="artifacts/phase1b_logistic",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase2a_km",
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


def prepare_test_frame(
    raw_data: pd.DataFrame,
    predictions: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Join frozen predictions to original SUPPORT2 survival fields."""

    horizon_view = build_support2_horizon_view(
        raw_data,
        horizon,
    ).frame

    survival_columns = [
        "patient_id",
        "death",
        "d.time",
        "hospdead",
        "slos",
        "outcome",
        "observed",
    ]

    merged = predictions.merge(
        horizon_view[survival_columns],
        on="patient_id",
        how="left",
        suffixes=(
            "_saved",
            "_rebuilt",
        ),
        validate="one_to_one",
    )

    if merged[
        [
            "death",
            "d.time",
            "hospdead",
            "slos",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Some test patients could not be matched to SUPPORT2"
        )

    saved_outcome = merged[
        "outcome_saved"
    ].astype(int)

    rebuilt_outcome = merged[
        "outcome_rebuilt"
    ].astype(int)

    if not saved_outcome.equals(
        rebuilt_outcome
    ):
        raise RuntimeError(
            "Saved and reconstructed outcomes disagree"
        )

    saved_observed = merged[
        "observed_saved"
    ].astype(int)

    rebuilt_observed = merged[
        "observed_rebuilt"
    ].astype(int)

    if not saved_observed.equals(
        rebuilt_observed
    ):
        raise RuntimeError(
            "Saved and reconstructed observation indicators disagree"
        )

    merged["outcome"] = saved_outcome
    merged["observed"] = saved_observed

    merged = make_hospital_restricted_survival(
        merged,
        horizon,
    )

    if not merged[
        "restricted_observed_at_horizon"
    ].astype(int).equals(
        merged["observed"].astype(int)
    ):
        raise RuntimeError(
            "Survival construction disagrees with Phase 1B observation rules"
        )

    return merged


def bootstrap_estimators(
    frame: pd.DataFrame,
    *,
    horizon: int,
    bin_edges: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap patients while keeping original risk-bin edges fixed."""

    rng = np.random.default_rng(seed)
    n = len(frame)

    rows: list[dict[str, float]] = []

    for bootstrap_id in range(replicates):
        sampled_index = rng.integers(
            0,
            n,
            size=n,
        )

        sample = (
            frame
            .iloc[sampled_index]
            .reset_index(drop=True)
        )

        result = evaluate_binned_estimators(
            sample,
            horizon=horizon,
            bin_edges=bin_edges,
        )

        row = {
            "bootstrap_id": bootstrap_id,
        }

        row.update(
            {
                key: float(value)
                for key, value
                in result.scalar_summary.items()
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def interval_table(
    point_summary: pd.Series,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Create percentile confidence intervals."""

    rows: list[dict[str, object]] = []

    for metric, estimate in point_summary.items():
        if metric not in bootstrap.columns:
            continue

        values = (
            bootstrap[metric]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate),
                "bootstrap_q025": float(
                    values.quantile(0.025)
                ),
                "bootstrap_median": float(
                    values.quantile(0.50)
                ),
                "bootstrap_q975": float(
                    values.quantile(0.975)
                ),
                "n_valid_bootstrap": int(
                    len(values)
                ),
            }
        )

    return pd.DataFrame(rows)


def determine_status(
    intervals: pd.DataFrame,
) -> str:
    """Determine whether KM recovers the reference better than CC."""

    indexed = intervals.set_index(
        "metric"
    )

    improvement = float(
        indexed.loc[
            "cc_minus_km_mae_improvement",
            "estimate",
        ]
    )

    improvement_lower = float(
        indexed.loc[
            "cc_minus_km_mae_improvement",
            "bootstrap_q025",
        ]
    )

    cc_mae = float(
        indexed.loc[
            "complete_case_mae_to_reference",
            "estimate",
        ]
    )

    km_mae = float(
        indexed.loc[
            "kaplan_meier_mae_to_reference",
            "estimate",
        ]
    )

    if (
        improvement_lower > 0
        and km_mae <= 0.5 * cc_mae
    ):
        return (
            "strong_km_recovery: Kaplan-Meier reduces reference-curve "
            "error by at least half with positive bootstrap support"
        )

    if improvement_lower > 0:
        return (
            "partial_km_recovery: Kaplan-Meier significantly improves "
            "over complete-case evaluation"
        )

    if improvement > 0:
        return (
            "weak_km_recovery: point estimate improves but bootstrap "
            "support is inconclusive"
        )

    return (
        "no_km_recovery: Kaplan-Meier does not improve over "
        "complete-case evaluation"
    )


def plot_curves(
    bins: pd.DataFrame,
    output_path: Path,
    horizon: int,
) -> None:
    """Plot reference, complete-case, and KM curves."""

    ordered = bins.sort_values(
        "mean_prediction"
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    axis.plot(
        ordered["mean_prediction"],
        ordered["reference_risk"],
        marker="o",
        linewidth=2,
        label="Reference follow-up",
    )

    axis.plot(
        ordered["mean_prediction"],
        ordered["complete_case_risk"],
        marker="s",
        linewidth=2,
        label="Naive complete case",
    )

    axis.plot(
        ordered["mean_prediction"],
        ordered["kaplan_meier_risk"],
        marker="^",
        linewidth=2,
        label="Kaplan-Meier",
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.set_xlabel(
        f"Predicted {horizon}-day mortality risk"
    )

    axis.set_ylabel(
        f"Estimated {horizon}-day mortality"
    )

    axis.set_title(
        "Reference vs Complete-Case vs Kaplan-Meier Calibration"
    )

    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_reference_error(
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot estimator minus reference risk by prediction bin."""

    ordered = bins.sort_values(
        "mean_prediction"
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    axis.plot(
        ordered["mean_prediction"],
        ordered[
            "complete_case_minus_reference"
        ],
        marker="s",
        linewidth=2,
        label="Complete case minus reference",
    )

    axis.plot(
        ordered["mean_prediction"],
        ordered[
            "kaplan_meier_minus_reference"
        ],
        marker="^",
        linewidth=2,
        label="Kaplan-Meier minus reference",
    )

    axis.set_xlabel(
        "Predicted mortality risk"
    )

    axis.set_ylabel(
        "Estimator minus reference event rate"
    )

    axis.set_title(
        "Estimator Error Relative to Reference Follow-Up"
    )

    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    args = parse_arguments()

    phase1b_dir = Path(
        args.phase1b_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        phase1b_dir
        / "test_predictions.csv"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 1B predictions: {predictions_path}"
        )

    raw_data = load_support2_csv(
        args.input
    )

    predictions = pd.read_csv(
        predictions_path
    )

    test_frame = prepare_test_frame(
        raw_data,
        predictions,
        args.horizon,
    )

    bin_edges = reference_quantile_edges(
        test_frame[
            "prediction"
        ].to_numpy(),
        args.n_bins,
    )

    point_result = evaluate_binned_estimators(
        test_frame,
        horizon=args.horizon,
        bin_edges=bin_edges,
    )

    print(
        f"Running {args.bootstrap_replicates} "
        "fixed-bin bootstrap replicates..."
    )

    bootstrap = bootstrap_estimators(
        test_frame,
        horizon=args.horizon,
        bin_edges=bin_edges,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 200,
    )

    intervals = interval_table(
        point_result.scalar_summary,
        bootstrap,
    )

    status = determine_status(
        intervals
    )

    point_result.bin_summary.to_csv(
        output_dir / "estimator_bin_summary.csv",
        index=False,
    )

    (
        point_result.scalar_summary
        .rename_axis("metric")
        .reset_index(name="value")
        .to_csv(
            output_dir / "estimator_scalar_summary.csv",
            index=False,
        )
    )

    bootstrap.to_csv(
        output_dir / "bootstrap_scalar_replicates.csv",
        index=False,
    )

    intervals.to_csv(
        output_dir / "bootstrap_intervals.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "edge_index": np.arange(
                len(bin_edges)
            ),
            "bin_edge": bin_edges,
        }
    ).to_csv(
        output_dir / "fixed_bin_edges.csv",
        index=False,
    )

    plot_curves(
        point_result.bin_summary,
        output_dir / "reference_complete_case_km.png",
        args.horizon,
    )

    plot_reference_error(
        point_result.bin_summary,
        output_dir / "estimator_error_vs_reference.png",
    )

    metadata = {
        "phase": "2A",
        "horizon_days": int(
            args.horizon
        ),
        "n_bins": int(
            args.n_bins
        ),
        "bin_definition": (
            "fixed quantile bins from original frozen test predictions"
        ),
        "bootstrap_replicates": int(
            args.bootstrap_replicates
        ),
        "kaplan_meier_assumption": (
            "independent discharge censoring within prediction-risk bin"
        ),
        "status": status,
        "test_rows": int(
            len(test_frame)
        ),
        "test_restricted_observed_rows": int(
            test_frame[
                "observed"
            ].sum()
        ),
        "test_discharge_censored_rows": int(
            test_frame[
                "restricted_censored_by_discharge"
            ].sum()
        ),
        "test_restricted_event_rows": int(
            test_frame[
                "restricted_event"
            ].sum()
        ),
        "random_seed": int(
            args.seed
        ),
    }

    (
        output_dir
        / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nPhase 2A complete"
    )

    print(
        "\nBin-level estimator comparison"
    )

    print(
        point_result.bin_summary.to_string(
            index=False
        )
    )

    print(
        "\nScalar estimator comparison"
    )

    print(
        point_result.scalar_summary.to_string()
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
        f"\nPhase 2A status: {status}"
    )

    print(
        f"\nArtifacts saved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
