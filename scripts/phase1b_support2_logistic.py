"""Phase 1B: frozen-model SUPPORT2 calibration phenomenon test."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from actionaudit import AuditConfig, binned_action_audit
from actionaudit.modeling import (
    create_feature_manifest,
    create_patient_split,
    tune_and_refit_logistic,
)
from actionaudit.support2 import (
    build_support2_horizon_view,
    load_support2_csv,
    prediction_feature_columns,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen SUPPORT2 logistic model and compare "
            "reference calibration against hospital-discharge-truncated "
            "calibration."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/support2.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase1b_logistic",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=25,
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


def additional_point_metrics(
    audit_frame: pd.DataFrame,
    audit_result,
) -> dict[str, float]:
    """Calculate prediction and observation-boundary metrics."""

    outcome = audit_frame["outcome"].astype(int)
    prediction = audit_frame["prediction"].astype(float)

    observed = audit_frame["observed"].astype(int).eq(1)
    hidden = ~observed

    eligible = audit_result.bin_summary.loc[
        audit_result.bin_summary["eligible"]
    ]

    weights = eligible["reference_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    curve_gap = eligible[
        "observed_minus_full"
    ].to_numpy(dtype=float)

    hidden_event_rate = (
        float(
            audit_frame.loc[
                hidden,
                "outcome",
            ].mean()
        )
        if hidden.any()
        else float("nan")
    )

    observed_event_rate = (
        float(
            audit_frame.loc[
                observed,
                "outcome",
            ].mean()
        )
        if observed.any()
        else float("nan")
    )

    reference_ici = float(
        audit_result.scalar_summary["reference_ici"]
    )

    restricted_ici = float(
        audit_result.scalar_summary[
            "restricted_complete_case_ici"
        ]
    )

    reference_calerr = float(
        audit_result.scalar_summary["reference_calerr"]
    )

    restricted_calerr = float(
        audit_result.scalar_summary[
            "restricted_complete_case_calerr"
        ]
    )

    return {
        "test_n": float(len(audit_frame)),
        "test_reference_event_rate": float(outcome.mean()),
        "test_restricted_observed_fraction": float(
            observed.mean()
        ),
        "test_restricted_observed_event_rate": (
            observed_event_rate
        ),
        "test_hidden_n": float(hidden.sum()),
        "test_hidden_event_n": float(
            audit_frame.loc[
                hidden,
                "outcome",
            ].sum()
        ),
        "test_hidden_event_rate": hidden_event_rate,
        "test_reference_log_loss": float(
            log_loss(
                outcome,
                prediction,
                labels=[0, 1],
            )
        ),
        "test_reference_brier": float(
            brier_score_loss(
                outcome,
                prediction,
            )
        ),
        "test_reference_roc_auc": float(
            roc_auc_score(
                outcome,
                prediction,
            )
        ),
        "test_reference_average_precision": float(
            average_precision_score(
                outcome,
                prediction,
            )
        ),
        "delta_ici_restricted_minus_reference": (
            restricted_ici - reference_ici
        ),
        "delta_calerr_restricted_minus_reference": (
            restricted_calerr - reference_calerr
        ),
        "max_abs_calibration_curve_gap": float(
            np.max(np.abs(curve_gap))
        ),
        "mean_abs_calibration_curve_gap": float(
            np.average(
                np.abs(curve_gap),
                weights=weights,
            )
        ),
    }


def run_bootstrap(
    audit_frame: pd.DataFrame,
    config: AuditConfig,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level bootstrap of scalar and bin-level metrics."""

    if replicates < 1:
        raise ValueError(
            "Bootstrap replicates must be at least 1"
        )

    rng = np.random.default_rng(seed)
    n = len(audit_frame)

    scalar_rows: list[dict[str, float]] = []
    bin_rows: list[dict[str, float]] = []

    for bootstrap_id in range(replicates):
        sampled_indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = (
            audit_frame
            .iloc[sampled_indices]
            .reset_index(drop=True)
        )

        try:
            result = binned_action_audit(
                sample,
                config,
                action_order=[
                    "alive_hospital_discharge",
                ],
            )
        except ValueError:
            continue

        eligible = result.bin_summary.loc[
            result.bin_summary["eligible"]
        ]

        if eligible.empty:
            continue

        extra = additional_point_metrics(
            sample,
            result,
        )

        scalar_rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "reference_ici": float(
                    result.scalar_summary["reference_ici"]
                ),
                "restricted_ici": float(
                    result.scalar_summary[
                        "restricted_complete_case_ici"
                    ]
                ),
                "delta_ici": float(
                    extra[
                        "delta_ici_restricted_minus_reference"
                    ]
                ),
                "reference_calerr": float(
                    result.scalar_summary[
                        "reference_calerr"
                    ]
                ),
                "restricted_calerr": float(
                    result.scalar_summary[
                        "restricted_complete_case_calerr"
                    ]
                ),
                "delta_calerr": float(
                    extra[
                        "delta_calerr_restricted_minus_reference"
                    ]
                ),
                "standardized_signed_gap": float(
                    result.scalar_summary[
                        "standardized_signed_gap"
                    ]
                ),
                "max_abs_curve_gap": float(
                    extra[
                        "max_abs_calibration_curve_gap"
                    ]
                ),
                "mean_abs_curve_gap": float(
                    extra[
                        "mean_abs_calibration_curve_gap"
                    ]
                ),
                "restricted_observed_fraction": float(
                    extra[
                        "test_restricted_observed_fraction"
                    ]
                ),
                "hidden_event_rate": float(
                    extra["test_hidden_event_rate"]
                ),
            }
        )

        for _, row in eligible.iterrows():
            bin_rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "bin": int(row["bin"]),
                    "mean_prediction": float(
                        row["mean_prediction"]
                    ),
                    "mu_full": float(row["mu_full"]),
                    "mu_observed": float(
                        row["mu_observed"]
                    ),
                    "observed_minus_full": float(
                        row["observed_minus_full"]
                    ),
                }
            )

    scalar_frame = pd.DataFrame(scalar_rows)
    bin_frame = pd.DataFrame(bin_rows)

    if scalar_frame.empty:
        raise RuntimeError(
            "Every bootstrap replicate failed"
        )

    return scalar_frame, bin_frame


def bootstrap_interval_table(
    point_metrics: dict[str, float],
    bootstrap_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create percentile bootstrap confidence intervals."""

    rows: list[dict[str, object]] = []

    for metric in point_metrics:
        values = (
            bootstrap_frame[metric]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        rows.append(
            {
                "metric": metric,
                "estimate": float(
                    point_metrics[metric]
                ),
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


def bootstrap_bin_bands(
    bootstrap_bins: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate percentile bands for every calibration bin."""

    value_columns = (
        "mean_prediction",
        "mu_full",
        "mu_observed",
        "observed_minus_full",
    )

    rows: list[dict[str, object]] = []

    for bin_id, group in bootstrap_bins.groupby(
        "bin",
        sort=True,
    ):
        row: dict[str, object] = {
            "bin": int(bin_id),
            "n_bootstrap": int(
                group["bootstrap_id"].nunique()
            ),
        }

        for column in value_columns:
            values = (
                group[column]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            row[f"{column}_q025"] = float(
                values.quantile(0.025)
            )

            row[f"{column}_q50"] = float(
                values.quantile(0.50)
            )

            row[f"{column}_q975"] = float(
                values.quantile(0.975)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_calibration_curves(
    bin_summary: pd.DataFrame,
    bin_bands: pd.DataFrame,
    output_path: Path,
    horizon_days: int,
) -> None:
    """Plot reference and restricted calibration curves."""

    eligible = (
        bin_summary.loc[
            bin_summary["eligible"]
        ]
        .sort_values("bin")
    )

    plotting = eligible.merge(
        bin_bands,
        on="bin",
        how="left",
        validate="one_to_one",
    )

    x = plotting[
        "mean_prediction"
    ].to_numpy(dtype=float)

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

    axis.fill_between(
        x,
        plotting[
            "mu_full_q025"
        ].to_numpy(dtype=float),
        plotting[
            "mu_full_q975"
        ].to_numpy(dtype=float),
        alpha=0.18,
    )

    axis.plot(
        x,
        plotting[
            "mu_full"
        ].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Reference follow-up",
    )

    axis.fill_between(
        x,
        plotting[
            "mu_observed_q025"
        ].to_numpy(dtype=float),
        plotting[
            "mu_observed_q975"
        ].to_numpy(dtype=float),
        alpha=0.18,
    )

    axis.plot(
        x,
        plotting[
            "mu_observed"
        ].to_numpy(dtype=float),
        marker="s",
        linewidth=2,
        label="Hospital-only complete case",
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.set_xlabel(
        f"Predicted {horizon_days}-day mortality risk"
    )

    axis.set_ylabel(
        f"Observed {horizon_days}-day mortality"
    )

    axis.set_title(
        "Frozen Predictions Under Two Observation Boundaries"
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


def plot_distortion_curve(
    bin_summary: pd.DataFrame,
    bin_bands: pd.DataFrame,
    output_path: Path,
    horizon_days: int,
) -> None:
    """Plot restricted-minus-reference calibration distortion."""

    eligible = (
        bin_summary.loc[
            bin_summary["eligible"]
        ]
        .sort_values("bin")
    )

    plotting = eligible.merge(
        bin_bands,
        on="bin",
        how="left",
        validate="one_to_one",
    )

    x = plotting[
        "mean_prediction"
    ].to_numpy(dtype=float)

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    axis.fill_between(
        x,
        plotting[
            "observed_minus_full_q025"
        ].to_numpy(dtype=float),
        plotting[
            "observed_minus_full_q975"
        ].to_numpy(dtype=float),
        alpha=0.20,
    )

    axis.plot(
        x,
        plotting[
            "observed_minus_full"
        ].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Hospital-only minus reference",
    )

    axis.set_xlabel(
        f"Predicted {horizon_days}-day mortality risk"
    )

    axis.set_ylabel(
        "Calibration curve difference"
    )

    axis.set_title(
        "Observation-Induced Calibration Distortion"
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


def determine_phase1b_status(
    interval_table: pd.DataFrame,
) -> str:
    """Apply the exploratory Phase 1 phenomenon gate."""

    indexed = interval_table.set_index("metric")

    delta_ici_lower = float(
        indexed.loc[
            "delta_ici",
            "bootstrap_q025",
        ]
    )

    max_gap = float(
        indexed.loc[
            "max_abs_curve_gap",
            "estimate",
        ]
    )

    signed_lower = float(
        indexed.loc[
            "standardized_signed_gap",
            "bootstrap_q025",
        ]
    )

    signed_upper = float(
        indexed.loc[
            "standardized_signed_gap",
            "bootstrap_q975",
        ]
    )

    if (
        delta_ici_lower > 0
        and max_gap >= 0.05
    ):
        return (
            "strong_phenomenon_supported: restricted evaluation "
            "worsens calibration and the curve distortion is "
            "practically large"
        )

    if (
        delta_ici_lower > 0
        or signed_lower > 0
        or signed_upper < 0
    ):
        return (
            "phenomenon_supported: bootstrap evidence indicates "
            "observation-induced calibration distortion"
        )

    return (
        "inconclusive: inspect bin-level distortion and "
        "alternative calibration estimators"
    )


def main() -> None:
    args = parse_arguments()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_support2_csv(
        args.input
    )

    horizon_view = build_support2_horizon_view(
        data,
        args.horizon,
    ).frame

    analysis = (
        horizon_view.loc[
            horizon_view["reference_known"].eq(1)
        ]
        .reset_index(drop=True)
    )

    if analysis.empty:
        raise RuntimeError(
            "No patients have known reference outcomes"
        )

    candidate_features = prediction_feature_columns(
        data
    )

    feature_manifest = create_feature_manifest(
        analysis,
        candidate_features,
    )

    included_features = (
        feature_manifest.loc[
            feature_manifest["included"],
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    if not included_features:
        raise RuntimeError(
            "No usable baseline features remain"
        )

    outcome = (
        analysis["outcome"]
        .astype(int)
        .reset_index(drop=True)
    )

    split = create_patient_split(
        analysis,
        outcome,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
        random_seed=args.seed,
    )

    selection = tune_and_refit_logistic(
        analysis,
        outcome,
        split,
        included_features,
        random_seed=args.seed,
    )

    test_data = (
        analysis
        .iloc[split.test_index]
        .copy()
        .reset_index(drop=True)
    )

    test_prediction = selection.model.predict_proba(
        test_data[included_features]
    )[:, 1]

    if not np.isfinite(test_prediction).all():
        raise RuntimeError(
            "Model produced non-finite test predictions"
        )

    audit_frame = pd.DataFrame(
        {
            "patient_id": test_data[
                "patient_id"
            ].astype(int),
            "prediction": test_prediction,
            "outcome": test_data[
                "outcome"
            ].astype(int),
            "observed": test_data[
                "observed"
            ].astype(int),
            "action": test_data["action"],
            "exit_time": test_data[
                "exit_time"
            ],
            "post_discharge_event_by_horizon": (
                test_data[
                    "post_discharge_event_by_horizon"
                ].astype(int)
            ),
        }
    )

    config = AuditConfig(
        horizon_days=args.horizon,
        boundary="hospital",
        n_bins=args.n_bins,
        min_bin_size=args.min_bin_size,
    )

    audit_result = binned_action_audit(
        audit_frame,
        config,
        action_order=[
            "alive_hospital_discharge",
        ],
    )

    if (
        audit_result.max_decomposition_residual
        >= 1e-10
    ):
        raise RuntimeError(
            "Action decomposition failed numerical verification"
        )

    extra = additional_point_metrics(
        audit_frame,
        audit_result,
    )

    point_metrics = {
        "reference_ici": float(
            audit_result.scalar_summary[
                "reference_ici"
            ]
        ),
        "restricted_ici": float(
            audit_result.scalar_summary[
                "restricted_complete_case_ici"
            ]
        ),
        "delta_ici": float(
            extra[
                "delta_ici_restricted_minus_reference"
            ]
        ),
        "reference_calerr": float(
            audit_result.scalar_summary[
                "reference_calerr"
            ]
        ),
        "restricted_calerr": float(
            audit_result.scalar_summary[
                "restricted_complete_case_calerr"
            ]
        ),
        "delta_calerr": float(
            extra[
                "delta_calerr_restricted_minus_reference"
            ]
        ),
        "standardized_signed_gap": float(
            audit_result.scalar_summary[
                "standardized_signed_gap"
            ]
        ),
        "max_abs_curve_gap": float(
            extra[
                "max_abs_calibration_curve_gap"
            ]
        ),
        "mean_abs_curve_gap": float(
            extra[
                "mean_abs_calibration_curve_gap"
            ]
        ),
        "restricted_observed_fraction": float(
            extra[
                "test_restricted_observed_fraction"
            ]
        ),
        "hidden_event_rate": float(
            extra["test_hidden_event_rate"]
        ),
    }

    print(
        f"Running {args.bootstrap_replicates} "
        "bootstrap replicates..."
    )

    bootstrap_scalars, bootstrap_bins = run_bootstrap(
        audit_frame,
        config,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 100,
    )

    interval_table = bootstrap_interval_table(
        point_metrics,
        bootstrap_scalars,
    )

    bin_bands = bootstrap_bin_bands(
        bootstrap_bins
    )

    phase_status = determine_phase1b_status(
        interval_table
    )

    split_labels = np.full(
        len(analysis),
        "",
        dtype=object,
    )

    split_labels[
        split.train_index
    ] = "train"

    split_labels[
        split.validation_index
    ] = "validation"

    split_labels[
        split.test_index
    ] = "test"

    split_manifest = pd.DataFrame(
        {
            "patient_id": analysis[
                "patient_id"
            ].astype(int),
            "split": split_labels,
            "outcome": outcome,
        }
    )

    if "hosp" in analysis.columns:
        split_manifest["hospital"] = analysis["hosp"]

    scalar_summary = (
        audit_result.scalar_summary
        .rename_axis("metric")
        .reset_index(name="value")
    )

    additional_summary = pd.DataFrame(
        {
            "metric": list(extra.keys()),
            "value": list(extra.values()),
        }
    )

    complete_scalar_summary = pd.concat(
        [
            scalar_summary,
            additional_summary,
        ],
        ignore_index=True,
    )

    feature_manifest.to_csv(
        output_dir / "feature_manifest.csv",
        index=False,
    )

    split_manifest.to_csv(
        output_dir / "split_manifest.csv",
        index=False,
    )

    selection.selection_table.to_csv(
        output_dir / "model_selection.csv",
        index=False,
    )

    audit_frame.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    audit_result.bin_summary.to_csv(
        output_dir / "calibration_bin_summary.csv",
        index=False,
    )

    audit_result.action_summary.to_csv(
        output_dir / "action_summary.csv",
        index=False,
    )

    complete_scalar_summary.to_csv(
        output_dir / "scalar_summary.csv",
        index=False,
    )

    bootstrap_scalars.to_csv(
        output_dir / "bootstrap_scalar_replicates.csv",
        index=False,
    )

    bootstrap_bins.to_csv(
        output_dir / "bootstrap_bin_replicates.csv",
        index=False,
    )

    interval_table.to_csv(
        output_dir / "bootstrap_intervals.csv",
        index=False,
    )

    bin_bands.to_csv(
        output_dir / "bootstrap_bin_bands.csv",
        index=False,
    )

    joblib.dump(
        selection.model,
        output_dir / "frozen_logistic_model.joblib",
    )

    plot_calibration_curves(
        audit_result.bin_summary,
        bin_bands,
        output_dir
        / "calibration_reference_vs_restricted.png",
        args.horizon,
    )

    plot_distortion_curve(
        audit_result.bin_summary,
        bin_bands,
        output_dir
        / "calibration_distortion_curve.png",
        args.horizon,
    )

    metadata = {
        "phase": "1B",
        "experiment": (
            "Frozen SUPPORT2 logistic model: reference versus "
            "hospital-discharge-truncated calibration"
        ),
        "horizon_days": int(args.horizon),
        "observation_boundary": "hospital",
        "model": "L2 logistic regression",
        "best_c": float(selection.best_c),
        "training_rows": int(
            len(split.train_index)
            + len(split.validation_index)
        ),
        "train_rows": int(
            len(split.train_index)
        ),
        "validation_rows": int(
            len(split.validation_index)
        ),
        "test_rows": int(
            len(split.test_index)
        ),
        "test_predictions_frozen": True,
        "test_used_for_model_selection": False,
        "stratification_method": (
            split.stratification_method
        ),
        "n_features_before_encoding": int(
            len(included_features)
        ),
        "included_features": included_features,
        "numeric_features": (
            selection.numeric_features
        ),
        "categorical_features": (
            selection.categorical_features
        ),
        "bootstrap_replicates_requested": int(
            args.bootstrap_replicates
        ),
        "bootstrap_replicates_valid": int(
            bootstrap_scalars[
                "bootstrap_id"
            ].nunique()
        ),
        "n_calibration_bins": int(args.n_bins),
        "phase_status": phase_status,
        "max_decomposition_residual": float(
            audit_result.max_decomposition_residual
        ),
        "random_seed": int(args.seed),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
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
        "\nPhase 1B frozen logistic phenomenon "
        "test complete"
    )

    print(
        f"\nSplit: "
        f"{len(split.train_index)} train, "
        f"{len(split.validation_index)} validation, "
        f"{len(split.test_index)} test"
    )

    print(
        f"Stratification: "
        f"{split.stratification_method}"
    )

    print(
        f"Included baseline features: "
        f"{len(included_features)}"
    )

    print(
        f"Selected logistic C: "
        f"{selection.best_c}"
    )

    print("\nValidation model selection")

    print(
        selection.selection_table.to_string(
            index=False
        )
    )

    print("\nTest discrimination and prediction")

    for metric in (
        "test_reference_event_rate",
        "test_reference_log_loss",
        "test_reference_brier",
        "test_reference_roc_auc",
        "test_reference_average_precision",
    ):
        print(
            f"{metric}: {extra[metric]:.6f}"
        )

    print("\nObservation-boundary audit")

    print(
        interval_table.to_string(
            index=False
        )
    )

    print(
        "\nMaximum decomposition residual: "
        f"{audit_result.max_decomposition_residual:.3e}"
    )

    print(
        f"\nPhase 1B status: {phase_status}"
    )

    print(
        f"\nArtifacts saved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
