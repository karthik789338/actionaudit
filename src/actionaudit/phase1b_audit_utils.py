"""Utilities for the Phase 1B frozen-prediction calibration audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from actionaudit import AuditConfig, binned_action_audit


def calculate_metrics(
    audit_frame: pd.DataFrame,
    audit_result,
) -> dict[str, float]:
    """Calculate prediction and observation-boundary metrics."""

    outcome = audit_frame["outcome"].astype(int)
    prediction = audit_frame["prediction"].astype(float)
    observed = audit_frame["observed"].astype(int).eq(1)
    hidden = ~observed

    eligible_bins = audit_result.bin_summary.loc[
        audit_result.bin_summary["eligible"]
    ]

    if eligible_bins.empty:
        raise ValueError("No eligible calibration bins are available.")

    weights = eligible_bins["reference_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    absolute_gap = (
        eligible_bins["observed_minus_full"]
        .abs()
        .to_numpy(dtype=float)
    )

    observed_event_rate = (
        float(audit_frame.loc[observed, "outcome"].mean())
        if observed.any()
        else float("nan")
    )

    hidden_event_rate = (
        float(audit_frame.loc[hidden, "outcome"].mean())
        if hidden.any()
        else float("nan")
    )

    return {
        "test_n": float(len(audit_frame)),
        "test_reference_event_rate": float(outcome.mean()),
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
        "restricted_observed_fraction": float(observed.mean()),
        "restricted_observed_n": float(observed.sum()),
        "restricted_observed_event_rate": observed_event_rate,
        "hidden_n": float(hidden.sum()),
        "hidden_event_n": float(
            audit_frame.loc[hidden, "outcome"].sum()
        ),
        "hidden_event_rate": hidden_event_rate,
        "reference_ici": float(
            audit_result.scalar_summary["reference_ici"]
        ),
        "restricted_ici": float(
            audit_result.scalar_summary[
                "restricted_complete_case_ici"
            ]
        ),
        "delta_ici": float(
            audit_result.scalar_summary[
                "restricted_complete_case_ici"
            ]
            - audit_result.scalar_summary["reference_ici"]
        ),
        "reference_calerr": float(
            audit_result.scalar_summary["reference_calerr"]
        ),
        "restricted_calerr": float(
            audit_result.scalar_summary[
                "restricted_complete_case_calerr"
            ]
        ),
        "delta_calerr": float(
            audit_result.scalar_summary[
                "restricted_complete_case_calerr"
            ]
            - audit_result.scalar_summary["reference_calerr"]
        ),
        "standardized_signed_gap": float(
            audit_result.scalar_summary[
                "standardized_signed_gap"
            ]
        ),
        "max_abs_curve_gap": float(absolute_gap.max()),
        "mean_abs_curve_gap": float(
            np.average(
                absolute_gap,
                weights=weights,
            )
        ),
        "curve_aci": float(
            audit_result.scalar_summary["curve_aci"]
        ),
        "global_signed_aci": float(
            audit_result.scalar_summary["global_signed_aci"]
        ),
        "max_decomposition_residual": float(
            audit_result.max_decomposition_residual
        ),
    }


def run_bootstrap(
    audit_frame: pd.DataFrame,
    config: AuditConfig,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a patient-level nonparametric bootstrap."""

    if replicates < 1:
        raise ValueError(
            "bootstrap-replicates must be at least 1"
        )

    rng = np.random.default_rng(seed)
    n = len(audit_frame)

    scalar_rows: list[dict[str, float]] = []
    bin_rows: list[dict[str, float]] = []

    for bootstrap_id in range(replicates):
        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = (
            audit_frame
            .iloc[sample_indices]
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

        metrics = calculate_metrics(
            sample,
            result,
        )

        scalar_rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "reference_ici": metrics["reference_ici"],
                "restricted_ici": metrics["restricted_ici"],
                "delta_ici": metrics["delta_ici"],
                "reference_calerr": metrics["reference_calerr"],
                "restricted_calerr": metrics["restricted_calerr"],
                "delta_calerr": metrics["delta_calerr"],
                "standardized_signed_gap": metrics[
                    "standardized_signed_gap"
                ],
                "max_abs_curve_gap": metrics[
                    "max_abs_curve_gap"
                ],
                "mean_abs_curve_gap": metrics[
                    "mean_abs_curve_gap"
                ],
                "restricted_observed_fraction": metrics[
                    "restricted_observed_fraction"
                ],
                "hidden_event_rate": metrics[
                    "hidden_event_rate"
                ],
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
                    "mu_observed": float(row["mu_observed"]),
                    "observed_minus_full": float(
                        row["observed_minus_full"]
                    ),
                }
            )

        if (bootstrap_id + 1) % 100 == 0:
            print(
                f"Completed {bootstrap_id + 1}/"
                f"{replicates} bootstrap replicates"
            )

    scalar_frame = pd.DataFrame(scalar_rows)
    bin_frame = pd.DataFrame(bin_rows)

    if scalar_frame.empty:
        raise RuntimeError(
            "All bootstrap replicates failed."
        )

    return scalar_frame, bin_frame


def create_interval_table(
    point_metrics: dict[str, float],
    bootstrap_scalars: pd.DataFrame,
) -> pd.DataFrame:
    """Create percentile bootstrap confidence intervals."""

    metric_names = [
        "reference_ici",
        "restricted_ici",
        "delta_ici",
        "reference_calerr",
        "restricted_calerr",
        "delta_calerr",
        "standardized_signed_gap",
        "max_abs_curve_gap",
        "mean_abs_curve_gap",
        "restricted_observed_fraction",
        "hidden_event_rate",
    ]

    rows: list[dict[str, object]] = []

    for metric in metric_names:
        values = (
            bootstrap_scalars[metric]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        rows.append(
            {
                "metric": metric,
                "estimate": point_metrics[metric],
                "bootstrap_q025": float(
                    values.quantile(0.025)
                ),
                "bootstrap_median": float(
                    values.quantile(0.50)
                ),
                "bootstrap_q975": float(
                    values.quantile(0.975)
                ),
                "n_valid_bootstrap": int(len(values)),
            }
        )

    return pd.DataFrame(rows)


def create_bin_bands(
    bootstrap_bins: pd.DataFrame,
) -> pd.DataFrame:
    """Create bootstrap confidence bands by score-bin position."""

    value_columns = [
        "mean_prediction",
        "mu_full",
        "mu_observed",
        "observed_minus_full",
    ]

    rows: list[dict[str, float]] = []

    for bin_id, group in bootstrap_bins.groupby(
        "bin",
        sort=True,
    ):
        row: dict[str, float] = {
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


def plot_calibration(
    bin_summary: pd.DataFrame,
    bin_bands: pd.DataFrame,
    output_path: Path,
    horizon: int,
) -> None:
    """Plot reference and hospital-only calibration curves."""

    eligible = (
        bin_summary.loc[
            bin_summary["eligible"]
        ]
        .sort_values("bin")
        .reset_index(drop=True)
    )

    bands = (
        bin_bands
        .set_index("bin")
        .loc[eligible["bin"]]
        .reset_index()
    )

    x = eligible["mean_prediction"].to_numpy(dtype=float)

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
        bands["mu_full_q025"].to_numpy(dtype=float),
        bands["mu_full_q975"].to_numpy(dtype=float),
        alpha=0.18,
    )

    axis.plot(
        x,
        eligible["mu_full"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Reference follow-up",
    )

    axis.fill_between(
        x,
        bands["mu_observed_q025"].to_numpy(dtype=float),
        bands["mu_observed_q975"].to_numpy(dtype=float),
        alpha=0.18,
    )

    axis.plot(
        x,
        eligible["mu_observed"].to_numpy(dtype=float),
        marker="s",
        linewidth=2,
        label="Hospital-only complete case",
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(
        f"Predicted {horizon}-day mortality risk"
    )
    axis.set_ylabel(
        f"Observed {horizon}-day mortality"
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


def plot_distortion(
    bin_summary: pd.DataFrame,
    bin_bands: pd.DataFrame,
    output_path: Path,
    horizon: int,
) -> None:
    """Plot hospital-only minus reference calibration."""

    eligible = (
        bin_summary.loc[
            bin_summary["eligible"]
        ]
        .sort_values("bin")
        .reset_index(drop=True)
    )

    bands = (
        bin_bands
        .set_index("bin")
        .loc[eligible["bin"]]
        .reset_index()
    )

    x = eligible["mean_prediction"].to_numpy(dtype=float)

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
        bands[
            "observed_minus_full_q025"
        ].to_numpy(dtype=float),
        bands[
            "observed_minus_full_q975"
        ].to_numpy(dtype=float),
        alpha=0.20,
    )

    axis.plot(
        x,
        eligible[
            "observed_minus_full"
        ].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Hospital-only minus reference",
    )

    axis.set_xlabel(
        f"Predicted {horizon}-day mortality risk"
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


def determine_status(
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
            "strong_phenomenon_supported: "
            "restricted evaluation worsens calibration "
            "and curve distortion is practically large"
        )

    if (
        delta_ici_lower > 0
        or signed_lower > 0
        or signed_upper < 0
    ):
        return (
            "phenomenon_supported: bootstrap evidence "
            "indicates observation-induced distortion"
        )

    return (
        "inconclusive: inspect alternative calibration "
        "estimators and score-region effects"
    )
