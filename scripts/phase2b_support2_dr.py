"""Phase 2B: held-out IPCW, outcome regression, and AIPW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
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

from actionaudit.dr import (
    evaluate_dr_estimators,
    tune_binary_logistic,
)
from actionaudit.km import (
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
            "Estimate calibration using held-out IPCW, outcome "
            "regression, and AIPW nuisance models."
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
        default="artifacts/phase2b_dr",
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
        "--propensity-clip",
        type=float,
        default=0.05,
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


def load_phase1b_metadata(
    phase1b_dir: Path,
) -> dict:
    """Load the frozen model experiment metadata."""

    metadata_path = (
        phase1b_dir
        / "experiment_metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}"
        )

    return json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )


def prepare_analysis_data(
    raw_data: pd.DataFrame,
    phase1b_dir: Path,
    horizon: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """Reconstruct development and test cohorts."""

    split_manifest = pd.read_csv(
        phase1b_dir
        / "split_manifest.csv"
    )

    test_predictions = pd.read_csv(
        phase1b_dir
        / "test_predictions.csv"
    )

    metadata = load_phase1b_metadata(
        phase1b_dir
    )

    features = list(
        metadata["included_features"]
    )

    horizon_view = build_support2_horizon_view(
        raw_data,
        horizon,
    ).frame

    analysis = horizon_view.merge(
        split_manifest[
            [
                "patient_id",
                "split",
            ]
        ],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if len(analysis) != len(split_manifest):
        raise RuntimeError(
            "Split manifest did not match the reconstructed cohort"
        )

    development = analysis.loc[
        analysis["split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ].copy()

    train = development.loc[
        development["split"].eq("train")
    ].copy()

    validation = development.loc[
        development["split"].eq(
            "validation"
        )
    ].copy()

    test = analysis.loc[
        analysis["split"].eq("test")
    ].copy()

    test = test.merge(
        test_predictions[
            [
                "patient_id",
                "prediction",
                "outcome",
                "observed",
            ]
        ],
        on="patient_id",
        how="inner",
        suffixes=(
            "_rebuilt",
            "_saved",
        ),
        validate="one_to_one",
    )

    if len(test) != len(test_predictions):
        raise RuntimeError(
            "Test predictions did not match the test cohort"
        )

    if not test[
        "outcome_rebuilt"
    ].astype(int).equals(
        test["outcome_saved"].astype(int)
    ):
        raise RuntimeError(
            "Reconstructed and saved test outcomes disagree"
        )

    if not test[
        "observed_rebuilt"
    ].astype(int).equals(
        test["observed_saved"].astype(int)
    ):
        raise RuntimeError(
            "Reconstructed and saved observation indicators disagree"
        )

    test["outcome"] = (
        test["outcome_saved"]
        .astype(int)
    )

    test["observed"] = (
        test["observed_saved"]
        .astype(int)
    )

    test = make_hospital_restricted_survival(
        test,
        horizon,
    )

    missing_features = [
        feature
        for feature in features
        if feature not in analysis.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing nuisance features: {missing_features}"
        )

    return (
        train,
        validation,
        test,
        features,
    )


def model_performance(
    outcome: pd.Series,
    probability: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Summarize binary nuisance-model performance."""

    outcome = outcome.astype(int)

    return {
        f"{prefix}_n": float(len(outcome)),
        f"{prefix}_event_rate": float(
            outcome.mean()
        ),
        f"{prefix}_log_loss": float(
            log_loss(
                outcome,
                probability,
                labels=[0, 1],
            )
        ),
        f"{prefix}_brier": float(
            brier_score_loss(
                outcome,
                probability,
            )
        ),
        f"{prefix}_roc_auc": float(
            roc_auc_score(
                outcome,
                probability,
            )
        ),
        f"{prefix}_average_precision": float(
            average_precision_score(
                outcome,
                probability,
            )
        ),
    }


def bootstrap_fixed_nuisance(
    frame: pd.DataFrame,
    *,
    horizon: int,
    bin_edges: np.ndarray,
    propensity_clip: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap test patients with held-out nuisance predictions fixed."""

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

        result = evaluate_dr_estimators(
            sample,
            horizon=horizon,
            bin_edges=bin_edges,
            propensity_clip=propensity_clip,
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
    """Create fixed-nuisance percentile bootstrap intervals."""

    rows: list[dict[str, object]] = []

    for metric, estimate in point_summary.items():
        if metric not in bootstrap.columns:
            continue

        values = (
            bootstrap[metric]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
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


def estimator_ranking(
    scalar_summary: pd.Series,
) -> pd.DataFrame:
    """Rank estimators by absolute error to the reference curve."""

    estimator_names = (
        "complete_case",
        "kaplan_meier",
        "outcome_regression",
        "ipcw_ht",
        "ipcw_hajek",
        "aipw",
        "aipw_clipped",
    )

    rows: list[dict[str, object]] = []

    for estimator in estimator_names:
        rows.append(
            {
                "estimator": estimator,
                "mae_to_reference": float(
                    scalar_summary[
                        f"{estimator}_mae_to_reference"
                    ]
                ),
                "signed_gap": float(
                    scalar_summary[
                        f"{estimator}_signed_gap"
                    ]
                ),
                "ici": float(
                    scalar_summary[
                        f"{estimator}_ici"
                    ]
                ),
                "improvement_over_complete_case": (
                    0.0
                    if estimator == "complete_case"
                    else float(
                        scalar_summary[
                            f"{estimator}_improvement_over_complete_case"
                        ]
                    )
                ),
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "mae_to_reference",
                "ici",
            ]
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    return ranking


def determine_status(
    ranking: pd.DataFrame,
    diagnostics: pd.Series,
) -> str:
    """Interpret the Phase 2B estimator comparison."""

    best = str(
        ranking.loc[
            0,
            "estimator",
        ]
    )

    best_mae = float(
        ranking.loc[
            0,
            "mae_to_reference",
        ]
    )

    km_row = ranking.loc[
        ranking["estimator"].eq(
            "kaplan_meier"
        )
    ].iloc[0]

    km_mae = float(
        km_row["mae_to_reference"]
    )

    clipped_fraction = float(
        diagnostics[
            "fraction_propensity_clipped_low"
        ]
        + diagnostics[
            "fraction_propensity_clipped_high"
        ]
    )

    if (
        best in {
            "aipw",
            "aipw_clipped",
        }
        and best_mae < km_mae
        and clipped_fraction <= 0.10
    ):
        return (
            "dr_recovery_supported: AIPW improves on Kaplan-Meier "
            "without severe positivity failure"
        )

    if (
        best in {
            "ipcw_ht",
            "ipcw_hajek",
            "outcome_regression",
        }
        and best_mae < km_mae
        and clipped_fraction <= 0.10
    ):
        return (
            "covariate_adjustment_supported: a covariate-adjusted "
            "estimator improves on Kaplan-Meier"
        )

    if clipped_fraction > 0.10:
        return (
            "positivity_concern: estimator comparison requires "
            "propensity-truncation sensitivity analysis"
        )

    return (
        "no_clear_dr_gain: Kaplan-Meier remains competitive or "
        "doubly robust adjustment does not recover the reference"
    )


def plot_estimator_curves(
    bins: pd.DataFrame,
    output_path: Path,
    horizon: int,
) -> None:
    """Plot all primary estimator calibration curves."""

    ordered = bins.sort_values(
        "mean_prediction"
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    plotting_columns = {
        "reference_risk": "Reference",
        "complete_case_risk": "Complete case",
        "kaplan_meier_risk": "Kaplan-Meier",
        "outcome_regression_risk": "Outcome regression",
        "ipcw_hajek_risk": "IPCW Hájek",
        "aipw_risk": "AIPW",
    }

    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
    ]

    for marker, (
        column,
        label,
    ) in zip(
        markers,
        plotting_columns.items(),
    ):
        axis.plot(
            ordered["mean_prediction"],
            ordered[column],
            marker=marker,
            linewidth=1.8,
            label=label,
        )

    axis.set_xlabel(
        f"Predicted {horizon}-day mortality risk"
    )

    axis.set_ylabel(
        f"Estimated {horizon}-day mortality"
    )

    axis.set_title(
        "Calibration Estimator Comparison"
    )

    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_reference_errors(
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot each estimator's deviation from the reference curve."""

    ordered = bins.sort_values(
        "mean_prediction"
    )

    reference = ordered[
        "reference_risk"
    ]

    columns = {
        "complete_case_risk": "Complete case",
        "kaplan_meier_risk": "Kaplan-Meier",
        "outcome_regression_risk": "Outcome regression",
        "ipcw_hajek_risk": "IPCW Hájek",
        "aipw_risk": "AIPW",
    }

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    for column, label in columns.items():
        axis.plot(
            ordered["mean_prediction"],
            ordered[column] - reference,
            marker="o",
            linewidth=1.7,
            label=label,
        )

    axis.set_xlabel(
        "Predicted mortality risk"
    )

    axis.set_ylabel(
        "Estimator minus reference risk"
    )

    axis.set_title(
        "Estimator Error Relative to Reference Follow-Up"
    )

    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

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

    raw_data = load_support2_csv(
        args.input
    )

    train, validation, test, features = (
        prepare_analysis_data(
            raw_data,
            phase1b_dir,
            args.horizon,
        )
    )

    propensity_model = tune_binary_logistic(
        train,
        train["observed"],
        validation,
        validation["observed"],
        features,
        random_seed=args.seed + 10,
    )

    train_observed = train.loc[
        train["observed"].astype(int).eq(1)
    ].copy()

    validation_observed = validation.loc[
        validation["observed"].astype(int).eq(1)
    ].copy()

    outcome_model = tune_binary_logistic(
        train_observed,
        train_observed["outcome"],
        validation_observed,
        validation_observed["outcome"],
        features,
        random_seed=args.seed + 20,
    )

    test["propensity"] = (
        propensity_model.model.predict_proba(
            test[features]
        )[:, 1]
    )

    test["outcome_regression"] = (
        outcome_model.model.predict_proba(
            test[features]
        )[:, 1]
    )

    propensity_test_metrics = model_performance(
        test["observed"],
        test["propensity"].to_numpy(),
        "propensity_test",
    )

    observed_test = test.loc[
        test["observed"].astype(int).eq(1)
    ]

    outcome_test_metrics = model_performance(
        observed_test["outcome"],
        observed_test[
            "outcome_regression"
        ].to_numpy(),
        "outcome_model_observed_test",
    )

    bin_edges = reference_quantile_edges(
        test["prediction"].to_numpy(),
        args.n_bins,
    )

    point_result = evaluate_dr_estimators(
        test,
        horizon=args.horizon,
        bin_edges=bin_edges,
        propensity_clip=args.propensity_clip,
    )

    print(
        f"Running {args.bootstrap_replicates} "
        "fixed-nuisance bootstrap replicates..."
    )

    bootstrap = bootstrap_fixed_nuisance(
        test,
        horizon=args.horizon,
        bin_edges=bin_edges,
        propensity_clip=args.propensity_clip,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 300,
    )

    intervals = interval_table(
        point_result.scalar_summary,
        bootstrap,
    )

    ranking = estimator_ranking(
        point_result.scalar_summary
    )

    status = determine_status(
        ranking,
        point_result.diagnostics,
    )

    point_result.bin_summary.to_csv(
        output_dir
        / "estimator_bin_summary.csv",
        index=False,
    )

    (
        point_result.scalar_summary
        .rename_axis("metric")
        .reset_index(name="value")
        .to_csv(
            output_dir
            / "estimator_scalar_summary.csv",
            index=False,
        )
    )

    (
        point_result.diagnostics
        .rename_axis("metric")
        .reset_index(name="value")
        .to_csv(
            output_dir
            / "positivity_diagnostics.csv",
            index=False,
        )
    )

    propensity_model.selection_table.to_csv(
        output_dir
        / "propensity_model_selection.csv",
        index=False,
    )

    outcome_model.selection_table.to_csv(
        output_dir
        / "outcome_model_selection.csv",
        index=False,
    )

    ranking.to_csv(
        output_dir
        / "estimator_ranking.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_scalar_replicates.csv",
        index=False,
    )

    intervals.to_csv(
        output_dir
        / "bootstrap_intervals.csv",
        index=False,
    )

    test[
        [
            "patient_id",
            "prediction",
            "outcome",
            "observed",
            "restricted_time",
            "restricted_event",
            "propensity",
            "outcome_regression",
        ]
    ].to_csv(
        output_dir
        / "test_nuisance_predictions.csv",
        index=False,
    )

    joblib.dump(
        propensity_model.model,
        output_dir
        / "propensity_model.joblib",
    )

    joblib.dump(
        outcome_model.model,
        output_dir
        / "outcome_regression_model.joblib",
    )

    plot_estimator_curves(
        point_result.bin_summary,
        output_dir
        / "all_estimator_calibration_curves.png",
        args.horizon,
    )

    plot_reference_errors(
        point_result.bin_summary,
        output_dir
        / "all_estimator_reference_errors.png",
    )

    nuisance_metrics = {
        **propensity_test_metrics,
        **outcome_test_metrics,
    }

    pd.DataFrame(
        {
            "metric": list(
                nuisance_metrics.keys()
            ),
            "value": list(
                nuisance_metrics.values()
            ),
        }
    ).to_csv(
        output_dir
        / "nuisance_test_performance.csv",
        index=False,
    )

    metadata = {
        "phase": "2B",
        "horizon_days": int(
            args.horizon
        ),
        "n_bins": int(
            args.n_bins
        ),
        "propensity_clip": float(
            args.propensity_clip
        ),
        "nuisance_training": (
            "train and validation only; test excluded from "
            "selection and fitting"
        ),
        "propensity_model_target": (
            "60-day outcome remains observable under hospital boundary"
        ),
        "outcome_model_training_population": (
            "patients whose 60-day outcome remains observable "
            "under hospital boundary"
        ),
        "nuisance_features": features,
        "propensity_best_c": float(
            propensity_model.best_c
        ),
        "outcome_model_best_c": float(
            outcome_model.best_c
        ),
        "bootstrap_type": (
            "patient bootstrap of untouched test cohort with "
            "held-out nuisance predictions fixed"
        ),
        "bootstrap_replicates": int(
            args.bootstrap_replicates
        ),
        "best_estimator": str(
            ranking.loc[
                0,
                "estimator",
            ]
        ),
        "status": status,
        "test_rows": int(
            len(test)
        ),
        "test_observed_rows": int(
            test["observed"].sum()
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
        "\nPhase 2B complete"
    )

    print(
        "\nPropensity model selection"
    )

    print(
        propensity_model.selection_table.to_string(
            index=False
        )
    )

    print(
        "\nOutcome model selection"
    )

    print(
        outcome_model.selection_table.to_string(
            index=False
        )
    )

    print(
        "\nEstimator ranking"
    )

    print(
        ranking.to_string(
            index=False
        )
    )

    print(
        "\nPositivity diagnostics"
    )

    print(
        point_result.diagnostics.to_string()
    )

    print(
        "\nBootstrap intervals"
    )

    selected_intervals = intervals.loc[
        intervals["metric"].str.contains(
            "mae_to_reference"
            "|improvement_over_complete_case"
        )
    ]

    print(
        selected_intervals.to_string(
            index=False
        )
    )

    print(
        f"\nPhase 2B status: {status}"
    )

    print(
        f"\nArtifacts saved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
