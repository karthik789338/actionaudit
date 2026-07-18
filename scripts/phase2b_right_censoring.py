"""Corrected Phase 2B: survival IPCW and survival outcome regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from actionaudit.discrete_survival import (
    TIME_FEATURES,
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
    expand_person_period,
    predict_hazard_grid,
)
from actionaudit.dr import tune_binary_logistic
from actionaudit.km import (
    make_hospital_restricted_survival,
    reference_quantile_edges,
)
from actionaudit.support2 import (
    build_support2_horizon_view,
    load_support2_csv,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        default="artifacts/phase2b_right_censoring",
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
        "--censor-survival-clip",
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


def prepare_data(
    raw_data: pd.DataFrame,
    phase1b_dir: Path,
    horizon: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    metadata = json.loads(
        (
            phase1b_dir
            / "experiment_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    features = list(
        metadata["included_features"]
    )

    split_manifest = pd.read_csv(
        phase1b_dir
        / "split_manifest.csv"
    )

    test_predictions = pd.read_csv(
        phase1b_dir
        / "test_predictions.csv"
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

    analysis = make_hospital_restricted_survival(
        analysis,
        horizon,
    )

    train = analysis.loc[
        analysis["split"].eq("train")
    ].copy()

    validation = analysis.loc[
        analysis["split"].eq(
            "validation"
        )
    ].copy()

    test = analysis.loc[
        analysis["split"].eq("test")
    ].copy()

    test = test.drop(
        columns=[
            "prediction",
        ],
        errors="ignore",
    )

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
            "Test cohort and frozen predictions disagree"
        )

    if "outcome_saved" in test.columns:
        test["outcome"] = (
            test["outcome_saved"]
            .astype(int)
        )

    if "observed_saved" in test.columns:
        test["observed"] = (
            test["observed_saved"]
            .astype(int)
        )

    missing_features = [
        feature
        for feature in features
        if feature not in analysis.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing features: {missing_features}"
        )

    return (
        train,
        validation,
        test,
        features,
    )


def bootstrap_estimators(
    frame: pd.DataFrame,
    *,
    horizon: int,
    bin_edges: np.ndarray,
    censor_survival_clip: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(frame)

    rows: list[dict[str, float]] = []

    for bootstrap_id in range(replicates):
        sample_index = rng.integers(
            0,
            n,
            size=n,
        )

        sample = (
            frame.iloc[
                sample_index
            ]
            .reset_index(drop=True)
        )

        result = evaluate_survival_estimators(
            sample,
            horizon=horizon,
            bin_edges=bin_edges,
            censor_survival_clip=(
                censor_survival_clip
            ),
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


def create_intervals(
    point_summary: pd.Series,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
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


def create_ranking(
    scalar_summary: pd.Series,
) -> pd.DataFrame:
    estimators = (
        "complete_case",
        "kaplan_meier",
        "survival_or",
        "ipcw_event",
    )

    rows: list[dict[str, object]] = []

    for estimator in estimators:
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
    best_estimator = str(
        ranking.loc[
            0,
            "estimator",
        ]
    )

    clipped_fraction = float(
        diagnostics[
            "fraction_event_g_clipped"
        ]
    )

    if clipped_fraction > 0.10:
        return (
            "severe_event_weight_positivity_concern"
        )

    if best_estimator == "ipcw_event":
        return (
            "conditional_censoring_ipcw_recovers_best"
        )

    if best_estimator == "survival_or":
        return (
            "survival_outcome_regression_recovers_best"
        )

    if best_estimator == "kaplan_meier":
        return (
            "kaplan_meier_remains_best: baseline covariate "
            "adjustment does not remove residual censoring bias"
        )

    return (
        "complete_case_unexpectedly_best: inspect implementation"
    )


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

    train, validation, test, features = prepare_data(
        raw_data,
        phase1b_dir,
        args.horizon,
    )

    model_features = (
        features
        + TIME_FEATURES
    )

    print(
        "Expanding train and validation patients "
        "into discrete at-risk intervals..."
    )

    train_event_rows = expand_person_period(
        train,
        features,
        horizon=args.horizon,
        target="event",
    )

    validation_event_rows = expand_person_period(
        validation,
        features,
        horizon=args.horizon,
        target="event",
    )

    train_censor_rows = expand_person_period(
        train,
        features,
        horizon=args.horizon,
        target="censor",
    )

    validation_censor_rows = expand_person_period(
        validation,
        features,
        horizon=args.horizon,
        target="censor",
    )

    print(
        f"Event hazard rows: "
        f"{len(train_event_rows):,} train and "
        f"{len(validation_event_rows):,} validation"
    )

    print(
        f"Censor hazard rows: "
        f"{len(train_censor_rows):,} train and "
        f"{len(validation_censor_rows):,} validation"
    )

    event_model = tune_binary_logistic(
        train_event_rows,
        train_event_rows[
            "hazard_target"
        ],
        validation_event_rows,
        validation_event_rows[
            "hazard_target"
        ],
        model_features,
        random_seed=args.seed + 30,
    )

    censor_model = tune_binary_logistic(
        train_censor_rows,
        train_censor_rows[
            "hazard_target"
        ],
        validation_censor_rows,
        validation_censor_rows[
            "hazard_target"
        ],
        model_features,
        random_seed=args.seed + 40,
    )

    print(
        "Predicting event and censoring hazards "
        "on the untouched test cohort..."
    )

    event_hazards = predict_hazard_grid(
        event_model.model,
        test,
        features,
        horizon=args.horizon,
    )

    censor_hazards = predict_hazard_grid(
        censor_model.model,
        test,
        features,
        horizon=args.horizon,
    )

    test = construct_survival_nuisance_predictions(
        test,
        event_hazards=event_hazards,
        censor_hazards=censor_hazards,
        horizon=args.horizon,
        censor_survival_clip=(
            args.censor_survival_clip
        ),
    )

    bin_edges = reference_quantile_edges(
        test[
            "prediction"
        ].to_numpy(),
        args.n_bins,
    )

    point_result = evaluate_survival_estimators(
        test,
        horizon=args.horizon,
        bin_edges=bin_edges,
        censor_survival_clip=(
            args.censor_survival_clip
        ),
    )

    print(
        f"Running {args.bootstrap_replicates} "
        "fixed-nuisance bootstrap replicates..."
    )

    bootstrap = bootstrap_estimators(
        test,
        horizon=args.horizon,
        bin_edges=bin_edges,
        censor_survival_clip=(
            args.censor_survival_clip
        ),
        replicates=args.bootstrap_replicates,
        seed=args.seed + 500,
    )

    intervals = create_intervals(
        point_result.scalar_summary,
        bootstrap,
    )

    ranking = create_ranking(
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
            / "event_weight_diagnostics.csv",
            index=False,
        )
    )

    event_model.selection_table.to_csv(
        output_dir
        / "event_hazard_model_selection.csv",
        index=False,
    )

    censor_model.selection_table.to_csv(
        output_dir
        / "censor_hazard_model_selection.csv",
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
            "restricted_censored_by_discharge",
            "survival_outcome_regression",
            "raw_censor_survival_before_observed_time",
            "ipcw_event_weight",
            "ipcw_event_contribution",
            "censor_survival_at_horizon",
        ]
    ].to_csv(
        output_dir
        / "test_survival_nuisance_predictions.csv",
        index=False,
    )

    joblib.dump(
        event_model.model,
        output_dir
        / "event_hazard_model.joblib",
    )

    joblib.dump(
        censor_model.model,
        output_dir
        / "censor_hazard_model.joblib",
    )

    metadata = {
        "phase": "2B-R",
        "estimator_family": (
            "right-censoring survival estimators"
        ),
        "horizon_days": int(
            args.horizon
        ),
        "n_bins": int(
            args.n_bins
        ),
        "time_discretization": (
            "daily intervals; events precede censoring "
            "within tied integer day"
        ),
        "ipcw_definition": (
            "observed in-hospital event weighted by "
            "1 / G(event time minus | baseline covariates)"
        ),
        "outcome_regression_definition": (
            "pooled logistic event hazard integrated through horizon"
        ),
        "censor_survival_clip": float(
            args.censor_survival_clip
        ),
        "event_hazard_best_c": float(
            event_model.best_c
        ),
        "censor_hazard_best_c": float(
            censor_model.best_c
        ),
        "event_hazard_train_rows": int(
            len(train_event_rows)
        ),
        "censor_hazard_train_rows": int(
            len(train_censor_rows)
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
        "aipcw_included": False,
        "aipcw_reason": (
            "survival-specific augmentation will be implemented "
            "only after the right-censoring influence-function "
            "estimator is mathematically verified"
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
        "\nCorrected Phase 2B-R complete"
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
        "\nEvent-weight diagnostics"
    )

    print(
        point_result.diagnostics.to_string()
    )

    print(
        "\nBootstrap intervals"
    )

    print(
        intervals.loc[
            intervals[
                "metric"
            ].str.contains(
                "mae_to_reference"
                "|improvement_over_complete_case"
            )
        ].to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"\nArtifacts saved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
