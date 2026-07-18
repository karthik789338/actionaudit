"""Phase 5G-D random-forest nuisance-family sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import phase5g_nonlinear_estimators as base


IDENTIFIERS = base.IDENTIFIERS
NUMERIC_FEATURES = base.NUMERIC_FEATURES
CATEGORICAL_FEATURES = base.CATEGORICAL_FEATURES
PREDICTION_FEATURES = base.PREDICTION_FEATURES
PROPENSITY_CLIPS = base.PROPENSITY_CLIPS


def clip_label(
    value: float,
) -> str:
    return base.clip_label(
        value
    )


def method_columns() -> list[str]:
    methods = list(
        base.method_columns()
    )

    methods.append(
        "rf_survival_or"
    )

    for clip in PROPENSITY_CLIPS:
        label = clip_label(
            clip
        )

        methods.extend(
            [
                f"rf_ipcw_clip_{label}",
                f"rf_aipw_clip_{label}",
            ]
        )

    return methods


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-forest nuisance-family "
            "sensitivity benchmark."
        )
    )

    parser.add_argument(
        "--rich-input",
        default=(
            "artifacts/phase5g_baseline_features/"
            "rich_test_analysis.csv.gz"
        ),
    )

    parser.add_argument(
        "--feature-manifest",
        default=(
            "artifacts/phase5g_baseline_features/"
            "baseline_feature_manifest.csv"
        ),
    )

    parser.add_argument(
        "--logistic-nuisance",
        default=(
            "artifacts/"
            "phase5g_rich_baseline_estimators/"
            "cross_fitted_rich_nuisance_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--score-summary",
        default=(
            "artifacts/phase5g_practitioner_estimators/"
            "estimator_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--rich-summary",
        default=(
            "artifacts/"
            "phase5g_rich_baseline_estimators/"
            "rich_estimator_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--hgb-nuisance",
        default=(
            "artifacts/"
            "phase5g_nonlinear_estimators/"
            "cross_fitted_hgb_nuisance_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--hgb-summary",
        default=(
            "artifacts/"
            "phase5g_nonlinear_estimators/"
            "nonlinear_estimator_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5g_random_forest_estimators"
        ),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--crossfit-repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-samples",
        type=float,
        default=0.80,
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


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="MISSING",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                NUMERIC_FEATURES
                + PREDICTION_FEATURES,
            ),
            (
                "categorical",
                categorical,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_rf_model(
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_samples: float,
    seed: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="log_loss",
        max_depth=max_depth,
        min_samples_leaf=(
            min_samples_leaf
        ),
        max_features="sqrt",
        bootstrap=True,
        max_samples=max_samples,
        class_weight=None,
        n_jobs=-1,
        random_state=seed,
    )


def load_inputs(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    (
        data,
        manifest,
        score_summary,
        rich_summary,
        edges,
    ) = base.load_inputs(
        rich_path=Path(
            args.rich_input
        ),
        manifest_path=Path(
            args.feature_manifest
        ),
        nuisance_path=Path(
            args.logistic_nuisance
        ),
        score_summary_path=Path(
            args.score_summary
        ),
        rich_summary_path=Path(
            args.rich_summary
        ),
        n_bins=args.n_bins,
    )

    hgb_nuisance = pd.read_csv(
        args.hgb_nuisance,
        low_memory=False,
    )

    required = (
        IDENTIFIERS
        + [
            "hgb_outcome_regression_risk",
            "hgb_observation_probability",
        ]
    )

    base.require_columns(
        hgb_nuisance,
        required,
        label="Frozen HGB nuisance input",
    )

    base.validate_unique(
        hgb_nuisance,
        label="Frozen HGB nuisance input",
    )

    data = data.merge(
        hgb_nuisance[
            required
        ],
        on=IDENTIFIERS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        data[
            "_merge"
        ].ne(
            "both"
        ).sum()
    )

    if unmatched:
        raise ValueError(
            f"{unmatched} admissions failed "
            "the HGB nuisance join"
        )

    data = data.drop(
        columns=[
            "_merge",
        ]
    )

    hgb_summary = pd.read_csv(
        args.hgb_summary
    )

    return (
        data.reset_index(
            drop=True
        ),
        manifest,
        score_summary,
        rich_summary,
        hgb_summary,
        edges,
    )


def cross_fitted_rf_predictions(
    data: pd.DataFrame,
    *,
    horizon: int,
    repeats: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_samples: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    n = len(
        data
    )

    outcome_sum = np.zeros(
        n,
        dtype=float,
    )

    observation_sum = np.zeros(
        n,
        dtype=float,
    )

    prediction_count = np.zeros(
        n,
        dtype=int,
    )

    diagnostics = []

    model_columns = (
        NUMERIC_FEATURES
        + PREDICTION_FEATURES
        + CATEGORICAL_FEATURES
    )

    for repeat in range(
        repeats
    ):
        folds = base.assign_subject_folds(
            data[
                "subject_id"
            ],
            seed=(
                seed
                + 1009
                * repeat
            ),
        )

        for held_out_fold in [
            0,
            1,
        ]:
            train_mask = (
                folds
                != held_out_fold
            )

            test_mask = (
                folds
                == held_out_fold
            )

            train = data.loc[
                train_mask
            ].reset_index(
                drop=True
            )

            test = data.loc[
                test_mask
            ].reset_index(
                drop=True
            )

            preprocessor = (
                make_preprocessor()
            )

            train_baseline = (
                preprocessor.fit_transform(
                    train[
                        model_columns
                    ]
                )
            )

            test_baseline = (
                preprocessor.transform(
                    test[
                        model_columns
                    ]
                )
            )

            (
                survival_design,
                survival_outcome,
            ) = base.build_person_period_design(
                train,
                train_baseline,
                horizon=horizon,
            )

            survival_model = make_rf_model(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=(
                    min_samples_leaf
                ),
                max_samples=max_samples,
                seed=(
                    seed
                    + repeat
                    * 100
                    + held_out_fold
                ),
            )

            survival_model.fit(
                survival_design,
                survival_outcome,
            )

            test_outcome_risk = (
                base.predict_horizon_risk(
                    survival_model,
                    test,
                    test_baseline,
                    horizon=horizon,
                )
            )

            observation_model = make_rf_model(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=(
                    min_samples_leaf
                ),
                max_samples=max_samples,
                seed=(
                    seed
                    + 5000
                    + repeat
                    * 100
                    + held_out_fold
                ),
            )

            observation_model.fit(
                train_baseline,
                train[
                    "observed"
                ].to_numpy(
                    dtype=int
                ),
            )

            test_observation_probability = (
                observation_model.predict_proba(
                    test_baseline
                )[
                    :,
                    1
                ]
            )

            test_observation_probability = (
                np.clip(
                    test_observation_probability,
                    1e-6,
                    1.0 - 1e-6,
                )
            )

            original_indices = np.flatnonzero(
                test_mask
            )

            outcome_sum[
                original_indices
            ] += test_outcome_risk

            observation_sum[
                original_indices
            ] += test_observation_probability

            prediction_count[
                original_indices
            ] += 1

            diagnostics.append(
                {
                    "repeat": repeat,
                    "held_out_fold": (
                        held_out_fold
                    ),
                    "n_train": len(
                        train
                    ),
                    "n_test": len(
                        test
                    ),
                    "baseline_columns": int(
                        train_baseline.shape[
                            1
                        ]
                    ),
                    "person_period_rows": int(
                        survival_design.shape[
                            0
                        ]
                    ),
                    "person_period_columns": int(
                        survival_design.shape[
                            1
                        ]
                    ),
                    "restricted_events_train": int(
                        train[
                            "restricted_event"
                        ].sum()
                    ),
                    "observed_train": int(
                        train[
                            "observed"
                        ].sum()
                    ),
                    "survival_trees": len(
                        survival_model.estimators_
                    ),
                    "observation_trees": len(
                        observation_model.estimators_
                    ),
                    "test_risk_min": float(
                        test_outcome_risk.min()
                    ),
                    "test_risk_max": float(
                        test_outcome_risk.max()
                    ),
                    "test_observation_probability_min": float(
                        test_observation_probability.min()
                    ),
                    "test_observation_probability_max": float(
                        test_observation_probability.max()
                    ),
                }
            )

    if not np.all(
        prediction_count
        == repeats
    ):
        raise RuntimeError(
            "Repeated RF cross-fitting did not "
            "cover every admission"
        )

    return (
        outcome_sum
        / prediction_count,
        observation_sum
        / prediction_count,
        prediction_count,
        pd.DataFrame(
            diagnostics
        ),
    )


def estimate_bin_curves(
    data: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    curves = base.estimate_bin_curves(
        data,
        horizon=horizon,
    )

    for row_index, row in curves.iterrows():
        risk_bin = int(
            row[
                "risk_bin"
            ]
        )

        group = data.loc[
            data[
                "risk_bin"
            ].eq(
                risk_bin
            )
        ]

        curves.loc[
            row_index,
            "rf_survival_or",
        ] = float(
            group[
                "rf_outcome_regression_risk"
            ].mean()
        )

        for clip in PROPENSITY_CLIPS:
            label = clip_label(
                clip
            )

            (
                rf_ipcw,
                rf_aipw,
                rf_aipw_raw,
            ) = base.weighted_estimators(
                group,
                outcome_model_column=(
                    "rf_outcome_regression_risk"
                ),
                propensity_column=(
                    "rf_observation_probability"
                ),
                clip=clip,
            )

            curves.loc[
                row_index,
                f"rf_ipcw_clip_{label}",
            ] = rf_ipcw

            curves.loc[
                row_index,
                f"rf_aipw_clip_{label}",
            ] = rf_aipw

            curves.loc[
                row_index,
                f"rf_aipw_raw_clip_{label}",
            ] = rf_aipw_raw

    return curves


def calculate_metrics(
    curves: pd.DataFrame,
) -> pd.DataFrame:
    weights = curves[
        "reference_weight"
    ].to_numpy(
        dtype=float
    )

    weights = (
        weights
        / weights.sum()
    )

    prediction = curves[
        "mean_prediction"
    ].to_numpy(
        dtype=float
    )

    reference = curves[
        "reference_oracle"
    ].to_numpy(
        dtype=float
    )

    reference_event_rate = float(
        np.sum(
            weights
            * reference
        )
    )

    reference_ici = float(
        np.sum(
            weights
            * np.abs(
                reference
                - prediction
            )
        )
    )

    rows = []

    for method in method_columns():
        estimate = curves[
            method
        ].to_numpy(
            dtype=float
        )

        estimated_event_rate = float(
            np.sum(
                weights
                * estimate
            )
        )

        estimated_ici = float(
            np.sum(
                weights
                * np.abs(
                    estimate
                    - prediction
                )
            )
        )

        rows.append(
            {
                "method": method,
                "estimated_event_rate": (
                    estimated_event_rate
                ),
                "reference_event_rate": (
                    reference_event_rate
                ),
                "event_rate_error": abs(
                    estimated_event_rate
                    - reference_event_rate
                ),
                "estimated_ici": (
                    estimated_ici
                ),
                "reference_ici": (
                    reference_ici
                ),
                "ici_error": abs(
                    estimated_ici
                    - reference_ici
                ),
                "bin_curve_mae": float(
                    np.sum(
                        weights
                        * np.abs(
                            estimate
                            - reference
                        )
                    )
                ),
                "signed_curve_error": float(
                    np.sum(
                        weights
                        * (
                            estimate
                            - reference
                        )
                    )
                ),
                "maximum_bin_error": float(
                    np.max(
                        np.abs(
                            estimate
                            - reference
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def run_bootstrap(
    data: pd.DataFrame,
    *,
    horizon: int,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    rows = []
    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        indices = (
            base.cluster_bootstrap_indices(
                data,
                rng,
            )
        )

        sample = (
            data.loc[
                indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        try:
            metrics = calculate_metrics(
                estimate_bin_curves(
                    sample,
                    horizon=horizon,
                )
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            invalid += 1
            continue

        for _, row in metrics.iterrows():
            rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    **row.to_dict(),
                }
            )

        if (
            bootstrap_id
            + 1
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


def paired_intervals(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pairs = [
        (
            "rf_survival_or",
            "score_pooled_survival_or",
        ),
        (
            "rf_survival_or",
            "rich_logistic_survival_or",
        ),
        (
            "rf_survival_or",
            "hgb_survival_or",
        ),
        (
            "rf_survival_or",
            "grouped_km",
        ),
    ]

    for clip in PROPENSITY_CLIPS:
        label = clip_label(
            clip
        )

        pairs.extend(
            [
                (
                    f"rf_ipcw_clip_{label}",
                    "score_ipcw",
                ),
                (
                    f"rf_ipcw_clip_{label}",
                    (
                        "rich_logistic_ipcw_"
                        f"clip_{label}"
                    ),
                ),
                (
                    f"rf_ipcw_clip_{label}",
                    f"hgb_ipcw_clip_{label}",
                ),
                (
                    f"rf_ipcw_clip_{label}",
                    "restricted_complete_case",
                ),
                (
                    f"rf_aipw_clip_{label}",
                    "score_aipw",
                ),
                (
                    f"rf_aipw_clip_{label}",
                    (
                        "rich_logistic_aipw_"
                        f"clip_{label}"
                    ),
                ),
                (
                    f"rf_aipw_clip_{label}",
                    f"hgb_aipw_clip_{label}",
                ),
                (
                    f"rf_aipw_clip_{label}",
                    "restricted_complete_case",
                ),
            ]
        )

    comparison_metrics = [
        "bin_curve_mae",
        "event_rate_error",
        "ici_error",
        "maximum_bin_error",
    ]

    lookup = metrics.set_index(
        "method"
    )

    point_rows = []
    interval_rows = []

    for method, comparator in pairs:
        for metric in comparison_metrics:
            point_rows.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(
                        lookup.loc[
                            method,
                            metric,
                        ]
                        - lookup.loc[
                            comparator,
                            metric,
                        ]
                    ),
                }
            )

            left = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        method
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: "method_value",
                    }
                )
            )

            right = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        comparator
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: (
                            "comparator_value"
                        ),
                    }
                )
            )

            paired = left.merge(
                right,
                on="bootstrap_id",
                validate="one_to_one",
            )

            difference = (
                paired[
                    "method_value"
                ]
                - paired[
                    "comparator_value"
                ]
            )

            interval_rows.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "metric": metric,
                    "q025": float(
                        difference.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        difference.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        difference.quantile(
                            0.975
                        )
                    ),
                    "n_valid": int(
                        difference.notna().sum()
                    ),
                }
            )

    return (
        pd.DataFrame(
            point_rows
        ),
        pd.DataFrame(
            interval_rows
        ),
    )


def positivity_diagnostics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    observed = data[
        "observed"
    ].to_numpy(
        dtype=float
    )

    raw = data[
        "rf_observation_probability"
    ].to_numpy(
        dtype=float
    )

    for clip in PROPENSITY_CLIPS:
        propensity = np.clip(
            raw,
            clip,
            1.0 - clip,
        )

        weight = (
            observed
            / propensity
        )

        rows.append(
            {
                "propensity_clip": clip,
                "raw_minimum_probability": float(
                    raw.min()
                ),
                "raw_q01_probability": float(
                    np.quantile(
                        raw,
                        0.01,
                    )
                ),
                "raw_median_probability": float(
                    np.median(
                        raw
                    )
                ),
                "maximum_inverse_weight": float(
                    1.0
                    / propensity.min()
                ),
                "weighted_observed_effective_sample_size": float(
                    weight.sum() ** 2
                    / (
                        np.sum(
                            weight**2
                        )
                        + 1e-12
                    )
                ),
                "fraction_clipped_low": float(
                    np.mean(
                        raw
                        < clip
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_validation_checks(
    *,
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    score_summary: pd.DataFrame,
    rich_summary: pd.DataFrame,
    hgb_summary: pd.DataFrame,
    curves: pd.DataFrame,
    diagnostics: pd.DataFrame,
    prediction_count: np.ndarray,
    repeats: int,
    n_estimators: int,
    bootstrap_rows: int,
    requested_bootstraps: int,
    invalid_bootstraps: int,
) -> pd.DataFrame:
    score_mapping = {
        "pooled_survival_or": (
            "score_pooled_survival_or"
        ),
        "score_ipcw": "score_ipcw",
        "score_aipw": "score_aipw",
        "grouped_km": "grouped_km",
        "restricted_complete_case": (
            "restricted_complete_case"
        ),
    }

    rich_mapping = {
        "rich_pooled_survival_or": (
            "rich_logistic_survival_or"
        ),
        "rich_ipcw_clip_0p01": (
            "rich_logistic_ipcw_clip_0p01"
        ),
        "rich_ipcw_clip_0p025": (
            "rich_logistic_ipcw_clip_0p025"
        ),
        "rich_ipcw_clip_0p05": (
            "rich_logistic_ipcw_clip_0p05"
        ),
        "rich_aipw_clip_0p01": (
            "rich_logistic_aipw_clip_0p01"
        ),
        "rich_aipw_clip_0p025": (
            "rich_logistic_aipw_clip_0p025"
        ),
        "rich_aipw_clip_0p05": (
            "rich_logistic_aipw_clip_0p05"
        ),
    }

    hgb_mapping = {
        method: method
        for method in base.method_columns()
    }

    score_error = base.maximum_reproduction_error(
        metrics,
        score_summary,
        score_mapping,
    )

    rich_error = base.maximum_reproduction_error(
        metrics,
        rich_summary,
        rich_mapping,
    )

    hgb_error = base.maximum_reproduction_error(
        metrics,
        hgb_summary,
        hgb_mapping,
    )

    expected_rows = (
        requested_bootstraps
        * len(
            method_columns()
        )
    )

    tree_budget_passed = bool(
        len(
            diagnostics
        )
        == repeats
        * 2
        and diagnostics[
            "survival_trees"
        ].eq(
            n_estimators
        ).all()
        and diagnostics[
            "observation_trees"
        ].eq(
            n_estimators
        ).all()
    )

    minimum_curve = float(
        curves[
            method_columns()
        ].min().min()
    )

    maximum_curve = float(
        curves[
            method_columns()
        ].max().max()
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "all RF cross-fitted "
                    "predictions complete"
                ),
                "value": int(
                    prediction_count.min()
                ),
                "threshold": repeats,
                "comparison": "==",
                "passed": bool(
                    np.all(
                        prediction_count
                        == repeats
                    )
                ),
            },
            {
                "check": (
                    "RF outcome risks within bounds"
                ),
                "value": (
                    f"{data['rf_outcome_regression_risk'].min()},"
                    f"{data['rf_outcome_regression_risk'].max()}"
                ),
                "threshold": "[0,1]",
                "comparison": "within",
                "passed": bool(
                    data[
                        "rf_outcome_regression_risk"
                    ]
                    .between(
                        0.0,
                        1.0,
                    )
                    .all()
                ),
            },
            {
                "check": (
                    "RF observation probabilities "
                    "strictly within bounds"
                ),
                "value": (
                    f"{data['rf_observation_probability'].min()},"
                    f"{data['rf_observation_probability'].max()}"
                ),
                "threshold": "(0,1)",
                "comparison": "within",
                "passed": bool(
                    (
                        data[
                            "rf_observation_probability"
                        ]
                        > 0.0
                    ).all()
                    and (
                        data[
                            "rf_observation_probability"
                        ]
                        < 1.0
                    ).all()
                ),
            },
            {
                "check": (
                    "all RF models completed "
                    "fixed tree budget"
                ),
                "value": int(
                    tree_budget_passed
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": tree_budget_passed,
            },
            {
                "check": (
                    "score benchmark reproduced"
                ),
                "value": score_error,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    score_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "rich logistic benchmark reproduced"
                ),
                "value": rich_error,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    rich_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "HGB benchmark reproduced"
                ),
                "value": hgb_error,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": (
                    hgb_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all curves within bounds"
                ),
                "value": (
                    f"{minimum_curve},"
                    f"{maximum_curve}"
                ),
                "threshold": "[0,1]",
                "comparison": "within",
                "passed": (
                    minimum_curve
                    >= 0.0
                    and maximum_curve
                    <= 1.0
                ),
            },
            {
                "check": (
                    "all bootstrap rows completed"
                ),
                "value": bootstrap_rows,
                "threshold": expected_rows,
                "comparison": "==",
                "passed": (
                    bootstrap_rows
                    == expected_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        data,
        manifest,
        score_summary,
        rich_summary,
        hgb_summary,
        bin_edges,
    ) = load_inputs(
        args
    )

    print(
        "Phase 5G-D random-forest sensitivity"
    )

    print(
        f"  admissions={len(data):,}"
    )

    print(
        f"  cross-fit repeats="
        f"{args.crossfit_repeats}"
    )

    print(
        f"  trees per model="
        f"{args.n_estimators}"
    )

    print(
        "\nFitting repeated cross-fitted "
        "random-forest nuisance models..."
    )

    (
        rf_outcome_risk,
        rf_observation_probability,
        prediction_count,
        diagnostics,
    ) = cross_fitted_rf_predictions(
        data,
        horizon=args.horizon,
        repeats=args.crossfit_repeats,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=(
            args.min_samples_leaf
        ),
        max_samples=args.max_samples,
        seed=args.seed,
    )

    data[
        "rf_outcome_regression_risk"
    ] = rf_outcome_risk

    data[
        "rf_observation_probability"
    ] = rf_observation_probability

    curves = estimate_bin_curves(
        data,
        horizon=args.horizon,
    )

    metrics = calculate_metrics(
        curves
    )

    ranking = (
        metrics.loc[
            ~metrics[
                "method"
            ].eq(
                "reference_oracle"
            )
        ]
        .sort_values(
            [
                "bin_curve_mae",
                "event_rate_error",
                "ici_error",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "rank"
    ] = np.arange(
        1,
        len(
            ranking
        )
        + 1,
    )

    print(
        "\nRunning conditional paired "
        "patient bootstrap..."
    )

    (
        bootstrap,
        invalid_bootstraps,
    ) = run_bootstrap(
        data,
        horizon=args.horizon,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
    )

    bootstrap_intervals = (
        base.summarize_intervals(
            bootstrap
        )
    )

    (
        point_differences,
        difference_intervals,
    ) = paired_intervals(
        metrics,
        bootstrap,
    )

    positivity = positivity_diagnostics(
        data
    )

    checks = build_validation_checks(
        data=data,
        metrics=metrics,
        score_summary=score_summary,
        rich_summary=rich_summary,
        hgb_summary=hgb_summary,
        curves=curves,
        diagnostics=diagnostics,
        prediction_count=(
            prediction_count
        ),
        repeats=args.crossfit_repeats,
        n_estimators=args.n_estimators,
        bootstrap_rows=len(
            bootstrap
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
        invalid_bootstraps=(
            invalid_bootstraps
        ),
    )

    preliminary_status = (
        "phase5g_random_forest_estimators_benchmarked"
        if checks[
            "passed"
        ].all()
        else (
            "phase5g_random_forest_estimators_"
            "require_review"
        )
    )

    status = (
        "phase5g_random_forest_estimators_smoke_completed"
        if args.smoke
        else preliminary_status
    )

    data[
        IDENTIFIERS
        + [
            "prediction",
            "risk_bin",
            "observed",
            "restricted_event",
            "restricted_time",
            "rf_outcome_regression_risk",
            "rf_observation_probability",
        ]
    ].to_csv(
        output_dir
        / "cross_fitted_rf_nuisance_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    curves.to_csv(
        output_dir
        / "random_forest_bin_curves.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "random_forest_metric_summary.csv",
        index=False,
    )

    ranking.to_csv(
        output_dir
        / "random_forest_ranking.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_metric_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_intervals.to_csv(
        output_dir
        / "bootstrap_metric_intervals.csv",
        index=False,
    )

    point_differences.to_csv(
        output_dir
        / "paired_method_point_differences.csv",
        index=False,
    )

    difference_intervals.to_csv(
        output_dir
        / "paired_method_difference_intervals.csv",
        index=False,
    )

    positivity.to_csv(
        output_dir
        / "rf_positivity_diagnostics.csv",
        index=False,
    )

    diagnostics.to_csv(
        output_dir
        / "rf_crossfit_fold_diagnostics.csv",
        index=False,
    )

    checks.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    evidence = {
        "phase": "5G-D",
        "status": status,
        "family": (
            "RandomForestClassifier"
        ),
        "n_estimators": (
            args.n_estimators
        ),
        "max_depth": args.max_depth,
        "min_samples_leaf": (
            args.min_samples_leaf
        ),
        "max_samples": (
            args.max_samples
        ),
        "crossfit_repeats": (
            args.crossfit_repeats
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "purpose": (
            "Model-family sensitivity only; "
            "not reference-outcome model selection."
        ),
        "oracle_use": (
            "Linked outcomes used only for "
            "retrospective evaluation."
        ),
    }

    (
        output_dir
        / "random_forest_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5g_random_forest_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nEstimator ranking"
    )

    print(
        ranking[
            [
                "rank",
                "method",
                "bin_curve_mae",
                "event_rate_error",
                "maximum_bin_error",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRF paired curve-MAE intervals"
    )

    print(
        difference_intervals.loc[
            difference_intervals[
                "metric"
            ].eq(
                "bin_curve_mae"
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nRF positivity"
    )

    print(
        positivity.to_string(
            index=False
        )
    )

    print(
        "\nValidation"
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


if __name__ == "__main__":
    main()
