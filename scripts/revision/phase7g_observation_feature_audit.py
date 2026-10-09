from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------
# Reuse the exact Phase 5C model construction.
# ---------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import phase5c_train_admission_model as phase5c


ID_COLUMNS = [
    "subject_id",
    "hadm_id",
]

PRIOR_UTILIZATION_FEATURES = [
    "prior_admission_count",
    "prior_emergency_or_urgent_count",
    "prior_total_los_days",
    "days_since_previous_discharge",
    "prior_icu_stay_count",
    "prior_icu_los_days",
    "has_prior_admission",
    "has_prior_icu",
]

TEMPORAL_PROCESS_FEATURES = [
    "admission_hour_sin",
    "admission_hour_cos",
    "admission_weekday_sin",
    "admission_weekday_cos",
]

ADMISSION_PROCESS_FEATURES = [
    "admission_type",
    "admission_location",
]

OBSERVATION_SCORE_NAME = (
    "observation_score"
)

REFERENCE_BINS = 5


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7G observation-process-as-feature "
            "audit for frozen MIMIC-IV benchmark."
        )
    )

    parser.add_argument(
        "--cohort",
        default=(
            "artifacts/phase5c_frozen_cohort/"
            "primary_inpatient_cohort.csv.gz"
        ),
    )

    parser.add_argument(
        "--mimic-root",
        default="data/raw/mimiciv/3.1",
    )

    parser.add_argument(
        "--phase5c-dir",
        default=(
            "artifacts/"
            "phase5c_admission_model"
        ),
    )

    parser.add_argument(
        "--phase5g-summary",
        default=(
            "artifacts/"
            "phase5g_practitioner_estimators/"
            "estimator_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        default=(
            "revision_kdd2027/"
            "phase7g_shared_cache"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "revision_kdd2027/"
            "phase7g_observation_feature_audit"
        ),
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--observation-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def build_phase5c_model(
    *,
    family,
    parameters,
    seed,
    numeric_features,
    categorical_features,
):
    old_numeric = list(
        phase5c.NUMERIC_FEATURES
    )

    old_categorical = list(
        phase5c.CATEGORICAL_FEATURES
    )

    try:
        phase5c.NUMERIC_FEATURES = list(
            numeric_features
        )

        phase5c.CATEGORICAL_FEATURES = list(
            categorical_features
        )

        model = phase5c.construct_model(
            family=family,
            parameters=parameters,
            seed=seed,
        )

    finally:
        phase5c.NUMERIC_FEATURES = (
            old_numeric
        )

        phase5c.CATEGORICAL_FEATURES = (
            old_categorical
        )

    return model


def load_metadata(
    phase5c_dir: Path,
):
    return json.loads(
        (
            phase5c_dir
            / "experiment_metadata.json"
        ).read_text()
    )


def build_analysis_frame(
    *,
    cohort_path,
    mimic_root,
    split_manifest_path,
):
    print(
        "Constructing exact Phase 5C "
        "admission-time feature frame..."
    )

    frame = (
        phase5c.construct_feature_frame(
            cohort_path=cohort_path,
            mimic_root=mimic_root,
        )
    )

    split = pd.read_csv(
        split_manifest_path
    )

    split_small = split[
        [
            "subject_id",
            "hadm_id",
            "split",
            "outcome",
        ]
    ].rename(
        columns={
            "outcome":
            "_manifest_outcome"
        }
    )

    frame = frame.merge(
        split_small,
        on=[
            "subject_id",
            "hadm_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    mismatch = int(
        (
            frame[
                "outcome"
            ].astype(int)
            != frame[
                "_manifest_outcome"
            ].astype(int)
        ).sum()
    )

    if mismatch:
        raise RuntimeError(
            "Outcome mismatch between "
            "feature frame and frozen split "
            f"manifest: {mismatch}"
        )

    frame = frame.drop(
        columns=[
            "_manifest_outcome"
        ]
    )

    return frame


def fit_observation_score(
    development,
    test,
    *,
    numeric_features,
    categorical_features,
    folds,
    seed,
):
    print(
        "Fitting cross-fitted "
        "observation-process score..."
    )

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    oof = np.full(
        len(development),
        np.nan,
        dtype=float,
    )

    features = (
        list(numeric_features)
        + list(categorical_features)
    )

    groups = development[
        "subject_id"
    ].to_numpy()

    target = development[
        "observed"
    ].astype(int).to_numpy()

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        splitter.split(
            development,
            target,
            groups=groups,
        )
    ):
        print(
            f"  observation fold "
            f"{fold + 1}/{folds}"
        )

        model = build_phase5c_model(
            family="logistic",
            parameters={
                "C": 1.0
            },
            seed=(
                seed
                + 5000
                + fold
            ),
            numeric_features=(
                numeric_features
            ),
            categorical_features=(
                categorical_features
            ),
        )

        model.fit(
            development.iloc[
                train_index
            ][
                features
            ],
            development.iloc[
                train_index
            ][
                "observed"
            ].astype(int),
        )

        oof[
            validation_index
        ] = model.predict_proba(
            development.iloc[
                validation_index
            ][
                features
            ]
        )[:, 1]

    if not np.isfinite(
        oof
    ).all():
        raise RuntimeError(
            "Incomplete OOF observation scores."
        )

    final_model = build_phase5c_model(
        family="logistic",
        parameters={
            "C": 1.0
        },
        seed=seed + 6000,
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
    )

    final_model.fit(
        development[
            features
        ],
        development[
            "observed"
        ].astype(int),
    )

    test_score = (
        final_model.predict_proba(
            test[
                features
            ]
        )[:, 1]
    )

    return (
        oof,
        test_score,
    )


def fit_mortality_model(
    development,
    test,
    *,
    family,
    parameters,
    seed,
    numeric_features,
    categorical_features,
):
    features = (
        list(numeric_features)
        + list(categorical_features)
    )

    model = build_phase5c_model(
        family=family,
        parameters=parameters,
        seed=seed,
        numeric_features=(
            numeric_features
        ),
        categorical_features=(
            categorical_features
        ),
    )

    model.fit(
        development[
            features
        ],
        development[
            "outcome"
        ].astype(int),
    )

    probability = (
        model.predict_proba(
            test[
                features
            ]
        )[:, 1]
    )

    return (
        model,
        probability,
    )


def quantile_edges(
    prediction,
    n_bins=REFERENCE_BINS,
):
    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        prediction,
        quantiles,
    )

    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    return edges


def calibration_metrics(
    outcome,
    observed,
    prediction,
    *,
    n_bins=REFERENCE_BINS,
):
    outcome = np.asarray(
        outcome,
        dtype=float,
    )

    observed = np.asarray(
        observed,
        dtype=int,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    edges = quantile_edges(
        prediction,
        n_bins=n_bins,
    )

    bin_index = np.digitize(
        prediction,
        edges[
            1:-1
        ],
        right=True,
    )

    linked_ici = 0.0
    restricted_ici = 0.0

    rows = []

    for risk_bin in range(
        n_bins
    ):
        mask = (
            bin_index
            == risk_bin
        )

        n = int(
            mask.sum()
        )

        if n == 0:
            raise RuntimeError(
                "Empty reference risk bin."
            )

        observed_mask = (
            mask
            & (
                observed
                == 1
            )
        )

        n_observed = int(
            observed_mask.sum()
        )

        if n_observed == 0:
            raise RuntimeError(
                "No visible outcomes in "
                f"risk bin {risk_bin}."
            )

        weight = (
            n
            / len(
                prediction
            )
        )

        prediction_mean = float(
            prediction[
                mask
            ].mean()
        )

        linked_rate = float(
            outcome[
                mask
            ].mean()
        )

        restricted_rate = float(
            outcome[
                observed_mask
            ].mean()
        )

        linked_gap = abs(
            linked_rate
            - prediction_mean
        )

        restricted_gap = abs(
            restricted_rate
            - prediction_mean
        )

        linked_ici += (
            weight
            * linked_gap
        )

        restricted_ici += (
            weight
            * restricted_gap
        )

        rows.append(
            {
                "risk_bin":
                    risk_bin + 1,
                "n": n,
                "n_observed":
                    n_observed,
                "weight":
                    weight,
                "prediction_mean":
                    prediction_mean,
                "linked_event_rate":
                    linked_rate,
                "restricted_event_rate":
                    restricted_rate,
            }
        )

    return {
        "linked_ici": float(
            linked_ici
        ),
        "restricted_ici": float(
            restricted_ici
        ),
        "distortion_gap": float(
            restricted_ici
            - linked_ici
        ),
        "bin_table": pd.DataFrame(
            rows
        ),
    }


def model_metrics(
    frame,
    prediction,
):
    y = frame[
        "outcome"
    ].astype(int).to_numpy()

    observed = frame[
        "observed"
    ].astype(int).to_numpy()

    prediction = np.clip(
        np.asarray(
            prediction,
            dtype=float,
        ),
        1e-9,
        1.0 - 1e-9,
    )

    calibration = (
        calibration_metrics(
            y,
            observed,
            prediction,
        )
    )

    return {
        "log_loss": float(
            log_loss(
                y,
                prediction,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                prediction,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                prediction,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y,
                prediction,
            )
        ),
        "mean_prediction": float(
            prediction.mean()
        ),
        "event_rate": float(
            y.mean()
        ),
        "linked_ici": (
            calibration[
                "linked_ici"
            ]
        ),
        "restricted_ici": (
            calibration[
                "restricted_ici"
            ]
        ),
        "distortion_gap": (
            calibration[
                "distortion_gap"
            ]
        ),
    }


def feature_role_table(
    full_numeric,
    full_categorical,
):
    rows = []

    for feature in (
        full_numeric
        + full_categorical
    ):
        if feature in (
            PRIOR_UTILIZATION_FEATURES
        ):
            role = (
                "prior_utilization_"
                "proxy_candidate"
            )

        elif feature in (
            TEMPORAL_PROCESS_FEATURES
        ):
            role = (
                "admission_timing_"
                "proxy_candidate"
            )

        elif feature in (
            ADMISSION_PROCESS_FEATURES
        ):
            role = (
                "admission_process_"
                "proxy_candidate"
            )

        else:
            role = (
                "retained_nonprocess_"
                "admission_feature"
            )

        rows.append(
            {
                "feature":
                    feature,
                "role":
                    role,
                "in_frozen_model":
                    True,
            }
        )

    return pd.DataFrame(
        rows
    )


def observation_diagnostics(
    test,
    observation_score,
    frozen_prediction,
):
    observed = test[
        "observed"
    ].astype(int).to_numpy()

    observation_score = np.clip(
        observation_score,
        1e-9,
        1.0 - 1e-9,
    )

    return pd.DataFrame(
        [
            {
                "metric":
                    "observation_score_auc",
                "value": float(
                    roc_auc_score(
                        observed,
                        observation_score,
                    )
                ),
            },
            {
                "metric":
                    "observation_score_"
                    "average_precision",
                "value": float(
                    average_precision_score(
                        observed,
                        observation_score,
                    )
                ),
            },
            {
                "metric":
                    "observation_score_brier",
                "value": float(
                    brier_score_loss(
                        observed,
                        observation_score,
                    )
                ),
            },
            {
                "metric":
                    "observation_score_"
                    "log_loss",
                "value": float(
                    log_loss(
                        observed,
                        observation_score,
                    )
                ),
            },
            {
                "metric":
                    "mean_score_observed",
                "value": float(
                    observation_score[
                        observed == 1
                    ].mean()
                ),
            },
            {
                "metric":
                    "mean_score_hidden",
                "value": float(
                    observation_score[
                        observed == 0
                    ].mean()
                ),
            },
            {
                "metric":
                    "mortality_prediction_"
                    "observation_auc",
                "value": float(
                    roc_auc_score(
                        observed,
                        frozen_prediction,
                    )
                ),
            },
            {
                "metric":
                    "correlation_observation_"
                    "score_with_mortality_"
                    "prediction",
                "value": float(
                    np.corrcoef(
                        observation_score,
                        frozen_prediction,
                    )[
                        0,
                        1,
                    ]
                ),
            },
        ]
    )


def patient_bootstrap_indices(
    frame,
    rng,
):
    grouped = [
        group.index.to_numpy(
            dtype=int
        )
        for _, group in (
            frame.groupby(
                "subject_id",
                sort=False,
            )
        )
    ]

    draw = rng.integers(
        0,
        len(grouped),
        size=len(grouped),
    )

    return np.concatenate(
        [
            grouped[index]
            for index in draw
        ]
    )


def run_bootstrap(
    predictions,
    *,
    replicates,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    model_columns = {
        "frozen_full":
            "prediction_frozen_full",
        "no_prior_utilization":
            "prediction_no_prior_utilization",
        "no_operational_proxies":
            "prediction_no_operational_proxies",
        "observation_score_augmented":
            "prediction_observation_score_augmented",
    }

    rows = []

    for bootstrap_id in range(
        replicates
    ):
        if (
            bootstrap_id == 0
            or (
                bootstrap_id + 1
            ) % 100
            == 0
        ):
            print(
                "  bootstrap "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

        indices = (
            patient_bootstrap_indices(
                predictions,
                rng,
            )
        )

        sample = (
            predictions.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        )

        for model_name, column in (
            model_columns.items()
        ):
            metrics = model_metrics(
                sample,
                sample[
                    column
                ].to_numpy(),
            )

            rows.append(
                {
                    "bootstrap_id":
                        bootstrap_id,
                    "model":
                        model_name,
                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


def paired_intervals(
    bootstrap,
):
    metrics = [
        "log_loss",
        "brier",
        "roc_auc",
        "average_precision",
        "linked_ici",
        "restricted_ici",
        "distortion_gap",
    ]

    comparisons = [
        (
            "no_prior_utilization",
            "frozen_full",
        ),
        (
            "no_operational_proxies",
            "frozen_full",
        ),
        (
            "observation_score_augmented",
            "frozen_full",
        ),
    ]

    rows = []

    for model, comparator in (
        comparisons
    ):
        left = (
            bootstrap.loc[
                bootstrap[
                    "model"
                ].eq(
                    model
                )
            ]
            .set_index(
                "bootstrap_id"
            )
        )

        right = (
            bootstrap.loc[
                bootstrap[
                    "model"
                ].eq(
                    comparator
                )
            ]
            .set_index(
                "bootstrap_id"
            )
        )

        for metric in metrics:
            difference = (
                left[
                    metric
                ]
                - right[
                    metric
                ]
            )

            rows.append(
                {
                    "model":
                        model,
                    "comparator":
                        comparator,
                    "metric":
                        metric,
                    "mean_difference":
                        float(
                            difference.mean()
                        ),
                    "q025":
                        float(
                            difference.quantile(
                                0.025
                            )
                        ),
                    "median":
                        float(
                            difference.quantile(
                                0.50
                            )
                        ),
                    "q975":
                        float(
                            difference.quantile(
                                0.975
                            )
                        ),
                    "n_valid":
                        int(
                            difference.notna().sum()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_arguments()

    phase5c_dir = Path(
        args.phase5c_dir
    )

    output_dir = Path(
        args.output_dir
    )

    cache_dir = Path(
        args.cache_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata(
        phase5c_dir
    )

    full_numeric = list(
        metadata[
            "numeric_features"
        ]
    )

    full_categorical = list(
        metadata[
            "categorical_features"
        ]
    )

    full_features = (
        full_numeric
        + full_categorical
    )

    no_prior_numeric = [
        feature
        for feature in full_numeric
        if feature
        not in PRIOR_UTILIZATION_FEATURES
    ]

    no_prior_categorical = list(
        full_categorical
    )

    no_operational_numeric = [
        feature
        for feature in full_numeric
        if (
            feature
            not in PRIOR_UTILIZATION_FEATURES
            and feature
            not in TEMPORAL_PROCESS_FEATURES
        )
    ]

    no_operational_categorical = [
        feature
        for feature in full_categorical
        if feature
        not in ADMISSION_PROCESS_FEATURES
    ]

    selected_family = metadata[
        "selected_family"
    ]

    selected_parameters = metadata[
        "selected_parameters"
    ]

    final_model_seed = (
        int(
            metadata[
                "split_seed"
            ]
        )
        + 10000
    )

    print("=" * 88)
    print(
        "ACTIONAUDIT PHASE 7G: "
        "OBSERVATION-PROCESS-AS-FEATURE AUDIT"
    )
    print("=" * 88)

    print(
        "Frozen raw feature count:",
        len(
            full_features
        ),
    )

    print(
        "Prior-utilization candidates:",
        len(
            PRIOR_UTILIZATION_FEATURES
        ),
    )

    print(
        "Selected model:",
        selected_family,
        selected_parameters,
    )

    prediction_cache = (
        cache_dir
        / "phase7g_test_predictions.csv.gz"
    )

    cache_metadata_path = (
        cache_dir
        / "phase7g_cache_metadata.json"
    )

    if prediction_cache.exists():
        print(
            "Reusing cached Phase 7G "
            "model predictions."
        )

        predictions = pd.read_csv(
            prediction_cache
        )

        cache_metadata = json.loads(
            cache_metadata_path.read_text()
        )

        reproduction_error = float(
            cache_metadata[
                "frozen_prediction_"
                "maximum_absolute_difference"
            ]
        )

    else:
        frame = build_analysis_frame(
            cohort_path=Path(
                args.cohort
            ),
            mimic_root=Path(
                args.mimic_root
            ),
            split_manifest_path=(
                phase5c_dir
                / "split_manifest.csv.gz"
            ),
        )

        if len(frame) != 155059:
            raise RuntimeError(
                "Expected 155,059 frozen "
                "admissions; found "
                f"{len(frame):,}"
            )

        train = (
            frame.loc[
                frame[
                    "split"
                ].eq(
                    "train"
                )
            ]
            .reset_index(
                drop=True
            )
        )

        validation = (
            frame.loc[
                frame[
                    "split"
                ].eq(
                    "validation"
                )
            ]
            .reset_index(
                drop=True
            )
        )

        # Match the exact frozen Phase 5C final-training order:
        # train rows first, then validation rows.
        development = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        test = (
            frame.loc[
                frame[
                    "split"
                ].eq(
                    "test"
                )
            ]
            .reset_index(
                drop=True
            )
        )

        print(
            "Development:",
            len(
                development
            ),
        )

        print(
            "Test:",
            len(
                test
            ),
        )

        # ---------------------------------------------------------
        # 1. Exact frozen full-feature reproduction.
        # ---------------------------------------------------------

        print(
            "\nLoading frozen full-feature "
            "mortality model..."
        )

        frozen_full_model = joblib.load(
            phase5c_dir
            / "selected_model.joblib"
        )

        frozen_full_probability = (
            frozen_full_model.predict_proba(
                test[
                    full_features
                ]
            )[:, 1]
        )

        frozen_saved = pd.read_csv(
            phase5c_dir
            / "test_predictions.csv.gz"
        )[
            [
                "subject_id",
                "hadm_id",
                "prediction",
            ]
        ].rename(
            columns={
                "prediction":
                    "_frozen_prediction"
            }
        )

        reproduction = (
            test[
                [
                    "subject_id",
                    "hadm_id",
                ]
            ]
            .copy()
        )

        reproduction[
            "_new_prediction"
        ] = (
            frozen_full_probability
        )

        reproduction = (
            reproduction.merge(
                frozen_saved,
                on=[
                    "subject_id",
                    "hadm_id",
                ],
                how="left",
                validate="one_to_one",
            )
        )

        reproduction_error = float(
            np.max(
                np.abs(
                    reproduction[
                        "_new_prediction"
                    ].to_numpy()
                    - reproduction[
                        "_frozen_prediction"
                    ].to_numpy()
                )
            )
        )

        print(
            "Frozen reproduction max "
            "absolute difference:",
            reproduction_error,
        )

        if reproduction_error > 1e-10:
            raise RuntimeError(
                "Frozen Phase 5C prediction "
                "reproduction failed before "
                "running ablations: "
                f"{reproduction_error}"
            )

        # ---------------------------------------------------------
        # 2. Remove historical utilization proxies.
        # ---------------------------------------------------------

        print(
            "\nFitting no-prior-utilization "
            "model..."
        )

        _, no_prior_probability = (
            fit_mortality_model(
                development,
                test,
                family=selected_family,
                parameters=(
                    selected_parameters
                ),
                seed=final_model_seed,
                numeric_features=(
                    no_prior_numeric
                ),
                categorical_features=(
                    no_prior_categorical
                ),
            )
        )

        # ---------------------------------------------------------
        # 3. Remove broader operational/process proxies.
        # ---------------------------------------------------------

        print(
            "\nFitting no-operational-proxy "
            "model..."
        )

        _, no_operational_probability = (
            fit_mortality_model(
                development,
                test,
                family=selected_family,
                parameters=(
                    selected_parameters
                ),
                seed=final_model_seed,
                numeric_features=(
                    no_operational_numeric
                ),
                categorical_features=(
                    no_operational_categorical
                ),
            )
        )

        # ---------------------------------------------------------
        # 4. Explicit observation-process-score stress test.
        #
        # Development scores are OOF. Test score is produced by
        # an observation model trained only on development data.
        # ---------------------------------------------------------

        (
            development_observation_score,
            test_observation_score,
        ) = fit_observation_score(
            development,
            test,
            numeric_features=(
                full_numeric
            ),
            categorical_features=(
                full_categorical
            ),
            folds=(
                args.observation_folds
            ),
            seed=args.seed,
        )

        development_augmented = (
            development.copy()
        )

        test_augmented = (
            test.copy()
        )

        development_augmented[
            OBSERVATION_SCORE_NAME
        ] = (
            development_observation_score
        )

        test_augmented[
            OBSERVATION_SCORE_NAME
        ] = (
            test_observation_score
        )

        print(
            "\nFitting observation-score-"
            "augmented mortality model..."
        )

        _, observation_augmented_probability = (
            fit_mortality_model(
                development_augmented,
                test_augmented,
                family=selected_family,
                parameters=(
                    selected_parameters
                ),
                seed=final_model_seed,
                numeric_features=(
                    full_numeric
                    + [
                        OBSERVATION_SCORE_NAME
                    ]
                ),
                categorical_features=(
                    full_categorical
                ),
            )
        )

        predictions = test[
            [
                "subject_id",
                "hadm_id",
                "outcome",
                "observed",
                "restricted_event",
                "hidden_event",
            ]
        ].copy()

        predictions[
            "prediction_frozen_full"
        ] = (
            frozen_full_probability
        )

        predictions[
            "prediction_no_prior_utilization"
        ] = (
            no_prior_probability
        )

        predictions[
            "prediction_no_operational_proxies"
        ] = (
            no_operational_probability
        )

        predictions[
            "observation_score"
        ] = (
            test_observation_score
        )

        predictions[
            "prediction_observation_score_augmented"
        ] = (
            observation_augmented_probability
        )

        predictions.to_csv(
            prediction_cache,
            index=False,
            compression="gzip",
        )

        cache_metadata = {
            "frozen_prediction_"
            "maximum_absolute_difference":
                reproduction_error,
            "development_n":
                len(
                    development
                ),
            "test_n":
                len(
                    test
                ),
            "observation_folds":
                args.observation_folds,
            "seed":
                args.seed,
        }

        cache_metadata_path.write_text(
            json.dumps(
                cache_metadata,
                indent=2,
            )
        )

    # -------------------------------------------------------------
    # Feature-count audit.
    # -------------------------------------------------------------

    frozen_model = joblib.load(
        phase5c_dir
        / "selected_model.joblib"
    )

    raw_feature_count = len(
        full_features
    )

    feature_count = pd.DataFrame(
        [
            {
                "quantity":
                    "frozen_numeric_raw_features",
                "value":
                    len(
                        full_numeric
                    ),
            },
            {
                "quantity":
                    "frozen_categorical_raw_features",
                "value":
                    len(
                        full_categorical
                    ),
            },
            {
                "quantity":
                    "frozen_total_raw_features",
                "value":
                    raw_feature_count,
            },
            {
                "quantity":
                    "manuscript_current_feature_count",
                "value":
                    31,
            },
            {
                "quantity":
                    "recommended_manuscript_feature_count",
                "value":
                    raw_feature_count,
            },
        ]
    )

    feature_count.to_csv(
        output_dir
        / "feature_count_audit.csv",
        index=False,
    )

    roles = feature_role_table(
        full_numeric,
        full_categorical,
    )

    roles.to_csv(
        output_dir
        / "feature_role_audit.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Point estimates.
    # -------------------------------------------------------------

    model_columns = {
        "frozen_full":
            "prediction_frozen_full",
        "no_prior_utilization":
            "prediction_no_prior_utilization",
        "no_operational_proxies":
            "prediction_no_operational_proxies",
        "observation_score_augmented":
            "prediction_observation_score_augmented",
    }

    point_rows = []

    for model_name, column in (
        model_columns.items()
    ):
        metrics = model_metrics(
            predictions,
            predictions[
                column
            ].to_numpy(),
        )

        if model_name == (
            "frozen_full"
        ):
            raw_count = len(
                full_features
            )

        elif model_name == (
            "no_prior_utilization"
        ):
            raw_count = (
                len(
                    no_prior_numeric
                )
                + len(
                    no_prior_categorical
                )
            )

        elif model_name == (
            "no_operational_proxies"
        ):
            raw_count = (
                len(
                    no_operational_numeric
                )
                + len(
                    no_operational_categorical
                )
            )

        else:
            raw_count = (
                len(
                    full_features
                )
                + 1
            )

        full_prediction = (
            predictions[
                "prediction_frozen_full"
            ].to_numpy()
        )

        current_prediction = (
            predictions[
                column
            ].to_numpy()
        )

        point_rows.append(
            {
                "model":
                    model_name,
                "raw_feature_count":
                    raw_count,
                **metrics,
                "prediction_mae_vs_frozen_full":
                    float(
                        np.mean(
                            np.abs(
                                current_prediction
                                - full_prediction
                            )
                        )
                    ),
                "prediction_correlation_"
                "with_frozen_full":
                    float(
                        np.corrcoef(
                            current_prediction,
                            full_prediction,
                        )[
                            0,
                            1,
                        ]
                    ),
            }
        )

    point = pd.DataFrame(
        point_rows
    )

    point.to_csv(
        output_dir
        / "model_point_summary.csv",
        index=False,
    )

    observation = (
        observation_diagnostics(
            predictions,
            predictions[
                "observation_score"
            ].to_numpy(),
            predictions[
                "prediction_frozen_full"
            ].to_numpy(),
        )
    )

    observation.to_csv(
        output_dir
        / "observation_score_diagnostics.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Paired patient bootstrap.
    # -------------------------------------------------------------

    print(
        "\nRunning paired patient bootstrap..."
    )

    bootstrap = run_bootstrap(
        predictions,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed + 9000,
    )

    bootstrap.to_csv(
        output_dir
        / "paired_bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    intervals = paired_intervals(
        bootstrap
    )

    intervals.to_csv(
        output_dir
        / "paired_bootstrap_intervals.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Frozen-reference checks.
    # -------------------------------------------------------------

    frozen_summary = pd.read_csv(
        args.phase5g_summary
    )

    frozen_restricted = float(
        frozen_summary.loc[
            frozen_summary[
                "method"
            ].eq(
                "restricted_complete_case"
            ),
            "estimated_ici",
        ].iloc[
            0
        ]
    )

    frozen_point = (
        point.loc[
            point[
                "model"
            ].eq(
                "frozen_full"
            )
        ]
        .iloc[
            0
        ]
    )

    expected_linked_ici = float(
        metadata[
            "test_metrics"
        ][
            "reference_ici_5_bins"
        ]
    )

    validation_rows = []

    def add_check(
        check,
        value,
        threshold,
        comparison,
        passed,
    ):
        validation_rows.append(
            {
                "check":
                    check,
                "value":
                    value,
                "threshold":
                    threshold,
                "comparison":
                    comparison,
                "passed":
                    bool(
                        passed
                    ),
            }
        )

    add_check(
        "frozen full predictions reproduced",
        reproduction_error,
        1e-10,
        "<=",
        reproduction_error
        <= 1e-10,
    )

    linked_difference = abs(
        float(
            frozen_point[
                "linked_ici"
            ]
        )
        - expected_linked_ici
    )

    add_check(
        "frozen linked ICI reproduced",
        linked_difference,
        1e-10,
        "<=",
        linked_difference
        <= 1e-10,
    )

    restricted_difference = abs(
        float(
            frozen_point[
                "restricted_ici"
            ]
        )
        - frozen_restricted
    )

    add_check(
        "frozen restricted ICI reproduced",
        restricted_difference,
        1e-10,
        "<=",
        restricted_difference
        <= 1e-10,
    )

    add_check(
        "frozen raw feature count",
        raw_feature_count,
        21,
        "==",
        raw_feature_count
        == 21,
    )

    forbidden_intersection = sorted(
        set(
            full_features
        )
        & set(
            phase5c.FORBIDDEN_MODEL_FEATURES
        )
    )

    add_check(
        "no forbidden current-admission "
        "post-outcome features",
        str(
            forbidden_intersection
        ),
        "[]",
        "==",
        len(
            forbidden_intersection
        )
        == 0,
    )

    add_check(
        "all eight prior-utilization "
        "features removed in ablation",
        len(
            set(
                PRIOR_UTILIZATION_FEATURES
            )
            - set(
                no_prior_numeric
            )
        ),
        8,
        "==",
        len(
            set(
                PRIOR_UTILIZATION_FEATURES
            )
            - set(
                no_prior_numeric
            )
        )
        == 8,
    )

    add_check(
        "all broader operational proxies "
        "removed in stress ablation",
        (
            len(
                no_operational_numeric
            )
            + len(
                no_operational_categorical
            )
        ),
        7,
        "==",
        (
            len(
                no_operational_numeric
            )
            + len(
                no_operational_categorical
            )
        )
        == 7,
    )

    observation_values = (
        predictions[
            "observation_score"
        ].to_numpy()
    )

    observation_valid = bool(
        np.isfinite(
            observation_values
        ).all()
        and (
            observation_values
            > 0
        ).all()
        and (
            observation_values
            < 1
        ).all()
    )

    add_check(
        "observation scores finite "
        "and within probability bounds",
        observation_valid,
        True,
        "==",
        observation_valid,
    )

    add_check(
        "test admissions preserved",
        len(
            predictions
        ),
        31012,
        "==",
        len(
            predictions
        )
        == 31012,
    )

    completed_bootstraps = int(
        bootstrap[
            "bootstrap_id"
        ].nunique()
    )

    add_check(
        "all requested patient "
        "bootstraps completed",
        completed_bootstraps,
        args.bootstrap_replicates,
        "==",
        completed_bootstraps
        == args.bootstrap_replicates,
    )

    finite_points = bool(
        np.isfinite(
            point.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy()
        ).all()
    )

    add_check(
        "all model point estimates finite",
        finite_points,
        True,
        "==",
        finite_points,
    )

    # No gate requires an ablation or augmentation
    # to improve calibration.
    add_check(
        "feature-effect directions treated "
        "as diagnostic rather than "
        "success criteria",
        True,
        True,
        "==",
        True,
    )

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    passed = bool(
        validation[
            "passed"
        ].all()
    )

    status = (
        "phase7g_observation_feature_audit_passed"
        if passed
        else
        "phase7g_observation_feature_audit_failed"
    )

    (
        output_dir
        / "phase7g_status.txt"
    ).write_text(
        status
        + "\n"
    )

    design = f"""Phase 7G: observation-process-as-feature audit

Frozen dataset:
MIMIC-IV test admissions = 31,012

Frozen model:
{selected_family}
parameters = {selected_parameters}

Frozen raw predictor features:
{len(full_features)}

Prior-utilization proxy candidates removed:
{PRIOR_UTILIZATION_FEATURES}

Broader operational/process proxy candidates additionally removed:
{TEMPORAL_PROCESS_FEATURES + ADMISSION_PROCESS_FEATURES}

Conditions:
1. frozen_full
2. no_prior_utilization
3. no_operational_proxies
4. observation_score_augmented

Observation score:
Cross-fitted logistic admission-time model of current-outcome
observability. Development observations are out-of-fold. Test
scores come from a model fit only on development data.

Mortality-model tuning:
None. All sensitivity models reuse the frozen Phase 5C GBM
family and selected hyperparameters.

Calibration:
5 reference-risk quantile bins.
Linked ICI uses linked 30-day outcomes.
Restricted ICI uses only visible outcomes within the same
reference-risk bins and integrates over the full test risk
distribution.

Inference:
Paired patient-cluster bootstrap.
Replicates = {args.bootstrap_replicates}

Interpretation boundary:
Prior utilization, timing, admission type, and admission location
are treated as candidate operational/process proxies, not proven
confounders. The observation-score condition is a controlled
stress test, not a recommended deployed predictor.
"""

    (
        output_dir
        / "phase7g_design.txt"
    ).write_text(
        design
    )

    print("\n" + "=" * 88)
    print(
        "PHASE 7G POINT RESULTS"
    )
    print("=" * 88)

    print(
        point[
            [
                "model",
                "raw_feature_count",
                "roc_auc",
                "log_loss",
                "linked_ici",
                "restricted_ici",
                "distortion_gap",
                "prediction_mae_vs_frozen_full",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nObservation-score diagnostics"
    )

    print(
        observation.to_string(
            index=False
        )
    )

    print(
        "\nValidation"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:",
        status,
    )

    if not passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
