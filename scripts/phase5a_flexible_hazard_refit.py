"""Phase 5A: flexible SUPPORT2 event- and censoring-hazard refit.

The prediction model, test patients, frozen predictions, score bins,
outcomes, and observation boundary remain unchanged. Only the nuisance
hazard models are varied.

Model families:
1. Linear pooled logistic regression.
2. Additive spline pooled logistic regression.
3. Histogram gradient boosting with nonlinearities and interactions.

All model and hyperparameter selection uses the original validation split.
Test outcomes are used only for the final reference-follow-up audit.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

from actionaudit.discrete_survival import (
    TIME_FEATURES,
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
    expand_person_period,
    predict_hazard_grid,
)
from actionaudit.km import reference_quantile_edges
from actionaudit.support2 import (
    build_support2_horizon_view,
    load_support2_csv,
)


MODEL_FAMILIES = [
    "linear",
    "spline",
    "gbm",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit flexible event and censoring hazards on the "
            "frozen SUPPORT2 split."
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
        default="artifacts/phase5a_flexible_hazards_60d",
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

    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use one configuration per family. "
            "Intended only for the smoke test."
        ),
    )

    return parser.parse_args()


def read_training_metadata(
    phase1b_dir: Path,
) -> dict[str, Any]:
    """Read feature-type metadata from either project layout."""

    candidates = [
        phase1b_dir / "training_metadata.json",
        phase1b_dir / "experiment_metadata.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        metadata = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            "training_metadata" in metadata
            and isinstance(
                metadata["training_metadata"],
                dict,
            )
        ):
            metadata = metadata[
                "training_metadata"
            ]

        if (
            "numeric_features" in metadata
            and "categorical_features" in metadata
        ):
            return metadata

    raise FileNotFoundError(
        "Could not find numeric/categorical feature metadata "
        f"under {phase1b_dir}"
    )


def included_feature_names(
    phase1b_dir: Path,
) -> list[str]:
    """Read the leakage-safe Phase 1B feature manifest."""

    path = (
        phase1b_dir
        / "feature_manifest.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature manifest: {path}"
        )

    manifest = pd.read_csv(
        path
    )

    if "included" not in manifest.columns:
        raise ValueError(
            "feature_manifest.csv has no included column"
        )

    included = manifest[
        "included"
    ]

    if pd.api.types.is_bool_dtype(
        included.dtype
    ):
        mask = included
    else:
        mask = (
            included.astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        )

    features = (
        manifest.loc[
            mask,
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    if not features:
        raise RuntimeError(
            "No included Phase 1B features found"
        )

    return features


def add_restricted_survival_columns(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Construct hospital-boundary survival outcomes.

    Event-before-discharge ordering is used on tied days.

    restricted_event:
        Death observed inside the hospital by the horizon.

    restricted_censored_by_discharge:
        Alive hospital discharge before the horizon.

    restricted_time:
        Death time for restricted events, discharge time for
        discharge censoring, and the horizon otherwise.
    """

    required = {
        "outcome",
        "observed",
        "action",
        "exit_time",
        "d.time",
    }

    missing = sorted(
        required.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Cannot construct restricted survival columns; "
            f"missing: {missing}"
        )

    result = frame.copy()

    outcome = (
        result[
            "outcome"
        ]
        .astype(int)
        .to_numpy()
    )

    observed = (
        result[
            "observed"
        ]
        .astype(int)
        .to_numpy()
    )

    event_time = pd.to_numeric(
        result[
            "d.time"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    exit_time = pd.to_numeric(
        result[
            "exit_time"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    action = (
        result[
            "action"
        ]
        .astype(str)
        .to_numpy()
    )

    restricted_event = (
        (outcome == 1)
        & (observed == 1)
        & np.isfinite(
            event_time
        )
        & (
            event_time
            <= horizon
        )
    )

    discharge_censored = (
        (observed == 0)
        & (
            action
            == "alive_hospital_discharge"
        )
        & np.isfinite(
            exit_time
        )
        & (
            exit_time
            <= horizon
        )
    )

    # Event wins on a tied day. This is mainly a defensive
    # safeguard because alive discharge and hospital death
    # should already be mutually exclusive.
    discharge_censored = (
        discharge_censored
        & ~restricted_event
    )

    restricted_time = np.full(
        len(result),
        float(
            horizon
        ),
        dtype=float,
    )

    restricted_time[
        discharge_censored
    ] = exit_time[
        discharge_censored
    ]

    restricted_time[
        restricted_event
    ] = event_time[
        restricted_event
    ]

    restricted_time = np.clip(
        restricted_time,
        1.0,
        float(
            horizon
        ),
    )

    result[
        "restricted_event"
    ] = restricted_event.astype(
        int
    )

    result[
        "restricted_censored_by_discharge"
    ] = discharge_censored.astype(
        int
    )

    result[
        "restricted_time"
    ] = restricted_time

    overlap = (
        result[
            "restricted_event"
        ].astype(bool)
        & result[
            "restricted_censored_by_discharge"
        ].astype(bool)
    )

    if overlap.any():
        raise RuntimeError(
            "Patients cannot be both restricted events "
            "and discharge-censored"
        )

    unexpected_hidden = (
        result[
            "observed"
        ].astype(int).eq(0)
        & result[
            "restricted_censored_by_discharge"
        ].astype(int).eq(0)
    )

    if unexpected_hidden.any():
        examples = (
            result.loc[
                unexpected_hidden,
                [
                    "patient_id",
                    "outcome",
                    "observed",
                    "action",
                    "exit_time",
                    "d.time",
                ],
            ]
            .head()
            .to_dict(
                orient="records"
            )
        )

        raise RuntimeError(
            "Some hospital-hidden outcomes were not mapped "
            f"to alive discharge: {examples}"
        )

    return result


def load_frozen_analysis(
    *,
    input_path: str,
    phase1b_dir: Path,
    horizon: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    """Reconstruct the original split and attach frozen predictions."""

    split_path = (
        phase1b_dir
        / "split_manifest.csv"
    )

    prediction_path = (
        phase1b_dir
        / "test_predictions.csv"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split manifest: {split_path}"
        )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing frozen predictions: {prediction_path}"
        )

    raw = load_support2_csv(
        input_path
    )

    analysis = (
        build_support2_horizon_view(
            raw,
            horizon,
        )
        .frame
        .loc[
            lambda frame: frame[
                "reference_known"
            ].eq(1)
        ]
        .copy()
        .reset_index(drop=True)
    )

    analysis = add_restricted_survival_columns(
        analysis,
        horizon=horizon,
    )

    split_manifest = pd.read_csv(
        split_path
    )[
        [
            "patient_id",
            "split",
        ]
    ]

    split_manifest[
        "patient_id"
    ] = split_manifest[
        "patient_id"
    ].astype(int)

    analysis[
        "patient_id"
    ] = analysis[
        "patient_id"
    ].astype(int)

    analysis = analysis.merge(
        split_manifest,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    if set(
        analysis["split"].unique()
    ) != expected_splits:
        raise RuntimeError(
            "Reconstructed split does not contain exactly "
            "train, validation, and test"
        )

    features = included_feature_names(
        phase1b_dir
    )

    metadata = read_training_metadata(
        phase1b_dir
    )

    numeric_features = [
        feature
        for feature in metadata[
            "numeric_features"
        ]
        if feature in features
    ]

    categorical_features = [
        feature
        for feature in metadata[
            "categorical_features"
        ]
        if feature in features
    ]

    uncovered = sorted(
        set(features).difference(
            numeric_features
            + categorical_features
        )
    )

    for feature in uncovered:
        if pd.api.types.is_numeric_dtype(
            analysis[feature]
        ):
            numeric_features.append(
                feature
            )
        else:
            categorical_features.append(
                feature
            )

    train = (
        analysis.loc[
            analysis[
                "split"
            ].eq(
                "train"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        analysis.loc[
            analysis[
                "split"
            ].eq(
                "validation"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        analysis.loc[
            analysis[
                "split"
            ].eq(
                "test"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    frozen_predictions = pd.read_csv(
        prediction_path
    )[
        [
            "patient_id",
            "prediction",
            "outcome",
        ]
    ]

    frozen_predictions[
        "patient_id"
    ] = frozen_predictions[
        "patient_id"
    ].astype(int)

    frozen_predictions = (
        frozen_predictions.rename(
            columns={
                "outcome": (
                    "frozen_reference_outcome"
                )
            }
        )
    )

    test = test.merge(
        frozen_predictions,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if len(test) != len(
        frozen_predictions
    ):
        raise RuntimeError(
            "Frozen prediction rows do not exactly match "
            "the reconstructed test split"
        )

    outcome_mismatch = (
        test[
            "outcome"
        ].astype(int)
        != test[
            "frozen_reference_outcome"
        ].astype(int)
    )

    if outcome_mismatch.any():
        raise RuntimeError(
            "Reference outcomes differ from the frozen "
            "Phase 1B prediction file"
        )

    return (
        train,
        validation,
        test,
        features,
        numeric_features,
        categorical_features,
    )


def numeric_feature_groups(
    train: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[list[str], list[str]]:
    """Separate spline-eligible and low-cardinality numeric features."""

    continuous: list[str] = []
    low_cardinality: list[str] = []

    for feature in numeric_features:
        n_unique = int(
            train[
                feature
            ].nunique(
                dropna=True
            )
        )

        if n_unique > 10:
            continuous.append(
                feature
            )
        else:
            low_cardinality.append(
                feature
            )

    return (
        continuous,
        low_cardinality,
    )


def logistic_classifier(
    *,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    """Create the pooled-logistic classifier."""

    return LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        tol=1e-6,
        random_state=seed,
    )


def build_linear_model(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    c_value: float,
    seed: int,
) -> Pipeline:
    """Original linear pooled-logistic nuisance family."""

    numeric_columns = (
        numeric_features
        + TIME_FEATURES
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        sparse_threshold=1.0,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                logistic_classifier(
                    c_value=c_value,
                    seed=seed,
                ),
            ),
        ]
    )


def build_spline_model(
    *,
    continuous_features: list[str],
    low_cardinality_features: list[str],
    categorical_features: list[str],
    c_value: float,
    n_knots: int,
    seed: int,
) -> Pipeline:
    """Additive nonlinear pooled-logistic hazard model."""

    spline_columns = (
        continuous_features
        + [
            "time_day_scaled",
        ]
    )

    spline_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "splines",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=False
                ),
            ),
        ]
    )

    linear_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    transformers = [
        (
            "splines",
            spline_pipeline,
            spline_columns,
        ),
    ]

    if low_cardinality_features:
        transformers.append(
            (
                "low_cardinality_numeric",
                linear_numeric_pipeline,
                low_cardinality_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                logistic_classifier(
                    c_value=c_value,
                    seed=seed,
                ),
            ),
        ]
    )


def build_gbm_model(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
    seed: int,
) -> Pipeline:
    """Nonlinear interaction-capable pooled-hazard model."""

    numeric_columns = (
        numeric_features
        + TIME_FEATURES
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )

    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=250,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=40,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def candidate_grid(
    family: str,
    *,
    fast: bool,
) -> list[dict[str, float | int]]:
    """Predeclared compact nuisance-model grids."""

    if family == "linear":
        if fast:
            return [
                {
                    "C": 0.1,
                }
            ]

        return [
            {
                "C": value,
            }
            for value in (
                0.01,
                0.1,
                1.0,
                10.0,
                100.0,
            )
        ]

    if family == "spline":
        if fast:
            return [
                {
                    "C": 1.0,
                    "n_knots": 4,
                }
            ]

        return [
            {
                "C": c_value,
                "n_knots": n_knots,
            }
            for c_value in (
                0.1,
                1.0,
                10.0,
                100.0,
            )
            for n_knots in (
                4,
                6,
                8,
            )
        ]

    if family == "gbm":
        if fast:
            return [
                {
                    "learning_rate": 0.05,
                    "max_leaf_nodes": 15,
                    "l2_regularization": 1.0,
                }
            ]

        return [
            {
                "learning_rate": 0.05,
                "max_leaf_nodes": leaves,
                "l2_regularization": l2_value,
            }
            for leaves in (
                15,
                31,
            )
            for l2_value in (
                1.0,
                10.0,
            )
        ]

    raise ValueError(
        f"Unknown family: {family}"
    )


def build_candidate(
    *,
    family: str,
    parameters: dict[str, float | int],
    numeric_features: list[str],
    categorical_features: list[str],
    continuous_features: list[str],
    low_cardinality_features: list[str],
    seed: int,
) -> Pipeline:
    """Construct one model from its family and parameters."""

    if family == "linear":
        return build_linear_model(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            c_value=float(
                parameters["C"]
            ),
            seed=seed,
        )

    if family == "spline":
        return build_spline_model(
            continuous_features=continuous_features,
            low_cardinality_features=(
                low_cardinality_features
            ),
            categorical_features=categorical_features,
            c_value=float(
                parameters["C"]
            ),
            n_knots=int(
                parameters[
                    "n_knots"
                ]
            ),
            seed=seed,
        )

    if family == "gbm":
        return build_gbm_model(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            learning_rate=float(
                parameters[
                    "learning_rate"
                ]
            ),
            max_leaf_nodes=int(
                parameters[
                    "max_leaf_nodes"
                ]
            ),
            l2_regularization=float(
                parameters[
                    "l2_regularization"
                ]
            ),
            seed=seed,
        )

    raise ValueError(
        f"Unknown family: {family}"
    )


def select_and_refit_target_models(
    *,
    train_expanded: pd.DataFrame,
    validation_expanded: pd.DataFrame,
    validation_patient_frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    continuous_features: list[str],
    low_cardinality_features: list[str],
    fast: bool,
    seed: int,
) -> tuple[
    dict[str, Pipeline],
    pd.DataFrame,
]:
    """Tune each family on validation and refit on train+validation."""

    model_columns = (
        numeric_features
        + categorical_features
        + TIME_FEATURES
    )

    train_x = train_expanded[
        model_columns
    ]

    train_y = (
        train_expanded[
            "hazard_target"
        ]
        .astype(int)
    )

    validation_x = validation_expanded[
        model_columns
    ]

    validation_y = (
        validation_expanded[
            "hazard_target"
        ]
        .astype(int)
    )

    selection_rows: list[
        dict[str, object]
    ] = []

    selected_parameters: dict[
        str,
        dict[str, float | int]
    ] = {}

    for family_index, family in enumerate(
        MODEL_FAMILIES
    ):
        best_key: tuple[
            float,
            ...
        ] | None = None

        best_parameters: dict[
            str,
            float | int
        ] | None = None

        for candidate_index, parameters in enumerate(
            candidate_grid(
                family,
                fast=fast,
            )
        ):
            candidate_seed = (
                seed
                + 1000 * family_index
                + candidate_index
            )

            model = build_candidate(
                family=family,
                parameters=parameters,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                continuous_features=continuous_features,
                low_cardinality_features=(
                    low_cardinality_features
                ),
                seed=candidate_seed,
            )

            model.fit(
                train_x,
                train_y,
            )

            validation_probability = (
                model.predict_proba(
                    validation_x
                )[:, 1]
            )

            validation_probability = np.clip(
                validation_probability,
                1e-8,
                1.0 - 1e-8,
            )

            candidate_log_loss = float(
                log_loss(
                    validation_y,
                    validation_probability,
                    labels=[
                        0,
                        1,
                    ],
                )
            )

            candidate_brier = float(
                brier_score_loss(
                    validation_y,
                    validation_probability,
                )
            )

            if target == "event":
                validation_hazards = (
                    predict_hazard_grid(
                        model,
                        validation_patient_frame,
                        feature_columns,
                        horizon=horizon,
                    )
                )

                validation_horizon_risk = (
                    1.0
                    - np.prod(
                        1.0
                        - validation_hazards,
                        axis=1,
                    )
                )

                validation_horizon_risk = (
                    np.clip(
                        validation_horizon_risk,
                        1e-8,
                        1.0 - 1e-8,
                    )
                )

                validation_horizon_outcome = (
                    validation_patient_frame[
                        "outcome"
                    ]
                    .astype(int)
                    .to_numpy()
                )

                validation_horizon_log_loss = float(
                    log_loss(
                        validation_horizon_outcome,
                        validation_horizon_risk,
                        labels=[
                            0,
                            1,
                        ],
                    )
                )

                validation_horizon_brier = float(
                    brier_score_loss(
                        validation_horizon_outcome,
                        validation_horizon_risk,
                    )
                )
            else:
                validation_horizon_log_loss = float(
                    "nan"
                )

                validation_horizon_brier = float(
                    "nan"
                )

            selection_rows.append(
                {
                    "target": target,
                    "family": family,
                    "candidate_index": (
                        candidate_index
                    ),
                    "parameters": json.dumps(
                        parameters,
                        sort_keys=True,
                    ),
                    "validation_log_loss": (
                        candidate_log_loss
                    ),
                    "validation_brier": (
                        candidate_brier
                    ),
                    "validation_horizon_log_loss": (
                        validation_horizon_log_loss
                    ),
                    "validation_horizon_brier": (
                        validation_horizon_brier
                    ),
                }
            )

            if target == "event":
                candidate_key = (
                    validation_horizon_log_loss,
                    validation_horizon_brier,
                    candidate_log_loss,
                )
            else:
                candidate_key = (
                    candidate_log_loss,
                    candidate_brier,
                )

            if (
                best_key is None
                or candidate_key
                < best_key
            ):
                best_key = candidate_key
                best_parameters = dict(
                    parameters
                )

        if best_parameters is None:
            raise RuntimeError(
                f"No valid {target} {family} model"
            )

        selected_parameters[
            family
        ] = best_parameters

    selection_table = pd.DataFrame(
        selection_rows
    )

    selection_table[
        "selected_within_family"
    ] = False

    final_models: dict[
        str,
        Pipeline
    ] = {}

    combined_x = pd.concat(
        [
            train_x,
            validation_x,
        ],
        ignore_index=True,
    )

    combined_y = pd.concat(
        [
            train_y,
            validation_y,
        ],
        ignore_index=True,
    )

    for family_index, family in enumerate(
        MODEL_FAMILIES
    ):
        parameters = selected_parameters[
            family
        ]

        parameter_text = json.dumps(
            parameters,
            sort_keys=True,
        )

        selection_table.loc[
            selection_table[
                "target"
            ].eq(
                target
            )
            & selection_table[
                "family"
            ].eq(
                family
            )
            & selection_table[
                "parameters"
            ].eq(
                parameter_text
            ),
            "selected_within_family",
        ] = True

        final_model = build_candidate(
            family=family,
            parameters=parameters,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            low_cardinality_features=(
                low_cardinality_features
            ),
            seed=(
                seed
                + 10000
                + family_index
            ),
        )

        final_model.fit(
            combined_x,
            combined_y,
        )

        final_models[
            family
        ] = final_model

    if target == "event":
        family_sort_columns = [
            "validation_horizon_log_loss",
            "validation_horizon_brier",
            "validation_log_loss",
        ]
    else:
        family_sort_columns = [
            "validation_log_loss",
            "validation_brier",
        ]

    selected_family_rows = (
        selection_table.loc[
            selection_table[
                "selected_within_family"
            ]
        ]
        .sort_values(
            family_sort_columns
        )
    )

    best_overall_family = str(
        selected_family_rows.iloc[
            0
        ][
            "family"
        ]
    )

    selection_table[
        "selected_overall_target"
    ] = (
        selection_table[
            "selected_within_family"
        ]
        & selection_table[
            "family"
        ].eq(
            best_overall_family
        )
    )

    return (
        final_models,
        selection_table,
    )


def scalar_row(
    *,
    estimator_label: str,
    event_family: str,
    censor_family: str,
    result,
) -> dict[str, object]:
    """Extract point-estimate outputs from one model pairing."""

    scalar = result.scalar_summary
    diagnostics = result.diagnostics

    return {
        "estimator_label": estimator_label,
        "event_family": event_family,
        "censor_family": censor_family,
        "reference_ici": float(
            scalar[
                "reference_ici"
            ]
        ),
        "complete_case_mae": float(
            scalar[
                "complete_case_mae_to_reference"
            ]
        ),
        "kaplan_meier_mae": float(
            scalar[
                "kaplan_meier_mae_to_reference"
            ]
        ),
        "survival_or_mae": float(
            scalar[
                "survival_or_mae_to_reference"
            ]
        ),
        "survival_or_signed_gap": float(
            scalar[
                "survival_or_signed_gap"
            ]
        ),
        "survival_or_ici": float(
            scalar[
                "survival_or_ici"
            ]
        ),
        "ipcw_event_mae": float(
            scalar[
                "ipcw_event_mae_to_reference"
            ]
        ),
        "ipcw_event_signed_gap": float(
            scalar[
                "ipcw_event_signed_gap"
            ]
        ),
        "ipcw_event_ici": float(
            scalar[
                "ipcw_event_ici"
            ]
        ),
        "fraction_event_g_clipped": float(
            diagnostics[
                "fraction_event_g_clipped"
            ]
        ),
        "maximum_event_weight": float(
            diagnostics[
                "maximum_event_weight"
            ]
        ),
        "event_weight_effective_sample_size": float(
            diagnostics[
                "event_weight_effective_sample_size"
            ]
        ),
    }


def bootstrap_fixed_nuisance(
    *,
    prepared_frames: dict[str, pd.DataFrame],
    pair_definitions: dict[
        str,
        tuple[str, str]
    ],
    horizon: int,
    bin_edges: np.ndarray,
    censor_survival_clip: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired patient bootstrap with nuisance predictions frozen."""

    rng = np.random.default_rng(
        seed
    )

    first_frame = next(
        iter(
            prepared_frames.values()
        )
    )

    n = len(
        first_frame
    )

    rows: list[
        dict[str, object]
    ] = []

    for bootstrap_id in range(
        replicates
    ):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        current_metrics: dict[
            str,
            dict[str, float]
        ] = {}

        for label, frame in prepared_frames.items():
            sample = (
                frame.iloc[
                    indices
                ]
                .copy()
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

            current_metrics[
                label
            ] = {
                "survival_or_mae": float(
                    result.scalar_summary[
                        "survival_or_mae_to_reference"
                    ]
                ),
                "survival_or_signed_gap": float(
                    result.scalar_summary[
                        "survival_or_signed_gap"
                    ]
                ),
                "survival_or_ici": float(
                    result.scalar_summary[
                        "survival_or_ici"
                    ]
                ),
                "ipcw_event_mae": float(
                    result.scalar_summary[
                        "ipcw_event_mae_to_reference"
                    ]
                ),
                "ipcw_event_signed_gap": float(
                    result.scalar_summary[
                        "ipcw_event_signed_gap"
                    ]
                ),
            }

        linear_or_mae = current_metrics[
            "linear"
        ][
            "survival_or_mae"
        ]

        linear_ipcw_mae = current_metrics[
            "linear"
        ][
            "ipcw_event_mae"
        ]

        for label, metrics in current_metrics.items():
            event_family, censor_family = (
                pair_definitions[
                    label
                ]
            )

            rows.append(
                {
                    "bootstrap_id": (
                        bootstrap_id
                    ),
                    "estimator_label": label,
                    "event_family": (
                        event_family
                    ),
                    "censor_family": (
                        censor_family
                    ),
                    **metrics,
                    "survival_or_mae_difference_vs_linear": (
                        metrics[
                            "survival_or_mae"
                        ]
                        - linear_or_mae
                    ),
                    "ipcw_mae_difference_vs_linear": (
                        metrics[
                            "ipcw_event_mae"
                        ]
                        - linear_ipcw_mae
                    ),
                }
            )

        if (
            bootstrap_id + 1
        ) % 100 == 0:
            print(
                "  bootstrap "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

    return pd.DataFrame(
        rows
    )


def bootstrap_interval_table(
    *,
    point_summary: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired bootstrap distributions."""

    metrics = [
        "survival_or_mae",
        "survival_or_signed_gap",
        "survival_or_ici",
        "ipcw_event_mae",
        "ipcw_event_signed_gap",
        "survival_or_mae_difference_vs_linear",
        "ipcw_mae_difference_vs_linear",
    ]

    rows: list[
        dict[str, object]
    ] = []

    point_index = point_summary.set_index(
        "estimator_label"
    )

    for label, group in bootstrap_results.groupby(
        "estimator_label",
        sort=False,
    ):
        for metric in metrics:
            values = (
                group[
                    metric
                ]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            if metric in point_index.columns:
                estimate = float(
                    point_index.loc[
                        label,
                        metric,
                    ]
                )
            else:
                if label == "linear":
                    estimate = 0.0
                elif metric == (
                    "survival_or_mae_difference_vs_linear"
                ):
                    estimate = float(
                        point_index.loc[
                            label,
                            "survival_or_mae",
                        ]
                        - point_index.loc[
                            "linear",
                            "survival_or_mae",
                        ]
                    )
                elif metric == (
                    "ipcw_mae_difference_vs_linear"
                ):
                    estimate = float(
                        point_index.loc[
                            label,
                            "ipcw_event_mae",
                        ]
                        - point_index.loc[
                            "linear",
                            "ipcw_event_mae",
                        ]
                    )
                else:
                    estimate = float(
                        "nan"
                    )

            rows.append(
                {
                    "estimator_label": label,
                    "metric": metric,
                    "estimate": estimate,
                    "bootstrap_q025": float(
                        values.quantile(
                            0.025
                        )
                    ),
                    "bootstrap_median": float(
                        values.quantile(
                            0.50
                        )
                    ),
                    "bootstrap_q975": float(
                        values.quantile(
                            0.975
                        )
                    ),
                    "n_valid_bootstrap": int(
                        len(values)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def determine_status(
    *,
    point_summary: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    """Classify whether flexibility explains the linear OR residual."""

    point_index = point_summary.set_index(
        "estimator_label"
    )

    flexible_labels = [
        "spline",
        "gbm",
    ]

    best_flexible_label = min(
        flexible_labels,
        key=lambda label: float(
            point_index.loc[
                label,
                "survival_or_mae",
            ]
        ),
    )

    linear_mae = float(
        point_index.loc[
            "linear",
            "survival_or_mae",
        ]
    )

    best_flexible_mae = float(
        point_index.loc[
            best_flexible_label,
            "survival_or_mae",
        ]
    )

    improvement = (
        linear_mae
        - best_flexible_mae
    )

    relative_reduction = (
        improvement
        / linear_mae
        if linear_mae > 0
        else float(
            "nan"
        )
    )

    interval_index = intervals.set_index(
        [
            "estimator_label",
            "metric",
        ]
    )

    difference_upper = float(
        interval_index.loc[
            (
                best_flexible_label,
                "survival_or_mae_difference_vs_linear",
            ),
            "bootstrap_q975",
        ]
    )

    spline_signed_lower = float(
        interval_index.loc[
            (
                "spline",
                "survival_or_signed_gap",
            ),
            "bootstrap_q025",
        ]
    )

    gbm_signed_lower = float(
        interval_index.loc[
            (
                "gbm",
                "survival_or_signed_gap",
            ),
            "bootstrap_q025",
        ]
    )

    if (
        best_flexible_mae <= 0.025
        and improvement >= 0.02
        and difference_upper < 0
    ):
        status = (
            "phase5a_flexibility_explains_most_residual"
        )
    elif (
        best_flexible_mae >= 0.04
        and spline_signed_lower > 0
        and gbm_signed_lower > 0
    ):
        status = (
            "phase5a_positive_residual_persists_after_"
            "flexible_refit"
        )
    else:
        status = (
            "phase5a_nuisance_model_uncertainty"
        )

    evidence = {
        "linear_survival_or_mae": (
            linear_mae
        ),
        "best_flexible_family": (
            best_flexible_label
        ),
        "best_flexible_survival_or_mae": (
            best_flexible_mae
        ),
        "absolute_mae_reduction": (
            improvement
        ),
        "relative_mae_reduction": (
            relative_reduction
        ),
        "paired_difference_bootstrap_q975": (
            difference_upper
        ),
        "spline_signed_gap_bootstrap_q025": (
            spline_signed_lower
        ),
        "gbm_signed_gap_bootstrap_q025": (
            gbm_signed_lower
        ),
    }

    return (
        status,
        evidence,
    )


def main() -> None:
    args = parse_arguments()

    if not (
        0
        < args.censor_survival_clip
        < 1
    ):
        raise ValueError(
            "censor-survival-clip must lie in (0,1)"
        )

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

    (
        train,
        validation,
        test,
        features,
        numeric_features,
        categorical_features,
    ) = load_frozen_analysis(
        input_path=args.input,
        phase1b_dir=phase1b_dir,
        horizon=args.horizon,
    )

    print(
        "Frozen cohort:"
    )

    print(
        f"  train={len(train):,}, "
        f"validation={len(validation):,}, "
        f"test={len(test):,}"
    )

    print(
        f"  features={len(features)}, "
        f"numeric={len(numeric_features)}, "
        f"categorical={len(categorical_features)}"
    )

    print(
        "  restricted test events="
        f"{int(test['restricted_event'].sum()):,}, "
        "discharge-censored="
        f"{int(test['restricted_censored_by_discharge'].sum()):,}"
    )

    (
        continuous_features,
        low_cardinality_features,
    ) = numeric_feature_groups(
        train,
        numeric_features,
    )

    print(
        "Expanding person-period datasets..."
    )

    train_event = expand_person_period(
        train,
        features,
        horizon=args.horizon,
        target="event",
    )

    validation_event = expand_person_period(
        validation,
        features,
        horizon=args.horizon,
        target="event",
    )

    train_censor = expand_person_period(
        train,
        features,
        horizon=args.horizon,
        target="censor",
    )

    validation_censor = expand_person_period(
        validation,
        features,
        horizon=args.horizon,
        target="censor",
    )

    print(
        f"  event rows: {len(train_event):,} train, "
        f"{len(validation_event):,} validation"
    )

    print(
        f"  censor rows: {len(train_censor):,} train, "
        f"{len(validation_censor):,} validation"
    )

    print(
        "Selecting event-hazard models..."
    )

    (
        event_models,
        event_selection,
    ) = select_and_refit_target_models(
        train_expanded=train_event,
        validation_expanded=(
            validation_event
        ),
        validation_patient_frame=validation,
        feature_columns=features,
        horizon=args.horizon,
        target="event",
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
        continuous_features=(
            continuous_features
        ),
        low_cardinality_features=(
            low_cardinality_features
        ),
        fast=args.fast,
        seed=args.seed + 100,
    )

    print(
        "Selecting censoring-hazard models..."
    )

    (
        censor_models,
        censor_selection,
    ) = select_and_refit_target_models(
        train_expanded=train_censor,
        validation_expanded=(
            validation_censor
        ),
        validation_patient_frame=validation,
        feature_columns=features,
        horizon=args.horizon,
        target="censor",
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
        continuous_features=(
            continuous_features
        ),
        low_cardinality_features=(
            low_cardinality_features
        ),
        fast=args.fast,
        seed=args.seed + 200,
    )

    selection_table = pd.concat(
        [
            event_selection,
            censor_selection,
        ],
        ignore_index=True,
    )

    selection_table.to_csv(
        output_dir
        / "nuisance_model_selection.csv",
        index=False,
    )

    best_event_family = str(
        event_selection.loc[
            event_selection[
                "selected_overall_target"
            ],
            "family",
        ].iloc[0]
    )

    best_censor_family = str(
        censor_selection.loc[
            censor_selection[
                "selected_overall_target"
            ],
            "family",
        ].iloc[0]
    )

    print(
        "Validation-selected families:"
    )

    print(
        f"  event={best_event_family}"
    )

    print(
        f"  censor={best_censor_family}"
    )

    event_hazards: dict[
        str,
        np.ndarray
    ] = {}

    censor_hazards: dict[
        str,
        np.ndarray
    ] = {}

    print(
        "Predicting untouched test hazards..."
    )

    for family in MODEL_FAMILIES:
        event_hazards[
            family
        ] = predict_hazard_grid(
            event_models[
                family
            ],
            test,
            features,
            horizon=args.horizon,
        )

        censor_hazards[
            family
        ] = predict_hazard_grid(
            censor_models[
                family
            ],
            test,
            features,
            horizon=args.horizon,
        )

        joblib.dump(
            event_models[
                family
            ],
            output_dir
            / f"event_model_{family}.joblib",
        )

        joblib.dump(
            censor_models[
                family
            ],
            output_dir
            / f"censor_model_{family}.joblib",
        )

    pair_definitions = {
        "linear": (
            "linear",
            "linear",
        ),
        "spline": (
            "spline",
            "spline",
        ),
        "gbm": (
            "gbm",
            "gbm",
        ),
        "best_validation": (
            best_event_family,
            best_censor_family,
        ),
    }

    bin_edges = reference_quantile_edges(
        test[
            "prediction"
        ].to_numpy(),
        args.n_bins,
    )

    prepared_frames: dict[
        str,
        pd.DataFrame
    ] = {}

    point_rows: list[
        dict[str, object]
    ] = []

    patient_output = test[
        [
            "patient_id",
            "prediction",
            "outcome",
        ]
    ].copy()

    for label, (
        event_family,
        censor_family,
    ) in pair_definitions.items():
        prepared = (
            construct_survival_nuisance_predictions(
                test,
                event_hazards=event_hazards[
                    event_family
                ],
                censor_hazards=censor_hazards[
                    censor_family
                ],
                horizon=args.horizon,
                censor_survival_clip=(
                    args.censor_survival_clip
                ),
            )
        )

        result = evaluate_survival_estimators(
            prepared,
            horizon=args.horizon,
            bin_edges=bin_edges,
            censor_survival_clip=(
                args.censor_survival_clip
            ),
        )

        prepared_frames[
            label
        ] = prepared

        point_rows.append(
            scalar_row(
                estimator_label=label,
                event_family=event_family,
                censor_family=censor_family,
                result=result,
            )
        )

        bin_output = (
            result.bin_summary.copy()
        )

        bin_output[
            "estimator_label"
        ] = label

        bin_output[
            "event_family"
        ] = event_family

        bin_output[
            "censor_family"
        ] = censor_family

        bin_output.to_csv(
            output_dir
            / f"bin_summary_{label}.csv",
            index=False,
        )

        patient_output[
            f"{label}_survival_or_risk"
        ] = prepared[
            "survival_outcome_regression"
        ].to_numpy()

        patient_output[
            f"{label}_raw_g_before"
        ] = prepared[
            "raw_censor_survival_before_observed_time"
        ].to_numpy()

        patient_output[
            f"{label}_ipcw_contribution"
        ] = prepared[
            "ipcw_event_contribution"
        ].to_numpy()

    point_summary = pd.DataFrame(
        point_rows
    )

    point_summary.to_csv(
        output_dir
        / "test_estimator_summary.csv",
        index=False,
    )

    patient_output.to_csv(
        output_dir
        / "test_nuisance_predictions.csv",
        index=False,
    )

    print(
        "Running paired fixed-nuisance bootstrap..."
    )

    bootstrap_results = (
        bootstrap_fixed_nuisance(
            prepared_frames=prepared_frames,
            pair_definitions=pair_definitions,
            horizon=args.horizon,
            bin_edges=bin_edges,
            censor_survival_clip=(
                args.censor_survival_clip
            ),
            replicates=(
                args.bootstrap_replicates
            ),
            seed=args.seed + 300,
        )
    )

    bootstrap_results.to_csv(
        output_dir
        / "bootstrap_replicates.csv",
        index=False,
    )

    intervals = bootstrap_interval_table(
        point_summary=point_summary,
        bootstrap_results=(
            bootstrap_results
        ),
    )

    intervals.to_csv(
        output_dir
        / "bootstrap_intervals.csv",
        index=False,
    )

    status, evidence = determine_status(
        point_summary=point_summary,
        intervals=intervals,
    )

    if args.fast:
        preliminary_status = status

        status = (
            "phase5a_smoke_completed_not_decisive"
        )

        evidence[
            "preliminary_fast_grid_status"
        ] = preliminary_status

    (
        output_dir
        / "phase5a_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    metadata = {
        "phase": "5A",
        "experiment": (
            "Flexible event- and censoring-hazard "
            "refit on the frozen SUPPORT2 split"
        ),
        "horizon_days": int(
            args.horizon
        ),
        "n_bins": int(
            args.n_bins
        ),
        "censor_survival_clip": float(
            args.censor_survival_clip
        ),
        "bootstrap_replicates": int(
            args.bootstrap_replicates
        ),
        "fast_grid": bool(
            args.fast
        ),
        "train_n": int(
            len(train)
        ),
        "validation_n": int(
            len(validation)
        ),
        "test_n": int(
            len(test)
        ),
        "test_predictions_frozen": True,
        "test_used_for_nuisance_selection": False,
        "event_family_selected_on_validation": (
            best_event_family
        ),
        "censor_family_selected_on_validation": (
            best_censor_family
        ),
        "included_features": features,
        "continuous_spline_features": (
            continuous_features
        ),
        "low_cardinality_numeric_features": (
            low_cardinality_features
        ),
        "categorical_features": (
            categorical_features
        ),
        "decision_evidence": evidence,
        "status": status,
        "random_seed": int(
            args.seed
        ),
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
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
        "\nPhase 5A estimator comparison"
    )

    print(
        point_summary.to_string(
            index=False
        )
    )

    print(
        "\nDecision evidence"
    )

    for key, value in evidence.items():
        print(
            f"{key}: {value}"
        )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"\nArtifacts saved under: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
