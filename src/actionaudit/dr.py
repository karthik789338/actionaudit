"""Held-out IPCW, outcome-regression, and AIPW calibration estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .km import kaplan_meier_event_risk
from .modeling import (
    build_logistic_pipeline,
    infer_feature_types,
)


@dataclass
class TunedBinaryModel:
    """A selected binary logistic model and its validation table."""

    model: object
    selection_table: pd.DataFrame
    best_c: float


@dataclass
class DREstimatorResult:
    """Bin-level and scalar doubly robust estimator results."""

    bin_summary: pd.DataFrame
    scalar_summary: pd.Series
    diagnostics: pd.Series
    bin_edges: np.ndarray


def tune_binary_logistic(
    train_data: pd.DataFrame,
    train_outcome: pd.Series,
    validation_data: pd.DataFrame,
    validation_outcome: pd.Series,
    feature_columns: Sequence[str],
    *,
    c_grid: Sequence[float] = (
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ),
    random_seed: int = 2027,
) -> TunedBinaryModel:
    """Tune logistic regularization using validation log loss."""

    train_outcome = (
        train_outcome
        .astype(int)
        .reset_index(drop=True)
    )

    validation_outcome = (
        validation_outcome
        .astype(int)
        .reset_index(drop=True)
    )

    if train_outcome.nunique() < 2:
        raise ValueError(
            "Training outcome must contain both classes"
        )

    if validation_outcome.nunique() < 2:
        raise ValueError(
            "Validation outcome must contain both classes"
        )

    numeric_features, categorical_features = infer_feature_types(
        train_data,
        feature_columns,
    )

    rows: list[dict[str, float]] = []

    for c_value in c_grid:
        model = build_logistic_pipeline(
            numeric_features,
            categorical_features,
            c_value=float(c_value),
            random_seed=random_seed,
        )

        model.fit(
            train_data[list(feature_columns)],
            train_outcome,
        )

        probability = model.predict_proba(
            validation_data[list(feature_columns)]
        )[:, 1]

        rows.append(
            {
                "c_value": float(c_value),
                "validation_log_loss": float(
                    log_loss(
                        validation_outcome,
                        probability,
                        labels=[0, 1],
                    )
                ),
                "validation_brier": float(
                    brier_score_loss(
                        validation_outcome,
                        probability,
                    )
                ),
                "validation_roc_auc": float(
                    roc_auc_score(
                        validation_outcome,
                        probability,
                    )
                ),
                "validation_average_precision": float(
                    average_precision_score(
                        validation_outcome,
                        probability,
                    )
                ),
            }
        )

    selection_table = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "validation_log_loss",
                "validation_brier",
                "c_value",
            ]
        )
        .reset_index(drop=True)
    )

    best_c = float(
        selection_table.loc[
            0,
            "c_value",
        ]
    )

    final_model = build_logistic_pipeline(
        numeric_features,
        categorical_features,
        c_value=best_c,
        random_seed=random_seed,
    )

    combined_data = pd.concat(
        [
            train_data[list(feature_columns)],
            validation_data[list(feature_columns)],
        ],
        ignore_index=True,
    )

    combined_outcome = pd.concat(
        [
            train_outcome,
            validation_outcome,
        ],
        ignore_index=True,
    )

    final_model.fit(
        combined_data,
        combined_outcome,
    )

    return TunedBinaryModel(
        model=final_model,
        selection_table=selection_table,
        best_c=best_c,
    )


def _weighted_metric(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Calculate a normalized weighted metric over available bins."""

    valid = (
        values.notna()
        & weights.notna()
    )

    if not valid.any():
        return float("nan")

    selected_weights = weights.loc[
        valid
    ]

    selected_weights = (
        selected_weights
        / selected_weights.sum()
    )

    return float(
        (
            values.loc[valid]
            * selected_weights
        ).sum()
    )


def evaluate_dr_estimators(
    frame: pd.DataFrame,
    *,
    horizon: int,
    bin_edges: np.ndarray,
    propensity_clip: float = 0.05,
) -> DREstimatorResult:
    """Evaluate CC, KM, IPCW, OR, and AIPW against reference outcomes."""

    required = {
        "prediction",
        "outcome",
        "observed",
        "restricted_time",
        "restricted_event",
        "propensity",
        "outcome_regression",
    }

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing estimator columns: {missing}"
        )

    if not 0 < propensity_clip < 0.5:
        raise ValueError(
            "propensity_clip must lie between zero and 0.5"
        )

    data = frame.copy()

    data["_bin"] = pd.cut(
        data["prediction"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    if data["_bin"].isna().any():
        raise ValueError(
            "Some predictions were not assigned to bins"
        )

    outcome = data["outcome"].astype(float)
    observed = data["observed"].astype(int)

    raw_propensity = (
        data["propensity"]
        .astype(float)
        .clip(0.0, 1.0)
    )

    # Only the lower tail creates unstable inverse-observation weights.
    # Keep probabilities of exactly 1.0 unchanged so fully observed data
    # receive weight 1 rather than 1 / (1 - clip).
    propensity = raw_propensity.clip(
        lower=propensity_clip,
        upper=1.0,
    )

    outcome_regression = (
        data["outcome_regression"]
        .astype(float)
        .clip(0.0, 1.0)
    )

    observed_weight = (
        observed
        / propensity
    )

    ipcw_numerator = (
        observed
        * outcome
        / propensity
    )

    residual = np.where(
        observed.eq(1),
        outcome - outcome_regression,
        0.0,
    )

    aipw_pseudo_outcome = (
        outcome_regression
        + observed_weight * residual
    )

    data["_clipped_propensity"] = propensity
    data["_observed_weight"] = observed_weight
    data["_ipcw_numerator"] = ipcw_numerator
    data["_aipw_pseudo_outcome"] = aipw_pseudo_outcome

    n_total = len(data)
    rows: list[dict[str, object]] = []

    for bin_id, group in data.groupby(
        "_bin",
        sort=True,
        observed=True,
    ):
        observed_group = group.loc[
            group["observed"].astype(int).eq(1)
        ]

        reference_risk = float(
            group["outcome"].mean()
        )

        complete_case_risk = (
            float(
                observed_group["outcome"].mean()
            )
            if not observed_group.empty
            else float("nan")
        )

        km_risk = kaplan_meier_event_risk(
            group["restricted_time"].to_numpy(),
            group["restricted_event"].to_numpy(),
            horizon,
        )

        outcome_regression_risk = float(
            group["outcome_regression"].mean()
        )

        ipcw_ht_risk = float(
            group["_ipcw_numerator"].mean()
        )

        denominator = float(
            group["_observed_weight"].sum()
        )

        ipcw_hajek_risk = (
            float(
                group["_ipcw_numerator"].sum()
                / denominator
            )
            if denominator > 0
            else float("nan")
        )

        aipw_risk = float(
            group[
                "_aipw_pseudo_outcome"
            ].mean()
        )

        observed_weights = group.loc[
            group["observed"].astype(int).eq(1),
            "_observed_weight",
        ]

        effective_sample_size = (
            float(
                observed_weights.sum() ** 2
                / np.square(
                    observed_weights
                ).sum()
            )
            if (
                not observed_weights.empty
                and np.square(
                    observed_weights
                ).sum() > 0
            )
            else 0.0
        )

        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(group)),
                "reference_weight": float(
                    len(group) / n_total
                ),
                "mean_prediction": float(
                    group["prediction"].mean()
                ),
                "reference_risk": reference_risk,
                "complete_case_risk": complete_case_risk,
                "kaplan_meier_risk": km_risk,
                "outcome_regression_risk": (
                    outcome_regression_risk
                ),
                "ipcw_ht_risk": ipcw_ht_risk,
                "ipcw_hajek_risk": ipcw_hajek_risk,
                "aipw_risk": aipw_risk,
                "aipw_risk_clipped": float(
                    np.clip(
                        aipw_risk,
                        0.0,
                        1.0,
                    )
                ),
                "n_observed": int(
                    group["observed"].sum()
                ),
                "observed_fraction": float(
                    group["observed"].mean()
                ),
                "mean_raw_propensity": float(
                    group["propensity"].mean()
                ),
                "minimum_raw_propensity": float(
                    group["propensity"].min()
                ),
                "maximum_observed_weight": (
                    float(
                        observed_weights.max()
                    )
                    if not observed_weights.empty
                    else float("nan")
                ),
                "effective_sample_size": (
                    effective_sample_size
                ),
            }
        )

    bin_summary = pd.DataFrame(rows)

    estimators = {
        "reference": "reference_risk",
        "complete_case": "complete_case_risk",
        "kaplan_meier": "kaplan_meier_risk",
        "outcome_regression": "outcome_regression_risk",
        "ipcw_ht": "ipcw_ht_risk",
        "ipcw_hajek": "ipcw_hajek_risk",
        "aipw": "aipw_risk",
        "aipw_clipped": "aipw_risk_clipped",
    }

    scalar_values: dict[str, float] = {}

    reference = bin_summary["reference_risk"]
    prediction = bin_summary["mean_prediction"]
    weights = bin_summary["reference_weight"]

    for estimator_name, column in estimators.items():
        estimate = bin_summary[column]

        scalar_values[
            f"{estimator_name}_ici"
        ] = _weighted_metric(
            (estimate - prediction).abs(),
            weights,
        )

        if estimator_name != "reference":
            scalar_values[
                f"{estimator_name}_mae_to_reference"
            ] = _weighted_metric(
                (estimate - reference).abs(),
                weights,
            )

            scalar_values[
                f"{estimator_name}_signed_gap"
            ] = _weighted_metric(
                estimate - reference,
                weights,
            )

    complete_case_mae = scalar_values[
        "complete_case_mae_to_reference"
    ]

    for estimator_name in (
        "kaplan_meier",
        "outcome_regression",
        "ipcw_ht",
        "ipcw_hajek",
        "aipw",
        "aipw_clipped",
    ):
        scalar_values[
            f"{estimator_name}_improvement_over_complete_case"
        ] = (
            complete_case_mae
            - scalar_values[
                f"{estimator_name}_mae_to_reference"
            ]
        )

    clipped_low = (
        raw_propensity
        < propensity_clip
    )

    # Upper-tail clipping is not used for inverse observation weighting.
    clipped_high = pd.Series(
        False,
        index=raw_propensity.index,
    )

    observed_weights = data.loc[
        observed.eq(1),
        "_observed_weight",
    ]

    overall_ess = float(
        observed_weights.sum() ** 2
        / np.square(
            observed_weights
        ).sum()
    )

    diagnostics = pd.Series(
        {
            "n": float(len(data)),
            "observed_fraction": float(
                observed.mean()
            ),
            "propensity_clip": propensity_clip,
            "raw_propensity_min": float(
                raw_propensity.min()
            ),
            "raw_propensity_q01": float(
                raw_propensity.quantile(0.01)
            ),
            "raw_propensity_q05": float(
                raw_propensity.quantile(0.05)
            ),
            "raw_propensity_median": float(
                raw_propensity.median()
            ),
            "raw_propensity_q95": float(
                raw_propensity.quantile(0.95)
            ),
            "raw_propensity_max": float(
                raw_propensity.max()
            ),
            "fraction_propensity_clipped_low": float(
                clipped_low.mean()
            ),
            "fraction_propensity_clipped_high": float(
                clipped_high.mean()
            ),
            "maximum_observed_weight": float(
                observed_weights.max()
            ),
            "overall_weight_effective_sample_size": (
                overall_ess
            ),
            "aipw_pseudo_outcome_min": float(
                aipw_pseudo_outcome.min()
            ),
            "aipw_pseudo_outcome_max": float(
                aipw_pseudo_outcome.max()
            ),
        },
        name="value",
    )

    return DREstimatorResult(
        bin_summary=bin_summary,
        scalar_summary=pd.Series(
            scalar_values,
            name="value",
        ),
        diagnostics=diagnostics,
        bin_edges=bin_edges,
    )
