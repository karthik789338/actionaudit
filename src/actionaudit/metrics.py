"""Reference-follow-up ActionAudit metrics.

The initial implementation uses common reference-cohort quantile bins.

Using common bins ensures that:

1. Reference and restricted calibration are evaluated over the same score
   regions.
2. Integrated quantities use the full evaluation cohort as the common
   standardization population.
3. Action-specific contributions reconstruct the observed-versus-reference
   calibration gap up to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import AuditColumns, AuditConfig, validate_audit_frame


@dataclass
class ActionAuditResult:
    """Outputs from a reference-follow-up action-attributed audit."""

    bin_summary: pd.DataFrame
    action_summary: pd.DataFrame
    scalar_summary: pd.Series
    bin_edges: np.ndarray
    config: AuditConfig

    @property
    def max_decomposition_residual(self) -> float:
        """Maximum absolute error in the empirical decomposition."""

        values = (
            self.bin_summary["decomposition_residual"]
            .dropna()
            .abs()
        )

        return (
            float(values.max())
            if not values.empty
            else float("nan")
        )


def _reference_quantile_edges(
    prediction: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Construct quantile-bin edges from the full reference cohort."""

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.unique(
        np.quantile(
            prediction,
            quantiles,
        )
    )

    if edges.size < 3:
        raise ValueError(
            "Predictions have insufficient variation to construct "
            "at least two bins"
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def _weighted_sum(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Return the weighted sum over non-missing value-weight pairs."""

    mask = values.notna() & weights.notna()

    if not mask.any():
        return float("nan")

    return float(
        (
            values.loc[mask]
            * weights.loc[mask]
        ).sum()
    )


def binned_action_audit(
    frame: pd.DataFrame,
    config: AuditConfig,
    columns: AuditColumns = AuditColumns(),
    *,
    action_order: Iterable[str] | None = None,
) -> ActionAuditResult:
    """Compute an exact empirical action decomposition using common bins.

    Parameters
    ----------
    frame:
        One row per patient in a fixed, untouched evaluation cohort.

        Reference outcomes must be available for every included row.

    config:
        Horizon, observation boundary, and binning settings.

    columns:
        Mapping from conceptual variables to DataFrame columns.

    action_order:
        Optional stable action ordering for exported tables and figures.

    Returns
    -------
    ActionAuditResult
        Bin-level curves, action summaries, scalar metrics, and bin edges.
    """

    validate_audit_frame(
        frame,
        columns,
        require_reference_outcomes=True,
    )

    data = frame.copy()

    prediction = (
        data[columns.prediction]
        .astype(float)
        .to_numpy()
    )

    edges = _reference_quantile_edges(
        prediction,
        config.n_bins,
    )

    data["_bin"] = pd.cut(
        prediction,
        bins=edges,
        labels=False,
        include_lowest=True,
    )

    observed_mask = (
        data[columns.observed]
        .astype(int)
        .eq(1)
    )

    missing_mask = ~observed_mask

    discovered_actions = (
        data.loc[missing_mask, columns.action]
        .astype(str)
        .unique()
        .tolist()
    )

    if action_order is None:
        actions = sorted(discovered_actions)
    else:
        actions = list(action_order)

        omitted = sorted(
            set(discovered_actions).difference(actions)
        )

        if omitted:
            raise ValueError(
                f"action_order omitted observed actions: {omitted}"
            )

    bin_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []

    n_total = len(data)

    grouped = data.groupby(
        "_bin",
        observed=True,
        sort=True,
    )

    for bin_id, group in grouped:
        n_bin = len(group)

        n_observed = int(
            group[columns.observed]
            .astype(int)
            .sum()
        )

        eligible = (
            n_bin >= config.min_bin_size
            and n_observed > 0
        )

        mean_prediction = float(
            group[columns.prediction].mean()
        )

        mu_full = (
            float(group[columns.outcome].mean())
            if eligible
            else float("nan")
        )

        observed_group = group[
            group[columns.observed]
            .astype(int)
            .eq(1)
        ]

        mu_observed = (
            float(observed_group[columns.outcome].mean())
            if eligible
            else float("nan")
        )

        raw_gap = (
            mu_observed - mu_full
            if eligible
            else float("nan")
        )

        contribution_sum = 0.0
        absolute_contribution_sum = 0.0
        has_estimable_action = False

        for action in actions:
            action_group = group[
                group[columns.observed]
                .astype(int)
                .eq(0)
                & group[columns.action]
                .astype(str)
                .eq(action)
            ]

            n_action = len(action_group)
            pi_action = n_action / n_bin

            mu_action = (
                float(action_group[columns.outcome].mean())
                if n_action > 0
                else float("nan")
            )

            if eligible and n_action > 0:
                contribution = (
                    pi_action
                    * (mu_observed - mu_action)
                )
            elif eligible:
                contribution = 0.0
            else:
                contribution = float("nan")

            if eligible:
                contribution_sum += contribution
                absolute_contribution_sum += abs(contribution)

                if n_action > 0:
                    has_estimable_action = True

            action_rows.append(
                {
                    "bin": int(bin_id),
                    "action": action,
                    "n_action": n_action,
                    "pi_action": pi_action,
                    "mu_action": mu_action,
                    "contribution": contribution,
                    "reference_weight": n_bin / n_total,
                }
            )

        if eligible and has_estimable_action:
            pointwise_aci = (
                1.0
                - abs(contribution_sum)
                / (
                    absolute_contribution_sum
                    + config.epsilon
                )
            )
        elif eligible:
            pointwise_aci = 0.0
        else:
            pointwise_aci = float("nan")

        residual = (
            raw_gap - contribution_sum
            if eligible
            else float("nan")
        )

        bin_rows.append(
            {
                "bin": int(bin_id),
                "n": n_bin,
                "n_observed": n_observed,
                "observed_fraction": n_observed / n_bin,
                "reference_weight": n_bin / n_total,
                "mean_prediction": mean_prediction,
                "mu_full": mu_full,
                "mu_observed": mu_observed,
                "observed_minus_full": raw_gap,
                "contribution_sum": (
                    contribution_sum
                    if eligible
                    else float("nan")
                ),
                "absolute_contribution_sum": (
                    absolute_contribution_sum
                    if eligible
                    else float("nan")
                ),
                "pointwise_aci": pointwise_aci,
                "decomposition_residual": residual,
                "eligible": eligible,
            }
        )

    bin_summary = (
        pd.DataFrame(bin_rows)
        .sort_values("bin")
        .reset_index(drop=True)
    )

    action_columns = [
        "bin",
        "action",
        "n_action",
        "pi_action",
        "mu_action",
        "contribution",
        "reference_weight",
    ]

    action_long = pd.DataFrame(
        action_rows,
        columns=action_columns,
    )

    if action_long.empty:
        action_summary = pd.DataFrame(
            columns=[
                "action",
                "sab",
                "aacd",
                "hidden_event_rate",
                "n_hidden",
            ]
        )
    else:
        action_long["weighted_contribution"] = (
            action_long["contribution"]
            * action_long["reference_weight"]
        )

        action_long["weighted_absolute_contribution"] = (
            action_long["contribution"].abs()
            * action_long["reference_weight"]
        )

        action_summary = (
            action_long
            .groupby(
                "action",
                sort=False,
            )
            .agg(
                sab=(
                    "weighted_contribution",
                    "sum",
                ),
                aacd=(
                    "weighted_absolute_contribution",
                    "sum",
                ),
                hidden_event_rate=(
                    "mu_action",
                    lambda _: float("nan"),
                ),
                n_hidden=(
                    "n_action",
                    "sum",
                ),
            )
            .reset_index()
        )

        # Hidden event rate must be calculated at the patient level.
        # It should not be an unweighted mean of bin-level event rates.
        for row_index, action in action_summary["action"].items():
            action_group = data[
                data[columns.observed]
                .astype(int)
                .eq(0)
                & data[columns.action]
                .astype(str)
                .eq(action)
            ]

            action_summary.loc[
                row_index,
                "hidden_event_rate",
            ] = (
                float(
                    action_group[columns.outcome].mean()
                )
                if not action_group.empty
                else float("nan")
            )

    eligible_bins = bin_summary["eligible"]

    weights = bin_summary.loc[
        eligible_bins,
        "reference_weight",
    ]

    normalized_weights = weights / weights.sum()

    full_error = (
        bin_summary.loc[
            eligible_bins,
            "mu_full",
        ]
        - bin_summary.loc[
            eligible_bins,
            "mean_prediction",
        ]
    )

    observed_error = (
        bin_summary.loc[
            eligible_bins,
            "mu_observed",
        ]
        - bin_summary.loc[
            eligible_bins,
            "mean_prediction",
        ]
    )

    total_sab = (
        float(action_summary["sab"].sum())
        if not action_summary.empty
        else 0.0
    )

    total_absolute_sab = (
        float(action_summary["sab"].abs().sum())
        if not action_summary.empty
        else 0.0
    )

    global_signed_aci = (
        1.0
        - abs(total_sab)
        / (
            total_absolute_sab
            + config.epsilon
        )
    )

    decomposition_values = (
        bin_summary["decomposition_residual"]
        .dropna()
        .abs()
    )

    max_residual = (
        float(decomposition_values.max())
        if not decomposition_values.empty
        else float("nan")
    )

    scalar_summary = pd.Series(
        {
            "n": n_total,
            "horizon_days": config.horizon_days,
            "reference_observed_fraction": 1.0,
            "restricted_observed_fraction": float(
                observed_mask.mean()
            ),
            "reference_calerr": _weighted_sum(
                full_error.pow(2),
                normalized_weights,
            ),
            "restricted_complete_case_calerr": _weighted_sum(
                observed_error.pow(2),
                normalized_weights,
            ),
            "reference_ici": _weighted_sum(
                full_error.abs(),
                normalized_weights,
            ),
            "restricted_complete_case_ici": _weighted_sum(
                observed_error.abs(),
                normalized_weights,
            ),
            "standardized_signed_gap": _weighted_sum(
                bin_summary.loc[
                    eligible_bins,
                    "observed_minus_full",
                ],
                normalized_weights,
            ),
            "curve_aci": _weighted_sum(
                bin_summary.loc[
                    eligible_bins,
                    "pointwise_aci",
                ],
                normalized_weights,
            ),
            "global_signed_aci": global_signed_aci,
            "max_decomposition_residual": max_residual,
        },
        name="value",
    )

    return ActionAuditResult(
        bin_summary=bin_summary,
        action_summary=action_summary,
        scalar_summary=scalar_summary,
        bin_edges=edges,
        config=config,
    )
