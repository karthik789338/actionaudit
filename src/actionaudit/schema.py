"""Input schema and validation for an ActionAudit analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AuditColumns:
    """Column names required by the fixed-horizon audit.

    `outcome` is the reference binary outcome and must be known for the
    reference-follow-up study.

    `observed` describes whether that outcome would be observable under the
    selected restricted observation boundary.

    `action` is required only when `observed == 0`.
    """

    prediction: Hashable = "prediction"
    outcome: Hashable = "outcome"
    observed: Hashable = "observed"
    action: Hashable = "action"
    exit_time: Hashable = "exit_time"
    patient_id: Hashable = "patient_id"


@dataclass(frozen=True)
class AuditConfig:
    """Frozen design choices for one fixed-horizon audit."""

    horizon_days: int
    boundary: str
    n_bins: int = 10
    min_bin_size: int = 25
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")

        if not self.boundary.strip():
            raise ValueError("boundary must be a non-empty label")

        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")

        if self.min_bin_size < 1:
            raise ValueError("min_bin_size must be at least 1")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def validate_audit_frame(
    frame: pd.DataFrame,
    columns: AuditColumns = AuditColumns(),
    *,
    require_reference_outcomes: bool = True,
) -> None:
    """Validate an ActionAudit input frame.

    Raises
    ------
    ValueError
        If required columns are missing or contain invalid values.
    """

    required = [
        columns.prediction,
        columns.observed,
        columns.action,
    ]

    if require_reference_outcomes:
        required.append(columns.outcome)

    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if frame.empty:
        raise ValueError("Audit frame is empty")

    prediction = pd.to_numeric(
        frame[columns.prediction],
        errors="coerce",
    )

    if prediction.isna().any():
        raise ValueError("Predictions must be numeric and non-missing")

    if np.isinf(prediction.to_numpy()).any():
        raise ValueError("Predictions must be finite")

    if not prediction.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Predictions must lie in [0, 1]")

    observed = pd.to_numeric(
        frame[columns.observed],
        errors="coerce",
    )

    if observed.isna().any() or not observed.isin([0, 1]).all():
        raise ValueError("Observed indicator must contain only 0/1")

    if require_reference_outcomes:
        outcome = pd.to_numeric(
            frame[columns.outcome],
            errors="coerce",
        )

        if outcome.isna().any() or not outcome.isin([0, 1]).all():
            raise ValueError("Reference outcome must contain only 0/1")

    missing_actions = frame.loc[
        observed.eq(0),
        columns.action,
    ].isna()

    if missing_actions.any():
        raise ValueError(
            "Every restricted-unobserved row must have an exit action"
        )
