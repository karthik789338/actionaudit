"""Tests for ActionAudit decomposition metrics."""

import numpy as np
import pandas as pd
import pytest

from actionaudit import (
    AuditConfig,
    binned_action_audit,
)


def make_frame(
    seed: int = 7,
    n: int = 4000,
) -> pd.DataFrame:
    """Create synthetic patients with two hidden exit pathways."""

    rng = np.random.default_rng(seed)

    prediction = rng.uniform(
        0.02,
        0.98,
        size=n,
    )

    outcome = rng.binomial(
        1,
        prediction,
    )

    home_logit = (
        -1.7
        + 2.1 * prediction
        - 1.0 * outcome
    )

    transfer_logit = (
        -2.2
        + 1.0 * prediction
        + 1.3 * outcome
    )

    home_probability = 1.0 / (
        1.0
        + np.exp(-home_logit)
    )

    transfer_probability = 1.0 / (
        1.0
        + np.exp(-transfer_logit)
    )

    home = (
        rng.uniform(size=n)
        < home_probability
    )

    transfer = (
        ~home
        & (
            rng.uniform(size=n)
            < transfer_probability
        )
    )

    observed = ~(home | transfer)

    action = np.full(
        n,
        None,
        dtype=object,
    )

    action[home] = "home"
    action[transfer] = "external_transfer"

    return pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "prediction": prediction,
            "outcome": outcome,
            "observed": observed.astype(int),
            "action": action,
            "exit_time": np.where(
                observed,
                np.nan,
                rng.uniform(
                    1,
                    59,
                    size=n,
                ),
            ),
        }
    )


def test_exact_decomposition() -> None:
    """Action contributions must exactly reconstruct the calibration gap."""

    result = binned_action_audit(
        make_frame(),
        AuditConfig(
            horizon_days=60,
            boundary="hospital",
            n_bins=10,
            min_bin_size=20,
        ),
    )

    assert (
        result.max_decomposition_residual
        < 1e-12
    )


def test_action_debt_nonnegative_and_aci_bounded() -> None:
    """Absolute action debt must be nonnegative and ACI must be bounded."""

    result = binned_action_audit(
        make_frame(),
        AuditConfig(
            horizon_days=60,
            boundary="hospital",
            n_bins=8,
            min_bin_size=20,
        ),
    )

    assert (
        result.action_summary["aacd"]
        >= 0
    ).all()

    aci = result.bin_summary.loc[
        result.bin_summary["eligible"],
        "pointwise_aci",
    ]

    assert (
        aci >= -1e-9
    ).all()

    assert (
        aci <= 1 + 1e-9
    ).all()


def test_all_observed_has_zero_gap() -> None:
    """No truncation must produce zero observation-induced gap."""

    frame = make_frame()

    frame["observed"] = 1
    frame["action"] = None

    result = binned_action_audit(
        frame,
        AuditConfig(
            horizon_days=60,
            boundary="hospital",
            n_bins=10,
            min_bin_size=20,
        ),
    )

    assert np.allclose(
        result.bin_summary["observed_minus_full"],
        0.0,
    )

    assert np.allclose(
        result.bin_summary["contribution_sum"],
        0.0,
    )


def test_rejects_missing_action_for_unobserved_rows() -> None:
    """Every outcome hidden by the boundary must have an action label."""

    frame = make_frame()

    first_hidden = frame.index[
        frame["observed"].eq(0)
    ][0]

    frame.loc[
        first_hidden,
        "action",
    ] = None

    with pytest.raises(
        ValueError,
        match="must have an exit action",
    ):
        binned_action_audit(
            frame,
            AuditConfig(
                horizon_days=60,
                boundary="hospital",
                n_bins=10,
            ),
        )
