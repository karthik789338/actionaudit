"""Tests for IPCW and AIPW calibration estimators."""

import numpy as np
import pandas as pd

from actionaudit.dr import (
    evaluate_dr_estimators,
)


def test_all_observed_with_perfect_outcome_model() -> None:
    """IPCW and AIPW equal reference when all outcomes are observed."""

    frame = pd.DataFrame(
        {
            "prediction": [
                0.1,
                0.2,
                0.7,
                0.9,
            ],
            "outcome": [
                0,
                0,
                1,
                1,
            ],
            "observed": [
                1,
                1,
                1,
                1,
            ],
            "restricted_time": [
                60,
                60,
                10,
                20,
            ],
            "restricted_event": [
                0,
                0,
                1,
                1,
            ],
            "propensity": [
                1,
                1,
                1,
                1,
            ],
            "outcome_regression": [
                0,
                0,
                1,
                1,
            ],
        }
    )

    result = evaluate_dr_estimators(
        frame,
        horizon=60,
        bin_edges=np.array(
            [
                -np.inf,
                0.5,
                np.inf,
            ]
        ),
        propensity_clip=0.01,
    )

    assert np.allclose(
        result.bin_summary[
            "reference_risk"
        ],
        result.bin_summary[
            "ipcw_ht_risk"
        ],
    )

    assert np.allclose(
        result.bin_summary[
            "reference_risk"
        ],
        result.bin_summary[
            "ipcw_hajek_risk"
        ],
    )

    assert np.allclose(
        result.bin_summary[
            "reference_risk"
        ],
        result.bin_summary[
            "aipw_risk"
        ],
    )


def test_propensity_clipping_limits_weights() -> None:
    """Small propensity values must be clipped before weighting."""

    frame = pd.DataFrame(
        {
            "prediction": [
                0.1,
                0.2,
                0.7,
                0.9,
            ],
            "outcome": [
                0,
                1,
                0,
                1,
            ],
            "observed": [
                1,
                0,
                1,
                0,
            ],
            "restricted_time": [
                60,
                5,
                60,
                10,
            ],
            "restricted_event": [
                0,
                0,
                0,
                0,
            ],
            "propensity": [
                0.001,
                0.1,
                0.02,
                0.2,
            ],
            "outcome_regression": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
        }
    )

    result = evaluate_dr_estimators(
        frame,
        horizon=60,
        bin_edges=np.array(
            [
                -np.inf,
                0.5,
                np.inf,
            ]
        ),
        propensity_clip=0.05,
    )

    assert np.isclose(
        result.diagnostics[
            "maximum_observed_weight"
        ],
        20.0,
    )

    assert (
        result.diagnostics[
            "fraction_propensity_clipped_low"
        ]
        == 0.5
    )
