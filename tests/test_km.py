"""Tests for Kaplan-Meier calibration utilities."""

import numpy as np
import pandas as pd

from actionaudit.km import (
    kaplan_meier_event_risk,
    make_hospital_restricted_survival,
)


def test_km_without_censoring_matches_empirical_risk() -> None:
    """Without censoring, KM risk equals the empirical event rate."""

    time = np.array(
        [1, 2, 5, 5],
        dtype=float,
    )

    event = np.array(
        [1, 1, 0, 0],
        dtype=int,
    )

    estimate = kaplan_meier_event_risk(
        time,
        event,
        horizon=5,
    )

    assert np.isclose(
        estimate,
        0.5,
    )


def test_km_with_censoring() -> None:
    """Check a hand-calculated Kaplan-Meier example."""

    time = np.array(
        [1, 2, 3, 4],
        dtype=float,
    )

    event = np.array(
        [1, 0, 1, 0],
        dtype=int,
    )

    estimate = kaplan_meier_event_risk(
        time,
        event,
        horizon=4,
    )

    # S = (1 - 1/4) * (1 - 1/2) = 0.375
    # Risk = 1 - 0.375 = 0.625
    assert np.isclose(
        estimate,
        0.625,
    )


def test_restricted_survival_construction() -> None:
    """Alive discharge censors while hospital death remains an event."""

    frame = pd.DataFrame(
        {
            "death": [1, 1, 0, 0],
            "d.time": [10, 40, 100, 100],
            "hospdead": [1, 0, 0, 0],
            "slos": [10, 5, 70, 20],
        }
    )

    result = make_hospital_restricted_survival(
        frame,
        horizon=60,
    )

    assert result.loc[
        0,
        "restricted_event",
    ] == 1

    assert result.loc[
        0,
        "restricted_time",
    ] == 10

    assert result.loc[
        1,
        "restricted_event",
    ] == 0

    assert result.loc[
        1,
        "restricted_time",
    ] == 5

    assert result.loc[
        2,
        "restricted_time",
    ] == 60

    assert result.loc[
        3,
        "restricted_censored_by_discharge",
    ] == 1
