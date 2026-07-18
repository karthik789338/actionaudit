"""Tests for controlled censoring simulations."""

import numpy as np

from actionaudit.simulation import (
    SimulationConfig,
    scenario_config,
    simulate_cohort,
)


def test_simulation_shapes_and_ranges() -> None:
    config = scenario_config(
        "observed_dependent",
        n=500,
        horizon=30,
        seed=10,
    )

    result = simulate_cohort(
        config
    )

    assert len(result.frame) == 500

    assert result.true_event_hazards.shape == (
        500,
        30,
    )

    assert result.true_censor_hazards.shape == (
        500,
        30,
    )

    assert result.frame[
        "prediction"
    ].between(
        0,
        1,
        inclusive="both",
    ).all()

    assert result.frame[
        "outcome"
    ].isin(
        [0, 1]
    ).all()

    assert result.frame[
        "observed"
    ].isin(
        [0, 1]
    ).all()


def test_no_censoring_makes_every_outcome_observed() -> None:
    config = SimulationConfig(
        scenario="no_censoring",
        n=1000,
        horizon=30,
        seed=12,
        censor_intercept=-50.0,
    )

    result = simulate_cohort(
        config
    )

    assert (
        result.frame[
            "observed"
        ]
        == 1
    ).all()

    assert (
        result.frame[
            "restricted_censored_by_discharge"
        ]
        == 0
    ).all()

    assert np.array_equal(
        result.frame[
            "restricted_event"
        ].to_numpy(),
        result.frame[
            "outcome"
        ].to_numpy(),
    )
