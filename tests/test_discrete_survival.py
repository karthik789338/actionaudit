"""Tests for corrected right-censoring estimators."""

import numpy as np
import pandas as pd

from actionaudit.discrete_survival import (
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
    expand_person_period,
)


def test_person_period_targets_final_interval() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "restricted_time": [2.0, 3.0],
            "restricted_event": [1, 0],
            "restricted_censored_by_discharge": [0, 1],
        }
    )

    event_rows = expand_person_period(
        frame,
        ["x"],
        horizon=5,
        target="event",
    )

    censor_rows = expand_person_period(
        frame,
        ["x"],
        horizon=5,
        target="censor",
    )

    assert len(event_rows) == 5
    assert event_rows["hazard_target"].sum() == 1
    assert censor_rows["hazard_target"].sum() == 1

    assert event_rows.loc[
        1,
        "hazard_target",
    ] == 1

    assert censor_rows.loc[
        4,
        "hazard_target",
    ] == 1


def test_no_censoring_ipcw_matches_empirical_event_risk() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.8, 0.9],
            "outcome": [0, 0, 1, 1],
            "observed": [1, 1, 1, 1],
            "restricted_time": [5, 5, 2, 3],
            "restricted_event": [0, 0, 1, 1],
            "restricted_censored_by_discharge": [0, 0, 0, 0],
        }
    )

    event_hazards = np.full(
        (4, 5),
        0.1,
    )

    censor_hazards = np.zeros(
        (4, 5),
    )

    augmented = construct_survival_nuisance_predictions(
        frame,
        event_hazards=event_hazards,
        censor_hazards=censor_hazards,
        horizon=5,
        censor_survival_clip=0.05,
    )

    result = evaluate_survival_estimators(
        augmented,
        horizon=5,
        bin_edges=np.array(
            [
                -np.inf,
                0.5,
                np.inf,
            ]
        ),
        censor_survival_clip=0.05,
    )

    assert np.allclose(
        result.bin_summary[
            "reference_risk"
        ],
        result.bin_summary[
            "ipcw_event_risk"
        ],
    )


def test_event_weight_uses_survival_before_event_day() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [0.5],
            "outcome": [1],
            "observed": [1],
            "restricted_time": [2],
            "restricted_event": [1],
            "restricted_censored_by_discharge": [0],
        }
    )

    event_hazards = np.full(
        (1, 3),
        0.1,
    )

    censor_hazards = np.array(
        [
            [
                0.2,
                0.3,
                0.4,
            ]
        ]
    )

    augmented = construct_survival_nuisance_predictions(
        frame,
        event_hazards=event_hazards,
        censor_hazards=censor_hazards,
        horizon=3,
        censor_survival_clip=0.01,
    )

    # Event on day 2 uses G(2-) = 1 - censor hazard on day 1.
    assert np.isclose(
        augmented.loc[
            0,
            "ipcw_event_weight",
        ],
        1 / 0.8,
    )
