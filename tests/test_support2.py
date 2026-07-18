"""Tests for SUPPORT2 horizon construction and feasibility checks."""

import numpy as np
import pandas as pd

from actionaudit.support2 import (
    build_support2_horizon_view,
    support2_feasibility_audit,
)


def small_support2() -> pd.DataFrame:
    """Create a small artificial SUPPORT2-style cohort."""

    return pd.DataFrame(
        {
            "death": [
                1,
                1,
                0,
                0,
                1,
                0,
            ],
            "d.time": [
                10,
                40,
                100,
                20,
                90,
                200,
            ],
            "hospdead": [
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            "slos": [
                10,
                5,
                70,
                5,
                10,
                80,
            ],
            "age": [
                60,
                55,
                70,
                45,
                80,
                66,
            ],
            "hday": [
                3,
                4,
                2,
                1,
                5,
                3,
            ],
        }
    )


def test_horizon_view_identifies_post_discharge_death() -> None:
    """A post-discharge death should be hidden by the hospital boundary."""

    view = build_support2_horizon_view(
        small_support2(),
        60,
    ).frame

    # Death occurred in hospital.
    assert view.loc[0, "observed"] == 1

    # Death occurred after alive discharge.
    assert view.loc[1, "observed"] == 0
    assert view.loc[1, "outcome"] == 1

    assert (
        view.loc[
            1,
            "post_discharge_event_by_horizon",
        ]
        == 1
    )

    # Final follow-up was before the horizon without a known death.
    assert np.isnan(
        view.loc[3, "outcome"]
    )


def test_feasibility_counts_reference_unknown() -> None:
    """The feasibility audit should identify unresolved outcomes."""

    result = support2_feasibility_audit(
        small_support2(),
        horizons=(60,),
    )

    row = result.horizon_summary.iloc[0]

    assert (
        row["n_reference_unknown"]
        == 1
    )

    assert (
        row["n_hidden_events"]
        == 1
    )
