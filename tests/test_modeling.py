"""Tests for frozen-model development utilities."""

import numpy as np
import pandas as pd

from actionaudit.modeling import (
    create_feature_manifest,
    create_patient_split,
    tune_and_refit_logistic,
)


def synthetic_model_frame(
    seed: int = 12,
    n: int = 800,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create mixed numeric/categorical prediction data."""

    rng = np.random.default_rng(seed)

    age = rng.normal(
        65,
        12,
        size=n,
    )

    biomarker = rng.normal(
        size=n,
    )

    sex = rng.choice(
        [
            "male",
            "female",
        ],
        size=n,
    )

    site = rng.choice(
        [
            "A",
            "B",
            "C",
        ],
        size=n,
    )

    missing = rng.uniform(
        size=n
    ) < 0.15

    biomarker[
        missing
    ] = np.nan

    logit = (
        -5.0
        + 0.055 * age
        + 0.8 * np.nan_to_num(
            biomarker,
            nan=0.0,
        )
        + 0.35 * (
            sex == "male"
        )
    )

    probability = 1.0 / (
        1.0
        + np.exp(-logit)
    )

    outcome = pd.Series(
        rng.binomial(
            1,
            probability,
        )
    )

    data = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "age": age,
            "biomarker": biomarker,
            "sex": sex,
            "hosp": site,
            "constant": 1,
        }
    )

    return (
        data,
        outcome,
    )


def test_split_is_complete_and_disjoint() -> None:
    """Every patient must appear in exactly one split."""

    data, outcome = synthetic_model_frame()

    split = create_patient_split(
        data,
        outcome,
        random_seed=2027,
    )

    combined = np.concatenate(
        [
            split.train_index,
            split.validation_index,
            split.test_index,
        ]
    )

    assert len(combined) == len(data)
    assert len(np.unique(combined)) == len(data)
    assert len(split.train_index) == 480
    assert len(split.validation_index) == 160
    assert len(split.test_index) == 160


def test_logistic_pipeline_handles_mixed_missing_data() -> None:
    """The model must fit mixed data and produce valid probabilities."""

    data, outcome = synthetic_model_frame()

    manifest = create_feature_manifest(
        data,
        [
            "patient_id",
            "age",
            "biomarker",
            "sex",
            "hosp",
            "constant",
        ],
    )

    included = (
        manifest.loc[
            manifest["included"],
            "feature",
        ]
        .tolist()
    )

    assert "patient_id" not in included
    assert "constant" not in included

    split = create_patient_split(
        data,
        outcome,
        random_seed=2027,
    )

    result = tune_and_refit_logistic(
        data,
        outcome,
        split,
        included,
        c_grid=(
            0.1,
            1.0,
        ),
        random_seed=2027,
    )

    probability = result.model.predict_proba(
        data.iloc[
            split.test_index
        ][included]
    )[:, 1]

    assert len(probability) == len(
        split.test_index
    )

    assert np.isfinite(
        probability
    ).all()

    assert (
        probability >= 0
    ).all()

    assert (
        probability <= 1
    ).all()
