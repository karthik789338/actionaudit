"""Analytically quantify oracle IPCW truncation bias."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actionaudit.simulation import (
    hazard_grid,
    scenario_config,
)


SCENARIOS = [
    "independent",
    "observed_dependent",
    "residual_same_direction",
    "residual_opposite_direction",
    "positivity_stress",
]

CLIPS = [
    0.0,
    1e-12,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    2.5e-2,
    5e-2,
    1e-1,
]

N = 20000
HORIZON = 60
SEED = 2027


def survival_before(
    hazards: np.ndarray,
) -> np.ndarray:
    """Return survival immediately before each discrete day."""

    through_day = np.cumprod(
        1.0 - hazards,
        axis=1,
    )

    before_day = np.ones_like(
        through_day
    )

    before_day[
        :,
        1:
    ] = through_day[
        :,
        :-1
    ]

    return before_day


def analyze_scenario(
    scenario: str,
    scenario_index: int,
) -> list[dict[str, object]]:
    """Calculate exact expected IPCW risk for one covariate sample."""

    scenario_seed = (
        SEED
        + 100000 * scenario_index
    )

    config = scenario_config(
        scenario,
        n=N,
        horizon=HORIZON,
        seed=scenario_seed,
    )

    rng = np.random.default_rng(
        scenario_seed
    )

    x1 = rng.normal(
        size=N
    )

    x2 = rng.binomial(
        1,
        0.45,
        size=N,
    ).astype(float)

    latent_u = rng.normal(
        size=N
    )

    event_hazards = hazard_grid(
        x1,
        x2,
        latent_u,
        horizon=HORIZON,
        intercept=config.event_intercept,
        time_slope=config.event_time_slope,
        x1_coefficient=config.event_x1,
        x2_coefficient=config.event_x2,
        u_coefficient=config.event_u,
    )

    censor_hazards = hazard_grid(
        x1,
        x2,
        latent_u,
        horizon=HORIZON,
        intercept=config.censor_intercept,
        time_slope=config.censor_time_slope,
        x1_coefficient=config.censor_x1,
        x2_coefficient=config.censor_x2,
        u_coefficient=config.censor_u,
    )

    event_survival_before = survival_before(
        event_hazards
    )

    censor_survival_before = survival_before(
        censor_hazards
    )

    event_probability_mass = (
        event_survival_before
        * event_hazards
    )

    patient_true_risk = (
        event_probability_mass.sum(
            axis=1
        )
    )

    true_risk = float(
        patient_true_risk.mean()
    )

    total_event_mass = float(
        event_probability_mass.sum()
    )

    rows: list[dict[str, object]] = []

    for clip in CLIPS:
        if clip == 0.0:
            # Exact oracle identity with no truncation:
            # G(t-) / G(t-) = 1 wherever positivity holds.
            observation_ratio = np.ones_like(
                censor_survival_before
            )

            clipped_event_mass = 0.0
        else:
            clipped_g = np.clip(
                censor_survival_before,
                clip,
                1.0,
            )

            observation_ratio = (
                censor_survival_before
                / clipped_g
            )

            clipped_event_mass = float(
                event_probability_mass[
                    censor_survival_before
                    < clip
                ].sum()
            )

        expected_ipcw_by_patient = (
            event_probability_mass
            * observation_ratio
        ).sum(
            axis=1
        )

        expected_ipcw_risk = float(
            expected_ipcw_by_patient.mean()
        )

        rows.append(
            {
                "scenario": scenario,
                "clip": clip,
                "n_covariate_draws": N,
                "true_event_risk": true_risk,
                "analytic_expected_ipcw_risk": (
                    expected_ipcw_risk
                ),
                "analytic_signed_bias": (
                    expected_ipcw_risk
                    - true_risk
                ),
                "analytic_absolute_bias": abs(
                    expected_ipcw_risk
                    - true_risk
                ),
                "fraction_event_probability_mass_clipped": (
                    clipped_event_mass
                    / total_event_mass
                    if total_event_mass > 0
                    else float("nan")
                ),
                "minimum_g_before": float(
                    censor_survival_before.min()
                ),
                "g_before_q001": float(
                    np.quantile(
                        censor_survival_before,
                        0.001,
                    )
                ),
                "g_before_q01": float(
                    np.quantile(
                        censor_survival_before,
                        0.01,
                    )
                ),
            }
        )

    return rows


def main() -> None:
    """Run the analytic truncation-bias diagnostic."""

    output_dir = Path(
        "artifacts/phase3c_ipcw_oracle_diagnostic"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(
        SCENARIOS
    ):
        print(
            f"Analyzing: {scenario}"
        )

        rows.extend(
            analyze_scenario(
                scenario,
                scenario_index,
            )
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "scenario",
                "clip",
            ]
        )
        .reset_index(drop=True)
    )

    results.to_csv(
        output_dir
        / "analytic_ipcw_truncation_bias.csv",
        index=False,
    )

    display_columns = [
        "scenario",
        "clip",
        "true_event_risk",
        "analytic_expected_ipcw_risk",
        "analytic_signed_bias",
        "fraction_event_probability_mass_clipped",
    ]

    print(
        "\nAnalytic IPCW truncation results"
    )

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    exactly_unclipped = results.loc[
        results["clip"].eq(
            0.0
        )
    ]

    maximum_unclipped_bias = float(
        exactly_unclipped[
            "analytic_absolute_bias"
        ].max()
    )

    print(
        "\nMaximum analytic bias with exactly "
        f"no clipping: {maximum_unclipped_bias:.12f}"
    )

    if maximum_unclipped_bias < 1e-12:
        print(
            "Oracle IPCW identity verified."
        )
    else:
        print(
            "Oracle IPCW identity failed; inspect tie ordering "
            "and censoring-survival construction."
        )

    practical_bias = results.loc[
        results["clip"].eq(
            1e-12
        ),
        "analytic_absolute_bias",
    ].max()

    print(
        "Maximum bias at clip=1e-12: "
        f"{practical_bias:.12f}"
    )

    print(
        "Any remaining bias at clip=1e-12 is truncation "
        "bias caused by censoring survival below 1e-12."
    )

    print(
        f"\nSaved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
