"""Controlled survival and discharge-censoring simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for one simulated survival cohort."""

    scenario: str
    n: int = 5000
    horizon: int = 60
    seed: int = 2027

    event_intercept: float = -4.70
    event_time_slope: float = 0.012
    event_x1: float = 0.70
    event_x2: float = 0.65
    event_u: float = 0.0

    censor_intercept: float = -4.10
    censor_time_slope: float = 0.012
    censor_x1: float = 0.0
    censor_x2: float = 0.0
    censor_u: float = 0.0

    prediction_x1: float = 0.70
    prediction_x2: float = 0.0


@dataclass
class SimulationOutput:
    """Simulated cohort and known nuisance hazard grids."""

    frame: pd.DataFrame
    true_event_hazards: np.ndarray
    true_censor_hazards: np.ndarray
    working_event_hazards: np.ndarray
    working_censor_hazards: np.ndarray


def expit(value: np.ndarray) -> np.ndarray:
    """Numerically stable inverse logit."""

    value = np.asarray(
        value,
        dtype=float,
    )

    result = np.empty_like(value)

    positive = value >= 0
    negative = ~positive

    result[positive] = (
        1.0
        / (
            1.0
            + np.exp(
                -value[positive]
            )
        )
    )

    exponential = np.exp(
        value[negative]
    )

    result[negative] = (
        exponential
        / (
            1.0
            + exponential
        )
    )

    return result


def scenario_config(
    scenario: str,
    *,
    n: int,
    horizon: int,
    seed: int,
) -> SimulationConfig:
    """Return the frozen configuration for a named scenario."""

    base = {
        "scenario": scenario,
        "n": n,
        "horizon": horizon,
        "seed": seed,
    }

    if scenario == "independent":
        return SimulationConfig(
            **base,
            censor_intercept=-4.10,
            censor_x1=0.0,
            censor_x2=0.0,
            censor_u=0.0,
        )

    if scenario == "observed_dependent":
        return SimulationConfig(
            **base,
            censor_intercept=-4.10,
            censor_x1=-0.75,
            censor_x2=1.00,
            censor_u=0.0,
        )

    if scenario == "residual_same_direction":
        return SimulationConfig(
            **base,
            event_u=0.85,
            censor_intercept=-4.10,
            censor_x1=-0.75,
            censor_x2=1.00,
            censor_u=0.90,
        )

    if scenario == "residual_opposite_direction":
        return SimulationConfig(
            **base,
            event_u=0.85,
            censor_intercept=-4.10,
            censor_x1=-0.75,
            censor_x2=1.00,
            censor_u=-0.90,
        )

    if scenario == "positivity_stress":
        return SimulationConfig(
            **base,
            censor_intercept=-3.35,
            censor_time_slope=0.018,
            censor_x1=-1.20,
            censor_x2=1.50,
            censor_u=0.0,
        )

    raise ValueError(
        f"Unknown simulation scenario: {scenario}"
    )


def hazard_grid(
    x1: np.ndarray,
    x2: np.ndarray,
    u: np.ndarray,
    *,
    horizon: int,
    intercept: float,
    time_slope: float,
    x1_coefficient: float,
    x2_coefficient: float,
    u_coefficient: float,
) -> np.ndarray:
    """Construct one daily discrete-time hazard grid."""

    days = np.arange(
        1,
        horizon + 1,
        dtype=float,
    )

    scaled_time = (
        days
        / horizon
    )

    linear_predictor = (
        intercept
        + time_slope * days[None, :]
        + x1_coefficient * x1[:, None]
        + x2_coefficient * x2[:, None]
        + u_coefficient * u[:, None]
        + 0.10 * np.square(
            scaled_time[None, :]
        )
    )

    return np.clip(
        expit(linear_predictor),
        1e-8,
        1.0 - 1e-8,
    )


def sample_first_time(
    hazards: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the first event day from discrete hazards."""

    if hazards.ndim != 2:
        raise ValueError(
            "hazards must be a two-dimensional array"
        )

    n, horizon = hazards.shape

    event_draws = (
        rng.uniform(
            size=hazards.shape
        )
        < hazards
    )

    any_event = event_draws.any(
        axis=1
    )

    first_time = np.full(
        n,
        horizon + 1,
        dtype=int,
    )

    first_time[
        any_event
    ] = (
        np.argmax(
            event_draws[
                any_event
            ],
            axis=1,
        )
        + 1
    )

    return first_time


def simulate_cohort(
    config: SimulationConfig,
) -> SimulationOutput:
    """Simulate latent event and discharge-censoring times.

    Events are ordered before censoring when both occur on the same
    discrete day.
    """

    if config.n < 10:
        raise ValueError(
            "Simulation size must be at least 10"
        )

    if config.horizon < 2:
        raise ValueError(
            "Simulation horizon must be at least 2"
        )

    rng = np.random.default_rng(
        config.seed
    )

    x1 = rng.normal(
        size=config.n
    )

    x2 = rng.binomial(
        1,
        0.45,
        size=config.n,
    ).astype(float)

    u = rng.normal(
        size=config.n
    )

    true_event_hazards = hazard_grid(
        x1,
        x2,
        u,
        horizon=config.horizon,
        intercept=config.event_intercept,
        time_slope=config.event_time_slope,
        x1_coefficient=config.event_x1,
        x2_coefficient=config.event_x2,
        u_coefficient=config.event_u,
    )

    true_censor_hazards = hazard_grid(
        x1,
        x2,
        u,
        horizon=config.horizon,
        intercept=config.censor_intercept,
        time_slope=config.censor_time_slope,
        x1_coefficient=config.censor_x1,
        x2_coefficient=config.censor_x2,
        u_coefficient=config.censor_u,
    )

    # Working nuisance functions omit latent U.
    working_event_hazards = hazard_grid(
        x1,
        x2,
        np.zeros_like(u),
        horizon=config.horizon,
        intercept=config.event_intercept,
        time_slope=config.event_time_slope,
        x1_coefficient=config.event_x1,
        x2_coefficient=config.event_x2,
        u_coefficient=0.0,
    )

    working_censor_hazards = hazard_grid(
        x1,
        x2,
        np.zeros_like(u),
        horizon=config.horizon,
        intercept=config.censor_intercept,
        time_slope=config.censor_time_slope,
        x1_coefficient=config.censor_x1,
        x2_coefficient=config.censor_x2,
        u_coefficient=0.0,
    )

    # The fixed prediction score intentionally omits X2 and U.
    prediction_hazards = hazard_grid(
        x1,
        np.zeros_like(x2),
        np.zeros_like(u),
        horizon=config.horizon,
        intercept=config.event_intercept,
        time_slope=config.event_time_slope,
        x1_coefficient=config.prediction_x1,
        x2_coefficient=config.prediction_x2,
        u_coefficient=0.0,
    )

    prediction = (
        1.0
        - np.prod(
            1.0 - prediction_hazards,
            axis=1,
        )
    )

    event_time = sample_first_time(
        true_event_hazards,
        rng,
    )

    censor_time = sample_first_time(
        true_censor_hazards,
        rng,
    )

    reference_event = (
        event_time
        <= config.horizon
    )

    restricted_event = (
        reference_event
        & (
            event_time
            <= censor_time
        )
    )

    discharge_censored = (
        censor_time
        < event_time
    ) & (
        censor_time
        <= config.horizon
    )

    restricted_time = np.where(
        restricted_event,
        event_time,
        np.where(
            discharge_censored,
            censor_time,
            config.horizon,
        ),
    )

    outcome_observed = (
        restricted_event
        | (
            censor_time
            > config.horizon
        )
    )

    hidden_event = (
        reference_event
        & ~outcome_observed
    )

    frame = pd.DataFrame(
        {
            "patient_id": np.arange(
                config.n,
                dtype=int,
            ),
            "x1": x1,
            "x2": x2,
            "latent_u": u,
            "prediction": prediction,
            "outcome": (
                reference_event.astype(int)
            ),
            "observed": (
                outcome_observed.astype(int)
            ),
            "restricted_time": (
                restricted_time.astype(float)
            ),
            "restricted_event": (
                restricted_event.astype(int)
            ),
            "restricted_censored_by_discharge": (
                discharge_censored.astype(int)
            ),
            "hidden_event": (
                hidden_event.astype(int)
            ),
            "latent_event_time": event_time,
            "latent_censor_time": censor_time,
            "action": np.where(
                discharge_censored,
                "alive_discharge",
                None,
            ),
        }
    )

    return SimulationOutput(
        frame=frame,
        true_event_hazards=true_event_hazards,
        true_censor_hazards=true_censor_hazards,
        working_event_hazards=working_event_hazards,
        working_censor_hazards=working_censor_hazards,
    )
