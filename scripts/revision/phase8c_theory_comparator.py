#!/usr/bin/env python3
"""
Phase 8C — Cross-fitted survival AIPCW / one-step comparator.

This is the ONE final new rebuttal experiment.

Goal
----
Benchmark a theory-motivated augmented inverse-probability-of-censoring
(AIPCW) one-step estimator against the existing complete-case, KM,
survival outcome-regression, and IPCW baselines while overlap is
progressively weakened at approximately fixed overall outcome visibility.

Important scope
---------------
This is NOT the Ulichney-Coston selective-label DML estimator verbatim.
Their estimator targets a general loss under selective labels/covariate
shift. Our data are right-censored time-to-event data, so binary label
observability and survival risk-set weighting are not interchangeable.

Instead, we use the standard discrete-time survival influence-function
augmentation:
    F_hat_tau(X)
    + sum_t [ S_hat_tau(X) / S_hat_t(X) ]
              / G_hat(t- | X)
              * { dN_t - Y_t lambda_hat_t(X) }

with cross-fitted event and censoring hazards.  This is the fixed-horizon
event-risk analogue of AIPCW / one-step estimation under conditionally
independent censoring.  It is doubly robust in the usual sense: the
augmentation is mean-zero when the event model is correct, and correct
censoring weights recover the full-data martingale contribution when the
censoring model is correct, subject to positivity/regularity.

Run from the ActionAudit repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path.cwd()
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

if not SRC.exists() or not SCRIPTS.exists():
    raise SystemExit("Run this script from the ActionAudit repository root.")

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from actionaudit.discrete_survival import (  # noqa: E402
    construct_survival_nuisance_predictions,
    evaluate_survival_estimators,
)
from actionaudit.km import (  # noqa: E402
    kaplan_meier_event_risk,
    reference_quantile_edges,
)
from actionaudit.simulation import (  # noqa: E402
    SimulationConfig,
    hazard_grid,
    simulate_cohort,
)
from phase3d_learned_nuisance_simulation import (  # noqa: E402
    fit_hazard_model,
    predict_hazard_grid,
)

warnings.filterwarnings(
    "ignore",
    message=".*penalty.*deprecated.*",
    category=FutureWarning,
)

STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
FEATURES = ["x1", "x2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="revision_kdd2027/phase8c_theory_comparator",
    )
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--test-n", type=int, default=5000)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--logistic-c", type=float, default=100.0)
    p.add_argument("--aipcw-clip", type=float, default=0.01)
    p.add_argument("--ipcw-clip", type=float, default=0.05)
    p.add_argument("--target-observed-fraction", type=float, default=0.30)
    p.add_argument("--calibration-n", type=int, default=50000)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def parse_strengths() -> list[float]:
    return list(STRENGTHS)


def baseline_event_hazards(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    return hazard_grid(
        x1,
        x2,
        np.zeros_like(x1),
        horizon=horizon,
        intercept=-4.70,
        time_slope=0.012,
        x1_coefficient=0.70,
        x2_coefficient=0.65,
        u_coefficient=0.0,
    )


def censor_coefficients(strength: float) -> tuple[float, float]:
    """
    Frozen overlap path.

    strength=0 has no covariate-dependent censoring.
    strength=1 produces a lower-tail censor-survival regime close in scale
    to the severe Phase-7B endpoint while the overall observed fraction is
    recalibrated to ~0.30.
    """
    return -0.75 * strength, 0.95 * strength


def expected_observed_fraction(
    event_hazards: np.ndarray,
    censor_hazards: np.ndarray,
) -> float:
    """
    E[R_tau] under discrete event-before-censor tie handling.

    P(outcome known by tau | X)
      = sum_t P(T=t, C>=t | X) + P(T>tau, C>tau | X).

    The second term must include event survival through tau. Adding only
    P(C>tau|X) would double-count subjects who experience an observed event
    before tau and also have C>tau.
    """
    n, horizon = event_hazards.shape

    event_survival_before = np.ones_like(event_hazards)
    censor_survival_before = np.ones_like(censor_hazards)

    if horizon > 1:
        event_survival_before[:, 1:] = np.cumprod(
            1.0 - event_hazards[:, :-1],
            axis=1,
        )
        censor_survival_before[:, 1:] = np.cumprod(
            1.0 - censor_hazards[:, :-1],
            axis=1,
        )

    event_observed = (
        event_survival_before
        * event_hazards
        * censor_survival_before
    ).sum(axis=1)

    event_survival_horizon = np.prod(
        1.0 - event_hazards,
        axis=1,
    )

    censor_survival_horizon = np.prod(
        1.0 - censor_hazards,
        axis=1,
    )

    no_event_and_no_censor_through_horizon = (
        event_survival_horizon
        * censor_survival_horizon
    )

    return float(
        np.mean(
            event_observed
            + no_event_and_no_censor_through_horizon
        )
    )


def calibrate_intercept(
    strength: float,
    *,
    target_observed_fraction: float,
    calibration_n: int,
    horizon: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    x1 = rng.normal(size=calibration_n)
    x2 = rng.binomial(1, 0.45, size=calibration_n).astype(float)

    event_hazards = baseline_event_hazards(
        x1,
        x2,
        horizon=horizon,
    )

    cx1, cx2 = censor_coefficients(strength)

    lo = -8.0
    hi = 2.0

    def evaluate(intercept: float):
        censor_hazards = hazard_grid(
            x1,
            x2,
            np.zeros_like(x1),
            horizon=horizon,
            intercept=intercept,
            time_slope=0.012,
            x1_coefficient=cx1,
            x2_coefficient=cx2,
            u_coefficient=0.0,
        )
        obs = expected_observed_fraction(
            event_hazards,
            censor_hazards,
        )
        g_tau = np.prod(
            1.0 - censor_hazards,
            axis=1,
        )
        return obs, g_tau

    lo_obs, _ = evaluate(lo)
    hi_obs, _ = evaluate(hi)

    if not (
        lo_obs >= target_observed_fraction
        and hi_obs <= target_observed_fraction
    ):
        raise RuntimeError(
            "Calibration bracket does not contain target: "
            f"strength={strength}, lo_obs={lo_obs}, hi_obs={hi_obs}"
        )

    mid = None
    mid_obs = None
    g_tau = None

    for _ in range(45):
        mid = (lo + hi) / 2.0
        mid_obs, g_tau = evaluate(mid)

        # More positive censor intercept -> more censoring -> less visibility.
        if mid_obs > target_observed_fraction:
            lo = mid
        else:
            hi = mid

    assert mid is not None
    assert mid_obs is not None
    assert g_tau is not None

    return {
        "strength": float(strength),
        "censor_intercept": float(mid),
        "censor_x1": float(cx1),
        "censor_x2": float(cx2),
        "expected_observed_fraction": float(mid_obs),
        "true_g_tau_q01": float(np.quantile(g_tau, 0.01)),
        "true_g_tau_q05": float(np.quantile(g_tau, 0.05)),
        "true_g_tau_median": float(np.median(g_tau)),
        "true_fraction_g_tau_lt_0p05": float(np.mean(g_tau < 0.05)),
    }


def make_config(
    *,
    strength: float,
    censor_intercept: float,
    n: int,
    horizon: int,
    seed: int,
) -> SimulationConfig:
    cx1, cx2 = censor_coefficients(strength)

    return SimulationConfig(
        scenario=f"phase8c_strength_{strength:g}",
        n=n,
        horizon=horizon,
        seed=seed,
        event_intercept=-4.70,
        event_time_slope=0.012,
        event_x1=0.70,
        event_x2=0.65,
        event_u=0.0,
        censor_intercept=censor_intercept,
        censor_time_slope=0.012,
        censor_x1=cx1,
        censor_x2=cx2,
        censor_u=0.0,
        prediction_x1=0.70,
        prediction_x2=0.0,
    )


def cross_fitted_hazards(
    frame: pd.DataFrame,
    *,
    horizon: int,
    folds: int,
    logistic_c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)

    event_hazards = np.empty((n, horizon), dtype=float)
    censor_hazards = np.empty((n, horizon), dtype=float)

    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame)):
        train = frame.iloc[train_idx].reset_index(drop=True)
        held = frame.iloc[test_idx].reset_index(drop=True)

        event_model = fit_hazard_model(
            train,
            FEATURES,
            horizon=horizon,
            target="event",
            c_value=logistic_c,
            seed=seed + 100 + fold,
        )

        censor_model = fit_hazard_model(
            train,
            FEATURES,
            horizon=horizon,
            target="censor",
            c_value=logistic_c,
            seed=seed + 200 + fold,
        )

        event_hazards[test_idx] = predict_hazard_grid(
            event_model,
            held,
            FEATURES,
            horizon=horizon,
        )

        censor_hazards[test_idx] = predict_hazard_grid(
            censor_model,
            held,
            FEATURES,
            horizon=horizon,
        )

    return event_hazards, censor_hazards


def aipcw_event_pseudo_outcome(
    frame: pd.DataFrame,
    *,
    event_hazards: np.ndarray,
    censor_hazards: np.ndarray,
    horizon: int,
    censor_survival_clip: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Cross-fitted one-step/AIPCW pseudo-outcome for F(tau).

    Event process:
      lambda_t(X) = P(T=t | T>=t, X)
      S_t(X) = product_{s<=t}(1-lambda_s(X))

    Censoring process:
      G(t-|X) = P(C>=t | X)

    One-step event-risk pseudo-outcome:
      1-S_tau(X)
      + sum_t [S_tau(X)/S_t(X)] / G(t-|X)
              * [dN_t - Y_t lambda_t(X)].
    """
    n = len(frame)

    if event_hazards.shape != (n, horizon):
        raise ValueError("event_hazards has wrong shape")
    if censor_hazards.shape != (n, horizon):
        raise ValueError("censor_hazards has wrong shape")

    event_survival = np.cumprod(
        1.0 - event_hazards,
        axis=1,
    )
    s_tau = event_survival[:, -1]
    event_risk = 1.0 - s_tau

    censor_survival_through = np.cumprod(
        1.0 - censor_hazards,
        axis=1,
    )
    g_before = np.ones_like(censor_survival_through)
    if horizon > 1:
        g_before[:, 1:] = censor_survival_through[:, :-1]

    clipped_g_before = np.clip(
        g_before,
        censor_survival_clip,
        1.0,
    )

    days = np.arange(1, horizon + 1, dtype=float)[None, :]
    restricted_time = (
        frame["restricted_time"]
        .astype(float)
        .to_numpy()[:, None]
    )

    at_risk = (restricted_time >= days).astype(float)

    restricted_event = (
        frame["restricted_event"]
        .astype(int)
        .to_numpy()[:, None]
    )

    event_increment = (
        (restricted_event == 1)
        & np.isclose(restricted_time, days)
    ).astype(float)

    martingale_residual = (
        event_increment
        - at_risk * event_hazards
    )

    survival_ratio = (
        s_tau[:, None]
        / np.clip(event_survival, 1e-12, 1.0)
    )

    clever_covariate = (
        survival_ratio
        / clipped_g_before
    )

    augmentation = (
        clever_covariate
        * martingale_residual
    ).sum(axis=1)

    pseudo = event_risk + augmentation

    # Relevant positivity diagnostic: G(t-) among at-risk person-days.
    raw_at_risk_g = g_before[at_risk.astype(bool)]

    diagnostics = {
        "aipcw_pseudo_min": float(np.min(pseudo)),
        "aipcw_pseudo_q01": float(np.quantile(pseudo, 0.01)),
        "aipcw_pseudo_median": float(np.median(pseudo)),
        "aipcw_pseudo_q99": float(np.quantile(pseudo, 0.99)),
        "aipcw_pseudo_max": float(np.max(pseudo)),
        "at_risk_g_min": float(np.min(raw_at_risk_g)),
        "at_risk_g_q01": float(np.quantile(raw_at_risk_g, 0.01)),
        "at_risk_g_q05": float(np.quantile(raw_at_risk_g, 0.05)),
        "fraction_at_risk_g_clipped": float(
            np.mean(raw_at_risk_g < censor_survival_clip)
        ),
    }

    return pseudo, diagnostics


def weighted_bin_mae(
    frame: pd.DataFrame,
    *,
    estimate_values: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    data = frame[["prediction", "outcome"]].copy()
    data["_estimate"] = estimate_values
    data["_bin"] = pd.cut(
        data["prediction"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    if data["_bin"].isna().any():
        raise RuntimeError("Unassigned prediction bin")

    n = len(data)
    total = 0.0

    for _, group in data.groupby("_bin", observed=True, sort=True):
        weight = len(group) / n
        reference = float(group["outcome"].mean())
        estimate = float(group["_estimate"].mean())
        total += weight * abs(estimate - reference)

    return float(total)


def run_replication(
    *,
    replication: int,
    strengths: list[float],
    design: pd.DataFrame,
    test_n: int,
    horizon: int,
    n_bins: int,
    folds: int,
    logistic_c: float,
    aipcw_clip: float,
    ipcw_clip: float,
    seed: int,
) -> tuple[list[dict[str, float]], dict[str, bool]]:
    rows: list[dict[str, float]] = []

    common_seed = seed + 1000 + replication

    reference_outcome = None
    reference_prediction = None

    event_invariant = True
    prediction_invariant = True

    for strength in strengths:
        drow = design.loc[
            np.isclose(design["strength"], strength)
        ].iloc[0]

        config = make_config(
            strength=strength,
            censor_intercept=float(drow["censor_intercept"]),
            n=test_n,
            horizon=horizon,
            seed=common_seed,
        )

        output = simulate_cohort(config)
        frame = output.frame.reset_index(drop=True)

        if reference_outcome is None:
            reference_outcome = frame["outcome"].to_numpy().copy()
            reference_prediction = frame["prediction"].to_numpy().copy()
        else:
            event_invariant &= np.array_equal(
                reference_outcome,
                frame["outcome"].to_numpy(),
            )
            prediction_invariant &= np.allclose(
                reference_prediction,
                frame["prediction"].to_numpy(),
                atol=0.0,
                rtol=0.0,
            )

        learned_event_hazards, learned_censor_hazards = (
            cross_fitted_hazards(
                frame,
                horizon=horizon,
                folds=folds,
                logistic_c=logistic_c,
                seed=seed + 100000 * replication + int(1000 * strength),
            )
        )

        bin_edges = reference_quantile_edges(
            frame["prediction"].to_numpy(),
            n_bins,
        )

        learned_survival_frame = construct_survival_nuisance_predictions(
            frame,
            event_hazards=learned_event_hazards,
            censor_hazards=learned_censor_hazards,
            horizon=horizon,
            censor_survival_clip=ipcw_clip,
        )

        learned_result = evaluate_survival_estimators(
            learned_survival_frame,
            horizon=horizon,
            bin_edges=bin_edges,
            censor_survival_clip=ipcw_clip,
        )

        aipcw_pseudo, aipcw_diag = aipcw_event_pseudo_outcome(
            frame,
            event_hazards=learned_event_hazards,
            censor_hazards=learned_censor_hazards,
            horizon=horizon,
            censor_survival_clip=aipcw_clip,
        )

        learned_aipcw_mae = weighted_bin_mae(
            frame,
            estimate_values=aipcw_pseudo,
            bin_edges=bin_edges,
        )

        oracle_aipcw_pseudo, _ = aipcw_event_pseudo_outcome(
            frame,
            event_hazards=output.true_event_hazards,
            censor_hazards=output.true_censor_hazards,
            horizon=horizon,
            censor_survival_clip=aipcw_clip,
        )

        oracle_aipcw_mae = weighted_bin_mae(
            frame,
            estimate_values=oracle_aipcw_pseudo,
            bin_edges=bin_edges,
        )

        scalar = learned_result.scalar_summary
        diagnostics = learned_result.diagnostics

        learned_g_tau = np.prod(
            1.0 - learned_censor_hazards,
            axis=1,
        )

        complete_case_mae = float(
            scalar["complete_case_mae_to_reference"]
        )

        row = {
            "replication": replication,
            "strength": float(strength),
            "test_n": int(test_n),
            "observed_fraction": float(frame["observed"].mean()),
            "reference_event_rate": float(frame["outcome"].mean()),
            "learned_g_tau_q01": float(np.quantile(learned_g_tau, 0.01)),
            "learned_g_tau_q05": float(np.quantile(learned_g_tau, 0.05)),
            "learned_g_tau_median": float(np.median(learned_g_tau)),
            "fraction_learned_g_tau_lt_0p05": float(
                np.mean(learned_g_tau < 0.05)
            ),
            "complete_case_mae": complete_case_mae,
            "kaplan_meier_mae": float(
                scalar["kaplan_meier_mae_to_reference"]
            ),
            "crossfit_survival_or_mae": float(
                scalar["survival_or_mae_to_reference"]
            ),
            "crossfit_ipcw_mae": float(
                scalar["ipcw_event_mae_to_reference"]
            ),
            "crossfit_aipcw_mae": float(learned_aipcw_mae),
            "oracle_aipcw_mae": float(oracle_aipcw_mae),
            "aipcw_fraction_gap_closed_vs_complete_case": float(
                (complete_case_mae - learned_aipcw_mae)
                / complete_case_mae
                if complete_case_mae > 0
                else np.nan
            ),
            "ipcw_event_ess": float(
                diagnostics["event_weight_effective_sample_size"]
            ),
            "ipcw_fraction_event_g_clipped": float(
                diagnostics["fraction_event_g_clipped"]
            ),
            **aipcw_diag,
        }

        rows.append(row)

    flags = {
        "event_process_exactly_invariant_across_strengths": bool(event_invariant),
        "prediction_exactly_invariant_across_strengths": bool(prediction_invariant),
    }

    return rows, flags


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "observed_fraction",
        "reference_event_rate",
        "learned_g_tau_q05",
        "fraction_learned_g_tau_lt_0p05",
        "complete_case_mae",
        "kaplan_meier_mae",
        "crossfit_survival_or_mae",
        "crossfit_ipcw_mae",
        "crossfit_aipcw_mae",
        "oracle_aipcw_mae",
        "aipcw_fraction_gap_closed_vs_complete_case",
        "ipcw_event_ess",
        "ipcw_fraction_event_g_clipped",
        "fraction_at_risk_g_clipped",
    ]

    rows = []

    for strength, group in results.groupby("strength", sort=True):
        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()

            rows.append({
                "strength": float(strength),
                "metric": metric,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "q025": float(values.quantile(0.025)),
                "median": float(values.quantile(0.50)),
                "q975": float(values.quantile(0.975)),
            })

    return pd.DataFrame(rows)


def wide_endpoint_summary(summary: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "observed_fraction",
        "learned_g_tau_q05",
        "complete_case_mae",
        "kaplan_meier_mae",
        "crossfit_survival_or_mae",
        "crossfit_ipcw_mae",
        "crossfit_aipcw_mae",
        "oracle_aipcw_mae",
        "aipcw_fraction_gap_closed_vs_complete_case",
    ]

    endpoints = summary.loc[
        summary["strength"].isin([0.0, 1.0])
        & summary["metric"].isin(selected)
    ].copy()

    return endpoints[
        ["strength", "metric", "mean", "q025", "median", "q975"]
    ].sort_values(["strength", "metric"])


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.replicates = min(args.replicates, 3)
        args.test_n = min(args.test_n, 1500)
        args.calibration_n = min(args.calibration_n, 10000)

    if args.folds < 2:
        raise ValueError("folds must be >=2")
    if not 0 < args.aipcw_clip < 1:
        raise ValueError("aipcw-clip must be in (0,1)")
    if not 0 < args.ipcw_clip < 1:
        raise ValueError("ipcw-clip must be in (0,1)")

    outdir = ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    strengths = parse_strengths()

    design_rows = []
    for strength in strengths:
        design_rows.append(
            calibrate_intercept(
                strength,
                target_observed_fraction=args.target_observed_fraction,
                calibration_n=args.calibration_n,
                horizon=args.horizon,
                seed=args.seed + 701,
            )
        )

    design = pd.DataFrame(design_rows)
    design.to_csv(
        outdir / "calibrated_overlap_design.csv",
        index=False,
    )

    all_rows = []
    invariant_flags = []

    for replication in range(args.replicates):
        rows, flags = run_replication(
            replication=replication,
            strengths=strengths,
            design=design,
            test_n=args.test_n,
            horizon=args.horizon,
            n_bins=args.n_bins,
            folds=args.folds,
            logistic_c=args.logistic_c,
            aipcw_clip=args.aipcw_clip,
            ipcw_clip=args.ipcw_clip,
            seed=args.seed,
        )

        all_rows.extend(rows)
        invariant_flags.append({
            "replication": replication,
            **flags,
        })

        print(
            f"replication {replication + 1}/{args.replicates} complete",
            flush=True,
        )

    results = pd.DataFrame(all_rows)
    flags_df = pd.DataFrame(invariant_flags)

    results.to_csv(
        outdir / "replicate_results.csv.gz",
        index=False,
        compression="gzip",
    )

    flags_df.to_csv(
        outdir / "common_random_number_checks.csv",
        index=False,
    )

    summary = summarize(results)
    summary.to_csv(
        outdir / "summary_by_strength.csv",
        index=False,
    )

    endpoint = wide_endpoint_summary(summary)
    endpoint.to_csv(
        outdir / "endpoint_comparison.csv",
        index=False,
    )

    # Direct method comparison at every overlap strength.
    comparison_rows = []
    for strength, group in results.groupby("strength", sort=True):
        for comparator in [
            "complete_case_mae",
            "kaplan_meier_mae",
            "crossfit_survival_or_mae",
            "crossfit_ipcw_mae",
        ]:
            diff = (
                group["crossfit_aipcw_mae"]
                - group[comparator]
            )
            comparison_rows.append({
                "strength": float(strength),
                "comparison": f"crossfit_aipcw_minus_{comparator}",
                "mean_difference": float(diff.mean()),
                "q025": float(diff.quantile(0.025)),
                "median": float(diff.quantile(0.50)),
                "q975": float(diff.quantile(0.975)),
            })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        outdir / "paired_method_comparisons.csv",
        index=False,
    )

    # Validation is deliberately about design/code integrity, not whether
    # the new method "wins".
    checks = []

    def add(name, value, expected, passed):
        checks.append({
            "check": name,
            "value": value,
            "expected": expected,
            "passed": bool(passed),
        })

    add(
        "expected_result_rows",
        len(results),
        args.replicates * len(strengths),
        len(results) == args.replicates * len(strengths),
    )

    add(
        "all_event_processes_invariant_within_replication",
        bool(
            flags_df[
                "event_process_exactly_invariant_across_strengths"
            ].all()
        ),
        True,
        bool(
            flags_df[
                "event_process_exactly_invariant_across_strengths"
            ].all()
        ),
    )

    add(
        "all_predictions_invariant_within_replication",
        bool(
            flags_df[
                "prediction_exactly_invariant_across_strengths"
            ].all()
        ),
        True,
        bool(
            flags_df[
                "prediction_exactly_invariant_across_strengths"
            ].all()
        ),
    )

    obs_mean = (
        results.groupby("strength")["observed_fraction"]
        .mean()
    )
    max_obs_deviation = float(
        (obs_mean - args.target_observed_fraction).abs().max()
    )

    add(
        "observed_fraction_near_fixed_target",
        max_obs_deviation,
        "<=0.03",
        max_obs_deviation <= 0.03,
    )

    reference_range = (
        results.groupby("replication")["reference_event_rate"]
        .agg(lambda x: float(x.max() - x.min()))
    )
    add(
        "reference_event_rate_exactly_invariant_across_strengths",
        float(reference_range.max()),
        "<=1e-12",
        float(reference_range.max()) <= 1e-12,
    )

    design_q05 = design.sort_values("strength")["true_g_tau_q05"].to_numpy()
    add(
        "true_g_tau_q05_nonincreasing",
        bool(np.all(np.diff(design_q05) <= 1e-12)),
        True,
        bool(np.all(np.diff(design_q05) <= 1e-12)),
    )

    add(
        "strength0_q05_in_expected_moderate_range",
        float(design.iloc[0]["true_g_tau_q05"]),
        "0.02 to 0.08",
        0.02 <= float(design.iloc[0]["true_g_tau_q05"]) <= 0.08,
    )

    add(
        "strength1_q05_severe_overlap",
        float(design.iloc[-1]["true_g_tau_q05"]),
        "<1e-5",
        float(design.iloc[-1]["true_g_tau_q05"]) < 1e-5,
    )

    numeric = results.select_dtypes(include=[np.number])
    add(
        "all_numeric_results_finite",
        bool(np.isfinite(numeric.to_numpy()).all()),
        True,
        bool(np.isfinite(numeric.to_numpy()).all()),
    )

    # Formula sanity check only: oracle nuisance AIPCW should be close at
    # the well-supported endpoint. This is not a performance claim.
    oracle_s0 = results.loc[
        np.isclose(results["strength"], 0.0),
        "oracle_aipcw_mae",
    ].mean()

    add(
        "oracle_aipcw_strength0_sanity",
        float(oracle_s0),
        "<0.08",
        float(oracle_s0) < 0.08,
    )

    validation = pd.DataFrame(checks)
    validation.to_csv(
        outdir / "validation_checks.csv",
        index=False,
    )

    metadata = {
        "phase": "8C",
        "experiment": (
            "cross_fitted_survival_aipcw_one_step_under_fixed_observation_overlap_sweep"
        ),
        "replicates": args.replicates,
        "test_n": args.test_n,
        "horizon": args.horizon,
        "n_bins": args.n_bins,
        "folds": args.folds,
        "logistic_c": args.logistic_c,
        "aipcw_censor_survival_clip": args.aipcw_clip,
        "baseline_ipcw_clip": args.ipcw_clip,
        "target_observed_fraction": args.target_observed_fraction,
        "strengths": strengths,
        "features": FEATURES,
        "event_process": {
            "event_intercept": -4.70,
            "event_time_slope": 0.012,
            "event_x1": 0.70,
            "event_x2": 0.65,
            "event_u": 0.0,
        },
        "overlap_path": {
            "censor_x1": "-0.75 * strength",
            "censor_x2": "0.95 * strength",
            "censor_intercept": (
                "calibrated per strength to target overall observed fraction"
            ),
            "censor_u": 0.0,
        },
        "primary_interpretation_boundary": (
            "The AIPCW comparator is theory-motivated under conditional "
            "independent censoring and positivity. It is not claimed to "
            "overcome actual positivity violations; severe lower-tail support "
            "remains a scope condition."
        ),
    }

    (outdir / "phase8c_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    status = (
        "phase8c_theory_comparator_passed"
        if validation["passed"].all()
        else "phase8c_theory_comparator_requires_review"
    )

    (outdir / "phase8c_status.txt").write_text(
        status + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(status)
    print("=" * 78)

    print("\n===== CALIBRATED OVERLAP DESIGN =====")
    print(design.to_string(index=False))

    print("\n===== ENDPOINT COMPARISON =====")
    print(endpoint.to_string(index=False))

    print("\n===== PAIRED METHOD COMPARISONS =====")
    print(comparison.to_string(index=False))

    print("\n===== VALIDATION =====")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
