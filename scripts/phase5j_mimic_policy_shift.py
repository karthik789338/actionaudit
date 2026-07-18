"""Phase 5J-C: MIMIC real-data-grounded policy-shift emulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import phase5j_support2_policy_shift as core


EXPECTED_INPUT_STATUS = (
    "phase5c_frozen_for_manuscript"
)

EXPECTED_N = 31012
HORIZON = 30.0


# The reused core functions look up these globals
# through their source module.
core.HORIZON = HORIZON


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 5E policy definitions "
            "on the fixed MIMIC test population."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/"
            "phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--input-status",
        default=(
            "artifacts/"
            "phase5c_frozen_for_manuscript/"
            "phase5c_freeze_status.txt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5j_mimic_policy_shift"
        ),
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def mimic_fixed_hash(
    frame: pd.DataFrame,
) -> str:
    columns = [
        "subject_id",
        "hadm_id",
        "prediction",
        "outcome",
        "latent_event_time",
        "u_exit",
        "u_time",
    ]

    hashed = pd.util.hash_pandas_object(
        frame[
            columns
        ],
        index=False,
    ).to_numpy()

    return hashlib.sha256(
        hashed.tobytes()
    ).hexdigest()


# Make reused audit functions use the MIMIC hash.
core.fixed_hash = mimic_fixed_hash


def prepare_fixed_population(
    data: pd.DataFrame,
    *,
    n_bins: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    str,
]:
    required = [
        "subject_id",
        "hadm_id",
        "prediction",
        "outcome",
        "observed",
        "exit_time",
    ]

    core.require_columns(
        data,
        required,
    )

    if data[
        "hadm_id"
    ].duplicated().any():
        raise ValueError(
            "MIMIC input contains duplicate "
            "admission IDs"
        )

    frame = data.copy()

    frame[
        "prediction"
    ] = pd.to_numeric(
        frame[
            "prediction"
        ],
        errors="raise",
    ).clip(
        core.EPSILON,
        1.0 - core.EPSILON,
    )

    frame[
        "outcome"
    ] = pd.to_numeric(
        frame[
            "outcome"
        ],
        errors="raise",
    ).astype(
        int
    )

    frame[
        "observed"
    ] = pd.to_numeric(
        frame[
            "observed"
        ],
        errors="raise",
    ).astype(
        int
    )

    frame[
        "exit_time"
    ] = pd.to_numeric(
        frame[
            "exit_time"
        ],
        errors="coerce",
    ).fillna(
        HORIZON
    ).clip(
        lower=0.0,
        upper=HORIZON,
    )

    prediction = frame[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    prediction_mean = float(
        prediction.mean()
    )

    prediction_sd = float(
        prediction.std(
            ddof=0
        )
    )

    if prediction_sd <= 0:
        raise ValueError(
            "Prediction standard deviation is zero"
        )

    frame[
        "standardized_risk"
    ] = (
        prediction
        - prediction_mean
    ) / prediction_sd

    edges = np.quantile(
        prediction,
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        ),
    )

    if len(
        np.unique(
            edges
        )
    ) != n_bins + 1:
        raise ValueError(
            "Prediction distribution does not "
            "support the requested risk bins"
        )

    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    frame[
        "risk_bin"
    ] = pd.cut(
        prediction,
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(
        int
    )

    rng = np.random.default_rng(
        seed
    )

    frame[
        "u_exit"
    ] = rng.uniform(
        size=len(
            frame
        )
    )

    frame[
        "u_time"
    ] = rng.uniform(
        size=len(
            frame
        )
    )

    hidden_event_uniform = rng.uniform(
        size=len(
            frame
        )
    )

    latent_event_time = np.full(
        len(
            frame
        ),
        np.inf,
        dtype=float,
    )

    observed_death = (
        frame[
            "outcome"
        ].eq(
            1
        )
        & frame[
            "observed"
        ].eq(
            1
        )
    )

    hidden_death = (
        frame[
            "outcome"
        ].eq(
            1
        )
        & frame[
            "observed"
        ].eq(
            0
        )
    )

    # In-hospital deaths retain the observed
    # event/exit time.
    latent_event_time[
        observed_death.to_numpy()
    ] = frame.loc[
        observed_death,
        "exit_time",
    ].to_numpy(
        dtype=float
    )

    # Linked post-discharge deaths receive one fixed
    # event-time realization after hospital exit.
    hidden_exit = frame.loc[
        hidden_death,
        "exit_time",
    ].to_numpy(
        dtype=float
    ).clip(
        0.0,
        HORIZON - 1e-8,
    )

    latent_event_time[
        hidden_death.to_numpy()
    ] = (
        hidden_exit
        + (
            HORIZON
            - hidden_exit
        )
        * hidden_event_uniform[
            hidden_death.to_numpy()
        ]
    )

    frame[
        "latent_event_time"
    ] = latent_event_time

    return (
        frame,
        edges,
        mimic_fixed_hash(
            frame
        ),
    )


def build_validation_checks(
    *,
    fixed: pd.DataFrame,
    input_status: str,
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
    policy_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    fixed_columns = [
        "subject_id",
        "hadm_id",
        "prediction",
        "outcome",
        "latent_event_time",
        "risk_bin",
    ]

    baseline = policy_frames[
        "baseline"
    ][
        fixed_columns
    ].reset_index(
        drop=True
    )

    fixed_components_identical = all(
        baseline.equals(
            frame[
                fixed_columns
            ].reset_index(
                drop=True
            )
        )
        for frame in policy_frames.values()
    )

    observed_patterns = {
        tuple(
            frame[
                "emulated_observed"
            ].to_numpy(
                dtype=int
            )
        )
        for frame in policy_frames.values()
    }

    reference_ici_range = float(
        point[
            "reference_ici"
        ].max()
        - point[
            "reference_ici"
        ].min()
    )

    reference_event_rate_range = float(
        point[
            "reference_event_rate"
        ].max()
        - point[
            "reference_event_rate"
        ].min()
    )

    observed_fraction_range = float(
        point[
            "observed_fraction"
        ].max()
        - point[
            "observed_fraction"
        ].min()
    )

    expected_bootstrap_rows = (
        requested_bootstraps
        * len(
            policy_frames
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "Phase 5C MIMIC benchmark "
                    "already frozen"
                ),
                "value": input_status,
                "threshold": (
                    EXPECTED_INPUT_STATUS
                ),
                "comparison": "==",
                "passed": (
                    input_status
                    == EXPECTED_INPUT_STATUS
                ),
            },
            {
                "check": (
                    "MIMIC test cohort contains "
                    "31012 unique admissions"
                ),
                "value": len(
                    fixed
                ),
                "threshold": EXPECTED_N,
                "comparison": "==",
                "passed": bool(
                    len(
                        fixed
                    )
                    == EXPECTED_N
                    and not fixed[
                        "hadm_id"
                    ].duplicated().any()
                ),
            },
            {
                "check": (
                    "five frozen observation "
                    "policies evaluated"
                ),
                "value": int(
                    point[
                        "policy"
                    ].nunique()
                ),
                "threshold": 5,
                "comparison": "==",
                "passed": (
                    point[
                        "policy"
                    ].nunique()
                    == 5
                ),
            },
            {
                "check": (
                    "predictions outcomes and latent "
                    "event process identical "
                    "across policies"
                ),
                "value": int(
                    fixed_components_identical
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    fixed_components_identical
                ),
            },
            {
                "check": (
                    "policies generate distinct "
                    "observation patterns"
                ),
                "value": len(
                    observed_patterns
                ),
                "threshold": 5,
                "comparison": "==",
                "passed": (
                    len(
                        observed_patterns
                    )
                    == 5
                ),
            },
            {
                "check": (
                    "reference ICI invariant "
                    "across policies"
                ),
                "value": (
                    reference_ici_range
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": bool(
                    reference_ici_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "reference event rate invariant "
                    "across policies"
                ),
                "value": (
                    reference_event_rate_range
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": bool(
                    reference_event_rate_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "observation policies materially "
                    "change observed fraction"
                ),
                "value": (
                    observed_fraction_range
                ),
                "threshold": 0.20,
                "comparison": ">",
                "passed": bool(
                    observed_fraction_range
                    > 0.20
                ),
            },
            {
                "check": (
                    "all requested paired "
                    "bootstrap rows completed"
                ),
                "value": len(
                    bootstrap
                ),
                "threshold": (
                    expected_bootstrap_rows
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap
                    )
                    == expected_bootstrap_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def build_core_h6_checks(
    *,
    point: pd.DataFrame,
    contrast_intervals: pd.DataFrame,
    validations: pd.DataFrame,
) -> pd.DataFrame:
    restricted_ici_range = float(
        point[
            "restricted_ici"
        ].max()
        - point[
            "restricted_ici"
        ].min()
    )

    aggressive_ici_q025 = (
        core.interval_value(
            contrast_intervals,
            policy="aggressive_early",
            metric=(
                "restricted_ici_difference"
            ),
            quantile="q025",
        )
    )

    aggressive_observation_q975 = (
        core.interval_value(
            contrast_intervals,
            policy="aggressive_early",
            metric=(
                "observed_fraction_difference"
            ),
            quantile="q975",
        )
    )

    fixed_components_passed = bool(
        validations.loc[
            validations[
                "check"
            ].eq(
                "predictions outcomes and latent "
                "event process identical "
                "across policies"
            ),
            "passed",
        ].iloc[
            0
        ]
    )

    reference_ici_passed = bool(
        validations.loc[
            validations[
                "check"
            ].eq(
                "reference ICI invariant "
                "across policies"
            ),
            "passed",
        ].iloc[
            0
        ]
    )

    reference_rate_passed = bool(
        validations.loc[
            validations[
                "check"
            ].eq(
                "reference event rate invariant "
                "across policies"
            ),
            "passed",
        ].iloc[
            0
        ]
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "fixed predictor and outcome "
                    "target across policies"
                ),
                "value": int(
                    fixed_components_passed
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    fixed_components_passed
                ),
            },
            {
                "check": (
                    "reference calibration target "
                    "invariant across policies"
                ),
                "value": int(
                    reference_ici_passed
                    and reference_rate_passed
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": bool(
                    reference_ici_passed
                    and reference_rate_passed
                ),
            },
            {
                "check": (
                    "restricted ICI changes "
                    "materially across policies"
                ),
                "value": (
                    restricted_ici_range
                ),
                "threshold": 0.03,
                "comparison": ">",
                "passed": bool(
                    restricted_ici_range
                    > 0.03
                ),
            },
            {
                "check": (
                    "aggressive policy increases "
                    "restricted ICI versus baseline"
                ),
                "value": (
                    aggressive_ici_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    aggressive_ici_q025
                    > 0
                ),
            },
            {
                "check": (
                    "aggressive policy lowers "
                    "observation fraction"
                ),
                "value": (
                    aggressive_observation_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": bool(
                    aggressive_observation_q975
                    < 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    input_path = Path(
        args.input
    )

    status_path = Path(
        args.input_status
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_status = status_path.read_text(
        encoding="utf-8"
    ).strip()

    raw = pd.read_csv(
        input_path,
        low_memory=False,
    )

    (
        fixed,
        bin_edges,
        population_hash,
    ) = prepare_fixed_population(
        raw,
        n_bins=args.n_bins,
        seed=args.seed,
    )

    policies = (
        core.policy_definitions()
    )

    policy_frames = {
        policy.name: core.apply_policy(
            fixed,
            policy,
        )
        for policy in policies
    }

    (
        point,
        bin_summary,
    ) = core.point_analysis(
        policy_frames
    )

    point_contrast_table = (
        core.point_contrasts(
            point
        )
    )

    print(
        "Phase 5J-C MIMIC "
        "observation-policy shift"
    )

    print(
        "\nRunning paired admission bootstrap..."
    )

    (
        bootstrap,
        invalid_bootstraps,
    ) = core.run_bootstrap(
        policy_frames,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed + 1,
    )

    scalar_intervals = (
        core.scalar_interval_table(
            bootstrap
        )
    )

    contrast_intervals = (
        core.contrast_interval_table(
            bootstrap
        )
    )

    validations = build_validation_checks(
        fixed=fixed,
        input_status=input_status,
        point=point,
        bootstrap=bootstrap,
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
        policy_frames=policy_frames,
    )

    full_evidence_checks = (
        core.build_h6_evidence_checks(
            point,
            contrast_intervals,
        )
    )

    core_h6_checks = (
        build_core_h6_checks(
            point=point,
            contrast_intervals=(
                contrast_intervals
            ),
            validations=validations,
        )
    )

    status = (
        "phase5j_mimic_policy_shift_benchmarked"
        if validations[
            "passed"
        ].all()
        else (
            "phase5j_mimic_policy_shift_"
            "requires_review"
        )
    )

    pd.DataFrame(
        [
            asdict(
                policy
            )
            for policy in policies
        ]
    ).to_csv(
        output_dir
        / "policy_definitions.csv",
        index=False,
    )

    point.to_csv(
        output_dir
        / "policy_scalar_summary.csv",
        index=False,
    )

    bin_summary.to_csv(
        output_dir
        / "policy_bin_summary.csv",
        index=False,
    )

    point_contrast_table.to_csv(
        output_dir
        / "point_policy_contrasts.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_scalar_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    scalar_intervals.to_csv(
        output_dir
        / "scalar_bootstrap_intervals.csv",
        index=False,
    )

    contrast_intervals.to_csv(
        output_dir
        / "policy_contrast_intervals.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    full_evidence_checks.to_csv(
        output_dir
        / "h6_policy_specific_checks.csv",
        index=False,
    )

    core_h6_checks.to_csv(
        output_dir
        / "h6_core_decision_checks.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "edge_index": np.arange(
                len(
                    bin_edges
                )
            ),
            "edge": bin_edges,
        }
    ).to_csv(
        output_dir
        / "frozen_risk_bin_edges.csv",
        index=False,
    )

    evidence = {
        "phase": "5J-C",
        "status": status,
        "dataset": "MIMIC-IV",
        "horizon_days": HORIZON,
        "fixed_population_hash": (
            population_hash
        ),
        "n_admissions": int(
            len(
                fixed
            )
        ),
        "n_bins": args.n_bins,
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstraps": (
            invalid_bootstraps
        ),
        "fixed_components": [
            "admission identities",
            "predictions",
            "linked 30-day outcomes",
            "latent event-time realization",
            "common random numbers",
        ],
        "shifted_components": [
            "hospital-exit probability",
            "risk-based exit selection",
            "hospital-exit timing",
            "emulated observation indicator",
        ],
        "policy_source": (
            "Exact policy coefficients from "
            "the frozen Phase 5E experiment "
            "and SUPPORT2 Phase 5J-B."
        ),
        "interpretation": (
            "Controlled real-data-grounded "
            "policy emulation, not a historical "
            "or causal MIMIC policy effect."
        ),
        "core_h6_supported": bool(
            core_h6_checks[
                "passed"
            ].all()
        ),
        "core_h6_checks_passed": int(
            core_h6_checks[
                "passed"
            ].sum()
        ),
        "core_h6_checks_total": int(
            len(
                core_h6_checks
            )
        ),
    }

    (
        output_dir
        / "mimic_policy_shift_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5j_mimic_policy_shift_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nPolicy summary"
    )

    print(
        point.to_string(
            index=False
        )
    )

    print(
        "\nPolicy contrasts"
    )

    print(
        point_contrast_table.to_string(
            index=False
        )
    )

    print(
        "\nStructural validation"
    )

    print(
        validations.to_string(
            index=False
        )
    )

    print(
        "\nCore H6 decision checks"
    )

    print(
        core_h6_checks.to_string(
            index=False
        )
    )

    print(
        "\nPolicy-specific directional checks"
    )

    print(
        full_evidence_checks.to_string(
            index=False
        )
    )

    print(
        "\nInvalid bootstrap replications: "
        f"{invalid_bootstraps}"
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
