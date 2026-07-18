"""Freeze validated Phase 5E observation-policy-shift results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(
    "artifacts/phase5e_observation_policy_shift"
)

OUTPUT_DIR = Path(
    "artifacts/phase5e_frozen_for_manuscript"
)

REQUIRED_STATUS = (
    "phase5e_observation_policy_shift_supported"
)

HASHED_FILES = [
    "scripts/phase5e_observation_policy_shift.py",
    (
        "artifacts/phase5e_observation_policy_shift/"
        "policy_definitions.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "policy_scalar_summary.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "policy_action_summary.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "scalar_bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "action_mix_l1_bootstrap_interval.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "action_mix_per_action_bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "validation_checks.csv"
    ),
    (
        "artifacts/phase5e_observation_policy_shift/"
        "policy_shift_evidence.json"
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def interval_value(
    intervals: pd.DataFrame,
    *,
    policy: str,
    metric: str,
    column: str,
) -> float:
    selected = intervals.loc[
        intervals["policy"].eq(policy)
        & intervals["metric"].eq(metric),
        column,
    ]

    if len(selected) != 1:
        raise ValueError(
            "Unable to locate interval for "
            f"{policy}: {metric}"
        )

    return float(
        selected.iloc[0]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        SOURCE_DIR
        / "phase5e_status.txt"
    )

    actual_status = (
        status_path.read_text(
            encoding="utf-8"
        ).strip()
        if status_path.exists()
        else "missing"
    )

    scalar = pd.read_csv(
        SOURCE_DIR
        / "policy_scalar_summary.csv"
    )

    intervals = pd.read_csv(
        SOURCE_DIR
        / "scalar_bootstrap_intervals.csv"
    )

    action_mix = pd.read_csv(
        SOURCE_DIR
        / "action_mix_l1_bootstrap_interval.csv"
    ).iloc[0]

    source_checks = pd.read_csv(
        SOURCE_DIR
        / "validation_checks.csv"
    )

    reference_ici_range = float(
        scalar["reference_ici"].max()
        - scalar["reference_ici"].min()
    )

    reference_event_rate_range = float(
        scalar["reference_event_rate"].max()
        - scalar["reference_event_rate"].min()
    )

    restricted_ici_range = float(
        scalar["restricted_ici"].max()
        - scalar["restricted_ici"].min()
    )

    aggressive_ici_q025 = interval_value(
        intervals,
        policy="aggressive_early",
        metric="restricted_ici_minus_baseline",
        column="q025",
    )

    conservative_ici_q975 = interval_value(
        intervals,
        policy="conservative_late",
        metric="restricted_ici_minus_baseline",
        column="q975",
    )

    aggressive_slope_q975 = interval_value(
        intervals,
        policy="aggressive_early",
        metric=(
            "restricted_calibration_slope_"
            "minus_baseline"
        ),
        column="q975",
    )

    validation = pd.DataFrame(
        [
            {
                "check": "required status passed",
                "value": actual_status,
                "threshold": REQUIRED_STATUS,
                "comparison": "==",
                "passed": (
                    actual_status
                    == REQUIRED_STATUS
                ),
            },
            {
                "check": (
                    "all original Phase 5E checks passed"
                ),
                "value": int(
                    source_checks["passed"].sum()
                ),
                "threshold": len(
                    source_checks
                ),
                "comparison": "==",
                "passed": bool(
                    source_checks["passed"].all()
                ),
            },
            {
                "check": (
                    "reference ICI exactly invariant"
                ),
                "value": reference_ici_range,
                "threshold": 1e-12,
                "comparison": "<",
                "passed": (
                    reference_ici_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "reference event rate exactly invariant"
                ),
                "value": (
                    reference_event_rate_range
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": (
                    reference_event_rate_range
                    < 1e-12
                ),
            },
            {
                "check": (
                    "restricted ICI range is material"
                ),
                "value": restricted_ici_range,
                "threshold": 0.05,
                "comparison": ">",
                "passed": (
                    restricted_ici_range
                    > 0.05
                ),
            },
            {
                "check": (
                    "aggressive ICI increase lower "
                    "bound is positive"
                ),
                "value": aggressive_ici_q025,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    aggressive_ici_q025
                    > 0
                ),
            },
            {
                "check": (
                    "conservative ICI difference upper "
                    "bound is negative"
                ),
                "value": conservative_ici_q975,
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    conservative_ici_q975
                    < 0
                ),
            },
            {
                "check": (
                    "aggressive slope decrease upper "
                    "bound is negative"
                ),
                "value": aggressive_slope_q975,
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    aggressive_slope_q975
                    < 0
                ),
            },
            {
                "check": (
                    "action-attribution L1 lower "
                    "bound exceeds materiality threshold"
                ),
                "value": float(
                    action_mix["q025"]
                ),
                "threshold": 0.01,
                "comparison": ">",
                "passed": (
                    float(
                        action_mix["q025"]
                    )
                    > 0.01
                ),
            },
        ]
    )

    status = (
        "phase5e_frozen_for_manuscript"
        if validation["passed"].all()
        else "phase5e_freeze_requires_review"
    )

    hashes = {
        path_text: sha256_file(
            Path(path_text)
        )
        for path_text in HASHED_FILES
    }

    manifest = {
        "phase": "5E",
        "status": status,
        "hypothesis": (
            "Observation-policy shift can mimic "
            "model drift."
        ),
        "core_claim": (
            "Changing only the observation policy "
            "produces substantial apparent calibration "
            "and proper-score drift while predictions "
            "and the event process remain fixed."
        ),
        "secondary_claim": (
            "Changing only the action mixture "
            "redistributes action-attributed distortion "
            "while preserving every global audit statistic."
        ),
        "reference_ici_range": (
            reference_ici_range
        ),
        "reference_event_rate_range": (
            reference_event_rate_range
        ),
        "restricted_ici_range": (
            restricted_ici_range
        ),
        "aggressive_ici_difference_q025": (
            aggressive_ici_q025
        ),
        "conservative_ici_difference_q975": (
            conservative_ici_q975
        ),
        "action_attribution_l1_q025": float(
            action_mix["q025"]
        ),
        "caveats": [
            (
                "The experiment is a controlled "
                "synthetic demonstration."
            ),
            (
                "Restricted proper scores are measured "
                "on policy-selected populations and are "
                "not common-population performance changes."
            ),
            (
                "Action attribution is descriptive and "
                "depends on the operational action taxonomy."
            ),
        ],
        "sha256": hashes,
    }

    validation.to_csv(
        OUTPUT_DIR
        / "freeze_validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5e_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5e_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5E freeze validation"
    )

    print(
        "\nFreeze checks"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
