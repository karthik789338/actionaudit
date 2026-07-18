"""Freeze validated Phase 5F recalibration-fragility results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(
    "artifacts/phase5f_recalibration_fragility"
)

OUTPUT_DIR = Path(
    "artifacts/phase5f_frozen_for_manuscript"
)

REQUIRED_STATUS = (
    "phase5f_recalibration_assumption_"
    "fragility_supported"
)

HASHED_FILES = [
    "scripts/phase5f_recalibration_fragility.py",
    (
        "artifacts/phase5f_recalibration_fragility/"
        "recalibration_parameters.csv"
    ),
    (
        "artifacts/phase5f_recalibration_fragility/"
        "evaluation_summary.csv"
    ),
    (
        "artifacts/phase5f_recalibration_fragility/"
        "point_contrasts.csv"
    ),
    (
        "artifacts/phase5f_recalibration_fragility/"
        "contrast_bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5f_recalibration_fragility/"
        "policy_comparison_intervals.csv"
    ),
    (
        "artifacts/phase5f_recalibration_fragility/"
        "validation_checks.csv"
    ),
    (
        "artifacts/phase5f_recalibration_fragility/"
        "recalibration_evidence.json"
    ),
]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def contrast_value(
    intervals: pd.DataFrame,
    *,
    policy: str,
    method: str,
    metric: str,
    column: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "fit_policy"
        ].eq(
            policy
        )
        & intervals[
            "method"
        ].eq(
            method
        )
        & intervals[
            "metric"
        ].eq(
            metric
        ),
        column,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Missing contrast interval: "
            f"{policy}, {method}, "
            f"{metric}, {column}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def policy_value(
    intervals: pd.DataFrame,
    *,
    method: str,
    metric: str,
    column: str,
) -> float:
    selected = intervals.loc[
        intervals[
            "method"
        ].eq(
            method
        )
        & intervals[
            "metric"
        ].eq(
            metric
        ),
        column,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Missing policy comparison: "
            f"{method}, {metric}, {column}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        SOURCE_DIR
        / "phase5f_status.txt"
    )

    actual_status = (
        status_path.read_text(
            encoding="utf-8"
        ).strip()
        if status_path.exists()
        else "missing"
    )

    source_checks = pd.read_csv(
        SOURCE_DIR
        / "validation_checks.csv"
    )

    parameters = pd.read_csv(
        SOURCE_DIR
        / "recalibration_parameters.csv"
    )

    contrast_intervals = pd.read_csv(
        SOURCE_DIR
        / "contrast_bootstrap_intervals.csv"
    )

    policy_intervals = pd.read_csv(
        SOURCE_DIR
        / "policy_comparison_intervals.csv"
    )

    logistic_parameters = (
        parameters.loc[
            parameters[
                "method"
            ].eq(
                "logistic"
            )
        ]
    )

    logistic_slope_range = float(
        logistic_parameters[
            "slope"
        ].max()
        - logistic_parameters[
            "slope"
        ].min()
    )

    logistic_restricted_ici_q975 = (
        contrast_value(
            contrast_intervals,
            policy="aggressive_early",
            method="logistic",
            metric="restricted_ici_change",
            column="q975",
        )
    )

    logistic_reference_ici_q025 = (
        contrast_value(
            contrast_intervals,
            policy="aggressive_early",
            method="logistic",
            metric="reference_ici_change",
            column="q025",
        )
    )

    logistic_restricted_logloss_q975 = (
        contrast_value(
            contrast_intervals,
            policy="aggressive_early",
            method="logistic",
            metric=(
                "restricted_log_loss_change"
            ),
            column="q975",
        )
    )

    logistic_reference_logloss_q025 = (
        contrast_value(
            contrast_intervals,
            policy="aggressive_early",
            method="logistic",
            metric=(
                "reference_log_loss_change"
            ),
            column="q025",
        )
    )

    intercept_restricted_ici_q975 = (
        contrast_value(
            contrast_intervals,
            policy="aggressive_early",
            method="intercept_only",
            metric="restricted_ici_change",
            column="q975",
        )
    )

    intercept_reference_ici_q025 = (
        contrast_value(
            contrast_intervals,
            policy="aggressive_early",
            method="intercept_only",
            metric="reference_ici_change",
            column="q025",
        )
    )

    transfer_ici_q025 = policy_value(
        policy_intervals,
        method="logistic",
        metric=(
            "baseline_fit_minus_own_fit_"
            "aggressive_early_restricted_ici"
        ),
        column="q025",
    )

    validation = pd.DataFrame(
        [
            {
                "check": (
                    "required Phase 5F status passed"
                ),
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
                    "all original Phase 5F checks passed"
                ),
                "value": int(
                    source_checks[
                        "passed"
                    ].sum()
                ),
                "threshold": len(
                    source_checks
                ),
                "comparison": "==",
                "passed": bool(
                    source_checks[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "logistic recalibration improves "
                    "restricted ICI"
                ),
                "value": (
                    logistic_restricted_ici_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    logistic_restricted_ici_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "logistic recalibration worsens "
                    "reference ICI"
                ),
                "value": (
                    logistic_reference_ici_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    logistic_reference_ici_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "logistic recalibration improves "
                    "restricted log loss"
                ),
                "value": (
                    logistic_restricted_logloss_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    logistic_restricted_logloss_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "logistic recalibration worsens "
                    "reference log loss"
                ),
                "value": (
                    logistic_reference_logloss_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    logistic_reference_logloss_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "intercept-only recalibration improves "
                    "restricted ICI"
                ),
                "value": (
                    intercept_restricted_ici_q975
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    intercept_restricted_ici_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "intercept-only recalibration worsens "
                    "reference ICI"
                ),
                "value": (
                    intercept_reference_ici_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    intercept_reference_ici_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "baseline recalibration transfers "
                    "poorly to aggressive policy"
                ),
                "value": (
                    transfer_ici_q025
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    transfer_ici_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "policy-specific logistic slopes "
                    "differ materially"
                ),
                "value": (
                    logistic_slope_range
                ),
                "threshold": 0.30,
                "comparison": ">",
                "passed": (
                    logistic_slope_range
                    > 0.30
                ),
            },
        ]
    )

    status = (
        "phase5f_frozen_for_manuscript"
        if validation[
            "passed"
        ].all()
        else "phase5f_freeze_requires_review"
    )

    hashes = {
        path_text: sha256_file(
            Path(
                path_text
            )
        )
        for path_text in HASHED_FILES
    }

    manifest = {
        "phase": "5F",
        "status": status,
        "hypothesis": (
            "Standard recalibration may be "
            "assumption-fragile under restricted "
            "follow-up."
        ),
        "core_claim": (
            "Recalibration fitted on policy-observed "
            "outcomes can improve held-out performance "
            "for the restricted observed target while "
            "degrading performance for the unchanged "
            "complete-population target."
        ),
        "secondary_claim": (
            "Recalibration parameters are strongly "
            "observation-policy dependent and do not "
            "reliably transfer across policies."
        ),
        "aggressive_logistic_restricted_ici_q975": (
            logistic_restricted_ici_q975
        ),
        "aggressive_logistic_reference_ici_q025": (
            logistic_reference_ici_q025
        ),
        "aggressive_logistic_restricted_logloss_q975": (
            logistic_restricted_logloss_q975
        ),
        "aggressive_logistic_reference_logloss_q025": (
            logistic_reference_logloss_q025
        ),
        "aggressive_intercept_restricted_ici_q975": (
            intercept_restricted_ici_q975
        ),
        "aggressive_intercept_reference_ici_q025": (
            intercept_reference_ici_q025
        ),
        "cross_policy_transfer_ici_q025": (
            transfer_ici_q025
        ),
        "logistic_slope_range": (
            logistic_slope_range
        ),
        "caveats": [
            (
                "This is a controlled synthetic "
                "demonstration rather than proof that "
                "every recalibration procedure fails."
            ),
            (
                "Recalibration can be appropriate when "
                "the policy-selected observed population "
                "is the intended deployment target."
            ),
            (
                "The concern arises when restricted "
                "outcomes are treated as representing "
                "the unchanged full-population target."
            ),
            (
                "Under conservative late discharge, "
                "recalibration slightly worsened the "
                "already well-calibrated restricted target."
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
        / "phase5f_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5f_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5F freeze validation"
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
