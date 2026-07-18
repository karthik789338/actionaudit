"""Freeze Phase 5G-A/B practitioner-estimator benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_DIR = Path(
    "artifacts/phase5g_practitioner_estimators"
)

FEATURE_DIR = Path(
    "artifacts/phase5g_baseline_features"
)

RICH_DIR = Path(
    "artifacts/phase5g_rich_baseline_estimators"
)

OUTPUT_DIR = Path(
    "artifacts/phase5g_frozen_for_manuscript"
)

EXPECTED_SCORE_STATUS = (
    "phase5g_practitioner_estimators_benchmarked"
)

EXPECTED_FEATURE_STATUS = (
    "phase5g_exact_baseline_features_ready"
)

EXPECTED_RICH_STATUS = (
    "phase5g_rich_baseline_estimators_benchmarked"
)

FROZEN_STATUS = (
    "phase5g_regularized_practitioner_"
    "benchmark_frozen_for_manuscript"
)

HASHED_FILES = [
    "scripts/phase5g_practitioner_estimators.py",
    "scripts/phase5g_build_baseline_features.py",
    "scripts/phase5g_rich_baseline_estimators.py",
    "scripts/phase5g_final_paired_diagnostics.py",
    (
        "artifacts/phase5g_practitioner_estimators/"
        "estimator_metric_summary.csv"
    ),
    (
        "artifacts/phase5g_practitioner_estimators/"
        "bootstrap_metric_intervals.csv"
    ),
    (
        "artifacts/phase5g_practitioner_estimators/"
        "validation_checks.csv"
    ),
    (
        "artifacts/phase5g_baseline_features/"
        "baseline_feature_manifest.csv"
    ),
    (
        "artifacts/phase5g_baseline_features/"
        "validation_checks.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "rich_estimator_metric_summary.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "rich_estimator_bin_curves.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "bootstrap_metric_intervals.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "expanded_paired_difference_intervals.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "positivity_diagnostics.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "crossfit_fold_diagnostics.csv"
    ),
    (
        "artifacts/phase5g_rich_baseline_estimators/"
        "validation_checks.csv"
    ),
]


def read_status(
    path: Path,
) -> str:
    if not path.exists():
        return "missing"

    return path.read_text(
        encoding="utf-8"
    ).strip()


def all_checks_pass(
    checks: pd.DataFrame,
) -> bool:
    return bool(
        checks[
            "passed"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            "true"
        )
        .all()
    )


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


def paired_value(
    intervals: pd.DataFrame,
    *,
    method: str,
    comparator: str,
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
            "comparator"
        ].eq(
            comparator
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
            "Unable to resolve paired interval: "
            f"{method}, {comparator}, "
            f"{metric}, {column}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def metric_value(
    metrics: pd.DataFrame,
    *,
    method: str,
    metric: str,
) -> float:
    selected = metrics.loc[
        metrics[
            "method"
        ].eq(
            method
        ),
        metric,
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            f"Unable to resolve metric: "
            f"{method}, {metric}"
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

    score_status = read_status(
        SCORE_DIR
        / "phase5g_status.txt"
    )

    feature_status = read_status(
        FEATURE_DIR
        / "phase5g_baseline_feature_status.txt"
    )

    rich_status = read_status(
        RICH_DIR
        / "phase5g_rich_status.txt"
    )

    score_checks = pd.read_csv(
        SCORE_DIR
        / "validation_checks.csv"
    )

    feature_checks = pd.read_csv(
        FEATURE_DIR
        / "validation_checks.csv"
    )

    rich_checks = pd.read_csv(
        RICH_DIR
        / "validation_checks.csv"
    )

    expanded_checks = pd.read_csv(
        RICH_DIR
        / "expanded_paired_validation_checks.csv"
    )

    score_metrics = pd.read_csv(
        SCORE_DIR
        / "estimator_metric_summary.csv"
    )

    rich_metrics = pd.read_csv(
        RICH_DIR
        / "rich_estimator_metric_summary.csv"
    )

    paired_intervals = pd.read_csv(
        RICH_DIR
        / "expanded_paired_difference_intervals.csv"
    )

    positivity = pd.read_csv(
        RICH_DIR
        / "positivity_diagnostics.csv"
    )

    convergence = pd.read_csv(
        RICH_DIR
        / "crossfit_fold_diagnostics.csv"
    )

    score_or_mae = metric_value(
        score_metrics,
        method="pooled_survival_or",
        metric="bin_curve_mae",
    )

    complete_case_mae = metric_value(
        score_metrics,
        method="restricted_complete_case",
        metric="bin_curve_mae",
    )

    rich_or_worse_q025 = paired_value(
        paired_intervals,
        method="rich_pooled_survival_or",
        comparator="score_pooled_survival_or",
        metric="bin_curve_mae",
        column="q025",
    )

    rich_ipcw_worse_q025 = paired_value(
        paired_intervals,
        method="rich_ipcw_clip_0p05",
        comparator="restricted_complete_case",
        metric="bin_curve_mae",
        column="q025",
    )

    aipw_0025_improves_q975 = paired_value(
        paired_intervals,
        method="rich_aipw_clip_0p025",
        comparator="restricted_complete_case",
        metric="bin_curve_mae",
        column="q975",
    )

    aipw_005_improves_q975 = paired_value(
        paired_intervals,
        method="rich_aipw_clip_0p05",
        comparator="restricted_complete_case",
        metric="bin_curve_mae",
        column="q975",
    )

    aipw_005_beats_score_q975 = paired_value(
        paired_intervals,
        method="rich_aipw_clip_0p05",
        comparator="score_aipw",
        metric="bin_curve_mae",
        column="q975",
    )

    aipw_005_max_error_q025 = paired_value(
        paired_intervals,
        method="rich_aipw_clip_0p05",
        comparator="restricted_complete_case",
        metric="maximum_bin_error",
        column="q025",
    )

    clip_005_row = positivity.loc[
        np.isclose(
            positivity[
                "propensity_clip"
            ],
            0.05,
        )
    ]

    if len(
        clip_005_row
    ) != 1:
        raise ValueError(
            "Unable to resolve positivity row "
            "for clipping level 0.05"
        )

    fraction_clipped_005 = float(
        clip_005_row[
            "fraction_clipped_low"
        ].iloc[
            0
        ]
    )

    all_survival_converged = bool(
        convergence[
            "survival_model_converged"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            "true"
        )
        .all()
    )

    all_observation_converged = bool(
        convergence[
            "observation_model_converged"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            "true"
        )
        .all()
    )

    all_interval_counts_valid = bool(
        paired_intervals[
            "n_valid"
        ].eq(
            1000
        ).all()
    )

    signed_equals_absolute = bool(
        np.allclose(
            rich_metrics[
                "signed_curve_error"
            ],
            rich_metrics[
                "bin_curve_mae"
            ],
            atol=1e-10,
            rtol=0.0,
        )
    )

    validation = pd.DataFrame(
        [
            {
                "check": (
                    "score-only benchmark status passed"
                ),
                "value": score_status,
                "threshold": EXPECTED_SCORE_STATUS,
                "comparison": "==",
                "passed": (
                    score_status
                    == EXPECTED_SCORE_STATUS
                ),
            },
            {
                "check": (
                    "exact baseline feature status passed"
                ),
                "value": feature_status,
                "threshold": EXPECTED_FEATURE_STATUS,
                "comparison": "==",
                "passed": (
                    feature_status
                    == EXPECTED_FEATURE_STATUS
                ),
            },
            {
                "check": (
                    "rich baseline benchmark status passed"
                ),
                "value": rich_status,
                "threshold": EXPECTED_RICH_STATUS,
                "comparison": "==",
                "passed": (
                    rich_status
                    == EXPECTED_RICH_STATUS
                ),
            },
            {
                "check": (
                    "all original score-only checks passed"
                ),
                "value": int(
                    all_checks_pass(
                        score_checks
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": all_checks_pass(
                    score_checks
                ),
            },
            {
                "check": (
                    "all feature assembly checks passed"
                ),
                "value": int(
                    all_checks_pass(
                        feature_checks
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": all_checks_pass(
                    feature_checks
                ),
            },
            {
                "check": (
                    "all rich benchmark checks passed"
                ),
                "value": int(
                    all_checks_pass(
                        rich_checks
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": all_checks_pass(
                    rich_checks
                ),
            },
            {
                "check": (
                    "all expanded paired checks passed"
                ),
                "value": int(
                    all_checks_pass(
                        expanded_checks
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": all_checks_pass(
                    expanded_checks
                ),
            },
            {
                "check": (
                    "all 10 survival nuisance fits converged"
                ),
                "value": int(
                    convergence[
                        "survival_model_converged"
                    ]
                    .astype(
                        str
                    )
                    .str.lower()
                    .eq(
                        "true"
                    )
                    .sum()
                ),
                "threshold": 10,
                "comparison": "==",
                "passed": (
                    all_survival_converged
                    and len(
                        convergence
                    )
                    == 10
                ),
            },
            {
                "check": (
                    "all 10 observation nuisance fits converged"
                ),
                "value": int(
                    convergence[
                        "observation_model_converged"
                    ]
                    .astype(
                        str
                    )
                    .str.lower()
                    .eq(
                        "true"
                    )
                    .sum()
                ),
                "threshold": 10,
                "comparison": "==",
                "passed": (
                    all_observation_converged
                    and len(
                        convergence
                    )
                    == 10
                ),
            },
            {
                "check": (
                    "all expanded intervals use "
                    "1000 bootstrap replications"
                ),
                "value": int(
                    paired_intervals[
                        "n_valid"
                    ].min()
                ),
                "threshold": 1000,
                "comparison": "==",
                "passed": (
                    all_interval_counts_valid
                ),
            },
            {
                "check": (
                    "score survival OR improves "
                    "over complete case"
                ),
                "value": score_or_mae,
                "threshold": complete_case_mae,
                "comparison": "<",
                "passed": (
                    score_or_mae
                    < complete_case_mae
                ),
            },
            {
                "check": (
                    "rich OR worse than score OR"
                ),
                "value": rich_or_worse_q025,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    rich_or_worse_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "rich IPCW clip 0.05 worse "
                    "than complete case"
                ),
                "value": rich_ipcw_worse_q025,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    rich_ipcw_worse_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "rich AIPW clip 0.025 improves "
                    "over complete case"
                ),
                "value": aipw_0025_improves_q975,
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    aipw_0025_improves_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "rich AIPW clip 0.05 improves "
                    "over complete case"
                ),
                "value": aipw_005_improves_q975,
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    aipw_005_improves_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "rich AIPW clip 0.05 improves "
                    "over score AIPW"
                ),
                "value": aipw_005_beats_score_q975,
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    aipw_005_beats_score_q975
                    < 0.0
                ),
            },
            {
                "check": (
                    "rich AIPW clip 0.05 worsens "
                    "maximum-bin error"
                ),
                "value": aipw_005_max_error_q025,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    aipw_005_max_error_q025
                    > 0.0
                ),
            },
            {
                "check": (
                    "clip 0.05 truncates over "
                    "40 percent of admissions"
                ),
                "value": fraction_clipped_005,
                "threshold": 0.40,
                "comparison": ">",
                "passed": (
                    fraction_clipped_005
                    > 0.40
                ),
            },
            {
                "check": (
                    "all estimator curve errors "
                    "are uniformly upward"
                ),
                "value": int(
                    signed_equals_absolute
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    signed_equals_absolute
                ),
            },
        ]
    )

    status = (
        FROZEN_STATUS
        if validation[
            "passed"
        ].all()
        else "phase5g_freeze_requires_review"
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
        "phase": "5G-A/B",
        "status": status,
        "scope": (
            "Score-only and exact admission-time "
            "regularized-logistic practitioner estimators."
        ),
        "primary_claim": (
            "Standard score-stratified survival estimators "
            "substantially reduce hospital-boundary "
            "calibration distortion but do not recover "
            "the linked reference curve."
        ),
        "rich_adjustment_claim": (
            "Adding the complete frozen admission-time "
            "feature set does not improve pooled survival "
            "outcome regression, while IPCW remains "
            "limited by severe positivity problems."
        ),
        "aipw_claim": (
            "Aggressively clipped AIPW reduces average "
            "curve error but leaves large uniformly upward "
            "distortion and worsens maximum-bin error."
        ),
        "score_or_curve_mae": score_or_mae,
        "complete_case_curve_mae": (
            complete_case_mae
        ),
        "rich_or_minus_score_or_q025": (
            rich_or_worse_q025
        ),
        "rich_ipcw_005_minus_complete_q025": (
            rich_ipcw_worse_q025
        ),
        "rich_aipw_0025_minus_complete_q975": (
            aipw_0025_improves_q975
        ),
        "rich_aipw_005_minus_complete_q975": (
            aipw_005_improves_q975
        ),
        "rich_aipw_005_minus_score_q975": (
            aipw_005_beats_score_q975
        ),
        "rich_aipw_005_max_error_q025": (
            aipw_005_max_error_q025
        ),
        "fraction_clipped_at_005": (
            fraction_clipped_005
        ),
        "inference": (
            "Patient-cluster bootstrap conditional on "
            "the repeated cross-fitted nuisance predictions."
        ),
        "caveats": [
            (
                "The benchmark covers regularized logistic "
                "nuisance models, not all possible nonlinear "
                "or machine-learning nuisance estimators."
            ),
            (
                "Linked post-discharge outcomes are used "
                "only for retrospective evaluation."
            ),
            (
                "The remaining error is compatible with "
                "residual censoring-outcome dependence or "
                "unmeasured severity, but does not identify "
                "the specific omitted mechanism."
            ),
            (
                "Rich AIPW improvement under clipping is "
                "not equivalent to unbiased reference-target "
                "recovery."
            ),
            (
                "Rich OR and grouped KM are not statistically "
                "distinguishable on curve MAE."
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
        / "phase5g_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5g_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5G-A/B freeze validation"
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
