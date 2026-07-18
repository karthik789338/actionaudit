"""Freeze the Phase 5G-C fixed-budget HGB benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGULARIZED_FREEZE_DIR = Path(
    "artifacts/"
    "phase5g_frozen_for_manuscript"
)

HGB_DIR = Path(
    "artifacts/"
    "phase5g_nonlinear_estimators"
)

OUTPUT_DIR = Path(
    "artifacts/"
    "phase5g_hgb_frozen_for_manuscript"
)

EXPECTED_REGULARIZED_STATUS = (
    "phase5g_regularized_practitioner_"
    "benchmark_frozen_for_manuscript"
)

EXPECTED_HGB_STATUS = (
    "phase5g_nonlinear_estimators_benchmarked"
)

FROZEN_STATUS = (
    "phase5g_hgb_nonlinear_benchmark_"
    "frozen_for_manuscript"
)

CLIP_LABELS = [
    "0p01",
    "0p025",
    "0p05",
]

HASHED_FILES = [
    (
        "scripts/"
        "phase5g_nonlinear_estimators.py"
    ),
    (
        "artifacts/"
        "phase5g_frozen_for_manuscript/"
        "phase5g_freeze_manifest.json"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "nonlinear_estimator_metric_summary.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "nonlinear_estimator_bin_curves.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "bootstrap_metric_intervals.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "paired_method_difference_intervals.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "hgb_positivity_diagnostics.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "hgb_crossfit_fold_diagnostics.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "validation_checks.csv"
    ),
    (
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "nonlinear_estimator_evidence.json"
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


def checks_pass(
    frame: pd.DataFrame,
) -> bool:
    return bool(
        frame[
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
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen file: {path}"
        )

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
            f"Unable to resolve metric "
            f"{method}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def paired_bounds(
    intervals: pd.DataFrame,
    *,
    method: str,
    comparator: str,
    metric: str,
) -> tuple[
    float,
    float,
]:
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
        )
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            "Unable to resolve paired interval: "
            f"{method}, {comparator}, {metric}"
        )

    return (
        float(
            selected[
                "q025"
            ].iloc[
                0
            ]
        ),
        float(
            selected[
                "q975"
            ].iloc[
                0
            ]
        ),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    regularized_status = read_status(
        REGULARIZED_FREEZE_DIR
        / "phase5g_freeze_status.txt"
    )

    hgb_status = read_status(
        HGB_DIR
        / "phase5g_nonlinear_status.txt"
    )

    validation_checks = pd.read_csv(
        HGB_DIR
        / "validation_checks.csv"
    )

    metrics = pd.read_csv(
        HGB_DIR
        / "nonlinear_estimator_metric_summary.csv"
    )

    intervals = pd.read_csv(
        HGB_DIR
        / "paired_method_difference_intervals.csv"
    )

    positivity = pd.read_csv(
        HGB_DIR
        / "hgb_positivity_diagnostics.csv"
    )

    diagnostics = pd.read_csv(
        HGB_DIR
        / "hgb_crossfit_fold_diagnostics.csv"
    )

    score_or_mae = metric_value(
        metrics,
        method="score_pooled_survival_or",
        metric="bin_curve_mae",
    )

    rich_or_mae = metric_value(
        metrics,
        method="rich_logistic_survival_or",
        metric="bin_curve_mae",
    )

    grouped_km_mae = metric_value(
        metrics,
        method="grouped_km",
        metric="bin_curve_mae",
    )

    hgb_or_mae = metric_value(
        metrics,
        method="hgb_survival_or",
        metric="bin_curve_mae",
    )

    complete_case_mae = metric_value(
        metrics,
        method="restricted_complete_case",
        metric="bin_curve_mae",
    )

    hgb_aipw_005_mae = metric_value(
        metrics,
        method="hgb_aipw_clip_0p05",
        metric="bin_curve_mae",
    )

    hgb_or_vs_score = paired_bounds(
        intervals,
        method="hgb_survival_or",
        comparator="score_pooled_survival_or",
        metric="bin_curve_mae",
    )

    hgb_or_vs_rich = paired_bounds(
        intervals,
        method="hgb_survival_or",
        comparator="rich_logistic_survival_or",
        metric="bin_curve_mae",
    )

    hgb_or_vs_km = paired_bounds(
        intervals,
        method="hgb_survival_or",
        comparator="grouped_km",
        metric="bin_curve_mae",
    )

    hgb_or_max_vs_score = paired_bounds(
        intervals,
        method="hgb_survival_or",
        comparator="score_pooled_survival_or",
        metric="maximum_bin_error",
    )

    hgb_or_max_vs_rich = paired_bounds(
        intervals,
        method="hgb_survival_or",
        comparator="rich_logistic_survival_or",
        metric="maximum_bin_error",
    )

    hgb_aipw_005_vs_logistic = paired_bounds(
        intervals,
        method="hgb_aipw_clip_0p05",
        comparator=(
            "rich_logistic_aipw_clip_0p05"
        ),
        metric="bin_curve_mae",
    )

    hgb_aipw_005_vs_score = paired_bounds(
        intervals,
        method="hgb_aipw_clip_0p05",
        comparator="score_aipw",
        metric="bin_curve_mae",
    )

    hgb_aipw_005_vs_complete = paired_bounds(
        intervals,
        method="hgb_aipw_clip_0p05",
        comparator="restricted_complete_case",
        metric="bin_curve_mae",
    )

    hgb_aipw_max_vs_score = paired_bounds(
        intervals,
        method="hgb_aipw_clip_0p05",
        comparator="score_aipw",
        metric="maximum_bin_error",
    )

    hgb_aipw_max_vs_complete = paired_bounds(
        intervals,
        method="hgb_aipw_clip_0p05",
        comparator="restricted_complete_case",
        metric="maximum_bin_error",
    )

    ipcw_vs_score_lower_bounds = []

    ipcw_vs_complete_lower_bounds = []

    for label in CLIP_LABELS:
        lower_score, _ = paired_bounds(
            intervals,
            method=f"hgb_ipcw_clip_{label}",
            comparator="score_ipcw",
            metric="bin_curve_mae",
        )

        lower_complete, _ = paired_bounds(
            intervals,
            method=f"hgb_ipcw_clip_{label}",
            comparator=(
                "restricted_complete_case"
            ),
            metric="bin_curve_mae",
        )

        ipcw_vs_score_lower_bounds.append(
            lower_score
        )

        ipcw_vs_complete_lower_bounds.append(
            lower_complete
        )

    clip_005 = positivity.loc[
        np.isclose(
            positivity[
                "propensity_clip"
            ],
            0.05,
        )
    ]

    if len(
        clip_005
    ) != 1:
        raise ValueError(
            "Unable to resolve HGB clip 0.05 "
            "positivity diagnostics"
        )

    clipped_fraction_005 = float(
        clip_005[
            "fraction_clipped_low"
        ].iloc[
            0
        ]
    )

    minimum_probability = float(
        clip_005[
            "raw_minimum_probability"
        ].iloc[
            0
        ]
    )

    q01_probability = float(
        clip_005[
            "raw_q01_probability"
        ].iloc[
            0
        ]
    )

    effective_sample_size_005 = float(
        clip_005[
            (
                "weighted_observed_"
                "effective_sample_size"
            )
        ].iloc[
            0
        ]
    )

    fixed_budget_passed = bool(
        len(
            diagnostics
        )
        == 10
        and diagnostics[
            "survival_iterations"
        ].eq(
            150
        ).all()
        and diagnostics[
            "observation_iterations"
        ].eq(
            150
        ).all()
    )

    interval_count_passed = bool(
        intervals[
            "n_valid"
        ].eq(
            1000
        ).all()
    )

    uniformly_upward = bool(
        np.allclose(
            metrics[
                "signed_curve_error"
            ],
            metrics[
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
                    "regularized Phase 5G-A/B "
                    "benchmark already frozen"
                ),
                "value": regularized_status,
                "threshold": (
                    EXPECTED_REGULARIZED_STATUS
                ),
                "comparison": "==",
                "passed": (
                    regularized_status
                    == EXPECTED_REGULARIZED_STATUS
                ),
            },
            {
                "check": (
                    "HGB benchmark status passed"
                ),
                "value": hgb_status,
                "threshold": EXPECTED_HGB_STATUS,
                "comparison": "==",
                "passed": (
                    hgb_status
                    == EXPECTED_HGB_STATUS
                ),
            },
            {
                "check": (
                    "all original HGB validation "
                    "checks passed"
                ),
                "value": int(
                    checks_pass(
                        validation_checks
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": checks_pass(
                    validation_checks
                ),
            },
            {
                "check": (
                    "all 10 HGB model pairs completed "
                    "fixed 150-iteration budget"
                ),
                "value": len(
                    diagnostics
                ),
                "threshold": 10,
                "comparison": "==",
                "passed": fixed_budget_passed,
            },
            {
                "check": (
                    "all paired intervals use "
                    "1000 bootstrap replications"
                ),
                "value": int(
                    intervals[
                        "n_valid"
                    ].min()
                ),
                "threshold": 1000,
                "comparison": "==",
                "passed": (
                    interval_count_passed
                ),
            },
            {
                "check": (
                    "HGB OR worse than score OR"
                ),
                "value": hgb_or_vs_score[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    hgb_or_vs_score[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB OR worse than rich "
                    "logistic OR"
                ),
                "value": hgb_or_vs_rich[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    hgb_or_vs_rich[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB OR indistinguishable "
                    "from grouped KM"
                ),
                "value": (
                    f"{hgb_or_vs_km[0]},"
                    f"{hgb_or_vs_km[1]}"
                ),
                "threshold": (
                    "interval crosses zero"
                ),
                "comparison": "crosses",
                "passed": (
                    hgb_or_vs_km[
                        0
                    ]
                    < 0.0
                    < hgb_or_vs_km[
                        1
                    ]
                ),
            },
            {
                "check": (
                    "HGB OR worsens maximum-bin "
                    "error versus score OR"
                ),
                "value": hgb_or_max_vs_score[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    hgb_or_max_vs_score[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB OR worsens maximum-bin "
                    "error versus rich OR"
                ),
                "value": hgb_or_max_vs_rich[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    hgb_or_max_vs_rich[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB IPCW worse than score IPCW "
                    "at every clipping level"
                ),
                "value": min(
                    ipcw_vs_score_lower_bounds
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    min(
                        ipcw_vs_score_lower_bounds
                    )
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB IPCW worse than complete case "
                    "at every clipping level"
                ),
                "value": min(
                    ipcw_vs_complete_lower_bounds
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    min(
                        ipcw_vs_complete_lower_bounds
                    )
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB AIPW clip 0.05 improves "
                    "over score AIPW"
                ),
                "value": hgb_aipw_005_vs_score[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    hgb_aipw_005_vs_score[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "HGB AIPW clip 0.05 improves "
                    "over complete case"
                ),
                "value": hgb_aipw_005_vs_complete[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    hgb_aipw_005_vs_complete[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "HGB and logistic AIPW clip 0.05 "
                    "are indistinguishable"
                ),
                "value": (
                    f"{hgb_aipw_005_vs_logistic[0]},"
                    f"{hgb_aipw_005_vs_logistic[1]}"
                ),
                "threshold": (
                    "interval crosses zero"
                ),
                "comparison": "crosses",
                "passed": (
                    hgb_aipw_005_vs_logistic[
                        0
                    ]
                    < 0.0
                    < hgb_aipw_005_vs_logistic[
                        1
                    ]
                ),
            },
            {
                "check": (
                    "HGB AIPW clip 0.05 worsens "
                    "maximum-bin error versus score AIPW"
                ),
                "value": hgb_aipw_max_vs_score[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    hgb_aipw_max_vs_score[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "HGB AIPW maximum-bin difference "
                    "versus complete case is uncertain"
                ),
                "value": (
                    f"{hgb_aipw_max_vs_complete[0]},"
                    f"{hgb_aipw_max_vs_complete[1]}"
                ),
                "threshold": (
                    "interval crosses zero"
                ),
                "comparison": "crosses",
                "passed": (
                    hgb_aipw_max_vs_complete[
                        0
                    ]
                    < 0.0
                    < hgb_aipw_max_vs_complete[
                        1
                    ]
                ),
            },
            {
                "check": (
                    "clip 0.05 truncates nearly "
                    "half of admissions"
                ),
                "value": clipped_fraction_005,
                "threshold": 0.45,
                "comparison": ">",
                "passed": (
                    clipped_fraction_005
                    > 0.45
                ),
            },
            {
                "check": (
                    "all estimator curve errors "
                    "remain uniformly upward"
                ),
                "value": int(
                    uniformly_upward
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": uniformly_upward,
            },
        ]
    )

    status = (
        FROZEN_STATUS
        if validation[
            "passed"
        ].all()
        else (
            "phase5g_hgb_freeze_"
            "requires_review"
        )
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
        "phase": "5G-C",
        "status": status,
        "scope": (
            "Fixed-budget histogram-gradient-boosting "
            "outcome and observation nuisance models."
        ),
        "primary_hgb_claim": (
            "Under the prespecified HGB nuisance family, "
            "nonlinear admission-time adjustment did not "
            "improve linked-reference calibration recovery "
            "over score-only or regularized-logistic "
            "survival outcome regression."
        ),
        "weighting_claim": (
            "HGB IPCW remained worse than score IPCW and "
            "complete-case estimation under all evaluated "
            "clipping thresholds."
        ),
        "aipw_claim": (
            "HGB AIPW with clipping at 0.05 reduced average "
            "curve error relative to score AIPW and complete "
            "case, but was statistically indistinguishable "
            "from logistic AIPW and worsened maximum-bin "
            "error relative to score AIPW."
        ),
        "positivity_claim": (
            "The nonlinear observation model did not resolve "
            "the positivity limitation; approximately half "
            "the cohort was truncated at clip 0.05."
        ),
        "score_or_curve_mae": (
            score_or_mae
        ),
        "rich_logistic_or_curve_mae": (
            rich_or_mae
        ),
        "grouped_km_curve_mae": (
            grouped_km_mae
        ),
        "hgb_or_curve_mae": (
            hgb_or_mae
        ),
        "complete_case_curve_mae": (
            complete_case_mae
        ),
        "hgb_aipw_clip_005_curve_mae": (
            hgb_aipw_005_mae
        ),
        "hgb_or_minus_score_or_interval": (
            list(
                hgb_or_vs_score
            )
        ),
        "hgb_or_minus_rich_or_interval": (
            list(
                hgb_or_vs_rich
            )
        ),
        "hgb_aipw_005_minus_logistic_interval": (
            list(
                hgb_aipw_005_vs_logistic
            )
        ),
        "hgb_aipw_005_minus_score_interval": (
            list(
                hgb_aipw_005_vs_score
            )
        ),
        "hgb_aipw_005_minus_complete_interval": (
            list(
                hgb_aipw_005_vs_complete
            )
        ),
        "fraction_clipped_at_005": (
            clipped_fraction_005
        ),
        "minimum_observation_probability": (
            minimum_probability
        ),
        "observation_probability_q01": (
            q01_probability
        ),
        "effective_observed_sample_size_at_005": (
            effective_sample_size_005
        ),
        "fixed_boosting_iterations": 150,
        "crossfit_repeats": 5,
        "bootstrap_replicates": 1000,
        "inference": (
            "Patient-cluster bootstrap conditional on "
            "the repeated cross-fitted nuisance predictions."
        ),
        "caveats": [
            (
                "The nonlinear benchmark covers one "
                "prespecified histogram-gradient-boosting "
                "family and is not a universal result for "
                "all machine-learning estimators."
            ),
            (
                "Hyperparameters were prespecified without "
                "selecting against linked reference outcomes."
            ),
            (
                "The bootstrap conditions on the fitted "
                "cross-fitted nuisance predictions and does "
                "not refit nuisance models within every "
                "bootstrap replication."
            ),
            (
                "Linked post-discharge outcomes were used "
                "only for retrospective evaluation."
            ),
            (
                "Improvement from clipped AIPW does not imply "
                "unbiased reference-target recovery."
            ),
            (
                "The remaining distortion is compatible with "
                "residual outcome-censoring dependence or "
                "omitted severity but does not identify the "
                "specific omitted mechanism."
            ),
        ],
        "sha256": hashes,
    }

    validation.to_csv(
        OUTPUT_DIR
        / "hgb_freeze_validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5g_hgb_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5g_hgb_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5G-C HGB freeze validation"
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
