"""Freeze Phase 5G-D random-forest sensitivity benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HGB_FREEZE_DIR = Path(
    "artifacts/phase5g_hgb_frozen_for_manuscript"
)

RF_DIR = Path(
    "artifacts/phase5g_random_forest_estimators"
)

OUTPUT_DIR = Path(
    "artifacts/phase5g_rf_frozen_for_manuscript"
)

EXPECTED_HGB_STATUS = (
    "phase5g_hgb_nonlinear_benchmark_"
    "frozen_for_manuscript"
)

EXPECTED_RF_STATUS = (
    "phase5g_random_forest_estimators_benchmarked"
)

FROZEN_STATUS = (
    "phase5g_random_forest_benchmark_"
    "frozen_for_manuscript"
)

CLIPS = [
    "0p01",
    "0p025",
    "0p05",
]

HASHED_FILES = [
    "scripts/phase5g_random_forest_estimators.py",
    (
        "artifacts/phase5g_hgb_frozen_for_manuscript/"
        "phase5g_hgb_freeze_manifest.json"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "random_forest_metric_summary.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "random_forest_bin_curves.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "bootstrap_metric_intervals.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "paired_method_difference_intervals.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "rf_positivity_diagnostics.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "rf_crossfit_fold_diagnostics.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "validation_checks.csv"
    ),
    (
        "artifacts/phase5g_random_forest_estimators/"
        "random_forest_evidence.json"
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
            f"Missing freeze input: {path}"
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
            f"Missing metric: {method}, {metric}"
        )

    return float(
        selected.iloc[
            0
        ]
    )


def interval_bounds(
    intervals: pd.DataFrame,
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
            "Missing paired interval: "
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

    hgb_status = read_status(
        HGB_FREEZE_DIR
        / "phase5g_hgb_freeze_status.txt"
    )

    rf_status = read_status(
        RF_DIR
        / "phase5g_random_forest_status.txt"
    )

    original_checks = pd.read_csv(
        RF_DIR
        / "validation_checks.csv"
    )

    metrics = pd.read_csv(
        RF_DIR
        / "random_forest_metric_summary.csv"
    )

    intervals = pd.read_csv(
        RF_DIR
        / "paired_method_difference_intervals.csv"
    )

    positivity = pd.read_csv(
        RF_DIR
        / "rf_positivity_diagnostics.csv"
    )

    diagnostics = pd.read_csv(
        RF_DIR
        / "rf_crossfit_fold_diagnostics.csv"
    )

    rf_or_vs_score = interval_bounds(
        intervals,
        "rf_survival_or",
        "score_pooled_survival_or",
        "bin_curve_mae",
    )

    rf_or_vs_rich = interval_bounds(
        intervals,
        "rf_survival_or",
        "rich_logistic_survival_or",
        "bin_curve_mae",
    )

    rf_or_vs_hgb = interval_bounds(
        intervals,
        "rf_survival_or",
        "hgb_survival_or",
        "bin_curve_mae",
    )

    rf_or_vs_km = interval_bounds(
        intervals,
        "rf_survival_or",
        "grouped_km",
        "bin_curve_mae",
    )

    rf_max_vs_hgb = interval_bounds(
        intervals,
        "rf_survival_or",
        "hgb_survival_or",
        "maximum_bin_error",
    )

    rf_aipw_vs_score = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "score_aipw",
        "bin_curve_mae",
    )

    rf_aipw_vs_logistic = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "rich_logistic_aipw_clip_0p05",
        "bin_curve_mae",
    )

    rf_aipw_vs_hgb = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "hgb_aipw_clip_0p05",
        "bin_curve_mae",
    )

    rf_aipw_vs_complete = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "restricted_complete_case",
        "bin_curve_mae",
    )

    rf_aipw_max_vs_score = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "score_aipw",
        "maximum_bin_error",
    )

    rf_aipw_max_vs_logistic = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "rich_logistic_aipw_clip_0p05",
        "maximum_bin_error",
    )

    rf_aipw_max_vs_hgb = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "hgb_aipw_clip_0p05",
        "maximum_bin_error",
    )

    rf_aipw_max_vs_complete = interval_bounds(
        intervals,
        "rf_aipw_clip_0p05",
        "restricted_complete_case",
        "maximum_bin_error",
    )

    ipcw_vs_score_lower = []
    ipcw_vs_complete_lower = []
    ipcw_vs_logistic_upper = []

    for clip in CLIPS:
        lower, _ = interval_bounds(
            intervals,
            f"rf_ipcw_clip_{clip}",
            "score_ipcw",
            "bin_curve_mae",
        )

        ipcw_vs_score_lower.append(
            lower
        )

        lower, _ = interval_bounds(
            intervals,
            f"rf_ipcw_clip_{clip}",
            "restricted_complete_case",
            "bin_curve_mae",
        )

        ipcw_vs_complete_lower.append(
            lower
        )

        _, upper = interval_bounds(
            intervals,
            f"rf_ipcw_clip_{clip}",
            (
                "rich_logistic_ipcw_"
                f"clip_{clip}"
            ),
            "bin_curve_mae",
        )

        ipcw_vs_logistic_upper.append(
            upper
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
            "Missing RF positivity row "
            "for clip 0.05"
        )

    fraction_clipped = float(
        clip_005[
            "fraction_clipped_low"
        ].iloc[
            0
        ]
    )

    fixed_budget = bool(
        len(
            diagnostics
        )
        == 10
        and diagnostics[
            "survival_trees"
        ].eq(
            300
        ).all()
        and diagnostics[
            "observation_trees"
        ].eq(
            300
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

    all_interval_counts = bool(
        intervals[
            "n_valid"
        ].eq(
            1000
        ).all()
    )

    validation = pd.DataFrame(
        [
            {
                "check": "HGB benchmark already frozen",
                "value": hgb_status,
                "threshold": EXPECTED_HGB_STATUS,
                "comparison": "==",
                "passed": (
                    hgb_status
                    == EXPECTED_HGB_STATUS
                ),
            },
            {
                "check": "RF benchmark status passed",
                "value": rf_status,
                "threshold": EXPECTED_RF_STATUS,
                "comparison": "==",
                "passed": (
                    rf_status
                    == EXPECTED_RF_STATUS
                ),
            },
            {
                "check": (
                    "all original RF validation "
                    "checks passed"
                ),
                "value": int(
                    all_checks_pass(
                        original_checks
                    )
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": all_checks_pass(
                    original_checks
                ),
            },
            {
                "check": (
                    "all 10 RF model pairs completed "
                    "300-tree budget"
                ),
                "value": len(
                    diagnostics
                ),
                "threshold": 10,
                "comparison": "==",
                "passed": fixed_budget,
            },
            {
                "check": (
                    "all paired intervals use "
                    "1000 bootstraps"
                ),
                "value": int(
                    intervals[
                        "n_valid"
                    ].min()
                ),
                "threshold": 1000,
                "comparison": "==",
                "passed": all_interval_counts,
            },
            {
                "check": (
                    "RF OR worse than score OR"
                ),
                "value": rf_or_vs_score[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    rf_or_vs_score[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "RF OR worse than rich OR"
                ),
                "value": rf_or_vs_rich[
                    0
                ],
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    rf_or_vs_rich[
                        0
                    ]
                    > 0.0
                ),
            },
            {
                "check": (
                    "RF and HGB OR indistinguishable"
                ),
                "value": (
                    f"{rf_or_vs_hgb[0]},"
                    f"{rf_or_vs_hgb[1]}"
                ),
                "threshold": "crosses zero",
                "comparison": "crosses",
                "passed": (
                    rf_or_vs_hgb[
                        0
                    ]
                    < 0.0
                    < rf_or_vs_hgb[
                        1
                    ]
                ),
            },
            {
                "check": (
                    "RF and grouped KM "
                    "indistinguishable"
                ),
                "value": (
                    f"{rf_or_vs_km[0]},"
                    f"{rf_or_vs_km[1]}"
                ),
                "threshold": "crosses zero",
                "comparison": "crosses",
                "passed": (
                    rf_or_vs_km[
                        0
                    ]
                    < 0.0
                    < rf_or_vs_km[
                        1
                    ]
                ),
            },
            {
                "check": (
                    "RF OR improves maximum-bin "
                    "error versus HGB"
                ),
                "value": rf_max_vs_hgb[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    rf_max_vs_hgb[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF IPCW worse than score IPCW "
                    "at all clipping levels"
                ),
                "value": min(
                    ipcw_vs_score_lower
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    min(
                        ipcw_vs_score_lower
                    )
                    > 0.0
                ),
            },
            {
                "check": (
                    "RF IPCW worse than complete case "
                    "at all clipping levels"
                ),
                "value": min(
                    ipcw_vs_complete_lower
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    min(
                        ipcw_vs_complete_lower
                    )
                    > 0.0
                ),
            },
            {
                "check": (
                    "RF IPCW improves over logistic "
                    "IPCW at all clipping levels"
                ),
                "value": max(
                    ipcw_vs_logistic_upper
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    max(
                        ipcw_vs_logistic_upper
                    )
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF AIPW 0.05 improves over "
                    "score AIPW"
                ),
                "value": rf_aipw_vs_score[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    rf_aipw_vs_score[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF AIPW 0.05 improves over "
                    "logistic AIPW"
                ),
                "value": rf_aipw_vs_logistic[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    rf_aipw_vs_logistic[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF AIPW 0.05 improves over "
                    "HGB AIPW"
                ),
                "value": rf_aipw_vs_hgb[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    rf_aipw_vs_hgb[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF AIPW 0.05 improves over "
                    "complete case"
                ),
                "value": rf_aipw_vs_complete[
                    1
                ],
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    rf_aipw_vs_complete[
                        1
                    ]
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF AIPW 0.05 improves "
                    "maximum-bin error versus "
                    "score logistic and HGB AIPW"
                ),
                "value": max(
                    rf_aipw_max_vs_score[
                        1
                    ],
                    rf_aipw_max_vs_logistic[
                        1
                    ],
                    rf_aipw_max_vs_hgb[
                        1
                    ],
                ),
                "threshold": 0.0,
                "comparison": "<",
                "passed": (
                    max(
                        rf_aipw_max_vs_score[
                            1
                        ],
                        rf_aipw_max_vs_logistic[
                            1
                        ],
                        rf_aipw_max_vs_hgb[
                            1
                        ],
                    )
                    < 0.0
                ),
            },
            {
                "check": (
                    "RF AIPW maximum-bin difference "
                    "versus complete case uncertain"
                ),
                "value": (
                    f"{rf_aipw_max_vs_complete[0]},"
                    f"{rf_aipw_max_vs_complete[1]}"
                ),
                "threshold": "crosses zero",
                "comparison": "crosses",
                "passed": (
                    rf_aipw_max_vs_complete[
                        0
                    ]
                    < 0.0
                    < rf_aipw_max_vs_complete[
                        1
                    ]
                ),
            },
            {
                "check": (
                    "clip 0.05 truncates over "
                    "45 percent"
                ),
                "value": fraction_clipped,
                "threshold": 0.45,
                "comparison": ">",
                "passed": (
                    fraction_clipped
                    > 0.45
                ),
            },
            {
                "check": (
                    "all curve errors remain "
                    "uniformly upward"
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
        else "phase5g_rf_freeze_requires_review"
    )

    manifest = {
        "phase": "5G-D",
        "status": status,
        "scope": (
            "Fixed-budget random-forest "
            "nuisance-family sensitivity."
        ),
        "outcome_regression_claim": (
            "Random-forest survival outcome "
            "regression did not improve recovery "
            "over score-only or rich logistic OR."
        ),
        "weighting_claim": (
            "RF IPCW improved over logistic IPCW "
            "but remained worse than score IPCW and "
            "complete-case estimation."
        ),
        "augmentation_claim": (
            "RF-AIPW with clipping at 0.05 produced "
            "the lowest average error among evaluated "
            "augmented estimators, but retained large "
            "uniformly upward distortion."
        ),
        "rf_or_curve_mae": metric_value(
            metrics,
            "rf_survival_or",
            "bin_curve_mae",
        ),
        "rf_aipw_005_curve_mae": metric_value(
            metrics,
            "rf_aipw_clip_0p05",
            "bin_curve_mae",
        ),
        "rf_aipw_005_event_rate": metric_value(
            metrics,
            "rf_aipw_clip_0p05",
            "estimated_event_rate",
        ),
        "reference_event_rate": metric_value(
            metrics,
            "reference_oracle",
            "estimated_event_rate",
        ),
        "fraction_clipped_at_005": (
            fraction_clipped
        ),
        "crossfit_repeats": 5,
        "trees_per_model": 300,
        "bootstrap_replicates": 1000,
        "caveats": [
            (
                "Random forest was a prespecified "
                "model-family sensitivity, not selected "
                "against linked outcomes."
            ),
            (
                "RF-AIPW improvement is partial and "
                "does not constitute reference-target "
                "recovery."
            ),
            (
                "Maximum-bin improvement relative to "
                "complete case remains uncertain."
            ),
            (
                "Inference conditions on fitted "
                "cross-fitted nuisance predictions."
            ),
            (
                "Linked outcomes were used only for "
                "retrospective evaluation."
            ),
        ],
        "sha256": {
            item: sha256_file(
                Path(
                    item
                )
            )
            for item in HASHED_FILES
        },
    }

    validation.to_csv(
        OUTPUT_DIR
        / "rf_freeze_validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5g_rf_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5g_rf_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5G-D RF freeze validation"
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
