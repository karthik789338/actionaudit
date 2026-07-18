"""Validate and freeze the Phase 5A flexible-hazard experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_DIR = Path(
    "artifacts/phase5a_flexible_hazards_60d"
)

OUTPUT_DIR = Path(
    "artifacts/phase5a_flexible_hazards_60d_validation"
)


def main() -> None:
    summary_path = (
        INPUT_DIR
        / "test_estimator_summary.csv"
    )

    interval_path = (
        INPUT_DIR
        / "bootstrap_intervals.csv"
    )

    selection_path = (
        INPUT_DIR
        / "nuisance_model_selection.csv"
    )

    for path in [
        summary_path,
        interval_path,
        selection_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.read_csv(
        summary_path
    )

    intervals = pd.read_csv(
        interval_path
    )

    selection = pd.read_csv(
        selection_path
    )

    summary_index = summary.set_index(
        "estimator_label"
    )

    interval_index = intervals.set_index(
        [
            "estimator_label",
            "metric",
        ]
    )

    selected_event = selection.loc[
        selection[
            "target"
        ].eq(
            "event"
        )
        & selection[
            "selected_overall_target"
        ].astype(bool),
        "family",
    ]

    if len(selected_event) != 1:
        raise RuntimeError(
            "Expected exactly one validation-selected "
            "event family"
        )

    selected_event_family = str(
        selected_event.iloc[0]
    )

    linear_mae = float(
        summary_index.loc[
            "linear",
            "survival_or_mae",
        ]
    )

    spline_mae = float(
        summary_index.loc[
            "spline",
            "survival_or_mae",
        ]
    )

    gbm_mae = float(
        summary_index.loc[
            "gbm",
            "survival_or_mae",
        ]
    )

    linear_signed_lower = float(
        interval_index.loc[
            (
                "linear",
                "survival_or_signed_gap",
            ),
            "bootstrap_q025",
        ]
    )

    spline_signed_lower = float(
        interval_index.loc[
            (
                "spline",
                "survival_or_signed_gap",
            ),
            "bootstrap_q025",
        ]
    )

    gbm_signed_lower = float(
        interval_index.loc[
            (
                "gbm",
                "survival_or_signed_gap",
            ),
            "bootstrap_q025",
        ]
    )

    spline_difference_lower = float(
        interval_index.loc[
            (
                "spline",
                "survival_or_mae_difference_vs_linear",
            ),
            "bootstrap_q025",
        ]
    )

    gbm_difference_lower = float(
        interval_index.loc[
            (
                "gbm",
                "survival_or_mae_difference_vs_linear",
            ),
            "bootstrap_q025",
        ]
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "validation selects linear event model"
                ),
                "value": selected_event_family,
                "threshold": "linear",
                "comparison": "==",
                "passed": (
                    selected_event_family
                    == "linear"
                ),
            },
            {
                "check": (
                    "linear OR signed gap remains positive"
                ),
                "value": linear_signed_lower,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    linear_signed_lower
                    > 0
                ),
            },
            {
                "check": (
                    "spline OR signed gap remains positive"
                ),
                "value": spline_signed_lower,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    spline_signed_lower
                    > 0
                ),
            },
            {
                "check": (
                    "GBM OR signed gap remains positive"
                ),
                "value": gbm_signed_lower,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    gbm_signed_lower
                    > 0
                ),
            },
            {
                "check": (
                    "spline does not outperform linear OR"
                ),
                "value": spline_difference_lower,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    spline_difference_lower
                    > 0
                ),
            },
            {
                "check": (
                    "GBM does not outperform linear OR"
                ),
                "value": gbm_difference_lower,
                "threshold": 0.0,
                "comparison": ">",
                "passed": (
                    gbm_difference_lower
                    > 0
                ),
            },
            {
                "check": (
                    "best flexible OR remains materially biased"
                ),
                "value": min(
                    spline_mae,
                    gbm_mae,
                ),
                "threshold": 0.04,
                "comparison": ">",
                "passed": (
                    min(
                        spline_mae,
                        gbm_mae,
                    )
                    > 0.04
                ),
            },
        ]
    )

    status = (
        "phase5a_validated_positive_residual_persists"
        if checks[
            "passed"
        ].all()
        else "phase5a_validation_requires_review"
    )

    checks.to_csv(
        OUTPUT_DIR
        / "validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5a_validated_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    evidence = {
        "selected_event_family": (
            selected_event_family
        ),
        "linear_survival_or_mae": (
            linear_mae
        ),
        "spline_survival_or_mae": (
            spline_mae
        ),
        "gbm_survival_or_mae": (
            gbm_mae
        ),
        "linear_signed_gap_q025": (
            linear_signed_lower
        ),
        "spline_signed_gap_q025": (
            spline_signed_lower
        ),
        "gbm_signed_gap_q025": (
            gbm_signed_lower
        ),
        "spline_difference_vs_linear_q025": (
            spline_difference_lower
        ),
        "gbm_difference_vs_linear_q025": (
            gbm_difference_lower
        ),
        "interpretation": (
            "The positive residual persists across the "
            "tested baseline-covariate event-hazard model "
            "families. This is evidence against simple "
            "functional-form misspecification, not proof "
            "of residual dependent censoring."
        ),
        "status": status,
    }

    (
        OUTPUT_DIR
        / "validation_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Phase 5A validation"
    )

    print(
        "\nValidation checks"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"\nSaved under: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
