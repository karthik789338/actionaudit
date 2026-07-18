"""Expanded paired comparisons for Phase 5G-B."""

from pathlib import Path

import pandas as pd


ROOT = Path(
    "artifacts/"
    "phase5g_rich_baseline_estimators_smoke"
)

BOOTSTRAP_PATH = (
    ROOT
    / "bootstrap_metric_replicates.csv.gz"
)

POINT_PATH = (
    ROOT
    / "rich_estimator_metric_summary.csv"
)

COMPARISONS = [
    (
        "rich_pooled_survival_or",
        "score_pooled_survival_or",
    ),
    (
        "rich_pooled_survival_or",
        "grouped_km",
    ),
    (
        "rich_aipw_clip_0p01",
        "score_aipw",
    ),
    (
        "rich_aipw_clip_0p025",
        "score_aipw",
    ),
    (
        "rich_aipw_clip_0p05",
        "score_aipw",
    ),
    (
        "rich_aipw_clip_0p01",
        "restricted_complete_case",
    ),
    (
        "rich_aipw_clip_0p025",
        "restricted_complete_case",
    ),
    (
        "rich_aipw_clip_0p05",
        "restricted_complete_case",
    ),
    (
        "rich_ipcw_clip_0p01",
        "score_ipcw",
    ),
    (
        "rich_ipcw_clip_0p025",
        "score_ipcw",
    ),
    (
        "rich_ipcw_clip_0p05",
        "score_ipcw",
    ),
    (
        "rich_ipcw_clip_0p01",
        "restricted_complete_case",
    ),
    (
        "rich_ipcw_clip_0p025",
        "restricted_complete_case",
    ),
    (
        "rich_ipcw_clip_0p05",
        "restricted_complete_case",
    ),
]

METRICS = [
    "bin_curve_mae",
    "event_rate_error",
    "ici_error",
    "maximum_bin_error",
]


def main() -> None:
    bootstrap = pd.read_csv(
        BOOTSTRAP_PATH
    )

    point = pd.read_csv(
        POINT_PATH
    ).set_index(
        "method"
    )

    point_rows = []

    interval_rows = []

    for method, comparator in COMPARISONS:
        for metric in METRICS:
            point_rows.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(
                        point.loc[
                            method,
                            metric,
                        ]
                        - point.loc[
                            comparator,
                            metric,
                        ]
                    ),
                }
            )

            left = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        method
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: "method_value",
                    }
                )
            )

            right = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        comparator
                    ),
                    [
                        "bootstrap_id",
                        metric,
                    ],
                ]
                .rename(
                    columns={
                        metric: (
                            "comparator_value"
                        ),
                    }
                )
            )

            paired = left.merge(
                right,
                on="bootstrap_id",
                validate="one_to_one",
            )

            difference = (
                paired[
                    "method_value"
                ]
                - paired[
                    "comparator_value"
                ]
            )

            interval_rows.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "metric": metric,
                    "q025": float(
                        difference.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        difference.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        difference.quantile(
                            0.975
                        )
                    ),
                    "n_valid": int(
                        difference.notna().sum()
                    ),
                }
            )

    points = pd.DataFrame(
        point_rows
    )

    intervals = pd.DataFrame(
        interval_rows
    )

    points.to_csv(
        ROOT
        / "expanded_paired_point_differences.csv",
        index=False,
    )

    intervals.to_csv(
        ROOT
        / "expanded_paired_difference_intervals.csv",
        index=False,
    )

    print(
        intervals.loc[
            intervals[
                "metric"
            ].eq(
                "bin_curve_mae"
            )
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
