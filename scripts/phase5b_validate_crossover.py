"""Validate Phase 5B crossover claims using paired replications."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


INPUT_DIR = Path(
    "artifacts/phase5b_continuous_dependence_sweep"
)

OUTPUT_DIR = Path(
    "artifacts/phase5b_continuous_dependence_validation"
)

PRIMARY_CLIP = 0.05

MAE_METRICS = [
    "complete_case_mae",
    "kaplan_meier_mae",
    "learned_working_or_mae",
    "learned_working_ipcw_mae",
]

SIGNED_METRICS = [
    "complete_case_signed_gap",
    "kaplan_meier_signed_gap",
    "learned_working_or_signed_gap",
    "learned_working_ipcw_signed_gap",
]


def mean_confidence_interval(
    values: pd.Series,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Return mean, standard error, and 95% t interval."""

    clean = (
        values.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .astype(float)
    )

    n = len(
        clean
    )

    if n < 2:
        return (
            float(
                clean.mean()
            ),
            float(
                "nan"
            ),
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )

    mean = float(
        clean.mean()
    )

    standard_error = float(
        clean.std(
            ddof=1
        )
        / np.sqrt(
            n
        )
    )

    critical = float(
        t.ppf(
            0.975,
            df=n - 1,
        )
    )

    lower = (
        mean
        - critical
        * standard_error
    )

    upper = (
        mean
        + critical
        * standard_error
    )

    return (
        mean,
        standard_error,
        lower,
        upper,
    )


def paired_difference_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare practical estimators within matched replications."""

    rows: list[
        dict[str, object]
    ] = []

    for gamma, group in results.groupby(
        "gamma",
        sort=True,
    ):
        group = group.sort_values(
            "replication"
        )

        for metric_a, metric_b in combinations(
            MAE_METRICS,
            2,
        ):
            difference = (
                group[
                    metric_a
                ]
                - group[
                    metric_b
                ]
            )

            (
                mean_difference,
                standard_error,
                lower,
                upper,
            ) = mean_confidence_interval(
                difference
            )

            if upper < 0:
                conclusion = (
                    f"{metric_a}_better"
                )
            elif lower > 0:
                conclusion = (
                    f"{metric_b}_better"
                )
            else:
                conclusion = (
                    "not_distinguishable"
                )

            rows.append(
                {
                    "gamma": gamma,
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "mean_a": float(
                        group[
                            metric_a
                        ].mean()
                    ),
                    "mean_b": float(
                        group[
                            metric_b
                        ].mean()
                    ),
                    "mean_difference_a_minus_b": (
                        mean_difference
                    ),
                    "standard_error": (
                        standard_error
                    ),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "conclusion": conclusion,
                    "n_replications": int(
                        len(group)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def winner_inference_table(
    results: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """Determine whether each mean winner dominates all competitors."""

    rows: list[
        dict[str, object]
    ] = []

    for gamma, group in results.groupby(
        "gamma",
        sort=True,
    ):
        means = {
            metric: float(
                group[
                    metric
                ].mean()
            )
            for metric in MAE_METRICS
        }

        mean_winner = min(
            means,
            key=means.get,
        )

        comparisons = (
            pairwise.loc[
                np.isclose(
                    pairwise[
                        "gamma"
                    ],
                    gamma,
                )
                & (
                    pairwise[
                        "metric_a"
                    ].eq(
                        mean_winner
                    )
                    | pairwise[
                        "metric_b"
                    ].eq(
                        mean_winner
                    )
                )
            ]
        )

        comparison_passes: list[
            bool
        ] = []

        for _, row in comparisons.iterrows():
            if (
                row[
                    "metric_a"
                ]
                == mean_winner
            ):
                comparison_passes.append(
                    row[
                        "ci95_upper"
                    ]
                    < 0
                )
            else:
                comparison_passes.append(
                    row[
                        "ci95_lower"
                    ]
                    > 0
                )

        statistically_dominant = (
            len(
                comparison_passes
            )
            == len(
                MAE_METRICS
            )
            - 1
            and all(
                comparison_passes
            )
        )

        sorted_means = sorted(
            means.items(),
            key=lambda item: item[
                1
            ],
        )

        rows.append(
            {
                "gamma": gamma,
                "mean_winner": (
                    mean_winner
                ),
                "mean_winner_mae": (
                    sorted_means[
                        0
                    ][
                        1
                    ]
                ),
                "runner_up": (
                    sorted_means[
                        1
                    ][
                        0
                    ]
                ),
                "runner_up_mae": (
                    sorted_means[
                        1
                    ][
                        1
                    ]
                ),
                "mean_margin": (
                    sorted_means[
                        1
                    ][
                        1
                    ]
                    - sorted_means[
                        0
                    ][
                        1
                    ]
                ),
                "statistically_dominant": (
                    statistically_dominant
                ),
                "winner_classification": (
                    mean_winner
                    if statistically_dominant
                    else "uncertain_boundary"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def signed_gap_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate mean signed gaps and their confidence intervals."""

    rows: list[
        dict[str, object]
    ] = []

    for gamma, group in results.groupby(
        "gamma",
        sort=True,
    ):
        for metric in SIGNED_METRICS:
            (
                mean,
                standard_error,
                lower,
                upper,
            ) = mean_confidence_interval(
                group[
                    metric
                ]
            )

            if lower > 0:
                direction = (
                    "positive"
                )
            elif upper < 0:
                direction = (
                    "negative"
                )
            else:
                direction = (
                    "not_distinguishable_from_zero"
                )

            rows.append(
                {
                    "gamma": gamma,
                    "metric": metric,
                    "mean": mean,
                    "standard_error": (
                        standard_error
                    ),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "direction": direction,
                    "n_replications": int(
                        len(group)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def anchor_direction(
    signed: pd.DataFrame,
    *,
    gamma: float,
    metric: str,
) -> str:
    selected = signed.loc[
        np.isclose(
            signed[
                "gamma"
            ],
            gamma,
        )
        & signed[
            "metric"
        ].eq(
            metric
        )
    ]

    if len(
        selected
    ) != 1:
        return (
            "missing"
        )

    return str(
        selected.iloc[
            0
        ][
            "direction"
        ]
    )


def main() -> None:
    input_path = (
        INPUT_DIR
        / "replication_results.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 5B results: {input_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = pd.read_csv(
        input_path
    )

    results = all_results.loc[
        np.isclose(
            all_results[
                "clip"
            ],
            PRIMARY_CLIP,
        )
    ].copy()

    pairwise = paired_difference_table(
        results
    )

    winners = winner_inference_table(
        results,
        pairwise,
    )

    signed = signed_gap_table(
        results
    )

    pairwise.to_csv(
        OUTPUT_DIR
        / "paired_mae_differences.csv",
        index=False,
    )

    winners.to_csv(
        OUTPUT_DIR
        / "winner_inference.csv",
        index=False,
    )

    signed.to_csv(
        OUTPUT_DIR
        / "signed_gap_inference.csv",
        index=False,
    )

    replications_per_gamma = (
        results.groupby(
            "gamma"
        )[
            "replication"
        ]
        .nunique()
    )

    common_seed_check = (
        results.groupby(
            "replication"
        )[
            [
                "development_seed",
                "test_seed",
            ]
        ]
        .nunique()
        .eq(1)
        .all()
        .all()
    )

    oracle_means = (
        results.groupby(
            "gamma"
        )[
            "learned_oracle_or_mae"
        ]
        .mean()
    )

    dominant_winners = sorted(
        winners.loc[
            winners[
                "statistically_dominant"
            ],
            "mean_winner",
        ]
        .unique()
        .tolist()
    )

    mean_winners = sorted(
        winners[
            "mean_winner"
        ]
        .unique()
        .tolist()
    )

    working_or_negative_anchor = (
        anchor_direction(
            signed,
            gamma=-0.90,
            metric=(
                "learned_working_or_signed_gap"
            ),
        )
    )

    working_or_positive_anchor = (
        anchor_direction(
            signed,
            gamma=0.90,
            metric=(
                "learned_working_or_signed_gap"
            ),
        )
    )

    km_negative_anchor = anchor_direction(
        signed,
        gamma=-0.90,
        metric="kaplan_meier_signed_gap",
    )

    km_positive_anchor = anchor_direction(
        signed,
        gamma=0.90,
        metric="kaplan_meier_signed_gap",
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "100 paired replications per gamma"
                ),
                "value": int(
                    replications_per_gamma.min()
                ),
                "threshold": 100,
                "comparison": "==",
                "passed": bool(
                    replications_per_gamma.eq(
                        100
                    ).all()
                ),
            },
            {
                "check": (
                    "common random-number seeds preserved"
                ),
                "value": (
                    common_seed_check
                ),
                "threshold": True,
                "comparison": "==",
                "passed": bool(
                    common_seed_check
                ),
            },
            {
                "check": (
                    "learned oracle OR mean MAE remains below 0.04"
                ),
                "value": float(
                    oracle_means.max()
                ),
                "threshold": 0.04,
                "comparison": "<",
                "passed": bool(
                    oracle_means.max()
                    < 0.04
                ),
            },
            {
                "check": (
                    "at least three mean winners occur"
                ),
                "value": len(
                    mean_winners
                ),
                "threshold": 3,
                "comparison": ">=",
                "passed": (
                    len(
                        mean_winners
                    )
                    >= 3
                ),
            },
            {
                "check": (
                    "at least two statistically dominant winners occur"
                ),
                "value": len(
                    dominant_winners
                ),
                "threshold": 2,
                "comparison": ">=",
                "passed": (
                    len(
                        dominant_winners
                    )
                    >= 2
                ),
            },
            {
                "check": (
                    "working OR signed gap reverses between anchors"
                ),
                "value": (
                    f"{working_or_negative_anchor}"
                    " -> "
                    f"{working_or_positive_anchor}"
                ),
                "threshold": (
                    "positive -> negative"
                ),
                "comparison": "==",
                "passed": (
                    working_or_negative_anchor
                    == "positive"
                    and working_or_positive_anchor
                    == "negative"
                ),
            },
            {
                "check": (
                    "Kaplan-Meier signed gap reverses between anchors"
                ),
                "value": (
                    f"{km_negative_anchor}"
                    " -> "
                    f"{km_positive_anchor}"
                ),
                "threshold": (
                    "positive -> negative"
                ),
                "comparison": "==",
                "passed": (
                    km_negative_anchor
                    == "positive"
                    and km_positive_anchor
                    == "negative"
                ),
            },
        ]
    )

    status = (
        "phase5b_validated_paired_estimator_crossover"
        if checks[
            "passed"
        ].all()
        else "phase5b_paired_validation_requires_review"
    )

    checks.to_csv(
        OUTPUT_DIR
        / "validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5b_validated_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    evidence = {
        "primary_clip": (
            PRIMARY_CLIP
        ),
        "mean_winners": (
            mean_winners
        ),
        "statistically_dominant_winners": (
            dominant_winners
        ),
        "uncertain_boundary_gammas": (
            winners.loc[
                ~winners[
                    "statistically_dominant"
                ],
                "gamma",
            ].tolist()
        ),
        "maximum_learned_oracle_or_mean_mae": float(
            oracle_means.max()
        ),
        "working_or_anchor_direction": {
            "-0.90": (
                working_or_negative_anchor
            ),
            "+0.90": (
                working_or_positive_anchor
            ),
        },
        "kaplan_meier_anchor_direction": {
            "-0.90": (
                km_negative_anchor
            ),
            "+0.90": (
                km_positive_anchor
            ),
        },
        "interpretation": (
            "The best practical estimator changes across "
            "latent dependence regions. Exact crossover "
            "boundaries are reported as uncertain where "
            "paired estimator differences include zero."
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
        "Phase 5B paired crossover validation"
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
        "\nWinner inference"
    )

    print(
        winners.to_string(
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
