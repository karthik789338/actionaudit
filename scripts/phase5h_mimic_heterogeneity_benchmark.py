"""Phase 5H-F: MIMIC clinical heterogeneity benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_INPUT_STATUS = (
    "phase5h_mimic_clinical_strata_ready"
)

EXPECTED_N = 31012


DIMENSIONS = {
    "clinical_chapter": {
        "column": "clinical_chapter",
        "evidence_tier": "primary",
    },
    "first_hospital_service": {
        "column": "first_service",
        "evidence_tier": "primary",
    },
    "admission_era": {
        "column": "anchor_year_group",
        "evidence_tier": "primary",
    },
    "admission_hour": {
        "column": "admission_hour_stratum",
        "evidence_tier": "primary",
    },
    "admission_weekpart": {
        "column": "admission_weekpart",
        "evidence_tier": "primary",
    },
    "icu_involvement": {
        "column": "icu_stratum",
        "evidence_tier": "primary",
    },
    "admission_type": {
        "column": "admission_type",
        "evidence_tier": "secondary",
    },
    "admission_location": {
        "column": "admission_location",
        "evidence_tier": "secondary",
    },
}


CHAPTER_MAP = {
    "mental_disorders": "mental_behavioral",
    "mental_behavioral": "mental_behavioral",

    "endocrine_metabolic_immunity": (
        "endocrine_metabolic"
    ),
    "endocrine_nutritional_metabolic": (
        "endocrine_metabolic"
    ),

    "blood_disorders": (
        "blood_and_immune_disorders"
    ),
    "blood_and_immune_disorders": (
        "blood_and_immune_disorders"
    ),

    "nervous_and_sense_organs": (
        "nervous_and_sensory"
    ),
    "nervous_system": (
        "nervous_and_sensory"
    ),
    "eye_and_ear": (
        "nervous_and_sensory"
    ),

    "injury_and_poisoning": (
        "injury_poisoning"
    ),
    "injury_poisoning": (
        "injury_poisoning"
    ),

    "symptoms_and_ill_defined": (
        "symptoms_abnormal_findings"
    ),
    "symptoms_and_abnormal_findings": (
        "symptoms_abnormal_findings"
    ),

    "supplementary_factors": (
        "factors_influencing_health"
    ),
    "factors_influencing_health": (
        "factors_influencing_health"
    ),
}


PRESPECIFIED_CONTRASTS = [
    {
        "contrast": "icu_minus_non_icu",
        "dimension": "icu_involvement",
        "first": "icu",
        "second": "non_icu",
    },
    {
        "contrast": "weekend_minus_weekday",
        "dimension": "admission_weekpart",
        "first": "weekend",
        "second": "weekday",
    },
    {
        "contrast": "latest_minus_earliest_era",
        "dimension": "admission_era",
        "first": "2020 - 2022",
        "second": "2008 - 2010",
    },
    {
        "contrast": "overnight_minus_morning",
        "dimension": "admission_hour",
        "first": "overnight_00_to_05",
        "second": "morning_06_to_11",
    },
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MIMIC calibration distortion "
            "heterogeneity across clinical strata."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/"
            "phase5h_mimic_clinical_strata/"
            "mimic_harmonized_clinical_test.csv.gz"
        ),
    )

    parser.add_argument(
        "--input-status",
        default=(
            "artifacts/"
            "phase5h_mimic_clinical_strata/"
            "phase5h_mimic_clinical_strata_status.txt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5h_mimic_heterogeneity_benchmark"
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
        "--minimum-stratum-n",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--minimum-observed-n",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--minimum-hidden-n",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def require_columns(
    data: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def harmonize_chapter(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return "missing_or_unclassified"

    label = str(
        value
    ).strip()

    if not label:
        return "missing_or_unclassified"

    return CHAPTER_MAP.get(
        label,
        label,
    )


def prepare_data(
    data: pd.DataFrame,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, list[str]],
]:
    required = [
        "subject_id",
        "hadm_id",
        "prediction",
        "outcome",
        "observed",
        "principal_icd_chapter",
        "first_service",
        "anchor_year_group",
        "admission_hour_stratum",
        "admission_weekpart",
        "has_icu_stay",
        "admission_type",
        "admission_location",
    ]

    require_columns(
        data,
        required,
    )

    if data[
        "hadm_id"
    ].duplicated().any():
        raise ValueError(
            "Input contains duplicate admissions"
        )

    result = data.copy()

    result[
        "clinical_chapter"
    ] = result[
        "principal_icd_chapter"
    ].map(
        harmonize_chapter
    )

    result[
        "icu_stratum"
    ] = np.where(
        pd.to_numeric(
            result[
                "has_icu_stay"
            ],
            errors="coerce",
        ).eq(
            1
        ),
        "icu",
        "non_icu",
    )

    for specification in DIMENSIONS.values():
        column = specification[
            "column"
        ]

        result[
            column
        ] = (
            result[
                column
            ]
            .astype(
                "string"
            )
            .fillna(
                "missing_or_unknown"
            )
            .str.strip()
            .replace(
                "",
                "missing_or_unknown",
            )
        )

    prediction = result[
        "prediction"
    ].to_numpy(
        dtype=float
    )

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

    result[
        "risk_bin"
    ] = pd.cut(
        prediction,
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(
        int
    )

    levels = {}

    for dimension, specification in DIMENSIONS.items():
        column = specification[
            "column"
        ]

        levels[
            dimension
        ] = sorted(
            result[
                column
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

    return (
        result,
        edges,
        levels,
    )


def compute_dimension(
    data: pd.DataFrame,
    *,
    dimension: str,
    column: str,
    levels: list[str],
    evidence_tier: str,
    minimum_stratum_n: int,
    minimum_observed_n: int,
    minimum_hidden_n: int,
) -> pd.DataFrame:
    work = data[
        [
            column,
            "risk_bin",
            "prediction",
            "outcome",
            "observed",
        ]
    ].copy()

    work[
        "_level"
    ] = work[
        column
    ].astype(
        str
    )

    work[
        "_observed_event"
    ] = (
        pd.to_numeric(
            work[
                "outcome"
            ],
            errors="raise",
        )
        * pd.to_numeric(
            work[
                "observed"
            ],
            errors="raise",
        )
    )

    grouped = (
        work.groupby(
            [
                "_level",
                "risk_bin",
            ],
            sort=True,
            observed=False,
        )
        .agg(
            bin_n=(
                "outcome",
                "size",
            ),
            outcome_sum=(
                "outcome",
                "sum",
            ),
            prediction_sum=(
                "prediction",
                "sum",
            ),
            observed_n=(
                "observed",
                "sum",
            ),
            observed_event_sum=(
                "_observed_event",
                "sum",
            ),
        )
        .reset_index()
    )

    rows = []

    for level in levels:
        bins = grouped.loc[
            grouped[
                "_level"
            ].eq(
                level
            )
        ].copy()

        if bins.empty:
            rows.append(
                {
                    "dimension": dimension,
                    "evidence_tier": evidence_tier,
                    "level": level,
                    "n": 0,
                    "observed_n": 0,
                    "hidden_n": 0,
                    "observed_fraction": np.nan,
                    "reference_event_rate": np.nan,
                    "restricted_event_rate": np.nan,
                    "mean_prediction": np.nan,
                    "reference_ici": np.nan,
                    "restricted_ici": np.nan,
                    "ici_inflation": np.nan,
                    "standardized_signed_gap": np.nan,
                    "maximum_absolute_bin_gap": np.nan,
                    "risk_signed_gap_range": np.nan,
                    "minimum_bin_observed_n": 0,
                    "complete_observed_support": False,
                    "eligible_for_inference": False,
                }
            )

            continue

        n = int(
            bins[
                "bin_n"
            ].sum()
        )

        observed_n = int(
            bins[
                "observed_n"
            ].sum()
        )

        hidden_n = int(
            n
            - observed_n
        )

        weights = (
            bins[
                "bin_n"
            ].to_numpy(
                dtype=float
            )
            / n
        )

        mu_full = (
            bins[
                "outcome_sum"
            ].to_numpy(
                dtype=float
            )
            / bins[
                "bin_n"
            ].to_numpy(
                dtype=float
            )
        )

        mean_prediction = (
            bins[
                "prediction_sum"
            ].to_numpy(
                dtype=float
            )
            / bins[
                "bin_n"
            ].to_numpy(
                dtype=float
            )
        )

        observed_bin_n = bins[
            "observed_n"
        ].to_numpy(
            dtype=float
        )

        complete_support = bool(
            np.all(
                observed_bin_n
                > 0
            )
        )

        minimum_bin_observed_n = int(
            observed_bin_n.min()
        )

        reference_ici = float(
            np.sum(
                weights
                * np.abs(
                    mu_full
                    - mean_prediction
                )
            )
        )

        if complete_support:
            mu_observed = (
                bins[
                    "observed_event_sum"
                ].to_numpy(
                    dtype=float
                )
                / observed_bin_n
            )

            signed_gap_by_bin = (
                mu_observed
                - mu_full
            )

            restricted_ici = float(
                np.sum(
                    weights
                    * np.abs(
                        mu_observed
                        - mean_prediction
                    )
                )
            )

            standardized_signed_gap = float(
                np.sum(
                    weights
                    * signed_gap_by_bin
                )
            )

            maximum_absolute_bin_gap = float(
                np.max(
                    np.abs(
                        signed_gap_by_bin
                    )
                )
            )

            risk_signed_gap_range = float(
                np.max(
                    signed_gap_by_bin
                )
                - np.min(
                    signed_gap_by_bin
                )
            )

            restricted_event_rate = float(
                bins[
                    "observed_event_sum"
                ].sum()
                / observed_n
            )
        else:
            restricted_ici = np.nan
            standardized_signed_gap = np.nan
            maximum_absolute_bin_gap = np.nan
            risk_signed_gap_range = np.nan

            restricted_event_rate = (
                float(
                    bins[
                        "observed_event_sum"
                    ].sum()
                    / observed_n
                )
                if observed_n > 0
                else np.nan
            )

        eligible = bool(
            n
            >= minimum_stratum_n
            and observed_n
            >= minimum_observed_n
            and hidden_n
            >= minimum_hidden_n
            and complete_support
        )

        rows.append(
            {
                "dimension": dimension,
                "evidence_tier": evidence_tier,
                "level": level,
                "n": n,
                "observed_n": observed_n,
                "hidden_n": hidden_n,
                "observed_fraction": float(
                    observed_n
                    / n
                ),
                "reference_event_rate": float(
                    bins[
                        "outcome_sum"
                    ].sum()
                    / n
                ),
                "restricted_event_rate": (
                    restricted_event_rate
                ),
                "mean_prediction": float(
                    bins[
                        "prediction_sum"
                    ].sum()
                    / n
                ),
                "reference_ici": reference_ici,
                "restricted_ici": (
                    restricted_ici
                ),
                "ici_inflation": (
                    float(
                        restricted_ici
                        - reference_ici
                    )
                    if np.isfinite(
                        restricted_ici
                    )
                    else np.nan
                ),
                "standardized_signed_gap": (
                    standardized_signed_gap
                ),
                "maximum_absolute_bin_gap": (
                    maximum_absolute_bin_gap
                ),
                "risk_signed_gap_range": (
                    risk_signed_gap_range
                ),
                "minimum_bin_observed_n": (
                    minimum_bin_observed_n
                ),
                "complete_observed_support": (
                    complete_support
                ),
                "eligible_for_inference": (
                    eligible
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def compute_all_dimensions(
    data: pd.DataFrame,
    *,
    levels: dict[str, list[str]],
    minimum_stratum_n: int,
    minimum_observed_n: int,
    minimum_hidden_n: int,
) -> pd.DataFrame:
    tables = []

    for dimension, specification in DIMENSIONS.items():
        tables.append(
            compute_dimension(
                data,
                dimension=dimension,
                column=specification[
                    "column"
                ],
                levels=levels[
                    dimension
                ],
                evidence_tier=specification[
                    "evidence_tier"
                ],
                minimum_stratum_n=(
                    minimum_stratum_n
                ),
                minimum_observed_n=(
                    minimum_observed_n
                ),
                minimum_hidden_n=(
                    minimum_hidden_n
                ),
            )
        )

    overall_data = data.copy()

    overall_data[
        "_overall"
    ] = "all_admissions"

    overall = compute_dimension(
        overall_data,
        dimension="overall",
        column="_overall",
        levels=[
            "all_admissions",
        ],
        evidence_tier="reference",
        minimum_stratum_n=(
            minimum_stratum_n
        ),
        minimum_observed_n=(
            minimum_observed_n
        ),
        minimum_hidden_n=(
            minimum_hidden_n
        ),
    )

    tables.append(
        overall
    )

    return pd.concat(
        tables,
        ignore_index=True,
    )


def run_bootstrap(
    data: pd.DataFrame,
    *,
    levels: dict[str, list[str]],
    replicates: int,
    seed: int,
    minimum_stratum_n: int,
    minimum_observed_n: int,
    minimum_hidden_n: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    tables = []
    invalid = 0

    for bootstrap_id in range(
        replicates
    ):
        indices = rng.integers(
            0,
            len(
                data
            ),
            size=len(
                data
            ),
        )

        sample = (
            data.iloc[
                indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        try:
            result = compute_all_dimensions(
                sample,
                levels=levels,
                minimum_stratum_n=(
                    minimum_stratum_n
                ),
                minimum_observed_n=(
                    minimum_observed_n
                ),
                minimum_hidden_n=(
                    minimum_hidden_n
                ),
            )
        except Exception:
            invalid += 1
            continue

        result[
            "bootstrap_id"
        ] = bootstrap_id

        tables.append(
            result
        )

        if (
            bootstrap_id
            + 1
        ) % 100 == 0:
            print(
                f"  completed "
                f"{bootstrap_id + 1}/"
                f"{replicates}"
            )

    if not tables:
        raise RuntimeError(
            "No valid bootstrap "
            "replications completed"
        )

    return (
        pd.concat(
            tables,
            ignore_index=True,
        ),
        invalid,
    )


def summarize_intervals(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "observed_fraction",
        "reference_event_rate",
        "restricted_event_rate",
        "reference_ici",
        "restricted_ici",
        "ici_inflation",
        "standardized_signed_gap",
        "maximum_absolute_bin_gap",
        "risk_signed_gap_range",
    ]

    rows = []

    grouped = bootstrap.groupby(
        [
            "dimension",
            "level",
        ],
        sort=True,
    )

    for (
        dimension,
        level,
    ), group in grouped:
        for metric in metrics:
            values = group[
                metric
            ].dropna()

            if values.empty:
                q025 = np.nan
                median = np.nan
                q975 = np.nan
            else:
                q025 = float(
                    values.quantile(
                        0.025
                    )
                )

                median = float(
                    values.quantile(
                        0.50
                    )
                )

                q975 = float(
                    values.quantile(
                        0.975
                    )
                )

            rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "metric": metric,
                    "q025": q025,
                    "median": median,
                    "q975": q975,
                    "n_valid": int(
                        len(
                            values
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def bootstrap_contrast(
    bootstrap: pd.DataFrame,
    *,
    dimension: str,
    first: str,
    second: str,
    metric: str,
) -> pd.Series:
    selected = bootstrap.loc[
        bootstrap[
            "dimension"
        ].eq(
            dimension
        )
        & bootstrap[
            "level"
        ].isin(
            [
                first,
                second,
            ]
        ),
        [
            "bootstrap_id",
            "level",
            metric,
        ],
    ]

    pivot = selected.pivot(
        index="bootstrap_id",
        columns="level",
        values=metric,
    )

    if (
        first not in pivot.columns
        or second not in pivot.columns
    ):
        return pd.Series(
            dtype=float
        )

    return (
        pivot[
            first
        ]
        - pivot[
            second
        ]
    ).dropna()


def build_prespecified_contrasts(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for specification in PRESPECIFIED_CONTRASTS:
        dimension = specification[
            "dimension"
        ]

        first = specification[
            "first"
        ]

        second = specification[
            "second"
        ]

        for metric in [
            "standardized_signed_gap",
            "ici_inflation",
        ]:
            first_rows = point.loc[
                point[
                    "dimension"
                ].eq(
                    dimension
                )
                & point[
                    "level"
                ].eq(
                    first
                )
            ]

            second_rows = point.loc[
                point[
                    "dimension"
                ].eq(
                    dimension
                )
                & point[
                    "level"
                ].eq(
                    second
                )
            ]

            if (
                first_rows.empty
                or second_rows.empty
            ):
                point_difference = np.nan
            else:
                point_difference = float(
                    first_rows[
                        metric
                    ].iloc[
                        0
                    ]
                    - second_rows[
                        metric
                    ].iloc[
                        0
                    ]
                )

            differences = bootstrap_contrast(
                bootstrap,
                dimension=dimension,
                first=first,
                second=second,
                metric=metric,
            )

            rows.append(
                {
                    "contrast": specification[
                        "contrast"
                    ],
                    "dimension": dimension,
                    "first_level": first,
                    "second_level": second,
                    "metric": metric,
                    "point_difference": (
                        point_difference
                    ),
                    "q025": (
                        float(
                            differences.quantile(
                                0.025
                            )
                        )
                        if not differences.empty
                        else np.nan
                    ),
                    "median": (
                        float(
                            differences.quantile(
                                0.50
                            )
                        )
                        if not differences.empty
                        else np.nan
                    ),
                    "q975": (
                        float(
                            differences.quantile(
                                0.975
                            )
                        )
                        if not differences.empty
                        else np.nan
                    ),
                    "n_valid": int(
                        len(
                            differences
                        )
                    ),
                    "interval_excludes_zero": bool(
                        not differences.empty
                        and (
                            differences.quantile(
                                0.025
                            )
                            > 0
                            or differences.quantile(
                                0.975
                            )
                            < 0
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_descriptive_extremes(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dimension in DIMENSIONS:
        eligible = point.loc[
            point[
                "dimension"
            ].eq(
                dimension
            )
            & point[
                "eligible_for_inference"
            ].eq(
                True
            )
            & point[
                "standardized_signed_gap"
            ].notna()
        ]

        if len(
            eligible
        ) < 2:
            continue

        maximum_row = eligible.loc[
            eligible[
                "standardized_signed_gap"
            ].idxmax()
        ]

        minimum_row = eligible.loc[
            eligible[
                "standardized_signed_gap"
            ].idxmin()
        ]

        first = str(
            maximum_row[
                "level"
            ]
        )

        second = str(
            minimum_row[
                "level"
            ]
        )

        differences = bootstrap_contrast(
            bootstrap,
            dimension=dimension,
            first=first,
            second=second,
            metric="standardized_signed_gap",
        )

        rows.append(
            {
                "dimension": dimension,
                "maximum_level": first,
                "minimum_level": second,
                "point_range": float(
                    maximum_row[
                        "standardized_signed_gap"
                    ]
                    - minimum_row[
                        "standardized_signed_gap"
                    ]
                ),
                "q025": (
                    float(
                        differences.quantile(
                            0.025
                        )
                    )
                    if not differences.empty
                    else np.nan
                ),
                "median": (
                    float(
                        differences.quantile(
                            0.50
                        )
                    )
                    if not differences.empty
                    else np.nan
                ),
                "q975": (
                    float(
                        differences.quantile(
                            0.975
                        )
                    )
                    if not differences.empty
                    else np.nan
                ),
                "n_valid": int(
                    len(
                        differences
                    )
                ),
                "selection_note": (
                    "Maximum and minimum levels were "
                    "selected from the full sample; "
                    "this is a descriptive contrast."
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_validations(
    *,
    data: pd.DataFrame,
    input_status: str,
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    expected_rows_per_replication = (
        1
        + sum(
            len(
                value
            )
            for value in levels.values()
        )
    )

    expected_bootstrap_rows = (
        requested_bootstraps
        * expected_rows_per_replication
    )

    legacy_labels = {
        key
        for key, value in CHAPTER_MAP.items()
        if key != value
    }

    remaining_legacy = sorted(
        set(
            data[
                "clinical_chapter"
            ].astype(
                str
            )
        ).intersection(
            legacy_labels
        )
    )

    overall = point.loc[
        point[
            "dimension"
        ].eq(
            "overall"
        )
    ]

    return pd.DataFrame(
        [
            {
                "check": (
                    "Phase 5H-E clinical strata ready"
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
                    "test cohort contains 31012 "
                    "unique admissions"
                ),
                "value": len(
                    data
                ),
                "threshold": EXPECTED_N,
                "comparison": "==",
                "passed": bool(
                    len(
                        data
                    )
                    == EXPECTED_N
                    and not data[
                        "hadm_id"
                    ].duplicated().any()
                ),
            },
            {
                "check": (
                    "all eight heterogeneity "
                    "dimensions evaluated"
                ),
                "value": int(
                    point[
                        "dimension"
                    ].isin(
                        DIMENSIONS
                    ).groupby(
                        point[
                            "dimension"
                        ]
                    ).any().sum()
                ),
                "threshold": 8,
                "comparison": "==",
                "passed": set(
                    DIMENSIONS
                ).issubset(
                    set(
                        point[
                            "dimension"
                        ]
                    )
                ),
            },
            {
                "check": (
                    "ICD9 and ICD10 chapter names "
                    "harmonized"
                ),
                "value": (
                    ",".join(
                        remaining_legacy
                    )
                    or "none"
                ),
                "threshold": "none",
                "comparison": "==",
                "passed": not bool(
                    remaining_legacy
                ),
            },
            {
                "check": (
                    "overall standardized distortion "
                    "is finite and positive"
                ),
                "value": float(
                    overall[
                        "standardized_signed_gap"
                    ].iloc[
                        0
                    ]
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    len(
                        overall
                    )
                    == 1
                    and np.isfinite(
                        overall[
                            "standardized_signed_gap"
                        ].iloc[
                            0
                        ]
                    )
                    and overall[
                        "standardized_signed_gap"
                    ].iloc[
                        0
                    ]
                    > 0
                ),
            },
            {
                "check": (
                    "all requested bootstrap rows "
                    "completed"
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
            {
                "check": (
                    "institution heterogeneity "
                    "not asserted"
                ),
                "value": int(
                    "institution"
                    not in DIMENSIONS
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    "institution"
                    not in DIMENSIONS
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
        data,
        bin_edges,
        levels,
    ) = prepare_data(
        raw,
        args.n_bins,
    )

    point = compute_all_dimensions(
        data,
        levels=levels,
        minimum_stratum_n=(
            args.minimum_stratum_n
        ),
        minimum_observed_n=(
            args.minimum_observed_n
        ),
        minimum_hidden_n=(
            args.minimum_hidden_n
        ),
    )

    print(
        "Phase 5H-F MIMIC heterogeneity benchmark"
    )

    print(
        "\nRunning admission-level bootstrap..."
    )

    (
        bootstrap,
        invalid_bootstraps,
    ) = run_bootstrap(
        data,
        levels=levels,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
        minimum_stratum_n=(
            args.minimum_stratum_n
        ),
        minimum_observed_n=(
            args.minimum_observed_n
        ),
        minimum_hidden_n=(
            args.minimum_hidden_n
        ),
    )

    intervals = summarize_intervals(
        bootstrap
    )

    prespecified = (
        build_prespecified_contrasts(
            point,
            bootstrap,
        )
    )

    descriptive_extremes = (
        build_descriptive_extremes(
            point,
            bootstrap,
        )
    )

    validations = build_validations(
        data=data,
        input_status=input_status,
        point=point,
        bootstrap=bootstrap,
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
        levels=levels,
    )

    status = (
        "phase5h_mimic_heterogeneity_benchmarked"
        if validations[
            "passed"
        ].all()
        else (
            "phase5h_mimic_heterogeneity_"
            "requires_review"
        )
    )

    point.to_csv(
        output_dir
        / "mimic_stratum_point_estimates.csv",
        index=False,
    )

    bootstrap.to_csv(
        output_dir
        / "bootstrap_stratum_estimates.csv.gz",
        index=False,
        compression="gzip",
    )

    intervals.to_csv(
        output_dir
        / "bootstrap_stratum_intervals.csv",
        index=False,
    )

    prespecified.to_csv(
        output_dir
        / "prespecified_contrasts.csv",
        index=False,
    )

    descriptive_extremes.to_csv(
        output_dir
        / "descriptive_extreme_contrasts.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
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
        "phase": "5H-F",
        "status": status,
        "dataset": "MIMIC-IV",
        "n": int(
            len(
                data
            )
        ),
        "risk_bins": (
            args.n_bins
        ),
        "dimensions": DIMENSIONS,
        "eligibility": {
            "minimum_stratum_n": (
                args.minimum_stratum_n
            ),
            "minimum_observed_n": (
                args.minimum_observed_n
            ),
            "minimum_hidden_n": (
                args.minimum_hidden_n
            ),
            "all_occupied_risk_bins_require_observed_outcomes": (
                True
            ),
        },
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstrap_replicates": (
            invalid_bootstraps
        ),
        "interpretation": (
            "Subgroup estimates are descriptive "
            "risk-standardized calibration-audit "
            "contrasts, not causal effects."
        ),
        "institution_heterogeneity": (
            "Unavailable and not asserted."
        ),
    }

    (
        output_dir
        / "mimic_heterogeneity_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5h_mimic_heterogeneity_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nOverall result"
    )

    print(
        point.loc[
            point[
                "dimension"
            ].eq(
                "overall"
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nEligible stratum estimates"
    )

    print(
        point.loc[
            point[
                "eligible_for_inference"
            ].eq(
                True
            )
            & point[
                "dimension"
            ].ne(
                "overall"
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nPrespecified contrasts"
    )

    print(
        prespecified.to_string(
            index=False
        )
    )

    print(
        "\nDescriptive extremes"
    )

    print(
        descriptive_extremes.to_string(
            index=False
        )
    )

    print(
        "\nValidation"
    )

    print(
        validations.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
