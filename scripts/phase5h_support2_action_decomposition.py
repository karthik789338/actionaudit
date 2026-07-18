"""Phase 5H-C: SUPPORT2 action-partition decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VIEWS = {
    "original_boundary_action": "boundary_action",
    "dnr_policy_partition": "dnr_partition_action",
    "disease_partition": "disease_partition_action",
    "hospital_timing_partition": "timing_partition_action",
}

EXPECTED_DEFINITION_STATUS = (
    "phase5h_support2_definitions_frozen"
)

EXPECTED_N = 1821
EPSILON = 1e-12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SUPPORT2 action decomposition "
            "under frozen descriptive partitions."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/"
            "phase5h_support2_definitions/"
            "support2_harmonized_test.csv.gz"
        ),
    )

    parser.add_argument(
        "--definition-manifest",
        default=(
            "artifacts/"
            "phase5h_support2_definitions/"
            "support2_definition_manifest.json"
        ),
    )

    parser.add_argument(
        "--definition-status",
        default=(
            "artifacts/"
            "phase5h_support2_definitions/"
            "phase5h_support2_definition_status.txt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5h_support2_action_decomposition"
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
        "--minimum-action-hidden-n",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
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


def assign_frozen_bins(
    data: pd.DataFrame,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    prediction = data[
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
            "support the requested bins"
        )

    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    result = data.copy()

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

    return (
        result,
        edges,
    )


def action_universes(
    data: pd.DataFrame,
) -> dict[
    str,
    list[str],
]:
    hidden = data[
        "observed"
    ].eq(
        0
    )

    universes: dict[
        str,
        list[str],
    ] = {}

    for view, column in VIEWS.items():
        actions = (
            data.loc[
                hidden,
                column,
            ]
            .astype(
                "string"
            )
            .dropna()
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        universes[
            view
        ] = sorted(
            actions
        )

    return universes


def compute_view(
    data: pd.DataFrame,
    *,
    view: str,
    action_column: str,
    actions: list[str],
    minimum_action_hidden_n: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    point_rows: list[
        dict[str, object]
    ] = []

    bin_rows: list[
        dict[str, object]
    ] = []

    risk_bins = sorted(
        data[
            "risk_bin"
        ].unique()
    )

    for risk_bin in risk_bins:
        group = data.loc[
            data[
                "risk_bin"
            ].eq(
                risk_bin
            )
        ]

        observed_group = group.loc[
            group[
                "observed"
            ].eq(
                1
            )
        ]

        if group.empty:
            raise ValueError(
                f"Empty risk bin {risk_bin}"
            )

        if observed_group.empty:
            raise ValueError(
                "No observed outcomes in "
                f"risk bin {risk_bin}"
            )

        n = len(
            group
        )

        reference_weight = float(
            n
            / len(
                data
            )
        )

        mean_prediction = float(
            group[
                "prediction"
            ].mean()
        )

        mu_full = float(
            group[
                "outcome"
            ].mean()
        )

        mu_observed = float(
            observed_group[
                "outcome"
            ].mean()
        )

        signed_gap = float(
            mu_observed
            - mu_full
        )

        contributions: list[
            float
        ] = []

        for action in actions:
            action_group = group.loc[
                group[
                    "observed"
                ].eq(
                    0
                )
                & group[
                    action_column
                ]
                .astype(
                    "string"
                )
                .eq(
                    action
                )
            ]

            action_n = int(
                len(
                    action_group
                )
            )

            action_prevalence = float(
                action_n
                / n
            )

            mu_action = (
                float(
                    action_group[
                        "outcome"
                    ].mean()
                )
                if action_n > 0
                else np.nan
            )

            contribution = (
                float(
                    action_prevalence
                    * (
                        mu_observed
                        - mu_action
                    )
                )
                if action_n > 0
                else 0.0
            )

            contributions.append(
                contribution
            )

            point_rows.append(
                {
                    "view": view,
                    "risk_bin": int(
                        risk_bin
                    ),
                    "action": action,
                    "bin_n": int(
                        n
                    ),
                    "reference_weight": (
                        reference_weight
                    ),
                    "mean_prediction": (
                        mean_prediction
                    ),
                    "mu_full": mu_full,
                    "mu_observed": (
                        mu_observed
                    ),
                    "signed_gap": (
                        signed_gap
                    ),
                    "action_n": (
                        action_n
                    ),
                    "action_prevalence": (
                        action_prevalence
                    ),
                    "mu_action": (
                        mu_action
                    ),
                    "contribution": (
                        contribution
                    ),
                }
            )

        contribution_sum = float(
            np.sum(
                contributions
            )
        )

        absolute_action_debt = float(
            np.sum(
                np.abs(
                    contributions
                )
            )
        )

        if (
            absolute_action_debt
            > EPSILON
        ):
            aci = (
                1.0
                - abs(
                    contribution_sum
                )
                / (
                    absolute_action_debt
                    + EPSILON
                )
            )
        else:
            aci = 0.0

        bin_rows.append(
            {
                "view": view,
                "risk_bin": int(
                    risk_bin
                ),
                "bin_n": int(
                    n
                ),
                "reference_weight": (
                    reference_weight
                ),
                "mean_prediction": (
                    mean_prediction
                ),
                "mu_full": (
                    mu_full
                ),
                "mu_observed": (
                    mu_observed
                ),
                "signed_gap": (
                    signed_gap
                ),
                "contribution_sum": (
                    contribution_sum
                ),
                "absolute_action_debt": (
                    absolute_action_debt
                ),
                "aci": float(
                    np.clip(
                        aci,
                        0.0,
                        1.0,
                    )
                ),
                "identity_residual": float(
                    signed_gap
                    - contribution_sum
                ),
                "observed_n": int(
                    len(
                        observed_group
                    )
                ),
                "hidden_n": int(
                    n
                    - len(
                        observed_group
                    )
                ),
            }
        )

    point = pd.DataFrame(
        point_rows
    )

    bins = pd.DataFrame(
        bin_rows
    )

    action_rows: list[
        dict[str, object]
    ] = []

    for action in actions:
        group = point.loc[
            point[
                "action"
            ].eq(
                action
            )
        ]

        hidden_n = int(
            group[
                "action_n"
            ].sum()
        )

        sab = float(
            np.sum(
                group[
                    "reference_weight"
                ]
                * group[
                    "contribution"
                ]
            )
        )

        aacd = float(
            np.sum(
                group[
                    "reference_weight"
                ]
                * np.abs(
                    group[
                        "contribution"
                    ]
                )
            )
        )

        action_rows.append(
            {
                "view": view,
                "action": action,
                "hidden_n": (
                    hidden_n
                ),
                "overall_hidden_prevalence": float(
                    hidden_n
                    / len(
                        data
                    )
                ),
                "sab": sab,
                "aacd": aacd,
                "minimum_bin_contribution": float(
                    group[
                        "contribution"
                    ].min()
                ),
                "maximum_bin_contribution": float(
                    group[
                        "contribution"
                    ].max()
                ),
                "eligible_for_inference": bool(
                    hidden_n
                    >= minimum_action_hidden_n
                ),
            }
        )

    action_summary = pd.DataFrame(
        action_rows
    )

    weights = bins[
        "reference_weight"
    ].to_numpy(
        dtype=float
    )

    signed_distortion = float(
        np.sum(
            weights
            * bins[
                "signed_gap"
            ].to_numpy(
                dtype=float
            )
        )
    )

    reference_ici = float(
        np.sum(
            weights
            * np.abs(
                bins[
                    "mu_full"
                ].to_numpy(
                    dtype=float
                )
                - bins[
                    "mean_prediction"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    restricted_ici = float(
        np.sum(
            weights
            * np.abs(
                bins[
                    "mu_observed"
                ].to_numpy(
                    dtype=float
                )
                - bins[
                    "mean_prediction"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    gross_pointwise_debt = float(
        np.sum(
            weights
            * bins[
                "absolute_action_debt"
            ].to_numpy(
                dtype=float
            )
        )
    )

    gross_integrated_debt = float(
        np.sum(
            np.abs(
                action_summary[
                    "sab"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    aci_curve = float(
        np.sum(
            weights
            * bins[
                "aci"
            ].to_numpy(
                dtype=float
            )
        )
    )

    if (
        gross_integrated_debt
        > EPSILON
    ):
        integrated_cancellation_ratio = (
            1.0
            - abs(
                signed_distortion
            )
            / (
                gross_integrated_debt
                + EPSILON
            )
        )
    else:
        integrated_cancellation_ratio = 0.0

    eligible = action_summary.loc[
        action_summary[
            "eligible_for_inference"
        ].eq(
            True
        )
    ]

    action_sab_range_all = (
        float(
            action_summary[
                "sab"
            ].max()
            - action_summary[
                "sab"
            ].min()
        )
        if len(
            action_summary
        ) >= 2
        else 0.0
    )

    action_sab_range_eligible = (
        float(
            eligible[
                "sab"
            ].max()
            - eligible[
                "sab"
            ].min()
        )
        if len(
            eligible
        ) >= 2
        else 0.0
    )

    action_aacd_range_eligible = (
        float(
            eligible[
                "aacd"
            ].max()
            - eligible[
                "aacd"
            ].min()
        )
        if len(
            eligible
        ) >= 2
        else 0.0
    )

    mixed_signed_sab = bool(
        (
            action_summary[
                "sab"
            ]
            > EPSILON
        ).any()
        and (
            action_summary[
                "sab"
            ]
            < -EPSILON
        ).any()
    )

    opposing_pointwise = False

    for _, group in point.groupby(
        "risk_bin",
        sort=True,
    ):
        has_positive = bool(
            (
                group[
                    "contribution"
                ]
                > EPSILON
            ).any()
        )

        has_negative = bool(
            (
                group[
                    "contribution"
                ]
                < -EPSILON
            ).any()
        )

        if (
            has_positive
            and has_negative
        ):
            opposing_pointwise = True
            break

    metrics = {
        "n": int(
            len(
                data
            )
        ),
        "n_actions": int(
            len(
                action_summary
            )
        ),
        "n_eligible_actions": int(
            action_summary[
                "eligible_for_inference"
            ].sum()
        ),
        "reference_event_rate": float(
            data[
                "outcome"
            ].mean()
        ),
        "observed_fraction": float(
            data[
                "observed"
            ].mean()
        ),
        "reference_ici": (
            reference_ici
        ),
        "restricted_ici": (
            restricted_ici
        ),
        "ici_inflation": float(
            restricted_ici
            - reference_ici
        ),
        "signed_distortion": (
            signed_distortion
        ),
        "gross_pointwise_action_debt": (
            gross_pointwise_debt
        ),
        "gross_integrated_action_debt": (
            gross_integrated_debt
        ),
        "aci_curve": (
            aci_curve
        ),
        "integrated_cancellation_ratio": float(
            np.clip(
                integrated_cancellation_ratio,
                0.0,
                1.0,
            )
        ),
        "maximum_absolute_bin_gap": float(
            np.abs(
                bins[
                    "signed_gap"
                ]
            ).max()
        ),
        "risk_signed_gap_range": float(
            bins[
                "signed_gap"
            ].max()
            - bins[
                "signed_gap"
            ].min()
        ),
        "action_sab_range_all": (
            action_sab_range_all
        ),
        "action_sab_range_eligible": (
            action_sab_range_eligible
        ),
        "action_aacd_range_eligible": (
            action_aacd_range_eligible
        ),
        "mixed_signed_sab": int(
            mixed_signed_sab
        ),
        "opposing_pointwise_contributions": int(
            opposing_pointwise
        ),
        "maximum_absolute_identity_residual": float(
            np.abs(
                bins[
                    "identity_residual"
                ]
            ).max()
        ),
    }

    return (
        point,
        action_summary,
        metrics,
    )


def compute_all_views(
    data: pd.DataFrame,
    *,
    universes: dict[
        str,
        list[str],
    ],
    minimum_action_hidden_n: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    point_tables: list[
        pd.DataFrame
    ] = []

    action_tables: list[
        pd.DataFrame
    ] = []

    metric_rows: list[
        dict[str, object]
    ] = []

    for view, action_column in VIEWS.items():
        (
            point,
            action_summary,
            metrics,
        ) = compute_view(
            data,
            view=view,
            action_column=action_column,
            actions=universes[
                view
            ],
            minimum_action_hidden_n=(
                minimum_action_hidden_n
            ),
        )

        point_tables.append(
            point
        )

        action_tables.append(
            action_summary
        )

        metric_rows.append(
            {
                "view": view,
                **metrics,
            }
        )

    return (
        pd.concat(
            point_tables,
            ignore_index=True,
        ),
        pd.concat(
            action_tables,
            ignore_index=True,
        ),
        pd.DataFrame(
            metric_rows
        ),
    )


def run_bootstrap(
    data: pd.DataFrame,
    *,
    universes: dict[
        str,
        list[str],
    ],
    minimum_action_hidden_n: int,
    replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    view_rows: list[
        pd.DataFrame
    ] = []

    action_rows: list[
        pd.DataFrame
    ] = []

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
            (
                _,
                action_summary,
                view_summary,
            ) = compute_all_views(
                sample,
                universes=universes,
                minimum_action_hidden_n=(
                    minimum_action_hidden_n
                ),
            )
        except ValueError:
            invalid += 1
            continue

        view_summary[
            "bootstrap_id"
        ] = bootstrap_id

        action_summary[
            "bootstrap_id"
        ] = bootstrap_id

        view_rows.append(
            view_summary
        )

        action_rows.append(
            action_summary
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

    if (
        not view_rows
        or not action_rows
    ):
        raise RuntimeError(
            "No valid bootstrap "
            "replications were completed"
        )

    return (
        pd.concat(
            view_rows,
            ignore_index=True,
        ),
        pd.concat(
            action_rows,
            ignore_index=True,
        ),
        invalid,
    )


def summarize_view_intervals(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "reference_ici",
        "restricted_ici",
        "ici_inflation",
        "signed_distortion",
        "gross_pointwise_action_debt",
        "gross_integrated_action_debt",
        "aci_curve",
        "integrated_cancellation_ratio",
        "maximum_absolute_bin_gap",
        "risk_signed_gap_range",
        "action_sab_range_all",
        "action_sab_range_eligible",
        "action_aacd_range_eligible",
    ]

    rows: list[
        dict[str, object]
    ] = []

    for view, group in bootstrap.groupby(
        "view",
        sort=True,
    ):
        for metric in metrics:
            values = group[
                metric
            ].dropna()

            rows.append(
                {
                    "view": view,
                    "metric": metric,
                    "q025": float(
                        values.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        values.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        values.quantile(
                            0.975
                        )
                    ),
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


def summarize_action_intervals(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    grouped = bootstrap.groupby(
        [
            "view",
            "action",
        ],
        sort=True,
    )

    for (
        view,
        action,
    ), group in grouped:
        for metric in [
            "sab",
            "aacd",
            "overall_hidden_prevalence",
        ]:
            values = group[
                metric
            ].dropna()

            rows.append(
                {
                    "view": view,
                    "action": action,
                    "metric": metric,
                    "q025": float(
                        values.quantile(
                            0.025
                        )
                    ),
                    "median": float(
                        values.quantile(
                            0.50
                        )
                    ),
                    "q975": float(
                        values.quantile(
                            0.975
                        )
                    ),
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


def build_validations(
    *,
    data: pd.DataFrame,
    definition_status: str,
    manifest: dict,
    actions: pd.DataFrame,
    views: pd.DataFrame,
    bootstrap_views: pd.DataFrame,
    bootstrap_actions: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
) -> pd.DataFrame:
    original_rows = views.loc[
        views[
            "view"
        ].eq(
            "original_boundary_action"
        )
    ]

    if len(
        original_rows
    ) != 1:
        raise ValueError(
            "Expected exactly one original "
            "boundary-action summary row"
        )

    original = original_rows.iloc[
        0
    ]

    expected_gap = float(
        manifest[
            "phase1b_scalar_reproduction"
        ][
            "standardized_signed_gap"
        ]
    )

    hidden = data[
        "observed"
    ].eq(
        0
    )

    partition_complete = all(
        data.loc[
            hidden,
            column,
        ].notna().all()
        for column in VIEWS.values()
    )

    expected_view_bootstrap_rows = (
        requested_bootstraps
        * len(
            VIEWS
        )
    )

    expected_action_bootstrap_rows = (
        requested_bootstraps
        * len(
            actions
        )
    )

    maximum_identity_residual = float(
        views[
            "maximum_absolute_identity_residual"
        ].max()
    )

    all_views_present = (
        set(
            views[
                "view"
            ].astype(
                str
            )
        )
        == set(
            VIEWS
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "SUPPORT2 definitions already frozen"
                ),
                "value": (
                    definition_status
                ),
                "threshold": (
                    EXPECTED_DEFINITION_STATUS
                ),
                "comparison": "==",
                "passed": (
                    definition_status
                    == EXPECTED_DEFINITION_STATUS
                ),
            },
            {
                "check": (
                    "harmonized cohort contains "
                    "1821 unique patients"
                ),
                "value": len(
                    data
                ),
                "threshold": (
                    EXPECTED_N
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        data
                    )
                    == EXPECTED_N
                    and not data[
                        "patient_id"
                    ].duplicated().any()
                ),
            },
            {
                "check": (
                    "all four prespecified "
                    "views evaluated"
                ),
                "value": len(
                    views
                ),
                "threshold": 4,
                "comparison": "==",
                "passed": bool(
                    len(
                        views
                    )
                    == 4
                    and all_views_present
                ),
            },
            {
                "check": (
                    "all hidden patients have "
                    "complete partition actions"
                ),
                "value": int(
                    partition_complete
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": bool(
                    partition_complete
                ),
            },
            {
                "check": (
                    "all decomposition identities exact"
                ),
                "value": (
                    maximum_identity_residual
                ),
                "threshold": 1e-12,
                "comparison": "<",
                "passed": bool(
                    maximum_identity_residual
                    < 1e-12
                ),
            },
            {
                "check": (
                    "harmonized 5-bin signed distortion "
                    "lies within frozen 10-bin interval"
                ),
                "value": float(
                    original[
                        "signed_distortion"
                    ]
                ),
                "threshold": (
                    "[0.33683309597878885,"
                    "0.46825066581273295]"
                ),
                "comparison": "within",
                "passed": bool(
                    0.33683309597878885
                    <= float(
                        original[
                            "signed_distortion"
                        ]
                    )
                    <= 0.46825066581273295
                ),
            },
            {
                "check": (
                    "original view has "
                    "exactly one action"
                ),
                "value": int(
                    original[
                        "n_actions"
                    ]
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    int(
                        original[
                            "n_actions"
                        ]
                    )
                    == 1
                ),
            },
            {
                "check": (
                    "original single-action ACI "
                    "is structurally zero"
                ),
                "value": float(
                    original[
                        "aci_curve"
                    ]
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": bool(
                    float(
                        original[
                            "aci_curve"
                        ]
                    )
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all requested bootstrap "
                    "view rows completed"
                ),
                "value": len(
                    bootstrap_views
                ),
                "threshold": (
                    expected_view_bootstrap_rows
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap_views
                    )
                    == expected_view_bootstrap_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
            {
                "check": (
                    "all requested bootstrap "
                    "action rows completed"
                ),
                "value": len(
                    bootstrap_actions
                ),
                "threshold": (
                    expected_action_bootstrap_rows
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap_actions
                    )
                    == expected_action_bootstrap_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    input_path = Path(
        args.input
    )

    manifest_path = Path(
        args.definition_manifest
    )

    status_path = Path(
        args.definition_status
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    definition_status = (
        status_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    data = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required_columns = [
        "patient_id",
        "prediction",
        "outcome",
        "observed",
        *VIEWS.values(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    (
        data,
        bin_edges,
    ) = assign_frozen_bins(
        data,
        args.n_bins,
    )

    universes = action_universes(
        data
    )

    if any(
        len(
            actions
        )
        == 0
        for actions in universes.values()
    ):
        raise ValueError(
            "At least one decomposition "
            "view has no actions"
        )

    (
        point,
        actions,
        views,
    ) = compute_all_views(
        data,
        universes=universes,
        minimum_action_hidden_n=(
            args.minimum_action_hidden_n
        ),
    )

    print(
        "Phase 5H-C SUPPORT2 "
        "action decomposition"
    )

    print(
        "\nRunning patient bootstrap..."
    )

    (
        bootstrap_views,
        bootstrap_actions,
        invalid_bootstraps,
    ) = run_bootstrap(
        data,
        universes=universes,
        minimum_action_hidden_n=(
            args.minimum_action_hidden_n
        ),
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
    )

    view_intervals = (
        summarize_view_intervals(
            bootstrap_views
        )
    )

    action_intervals = (
        summarize_action_intervals(
            bootstrap_actions
        )
    )

    validations = build_validations(
        data=data,
        definition_status=(
            definition_status
        ),
        manifest=manifest,
        actions=actions,
        views=views,
        bootstrap_views=(
            bootstrap_views
        ),
        bootstrap_actions=(
            bootstrap_actions
        ),
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
    )

    status = (
        "phase5h_support2_action_"
        "decomposition_benchmarked"
        if validations[
            "passed"
        ].all()
        else (
            "phase5h_support2_action_"
            "decomposition_requires_review"
        )
    )

    point.to_csv(
        output_dir
        / "support2_bin_action_contributions.csv",
        index=False,
    )

    actions.to_csv(
        output_dir
        / "support2_action_summary.csv",
        index=False,
    )

    views.to_csv(
        output_dir
        / "support2_view_summary.csv",
        index=False,
    )

    bootstrap_views.to_csv(
        output_dir
        / "bootstrap_view_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_actions.to_csv(
        output_dir
        / "bootstrap_action_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    view_intervals.to_csv(
        output_dir
        / "bootstrap_view_metric_intervals.csv",
        index=False,
    )

    action_intervals.to_csv(
        output_dir
        / "bootstrap_action_metric_intervals.csv",
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
        / "frozen_prediction_bin_edges.csv",
        index=False,
    )

    evidence = {
        "phase": "5H-C",
        "status": status,
        "dataset": "SUPPORT2",
        "horizon_days": 60,
        "harmonized_score_bins": args.n_bins,
        "historical_phase1b_score_bins": 10,
        "binning_note": (
            "Phase 5H uses five harmonized bins. "
            "The historical Phase 1B benchmark used "
            "ten bins, so agreement is assessed using "
            "the frozen ten-bin bootstrap interval "
            "rather than exact point equality."
        ),
        "views": VIEWS,
        "minimum_action_hidden_n": (
            args.minimum_action_hidden_n
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstrap_replicates": (
            invalid_bootstraps
        ),
        "interpretation_boundary": (
            "DNR, disease, and timing actions "
            "are descriptive partitions of alive "
            "hospital discharge, not causal "
            "treatment effects."
        ),
        "institution_heterogeneity": (
            "Unavailable in SUPPORT2."
        ),
        "sha256": {
            str(
                input_path
            ): sha256_file(
                input_path
            ),
            str(
                manifest_path
            ): sha256_file(
                manifest_path
            ),
            str(
                status_path
            ): sha256_file(
                status_path
            ),
        },
    }

    (
        output_dir
        / "support2_action_decomposition_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5h_support2_action_decomposition_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nView summary"
    )

    print(
        views.to_string(
            index=False
        )
    )

    print(
        "\nAction summary"
    )

    print(
        actions.to_string(
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
        "\nInvalid bootstrap replications: "
        f"{invalid_bootstraps}"
    )

    print(
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()
