"""Phase 5I-B: SUPPORT2 partition-specific sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit


EXPECTED_INPUT_STATUS = (
    "phase5h_support2_definitions_frozen"
)

EXPECTED_N = 1821
EPSILON = 1e-10


VIEWS = {
    "original_boundary_action": {
        "column": "boundary_action",
        "interpretation": (
            "Native hospital-boundary action."
        ),
    },
    "dnr_policy_partition": {
        "column": "dnr_partition_action",
        "interpretation": (
            "Descriptive DNR-policy partition "
            "of alive hospital discharge."
        ),
    },
    "disease_partition": {
        "column": "disease_partition_action",
        "interpretation": (
            "Descriptive disease-group partition "
            "of alive hospital discharge."
        ),
    },
    "hospital_timing_partition": {
        "column": "timing_partition_action",
        "interpretation": (
            "Descriptive hospital-timing partition "
            "of alive hospital discharge."
        ),
    },
}


ACTION_METRICS = [
    "oracle_matching_delta",
    "exact_integrated_sab_tipping_delta",
    "sab_at_oracle_matching_delta",
    "distance_oracle_to_tipping",
]


GLOBAL_METRICS = [
    "common_oracle_matching_delta",
    "common_signed_distortion_tipping_delta",
    "signed_distortion_at_common_oracle_delta",
    "empirical_signed_distortion",
    "minimum_action_tipping_delta",
    "maximum_action_tipping_delta",
    "action_tipping_delta_range",
    "minimum_oracle_to_tipping_distance",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 5D-compatible sensitivity "
            "analysis over SUPPORT2 partitions."
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
        "--input-status",
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
            "phase5i_support2_partition_sensitivity"
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
        "--root-lower",
        type=float,
        default=-8.0,
    )

    parser.add_argument(
        "--root-upper",
        type=float,
        default=8.0,
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


def prepare_data(
    data: pd.DataFrame,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, list[str]],
]:
    required = [
        "patient_id",
        "prediction",
        "outcome",
        "observed",
        *[
            specification[
                "column"
            ]
            for specification in VIEWS.values()
        ],
    ]

    require_columns(
        data,
        required,
    )

    if data[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "SUPPORT2 input contains "
            "duplicate patients"
        )

    result = data.copy()

    result[
        "prediction"
    ] = pd.to_numeric(
        result[
            "prediction"
        ],
        errors="raise",
    ).clip(
        EPSILON,
        1.0 - EPSILON,
    )

    result[
        "outcome"
    ] = pd.to_numeric(
        result[
            "outcome"
        ],
        errors="raise",
    ).astype(
        int
    )

    result[
        "observed"
    ] = pd.to_numeric(
        result[
            "observed"
        ],
        errors="raise",
    ).astype(
        int
    )

    result[
        "logit_prediction"
    ] = logit(
        result[
            "prediction"
        ].to_numpy(
            dtype=float
        )
    )

    edges = np.quantile(
        result[
            "prediction"
        ],
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
        result[
            "prediction"
        ],
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(
        int
    )

    hidden = result[
        "observed"
    ].eq(
        0
    )

    universes = {}

    for view, specification in VIEWS.items():
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
            .astype(
                str
            )
        )

        universes[
            view
        ] = sorted(
            result.loc[
                hidden,
                column,
            ]
            .unique()
            .tolist()
        )

    return (
        result,
        edges,
        universes,
    )


def safe_root(
    function,
    *,
    lower: float,
    upper: float,
) -> float:
    lower_value = float(
        function(
            lower
        )
    )

    upper_value = float(
        function(
            upper
        )
    )

    if not (
        np.isfinite(
            lower_value
        )
        and np.isfinite(
            upper_value
        )
    ):
        return np.nan

    if abs(
        lower_value
    ) <= 1e-12:
        return float(
            lower
        )

    if abs(
        upper_value
    ) <= 1e-12:
        return float(
            upper
        )

    if (
        lower_value
        * upper_value
        > 0
    ):
        return np.nan

    return float(
        brentq(
            function,
            lower,
            upper,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
        )
    )


def shifted_mean(
    logits: np.ndarray,
    delta: float,
) -> float:
    return float(
        expit(
            logits
            + delta
        ).mean()
    )


def oracle_matching_delta(
    data: pd.DataFrame,
    *,
    action_column: str | None,
    action: str | None,
    lower: float,
    upper: float,
) -> float:
    hidden = data.loc[
        data[
            "observed"
        ].eq(
            0
        )
    ]

    if action_column is not None:
        hidden = hidden.loc[
            hidden[
                action_column
            ].eq(
                action
            )
        ]

    if hidden.empty:
        return np.nan

    target = float(
        hidden[
            "outcome"
        ].mean()
    )

    logits = hidden[
        "logit_prediction"
    ].to_numpy(
        dtype=float
    )

    return safe_root(
        lambda delta: (
            shifted_mean(
                logits,
                delta,
            )
            - target
        ),
        lower=lower,
        upper=upper,
    )


def bin_observed_mean(
    group: pd.DataFrame,
) -> float:
    observed = group.loc[
        group[
            "observed"
        ].eq(
            1
        )
    ]

    if observed.empty:
        raise ValueError(
            "A risk bin contains no "
            "observed outcomes"
        )

    return float(
        observed[
            "outcome"
        ].mean()
    )


def action_sab(
    data: pd.DataFrame,
    *,
    action_column: str,
    action: str,
    delta: float,
) -> float:
    total_n = len(
        data
    )

    value = 0.0

    for risk_bin in sorted(
        data[
            "risk_bin"
        ].unique()
    ):
        group = data.loc[
            data[
                "risk_bin"
            ].eq(
                risk_bin
            )
        ]

        mu_observed = bin_observed_mean(
            group
        )

        action_rows = group.loc[
            group[
                "observed"
            ].eq(
                0
            )
            & group[
                action_column
            ].eq(
                action
            )
        ]

        if action_rows.empty:
            continue

        mu_action = shifted_mean(
            action_rows[
                "logit_prediction"
            ].to_numpy(
                dtype=float
            ),
            delta,
        )

        value += (
            len(
                action_rows
            )
            / total_n
        ) * (
            mu_observed
            - mu_action
        )

    return float(
        value
    )


def common_signed_distortion(
    data: pd.DataFrame,
    *,
    action_column: str,
    actions: list[str],
    delta: float,
) -> float:
    return float(
        sum(
            action_sab(
                data,
                action_column=action_column,
                action=action,
                delta=delta,
            )
            for action in actions
        )
    )


def empirical_signed_distortion(
    data: pd.DataFrame,
) -> float:
    total_n = len(
        data
    )

    value = 0.0

    for risk_bin in sorted(
        data[
            "risk_bin"
        ].unique()
    ):
        group = data.loc[
            data[
                "risk_bin"
            ].eq(
                risk_bin
            )
        ]

        mu_observed = bin_observed_mean(
            group
        )

        mu_full = float(
            group[
                "outcome"
            ].mean()
        )

        value += (
            len(
                group
            )
            / total_n
        ) * (
            mu_observed
            - mu_full
        )

    return float(
        value
    )


def exact_action_table(
    data: pd.DataFrame,
    *,
    universes: dict[str, list[str]],
    minimum_action_hidden_n: int,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    rows = []

    for view, specification in VIEWS.items():
        column = specification[
            "column"
        ]

        for action in universes[
            view
        ]:
            hidden = data.loc[
                data[
                    "observed"
                ].eq(
                    0
                )
                & data[
                    column
                ].eq(
                    action
                )
            ]

            hidden_n = int(
                len(
                    hidden
                )
            )

            hidden_event_rate = (
                float(
                    hidden[
                        "outcome"
                    ].mean()
                )
                if hidden_n > 0
                else np.nan
            )

            oracle_delta = (
                oracle_matching_delta(
                    data,
                    action_column=column,
                    action=action,
                    lower=lower,
                    upper=upper,
                )
            )

            tipping_delta = safe_root(
                lambda delta: action_sab(
                    data,
                    action_column=column,
                    action=action,
                    delta=delta,
                ),
                lower=lower,
                upper=upper,
            )

            sab_at_oracle = (
                action_sab(
                    data,
                    action_column=column,
                    action=action,
                    delta=oracle_delta,
                )
                if np.isfinite(
                    oracle_delta
                )
                else np.nan
            )

            distance = (
                float(
                    tipping_delta
                    - oracle_delta
                )
                if (
                    np.isfinite(
                        tipping_delta
                    )
                    and np.isfinite(
                        oracle_delta
                    )
                )
                else np.nan
            )

            rows.append(
                {
                    "view": view,
                    "view_interpretation": (
                        specification[
                            "interpretation"
                        ]
                    ),
                    "action": action,
                    "hidden_n": hidden_n,
                    "hidden_event_rate": (
                        hidden_event_rate
                    ),
                    "oracle_matching_delta": (
                        oracle_delta
                    ),
                    "exact_integrated_sab_tipping_delta": (
                        tipping_delta
                    ),
                    "sab_at_oracle_matching_delta": (
                        sab_at_oracle
                    ),
                    "distance_oracle_to_tipping": (
                        distance
                    ),
                    "oracle_is_below_tipping_point": bool(
                        np.isfinite(
                            distance
                        )
                        and distance > 0
                    ),
                    "eligible_for_inference": bool(
                        hidden_n
                        >= minimum_action_hidden_n
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def exact_global_table(
    data: pd.DataFrame,
    *,
    universes: dict[str, list[str]],
    action_table: pd.DataFrame,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    rows = []

    empirical_gap = (
        empirical_signed_distortion(
            data
        )
    )

    common_oracle = (
        oracle_matching_delta(
            data,
            action_column=None,
            action=None,
            lower=lower,
            upper=upper,
        )
    )

    for view, specification in VIEWS.items():
        column = specification[
            "column"
        ]

        actions = universes[
            view
        ]

        common_tipping = safe_root(
            lambda delta: common_signed_distortion(
                data,
                action_column=column,
                actions=actions,
                delta=delta,
            ),
            lower=lower,
            upper=upper,
        )

        signed_at_oracle = (
            common_signed_distortion(
                data,
                action_column=column,
                actions=actions,
                delta=common_oracle,
            )
            if np.isfinite(
                common_oracle
            )
            else np.nan
        )

        eligible = action_table.loc[
            action_table[
                "view"
            ].eq(
                view
            )
            & action_table[
                "eligible_for_inference"
            ].eq(
                True
            )
        ]

        finite_tipping = eligible.loc[
            eligible[
                "exact_integrated_sab_tipping_delta"
            ].notna()
        ]

        finite_distance = eligible.loc[
            eligible[
                "distance_oracle_to_tipping"
            ].notna()
        ]

        if finite_tipping.empty:
            minimum_tipping = np.nan
            maximum_tipping = np.nan
            tipping_range = np.nan
        else:
            minimum_tipping = float(
                finite_tipping[
                    "exact_integrated_sab_tipping_delta"
                ].min()
            )

            maximum_tipping = float(
                finite_tipping[
                    "exact_integrated_sab_tipping_delta"
                ].max()
            )

            tipping_range = float(
                maximum_tipping
                - minimum_tipping
            )

        if finite_distance.empty:
            minimum_distance = np.nan
            closest_action = ""
        else:
            closest_index = finite_distance[
                "distance_oracle_to_tipping"
            ].idxmin()

            closest_action = str(
                finite_distance.loc[
                    closest_index,
                    "action",
                ]
            )

            minimum_distance = float(
                finite_distance.loc[
                    closest_index,
                    "distance_oracle_to_tipping",
                ]
            )

        rows.append(
            {
                "view": view,
                "view_interpretation": (
                    specification[
                        "interpretation"
                    ]
                ),
                "n_actions": int(
                    len(
                        actions
                    )
                ),
                "n_eligible_actions": int(
                    eligible.shape[
                        0
                    ]
                ),
                "common_oracle_matching_delta": (
                    common_oracle
                ),
                "common_signed_distortion_tipping_delta": (
                    common_tipping
                ),
                "signed_distortion_at_common_oracle_delta": (
                    signed_at_oracle
                ),
                "empirical_signed_distortion": (
                    empirical_gap
                ),
                "minimum_action_tipping_delta": (
                    minimum_tipping
                ),
                "maximum_action_tipping_delta": (
                    maximum_tipping
                ),
                "action_tipping_delta_range": (
                    tipping_range
                ),
                "closest_action": (
                    closest_action
                ),
                "minimum_oracle_to_tipping_distance": (
                    minimum_distance
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def calculate_point_results(
    data: pd.DataFrame,
    *,
    universes: dict[str, list[str]],
    minimum_action_hidden_n: int,
    lower: float,
    upper: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    actions = exact_action_table(
        data,
        universes=universes,
        minimum_action_hidden_n=(
            minimum_action_hidden_n
        ),
        lower=lower,
        upper=upper,
    )

    global_table = exact_global_table(
        data,
        universes=universes,
        action_table=actions,
        lower=lower,
        upper=upper,
    )

    return (
        actions,
        global_table,
    )


def run_bootstrap(
    data: pd.DataFrame,
    *,
    universes: dict[str, list[str]],
    replicates: int,
    seed: int,
    minimum_action_hidden_n: int,
    lower: float,
    upper: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    rng = np.random.default_rng(
        seed
    )

    action_tables = []
    global_tables = []
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
                actions,
                global_table,
            ) = calculate_point_results(
                sample,
                universes=universes,
                minimum_action_hidden_n=(
                    minimum_action_hidden_n
                ),
                lower=lower,
                upper=upper,
            )
        except ValueError:
            invalid += 1
            continue

        actions[
            "bootstrap_id"
        ] = bootstrap_id

        global_table[
            "bootstrap_id"
        ] = bootstrap_id

        action_tables.append(
            actions
        )

        global_tables.append(
            global_table
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
        not action_tables
        or not global_tables
    ):
        raise RuntimeError(
            "No valid bootstrap "
            "replications completed"
        )

    return (
        pd.concat(
            action_tables,
            ignore_index=True,
        ),
        pd.concat(
            global_tables,
            ignore_index=True,
        ),
        invalid,
    )


def action_interval_table(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, point_row in point.iterrows():
        view = str(
            point_row[
                "view"
            ]
        )

        action = str(
            point_row[
                "action"
            ]
        )

        selected = bootstrap.loc[
            bootstrap[
                "view"
            ].eq(
                view
            )
            & bootstrap[
                "action"
            ].eq(
                action
            )
        ]

        for metric in ACTION_METRICS:
            values = selected[
                metric
            ].dropna()

            rows.append(
                {
                    "view": view,
                    "action": action,
                    "metric": metric,
                    "estimate": float(
                        point_row[
                            metric
                        ]
                    ),
                    "q025": (
                        float(
                            values.quantile(
                                0.025
                            )
                        )
                        if not values.empty
                        else np.nan
                    ),
                    "median": (
                        float(
                            values.quantile(
                                0.50
                            )
                        )
                        if not values.empty
                        else np.nan
                    ),
                    "q975": (
                        float(
                            values.quantile(
                                0.975
                            )
                        )
                        if not values.empty
                        else np.nan
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


def global_interval_table(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, point_row in point.iterrows():
        view = str(
            point_row[
                "view"
            ]
        )

        selected = bootstrap.loc[
            bootstrap[
                "view"
            ].eq(
                view
            )
        ]

        for metric in GLOBAL_METRICS:
            values = selected[
                metric
            ].dropna()

            rows.append(
                {
                    "view": view,
                    "metric": metric,
                    "estimate": float(
                        point_row[
                            metric
                        ]
                    ),
                    "q025": (
                        float(
                            values.quantile(
                                0.025
                            )
                        )
                        if not values.empty
                        else np.nan
                    ),
                    "median": (
                        float(
                            values.quantile(
                                0.50
                            )
                        )
                        if not values.empty
                        else np.nan
                    ),
                    "q975": (
                        float(
                            values.quantile(
                                0.975
                            )
                        )
                        if not values.empty
                        else np.nan
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
    input_status: str,
    universes: dict[str, list[str]],
    point_actions: pd.DataFrame,
    point_global: pd.DataFrame,
    bootstrap_actions: pd.DataFrame,
    bootstrap_global: pd.DataFrame,
    action_intervals: pd.DataFrame,
    invalid_bootstraps: int,
    requested_bootstraps: int,
) -> pd.DataFrame:
    expected_action_rows = (
        requested_bootstraps
        * sum(
            len(
                actions
            )
            for actions in universes.values()
        )
    )

    expected_global_rows = (
        requested_bootstraps
        * len(
            VIEWS
        )
    )

    eligible_actions = point_actions.loc[
        point_actions[
            "eligible_for_inference"
        ].eq(
            True
        )
    ]

    point_roots_finite = bool(
        eligible_actions[
            [
                "oracle_matching_delta",
                "exact_integrated_sab_tipping_delta",
                "distance_oracle_to_tipping",
            ]
        ].apply(
            np.isfinite
        ).all().all()
    )

    global_roots_finite = bool(
        point_global[
            [
                "common_oracle_matching_delta",
                "common_signed_distortion_tipping_delta",
            ]
        ].apply(
            np.isfinite
        ).all().all()
    )

    identity_error = float(
        np.max(
            np.abs(
                point_global[
                    "signed_distortion_at_common_oracle_delta"
                ]
                - point_global[
                    "empirical_signed_distortion"
                ]
            )
        )
    )

    eligible_keys = eligible_actions[
        [
            "view",
            "action",
        ]
    ].drop_duplicates()

    required_intervals = action_intervals.merge(
        eligible_keys,
        on=[
            "view",
            "action",
        ],
        how="inner",
    )

    required_intervals = required_intervals.loc[
        required_intervals[
            "metric"
        ].isin(
            [
                "oracle_matching_delta",
                "exact_integrated_sab_tipping_delta",
                "distance_oracle_to_tipping",
            ]
        )
    ]

    minimum_valid_interval_n = int(
        required_intervals[
            "n_valid"
        ].min()
    )

    original = point_global.loc[
        point_global[
            "view"
        ].eq(
            "original_boundary_action"
        )
    ].iloc[
        0
    ]

    original_action = point_actions.loc[
        point_actions[
            "view"
        ].eq(
            "original_boundary_action"
        )
    ].iloc[
        0
    ]

    original_tipping_difference = abs(
        float(
            original[
                "common_signed_distortion_tipping_delta"
            ]
        )
        - float(
            original_action[
                "exact_integrated_sab_tipping_delta"
            ]
        )
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "Phase 5H SUPPORT2 definitions "
                    "already frozen"
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
                    "SUPPORT2 contains 1821 "
                    "unique patients"
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
                        "patient_id"
                    ].duplicated().any()
                ),
            },
            {
                "check": (
                    "all four sensitivity views "
                    "evaluated"
                ),
                "value": int(
                    point_global[
                        "view"
                    ].nunique()
                ),
                "threshold": 4,
                "comparison": "==",
                "passed": (
                    point_global[
                        "view"
                    ].nunique()
                    == 4
                ),
            },
            {
                "check": (
                    "all eligible exact action "
                    "roots are finite"
                ),
                "value": int(
                    point_roots_finite
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    point_roots_finite
                ),
            },
            {
                "check": (
                    "all global exact roots "
                    "are finite"
                ),
                "value": int(
                    global_roots_finite
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    global_roots_finite
                ),
            },
            {
                "check": (
                    "common oracle reproduces "
                    "empirical signed distortion"
                ),
                "value": identity_error,
                "threshold": 1e-10,
                "comparison": "<",
                "passed": bool(
                    identity_error
                    < 1e-10
                ),
            },
            {
                "check": (
                    "original single-action and "
                    "global tipping points coincide"
                ),
                "value": (
                    original_tipping_difference
                ),
                "threshold": 1e-10,
                "comparison": "<",
                "passed": bool(
                    original_tipping_difference
                    < 1e-10
                ),
            },
            {
                "check": (
                    "all eligible oracle shifts "
                    "remain below tipping points"
                ),
                "value": int(
                    eligible_actions[
                        "oracle_is_below_tipping_point"
                    ].sum()
                ),
                "threshold": int(
                    len(
                        eligible_actions
                    )
                ),
                "comparison": "==",
                "passed": bool(
                    eligible_actions[
                        "oracle_is_below_tipping_point"
                    ].all()
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
                    expected_action_rows
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap_actions
                    )
                    == expected_action_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
            {
                "check": (
                    "all requested bootstrap "
                    "global rows completed"
                ),
                "value": len(
                    bootstrap_global
                ),
                "threshold": (
                    expected_global_rows
                ),
                "comparison": "==",
                "passed": bool(
                    len(
                        bootstrap_global
                    )
                    == expected_global_rows
                    and invalid_bootstraps
                    == 0
                ),
            },
            {
                "check": (
                    "eligible action intervals "
                    "have at least 900 valid "
                    "replications"
                ),
                "value": (
                    minimum_valid_interval_n
                ),
                "threshold": 900,
                "comparison": ">=",
                "passed": bool(
                    minimum_valid_interval_n
                    >= 900
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
        universes,
    ) = prepare_data(
        raw,
        args.n_bins,
    )

    (
        point_actions,
        point_global,
    ) = calculate_point_results(
        data,
        universes=universes,
        minimum_action_hidden_n=(
            args.minimum_action_hidden_n
        ),
        lower=args.root_lower,
        upper=args.root_upper,
    )

    print(
        "Phase 5I-B SUPPORT2 "
        "partition sensitivity"
    )

    print(
        "\nRunning patient bootstrap..."
    )

    (
        bootstrap_actions,
        bootstrap_global,
        invalid_bootstraps,
    ) = run_bootstrap(
        data,
        universes=universes,
        replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
        minimum_action_hidden_n=(
            args.minimum_action_hidden_n
        ),
        lower=args.root_lower,
        upper=args.root_upper,
    )

    action_intervals = action_interval_table(
        point_actions,
        bootstrap_actions,
    )

    global_intervals = global_interval_table(
        point_global,
        bootstrap_global,
    )

    validations = build_validations(
        data=data,
        input_status=input_status,
        universes=universes,
        point_actions=point_actions,
        point_global=point_global,
        bootstrap_actions=(
            bootstrap_actions
        ),
        bootstrap_global=(
            bootstrap_global
        ),
        action_intervals=(
            action_intervals
        ),
        invalid_bootstraps=(
            invalid_bootstraps
        ),
        requested_bootstraps=(
            args.bootstrap_replicates
        ),
    )

    status = (
        "phase5i_support2_partition_"
        "sensitivity_benchmarked"
        if validations[
            "passed"
        ].all()
        else (
            "phase5i_support2_partition_"
            "sensitivity_requires_review"
        )
    )

    point_actions.to_csv(
        output_dir
        / "exact_action_tipping_points.csv",
        index=False,
    )

    point_global.to_csv(
        output_dir
        / "exact_global_tipping_points.csv",
        index=False,
    )

    bootstrap_actions.to_csv(
        output_dir
        / "bootstrap_action_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_global.to_csv(
        output_dir
        / "bootstrap_global_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    action_intervals.to_csv(
        output_dir
        / "action_bootstrap_intervals.csv",
        index=False,
    )

    global_intervals.to_csv(
        output_dir
        / "global_bootstrap_intervals.csv",
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
        "phase": "5I-B",
        "status": status,
        "dataset": "SUPPORT2",
        "horizon_days": 60,
        "sensitivity_model": (
            "logit hidden outcome probability "
            "equals logit frozen prediction plus "
            "an action- or partition-specific delta."
        ),
        "risk_bins": args.n_bins,
        "root_range": [
            args.root_lower,
            args.root_upper,
        ],
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "invalid_bootstrap_replicates": (
            invalid_bootstraps
        ),
        "minimum_action_hidden_n": (
            args.minimum_action_hidden_n
        ),
        "views": VIEWS,
        "interpretation_boundary": (
            "Only alive hospital discharge is a "
            "native SUPPORT2 boundary action. DNR, "
            "disease, and timing categories are "
            "descriptive discharge-linked strata, "
            "not causal interventions."
        ),
        "h5_decision_rule": (
            "H5 is evaluated after observing whether "
            "eligible partition-specific tipping "
            "thresholds and oracle-to-tipping margins "
            "materially vary around the common/global "
            "threshold. This benchmark does not force "
            "a positive H5 result."
        ),
    }

    (
        output_dir
        / "support2_partition_sensitivity_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5i_support2_partition_sensitivity_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "\nExact action tipping points"
    )

    print(
        point_actions.to_string(
            index=False
        )
    )

    print(
        "\nExact global tipping points"
    )

    print(
        point_global.to_string(
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
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
