from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7H-A: subgroup/pathway size and "
            "bootstrap reselection robustness."
        )
    )

    parser.add_argument(
        "--predictions",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--phase5d-dir",
        default=(
            "artifacts/"
            "phase5d_bootstrap_tipping_points"
        ),
    )

    parser.add_argument(
        "--phase5h-dir",
        default=(
            "artifacts/"
            "phase5h_mimic_heterogeneity_benchmark"
        ),
    )

    parser.add_argument(
        "--phase5i-freeze-dir",
        default=(
            "artifacts/"
            "phase5i_cross_dataset_h5_"
            "frozen_for_manuscript"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "revision_kdd2027/"
            "phase7ha_subgroup_selection_robustness"
        ),
    )

    return parser.parse_args()


def summarize_interval(values):
    values = pd.Series(
        values,
        dtype=float,
    ).dropna()

    if values.empty:
        return {
            "mean": np.nan,
            "q025": np.nan,
            "median": np.nan,
            "q975": np.nan,
            "n_valid": 0,
        }

    return {
        "mean": float(
            values.mean()
        ),
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
            len(values)
        ),
    }


def build_action_support(
    predictions,
):
    action_column = (
        "analysis_action"
        if "analysis_action"
        in predictions.columns
        else "action"
    )

    grouped = (
        predictions.groupby(
            action_column,
            dropna=False,
            sort=False,
        )
        .agg(
            n=(
                "hadm_id",
                "size",
            ),
            observed_n=(
                "observed",
                "sum",
            ),
            event_n=(
                "outcome",
                "sum",
            ),
            hidden_event_n=(
                "hidden_event",
                "sum",
            ),
            mean_prediction=(
                "prediction",
                "mean",
            ),
        )
        .reset_index()
        .rename(
            columns={
                action_column:
                    "action"
            }
        )
    )

    grouped[
        "hidden_n"
    ] = (
        grouped[
            "n"
        ]
        - grouped[
            "observed_n"
        ]
    )

    grouped[
        "observed_fraction"
    ] = (
        grouped[
            "observed_n"
        ]
        / grouped[
            "n"
        ]
    )

    grouped[
        "hidden_event_rate"
    ] = np.where(
        grouped[
            "hidden_n"
        ]
        > 0,
        grouped[
            "hidden_event_n"
        ]
        / grouped[
            "hidden_n"
        ],
        np.nan,
    )

    return grouped[
        [
            "action",
            "n",
            "observed_n",
            "hidden_n",
            "observed_fraction",
            "event_n",
            "hidden_event_n",
            "hidden_event_rate",
            "mean_prediction",
        ]
    ]


def locate_heterogeneity_table(
    directory: Path,
):
    required = {
        "dimension",
        "level",
        "n",
        "observed_n",
        "hidden_n",
    }

    candidates = []

    files = []

    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue

        name = candidate.name.lower()

        if (
            name.endswith(".csv")
            or name.endswith(".csv.gz")
        ):
            files.append(candidate)

    print(
        "Scanning Phase 5H tabular files:",
        len(files),
    )

    inspected = []

    for path in sorted(files):
        try:
            frame = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception as error:
            inspected.append(
                (
                    str(path),
                    f"READ_ERROR:{error}",
                )
            )
            continue

        columns = set(
            frame.columns
        )

        inspected.append(
            (
                str(path),
                ",".join(
                    frame.columns
                ),
            )
        )

        if required.issubset(
            columns
        ):
            candidates.append(
                path
            )

    if not candidates:
        print(
            "\nPhase 5H files inspected:"
        )

        for filename, columns in inspected:
            print(
                "  ",
                filename,
            )
            print(
                "     columns:",
                columns,
            )

        raise FileNotFoundError(
            "Could not locate Phase 5H "
            "stratum table containing "
            f"{sorted(required)}"
        )

    # Prefer a point/stratum estimate table
    # rather than a large bootstrap file.
    preferred_words = (
        "stratum",
        "estimate",
        "summary",
        "point",
        "heterogeneity",
    )

    def score(candidate):
        name = candidate.name.lower()

        preference = sum(
            word in name
            for word
            in preferred_words
        )

        # Prefer smaller summary tables to
        # large bootstrap replicate tables.
        bootstrap_penalty = (
            1
            if "bootstrap" in name
            else 0
        )

        return (
            preference,
            -bootstrap_penalty,
            -candidate.stat().st_size,
        )

    candidates = sorted(
        candidates,
        key=score,
        reverse=True,
    )

    selected = candidates[0]

    print(
        "Using heterogeneity table:",
        selected,
    )

    return pd.read_csv(
        selected
    )


def build_heterogeneity_support(
    frame,
):
    keep = [
        column
        for column in [
            "dimension",
            "evidence_tier",
            "level",
            "n",
            "observed_n",
            "hidden_n",
            "observed_fraction",
            "reference_event_rate",
            "restricted_event_rate",
            "mean_prediction",
            "reference_ici",
            "restricted_ici",
            "ici_inflation",
            "standardized_signed_gap",
            "eligible_for_inference",
        ]
        if column
        in frame.columns
    ]

    result = frame[
        keep
    ].copy()

    if (
        "dimension"
        in result.columns
    ):
        result = result.loc[
            ~result[
                "dimension"
            ].eq(
                "overall"
            )
        ].copy()

    return result


def build_selection_robustness(
    action_replicates,
    global_replicates,
    point_actions,
):
    point_actions = (
        point_actions.sort_values(
            "distance_oracle_to_tipping"
        )
        .reset_index(
            drop=True
        )
    )

    point_selected_action = str(
        point_actions.loc[
            0,
            "action",
        ]
    )

    point_selected_margin = float(
        point_actions.loc[
            0,
            "distance_oracle_to_tipping",
        ]
    )

    if point_selected_action != "hospice":
        print(
            "NOTE: point-selected action is",
            point_selected_action,
            "rather than hospice."
        )

    required_action = {
        "bootstrap_id",
        "action",
        "distance_oracle_to_tipping",
    }

    missing = (
        required_action
        - set(
            action_replicates.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Action bootstrap file missing: "
            f"{sorted(missing)}"
        )

    required_global = {
        "bootstrap_id",
        "common_oracle_matching_delta",
        "common_signed_distortion_tipping_delta",
    }

    missing = (
        required_global
        - set(
            global_replicates.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Global bootstrap file missing: "
            f"{sorted(missing)}"
        )

    global_work = (
        global_replicates[
            [
                "bootstrap_id",
                "common_oracle_matching_delta",
                "common_signed_distortion_tipping_delta",
            ]
        ]
        .copy()
    )

    global_work[
        "global_margin"
    ] = (
        global_work[
            "common_signed_distortion_tipping_delta"
        ]
        - global_work[
            "common_oracle_matching_delta"
        ]
    )

    fixed = (
        action_replicates.loc[
            action_replicates[
                "action"
            ].eq(
                point_selected_action
            ),
            [
                "bootstrap_id",
                "distance_oracle_to_tipping",
            ],
        ]
        .rename(
            columns={
                "distance_oracle_to_tipping":
                    "fixed_selected_margin"
            }
        )
    )

    finite = (
        action_replicates.loc[
            np.isfinite(
                action_replicates[
                    "distance_oracle_to_tipping"
                ]
            )
        ]
        .copy()
    )

    index = (
        finite.groupby(
            "bootstrap_id"
        )[
            "distance_oracle_to_tipping"
        ]
        .idxmin()
    )

    reselected = (
        finite.loc[
            index,
            [
                "bootstrap_id",
                "action",
                "distance_oracle_to_tipping",
            ],
        ]
        .rename(
            columns={
                "action":
                    "reselected_action",
                "distance_oracle_to_tipping":
                    "reselected_margin",
            }
        )
    )

    result = (
        global_work.merge(
            fixed,
            on="bootstrap_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            reselected,
            on="bootstrap_id",
            how="inner",
            validate="one_to_one",
        )
    )

    result[
        "fixed_selected_action"
    ] = (
        point_selected_action
    )

    result[
        "fixed_advantage"
    ] = (
        result[
            "global_margin"
        ]
        - result[
            "fixed_selected_margin"
        ]
    )

    result[
        "reselected_advantage"
    ] = (
        result[
            "global_margin"
        ]
        - result[
            "reselected_margin"
        ]
    )

    result[
        "reselection_gain"
    ] = (
        result[
            "fixed_selected_margin"
        ]
        - result[
            "reselected_margin"
        ]
    )

    result[
        "same_as_point_selected"
    ] = (
        result[
            "reselected_action"
        ]
        .eq(
            point_selected_action
        )
    )

    return (
        result,
        point_selected_action,
        point_selected_margin,
    )


def build_selection_summary(
    replicate_table,
    point_selected_action,
    point_selected_margin,
):
    fixed = summarize_interval(
        replicate_table[
            "fixed_selected_margin"
        ]
    )

    reselected = summarize_interval(
        replicate_table[
            "reselected_margin"
        ]
    )

    fixed_advantage = (
        summarize_interval(
            replicate_table[
                "fixed_advantage"
            ]
        )
    )

    reselected_advantage = (
        summarize_interval(
            replicate_table[
                "reselected_advantage"
            ]
        )
    )

    reselection_gain = (
        summarize_interval(
            replicate_table[
                "reselection_gain"
            ]
        )
    )

    stable_fraction = float(
        replicate_table[
            "same_as_point_selected"
        ].mean()
    )

    rows = [
        {
            "analysis":
                "point_selected_action",
            "value":
                point_selected_action,
            "estimate":
                point_selected_margin,
            "q025":
                np.nan,
            "median":
                np.nan,
            "q975":
                np.nan,
            "n_valid":
                1,
        },
        {
            "analysis":
                "fixed_point_selected_margin",
            "value":
                point_selected_action,
            **fixed,
        },
        {
            "analysis":
                "reselected_minimum_margin",
            "value":
                "reselected_each_bootstrap",
            **reselected,
        },
        {
            "analysis":
                "fixed_localized_advantage",
            "value":
                point_selected_action,
            **fixed_advantage,
        },
        {
            "analysis":
                "reselected_localized_advantage",
            "value":
                "reselected_each_bootstrap",
            **reselected_advantage,
        },
        {
            "analysis":
                "reselection_gain",
            "value":
                (
                    "fixed_margin_minus_"
                    "reselected_margin"
                ),
            **reselection_gain,
        },
        {
            "analysis":
                "point_selected_action_"
                "selection_frequency",
            "value":
                point_selected_action,
            "estimate":
                stable_fraction,
            "q025":
                np.nan,
            "median":
                np.nan,
            "q975":
                np.nan,
            "n_valid":
                len(
                    replicate_table
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def build_selection_frequencies(
    replicate_table,
):
    counts = (
        replicate_table[
            "reselected_action"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "action"
        )
        .reset_index(
            name="bootstrap_count"
        )
    )

    counts[
        "bootstrap_fraction"
    ] = (
        counts[
            "bootstrap_count"
        ]
        / counts[
            "bootstrap_count"
        ].sum()
    )

    return counts


def main():
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase5d_dir = Path(
        args.phase5d_dir
    )

    phase5h_dir = Path(
        args.phase5h_dir
    )

    phase5i_dir = Path(
        args.phase5i_freeze_dir
    )

    print("=" * 88)
    print(
        "ACTIONAUDIT PHASE 7H-A: "
        "SUBGROUP/PATHWAY SELECTION ROBUSTNESS"
    )
    print("=" * 88)

    predictions = pd.read_csv(
        args.predictions
    )

    action_support = (
        build_action_support(
            predictions
        )
    )

    action_support.to_csv(
        output_dir
        / "mimic_action_sample_sizes.csv",
        index=False,
    )

    heterogeneity_raw = (
        locate_heterogeneity_table(
            phase5h_dir
        )
    )

    heterogeneity_support = (
        build_heterogeneity_support(
            heterogeneity_raw
        )
    )

    heterogeneity_support.to_csv(
        output_dir
        / "mimic_subgroup_sample_sizes.csv",
        index=False,
    )

    action_replicates = pd.read_csv(
        phase5d_dir
        / "bootstrap_action_replicates.csv.gz"
    )

    global_replicates = pd.read_csv(
        phase5d_dir
        / "bootstrap_global_replicates.csv.gz"
    )

    point_actions = pd.read_csv(
        phase5d_dir
        / "exact_action_tipping_points.csv"
    )

    (
        reselection,
        point_selected_action,
        point_selected_margin,
    ) = build_selection_robustness(
        action_replicates,
        global_replicates,
        point_actions,
    )

    reselection.to_csv(
        output_dir
        / "mimic_bootstrap_reselection_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    selection_summary = (
        build_selection_summary(
            reselection,
            point_selected_action,
            point_selected_margin,
        )
    )

    selection_summary.to_csv(
        output_dir
        / "mimic_bootstrap_reselection_summary.csv",
        index=False,
    )

    selection_frequency = (
        build_selection_frequencies(
            reselection
        )
    )

    selection_frequency.to_csv(
        output_dir
        / "mimic_reselected_action_frequency.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Reproduce the existing frozen H5 summary.
    # ---------------------------------------------------------

    frozen_h5 = pd.read_csv(
        phase5i_dir
        / "mimic_localized_sensitivity_summary.csv"
    ).iloc[
        0
    ]

    point_global = pd.read_csv(
        phase5d_dir
        / "exact_global_tipping_points.csv"
    ).iloc[
        0
    ]

    global_margin = float(
        point_global[
            "common_signed_distortion_tipping_delta"
        ]
        - point_global[
            "common_oracle_matching_delta"
        ]
    )

    point_advantage = (
        global_margin
        - point_selected_margin
    )

    reproduction = pd.DataFrame(
        [
            {
                "quantity":
                    "common_global_margin",
                "new_value":
                    global_margin,
                "frozen_value":
                    float(
                        frozen_h5[
                            "common_global_margin"
                        ]
                    ),
            },
            {
                "quantity":
                    "minimum_local_margin",
                "new_value":
                    point_selected_margin,
                "frozen_value":
                    float(
                        frozen_h5[
                            "minimum_local_margin"
                        ]
                    ),
            },
            {
                "quantity":
                    "localized_margin_advantage",
                "new_value":
                    point_advantage,
                "frozen_value":
                    float(
                        frozen_h5[
                            "localized_margin_advantage"
                        ]
                    ),
            },
        ]
    )

    reproduction[
        "absolute_difference"
    ] = (
        reproduction[
            "new_value"
        ]
        - reproduction[
            "frozen_value"
        ]
    ).abs()

    reproduction.to_csv(
        output_dir
        / "frozen_h5_reproduction.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Validation.
    # ---------------------------------------------------------

    checks = []

    def check(
        name,
        value,
        threshold,
        comparison,
        passed,
    ):
        checks.append(
            {
                "check":
                    name,
                "value":
                    value,
                "threshold":
                    threshold,
                "comparison":
                    comparison,
                "passed":
                    bool(
                        passed
                    ),
            }
        )

    check(
        "test admissions preserved",
        len(
            predictions
        ),
        31012,
        "==",
        len(
            predictions
        )
        == 31012,
    )

    check(
        "eight native MIMIC action "
        "pathways preserved",
        len(
            action_support
        ),
        8,
        "==",
        len(
            action_support
        )
        == 8,
    )

    check(
        "all action pathway sample "
        "sizes are positive",
        int(
            (
                action_support[
                    "n"
                ]
                > 0
            ).sum()
        ),
        len(
            action_support
        ),
        "==",
        bool(
            (
                action_support[
                    "n"
                ]
                > 0
            ).all()
        ),
    )

    check(
        "all action pathways report "
        "observed and hidden counts",
        bool(
            action_support[
                [
                    "observed_n",
                    "hidden_n",
                ]
            ]
            .notna()
            .all()
            .all()
        ),
        True,
        "==",
        bool(
            action_support[
                [
                    "observed_n",
                    "hidden_n",
                ]
            ]
            .notna()
            .all()
            .all()
        ),
    )

    check(
        "point-selected closest "
        "MIMIC action reproduced",
        point_selected_action,
        "hospice",
        "==",
        point_selected_action
        == "hospice",
    )

    n_bootstrap = int(
        reselection[
            "bootstrap_id"
        ].nunique()
    )

    check(
        "all 1000 bootstrap replicates "
        "included in reselection audit",
        n_bootstrap,
        1000,
        "==",
        n_bootstrap
        == 1000,
    )

    max_reproduction_error = float(
        reproduction[
            "absolute_difference"
        ].max()
    )

    check(
        "frozen H5 point summary "
        "reproduced",
        max_reproduction_error,
        1e-10,
        "<=",
        max_reproduction_error
        <= 1e-10,
    )

    finite_reselection = bool(
        np.isfinite(
            reselection[
                [
                    "global_margin",
                    "fixed_selected_margin",
                    "reselected_margin",
                    "fixed_advantage",
                    "reselected_advantage",
                ]
            ].to_numpy()
        ).all()
    )

    check(
        "all reselection statistics finite",
        finite_reselection,
        True,
        "==",
        finite_reselection,
    )

    # No success gate requires hospice to be
    # selected in every bootstrap replicate.
    check(
        "bootstrap subgroup identity "
        "is allowed to change",
        True,
        True,
        "==",
        True,
    )

    validation = pd.DataFrame(
        checks
    )

    validation.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    passed = bool(
        validation[
            "passed"
        ].all()
    )

    status = (
        "phase7ha_subgroup_selection_robustness_passed"
        if passed
        else
        "phase7ha_subgroup_selection_robustness_failed"
    )

    (
        output_dir
        / "phase7ha_status.txt"
    ).write_text(
        status
        + "\n"
    )

    print(
        "\nMIMIC action/pathway sample sizes"
    )

    print(
        action_support.to_string(
            index=False
        )
    )

    print(
        "\nBootstrap-reselected closest "
        "action frequencies"
    )

    print(
        selection_frequency.to_string(
            index=False
        )
    )

    print(
        "\nFixed versus reselected "
        "bootstrap summary"
    )

    print(
        selection_summary.to_string(
            index=False
        )
    )

    print(
        "\nValidation"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:",
        status,
    )

    if not passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
