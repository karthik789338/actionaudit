from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


QUANTILES = [
    0,
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
    1,
]

CLIPS = [
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ActionAudit KDD revision Phase 7A: "
            "direct overlap and positivity diagnostics."
        )
    )

    parser.add_argument(
        "--artifact-root",
        default="artifacts",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "revision_kdd2027/"
            "phase7a_overlap_diagnostics"
        ),
    )

    return parser.parse_args()


def effective_sample_size(
    weights: np.ndarray,
) -> float:
    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights[
        np.isfinite(weights)
        & (weights > 0)
    ]

    if len(weights) == 0:
        return float("nan")

    denominator = np.sum(
        weights ** 2
    )

    if denominator <= 0:
        return float("nan")

    return float(
        weights.sum() ** 2
        / denominator
    )


def standardized_mean_difference_numeric(
    series: pd.Series,
    observed: np.ndarray,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    visible = values[
        observed == 1
    ]

    hidden = values[
        observed == 0
    ]

    visible = visible[
        np.isfinite(
            visible
        )
    ]

    hidden = hidden[
        np.isfinite(
            hidden
        )
    ]

    if (
        len(visible) < 2
        or len(hidden) < 2
    ):
        return float("nan")

    pooled_sd = np.sqrt(
        (
            np.var(
                visible,
                ddof=1,
            )
            + np.var(
                hidden,
                ddof=1,
            )
        )
        / 2.0
    )

    if pooled_sd == 0:
        return 0.0

    return float(
        (
            np.mean(
                visible
            )
            - np.mean(
                hidden
            )
        )
        / pooled_sd
    )


def overlap_coefficient(
    observed_probability: np.ndarray,
    hidden_probability: np.ndarray,
) -> float:
    """
    Histogram overlap coefficient.

    1.0 = identical distributions.
    0.0 = no histogram overlap.

    This is a descriptive diagnostic, not
    an identification theorem.
    """

    edges = np.linspace(
        0.0,
        1.0,
        101,
    )

    observed_hist, _ = np.histogram(
        observed_probability,
        bins=edges,
    )

    hidden_hist, _ = np.histogram(
        hidden_probability,
        bins=edges,
    )

    observed_mass = (
        observed_hist
        / observed_hist.sum()
    )

    hidden_mass = (
        hidden_hist
        / hidden_hist.sum()
    )

    return float(
        np.minimum(
            observed_mass,
            hidden_mass,
        ).sum()
    )


def analyze_dataset(
    *,
    dataset: str,
    observation_model: str,
    data: pd.DataFrame,
    probability_column: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    pd.DataFrame,
]:
    observed = (
        data[
            "observed"
        ]
        .astype(int)
        .to_numpy()
    )

    probability = (
        data[
            probability_column
        ]
        .astype(float)
        .to_numpy()
    )

    prediction = (
        data[
            "prediction"
        ]
        .astype(float)
        .to_numpy()
    )

    if np.any(
        ~np.isfinite(
            probability
        )
    ):
        raise ValueError(
            f"Non-finite observation probabilities "
            f"in {dataset}/{observation_model}"
        )

    if np.any(
        (probability <= 0)
        | (probability >= 1)
    ):
        raise ValueError(
            f"Observation probabilities outside "
            f"(0,1) in {dataset}/{observation_model}"
        )

    visible_probability = (
        probability[
            observed == 1
        ]
    )

    hidden_probability = (
        probability[
            observed == 0
        ]
    )

    summary = {
        "dataset": dataset,
        "observation_model": observation_model,
        "n": int(
            len(
                data
            )
        ),
        "n_observed": int(
            observed.sum()
        ),
        "n_hidden": int(
            (
                observed
                == 0
            ).sum()
        ),
        "observed_fraction": float(
            observed.mean()
        ),
        "p_min": float(
            probability.min()
        ),
        "p01": float(
            np.quantile(
                probability,
                0.01,
            )
        ),
        "p05": float(
            np.quantile(
                probability,
                0.05,
            )
        ),
        "p50": float(
            np.quantile(
                probability,
                0.50,
            )
        ),
        "p95": float(
            np.quantile(
                probability,
                0.95,
            )
        ),
        "p_max": float(
            probability.max()
        ),
        "observed_p50": float(
            np.median(
                visible_probability
            )
        ),
        "hidden_p50": float(
            np.median(
                hidden_probability
            )
        ),
        "propensity_overlap_coefficient": (
            overlap_coefficient(
                visible_probability,
                hidden_probability,
            )
        ),
        "risk_score_smd_observed_vs_hidden": (
            standardized_mean_difference_numeric(
                pd.Series(
                    prediction
                ),
                observed,
            )
        ),
        "observation_auc": float(
            roc_auc_score(
                observed,
                probability,
            )
        ),
        "observation_brier": float(
            brier_score_loss(
                observed,
                probability,
            )
        ),
        "observation_logloss": float(
            log_loss(
                observed,
                np.clip(
                    probability,
                    1e-12,
                    1 - 1e-12,
                ),
            )
        ),
    }

    quantile_rows = []

    groups = [
        (
            "all",
            np.ones(
                len(
                    observed
                ),
                dtype=bool,
            ),
        ),
        (
            "observed",
            observed
            == 1,
        ),
        (
            "hidden",
            observed
            == 0,
        ),
    ]

    for (
        group_name,
        group_mask,
    ) in groups:
        values = probability[
            group_mask
        ]

        for quantile in QUANTILES:
            quantile_rows.append(
                {
                    "dataset": dataset,
                    "observation_model": (
                        observation_model
                    ),
                    "group": (
                        group_name
                    ),
                    "quantile": (
                        quantile
                    ),
                    "observation_probability": float(
                        np.quantile(
                            values,
                            quantile,
                        )
                    ),
                    "n": int(
                        group_mask.sum()
                    ),
                }
            )

    tail_rows = []

    for threshold in [
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
    ]:
        tail_rows.append(
            {
                "dataset": dataset,
                "observation_model": (
                    observation_model
                ),
                "threshold": (
                    threshold
                ),
                "fraction_all_below": float(
                    np.mean(
                        probability
                        < threshold
                    )
                ),
                "fraction_observed_below": float(
                    np.mean(
                        visible_probability
                        < threshold
                    )
                ),
                "fraction_hidden_below": float(
                    np.mean(
                        hidden_probability
                        < threshold
                    )
                ),
                "n_observed_below": int(
                    np.sum(
                        (
                            probability
                            < threshold
                        )
                        & (
                            observed
                            == 1
                        )
                    )
                ),
            }
        )

    ess_rows = []

    for clip in CLIPS:
        denominator = np.clip(
            probability,
            max(
                clip,
                1e-12,
            ),
            1.0,
        )

        weights = (
            observed
            / denominator
        )

        observed_weights = (
            weights[
                observed
                == 1
            ]
        )

        current_ess = (
            effective_sample_size(
                weights
            )
        )

        ess_rows.append(
            {
                "dataset": dataset,
                "observation_model": (
                    observation_model
                ),
                "clip": clip,
                "n_observed": int(
                    observed.sum()
                ),
                "ess": current_ess,
                "ess_over_observed_n": float(
                    current_ess
                    / observed.sum()
                ),
                "ess_over_full_n": float(
                    current_ess
                    / len(
                        observed
                    )
                ),
                "max_weight": float(
                    observed_weights.max()
                ),
                "p99_weight": float(
                    np.quantile(
                        observed_weights,
                        0.99,
                    )
                ),
                "fraction_observed_clipped": (
                    float(
                        np.mean(
                            visible_probability
                            < clip
                        )
                    )
                    if clip > 0
                    else 0.0
                ),
            }
        )

    calibration = pd.DataFrame(
        {
            "probability": (
                probability
            ),
            "observed": (
                observed
            ),
        }
    )

    calibration[
        "decile"
    ] = pd.qcut(
        calibration[
            "probability"
        ],
        10,
        labels=False,
        duplicates="drop",
    )

    calibration = (
        calibration
        .groupby(
            "decile",
            observed=True,
        )
        .agg(
            n=(
                "observed",
                "size",
            ),
            predicted_probability=(
                "probability",
                "mean",
            ),
            empirical_observed_fraction=(
                "observed",
                "mean",
            ),
            p_min=(
                "probability",
                "min",
            ),
            p_max=(
                "probability",
                "max",
            ),
        )
        .reset_index()
    )

    calibration.insert(
        0,
        "observation_model",
        observation_model,
    )

    calibration.insert(
        0,
        "dataset",
        dataset,
    )

    return (
        summary,
        quantile_rows,
        tail_rows,
        ess_rows,
        calibration,
    )


def mimic_covariate_balance(
    artifact_root: Path,
) -> pd.DataFrame:
    data_path = (
        artifact_root
        / "phase5g_baseline_features"
        / "rich_test_analysis.csv.gz"
    )

    manifest_path = (
        artifact_root
        / "phase5g_baseline_features"
        / "baseline_feature_manifest.csv"
    )

    if (
        not data_path.exists()
        or not manifest_path.exists()
    ):
        return pd.DataFrame()

    data = pd.read_csv(
        data_path,
        low_memory=False,
    )

    manifest = pd.read_csv(
        manifest_path
    )

    observed = (
        data[
            "observed"
        ]
        .astype(int)
        .to_numpy()
    )

    rows = []

    for _, row in (
        manifest.iterrows()
    ):
        feature = str(
            row[
                "feature"
            ]
        )

        feature_type = str(
            row[
                "feature_type"
            ]
        )

        if (
            feature
            not in data.columns
        ):
            continue

        if (
            feature_type
            == "numeric"
        ):
            smd = (
                standardized_mean_difference_numeric(
                    data[
                        feature
                    ],
                    observed,
                )
            )

            rows.append(
                {
                    "feature": feature,
                    "feature_type": (
                        feature_type
                    ),
                    "level": "",
                    "smd": smd,
                    "abs_smd": (
                        abs(
                            smd
                        )
                        if np.isfinite(
                            smd
                        )
                        else np.nan
                    ),
                }
            )

        else:
            values = (
                data[
                    feature
                ]
                .astype(
                    "string"
                )
                .fillna(
                    "<MISSING>"
                )
            )

            for level in sorted(
                values.unique().tolist()
            ):
                indicator = (
                    values
                    == level
                ).astype(
                    float
                )

                smd = (
                    standardized_mean_difference_numeric(
                        indicator,
                        observed,
                    )
                )

                rows.append(
                    {
                        "feature": (
                            feature
                        ),
                        "feature_type": (
                            feature_type
                        ),
                        "level": str(
                            level
                        ),
                        "smd": smd,
                        "abs_smd": (
                            abs(
                                smd
                            )
                            if np.isfinite(
                                smd
                            )
                            else np.nan
                        ),
                    }
                )

    output = pd.DataFrame(
        rows
    )

    return (
        output
        .sort_values(
            "abs_smd",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def estimator_error_table(
    artifact_root: Path,
    summary: pd.DataFrame,
    ess_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    support_path = (
        artifact_root
        / "phase2b_dr"
        / "estimator_ranking.csv"
    )

    if support_path.exists():
        table = pd.read_csv(
            support_path
        )

        for _, row in (
            table.iterrows()
        ):
            rows.append(
                {
                    "dataset": (
                        "SUPPORT2"
                    ),
                    "observation_model": (
                        "logistic_baseline"
                    ),
                    "method": str(
                        row[
                            "estimator"
                        ]
                    ),
                    "bin_curve_mae": float(
                        row[
                            "mae_to_reference"
                        ]
                    ),
                }
            )

    mimic_path = (
        artifact_root
        / "phase5g_practitioner_estimators"
        / "estimator_metric_summary.csv"
    )

    if mimic_path.exists():
        table = pd.read_csv(
            mimic_path
        )

        for _, row in (
            table.iterrows()
        ):
            rows.append(
                {
                    "dataset": (
                        "MIMIC-IV"
                    ),
                    "observation_model": (
                        "score_logistic"
                    ),
                    "method": str(
                        row[
                            "method"
                        ]
                    ),
                    "bin_curve_mae": float(
                        row[
                            "bin_curve_mae"
                        ]
                    ),
                }
            )

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    summary_columns = [
        "dataset",
        "observation_model",
        "observed_fraction",
        "p01",
        "p05",
        "p50",
        "propensity_overlap_coefficient",
        "observation_auc",
    ]

    output = output.merge(
        summary[
            summary_columns
        ],
        on=[
            "dataset",
            "observation_model",
        ],
        how="left",
    )

    clip_005 = (
        ess_table.loc[
            ess_table[
                "clip"
            ].eq(
                0.05
            ),
            [
                "dataset",
                "observation_model",
                "ess",
                "ess_over_observed_n",
                "ess_over_full_n",
                "max_weight",
                "fraction_observed_clipped",
            ],
        ]
        .copy()
    )

    clip_005 = (
        clip_005.rename(
            columns={
                column: (
                    "clip0p05_"
                    + column
                )
                for column
                in clip_005.columns
                if column
                not in [
                    "dataset",
                    "observation_model",
                ]
            }
        )
    )

    output = output.merge(
        clip_005,
        on=[
            "dataset",
            "observation_model",
        ],
        how="left",
    )

    return output


def make_plots(
    records: list[
        tuple[
            str,
            str,
            np.ndarray,
            np.ndarray,
        ]
    ],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        len(
            records
        ),
        1,
        figsize=(
            8,
            3.2
            * len(
                records
            ),
        ),
        squeeze=False,
    )

    for (
        axis,
        (
            dataset,
            model,
            probability,
            observed,
        ),
    ) in zip(
        axes[
            :,
            0,
        ],
        records,
    ):
        axis.hist(
            probability[
                observed
                == 1
            ],
            bins=50,
            density=True,
            alpha=0.55,
            label="observed",
        )

        axis.hist(
            probability[
                observed
                == 0
            ],
            bins=50,
            density=True,
            alpha=0.55,
            label="hidden",
        )

        axis.set_title(
            f"{dataset}: {model}"
        )

        axis.set_xlabel(
            "Estimated outcome-observation probability"
        )

        axis.set_ylabel(
            "Density"
        )

        axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "propensity_observed_vs_hidden.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    figure, axes = plt.subplots(
        len(
            records
        ),
        1,
        figsize=(
            8,
            3.2
            * len(
                records
            ),
        ),
        squeeze=False,
    )

    for (
        axis,
        (
            dataset,
            model,
            probability,
            observed,
        ),
    ) in zip(
        axes[
            :,
            0,
        ],
        records,
    ):
        log_probability = np.log10(
            np.clip(
                probability,
                1e-8,
                1.0,
            )
        )

        axis.hist(
            log_probability[
                observed
                == 1
            ],
            bins=50,
            density=True,
            alpha=0.55,
            label="observed",
        )

        axis.hist(
            log_probability[
                observed
                == 0
            ],
            bins=50,
            density=True,
            alpha=0.55,
            label="hidden",
        )

        axis.set_title(
            f"{dataset}: {model}"
        )

        axis.set_xlabel(
            "log10 estimated outcome-observation probability"
        )

        axis.set_ylabel(
            "Density"
        )

        axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "propensity_log10_observed_vs_hidden.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    arguments = parse_args()

    artifact_root = Path(
        arguments.artifact_root
    )

    output_dir = Path(
        arguments.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    specifications = [
        {
            "dataset": (
                "SUPPORT2"
            ),
            "model": (
                "logistic_baseline"
            ),
            "path": (
                artifact_root
                / "phase2b_dr"
                / "test_nuisance_predictions.csv"
            ),
            "probability_column": (
                "propensity"
            ),
            "expected_n": 1821,
        },
        {
            "dataset": (
                "MIMIC-IV"
            ),
            "model": (
                "score_logistic"
            ),
            "path": (
                artifact_root
                / "phase5g_practitioner_estimators"
                / "cross_fitted_nuisance_predictions.csv.gz"
            ),
            "probability_column": (
                "observation_probability"
            ),
            "expected_n": 31012,
        },
    ]

    summary_rows = []
    quantile_rows = []
    tail_rows = []
    ess_rows = []
    decile_tables = []
    plot_records = []
    validation_rows = []

    for specification in specifications:
        input_path = Path(
            specification[
                "path"
            ]
        )

        if (
            not input_path.exists()
        ):
            raise FileNotFoundError(
                "Missing required Phase 7A input: "
                f"{input_path}"
            )

        data = pd.read_csv(
            input_path
        )

        (
            summary,
            quantiles,
            tails,
            ess_values,
            deciles,
        ) = analyze_dataset(
            dataset=str(
                specification[
                    "dataset"
                ]
            ),
            observation_model=str(
                specification[
                    "model"
                ]
            ),
            data=data,
            probability_column=str(
                specification[
                    "probability_column"
                ]
            ),
        )

        summary_rows.append(
            summary
        )

        quantile_rows.extend(
            quantiles
        )

        tail_rows.extend(
            tails
        )

        ess_rows.extend(
            ess_values
        )

        decile_tables.append(
            deciles
        )

        plot_records.append(
            (
                str(
                    specification[
                        "dataset"
                    ]
                ),
                str(
                    specification[
                        "model"
                    ]
                ),
                data[
                    specification[
                        "probability_column"
                    ]
                ].to_numpy(
                    dtype=float
                ),
                data[
                    "observed"
                ].to_numpy(
                    dtype=int
                ),
            )
        )

        actual_n = int(
            len(
                data
            )
        )

        expected_n = int(
            specification[
                "expected_n"
            ]
        )

        validation_rows.append(
            {
                "check": (
                    f"{specification['dataset']}"
                    "_expected_n"
                ),
                "value": (
                    actual_n
                ),
                "expected": (
                    expected_n
                ),
                "passed": (
                    actual_n
                    == expected_n
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    quantiles = pd.DataFrame(
        quantile_rows
    )

    tails = pd.DataFrame(
        tail_rows
    )

    ess_table = pd.DataFrame(
        ess_rows
    )

    deciles = pd.concat(
        decile_tables,
        ignore_index=True,
    )

    covariate_balance = (
        mimic_covariate_balance(
            artifact_root
        )
    )

    estimator_errors = (
        estimator_error_table(
            artifact_root,
            summary,
            ess_table,
        )
    )

    summary.to_csv(
        output_dir
        / "overlap_summary.csv",
        index=False,
    )

    quantiles.to_csv(
        output_dir
        / "observation_probability_quantiles.csv",
        index=False,
    )

    tails.to_csv(
        output_dir
        / "positivity_tail_diagnostics.csv",
        index=False,
    )

    ess_table.to_csv(
        output_dir
        / "ipcw_effective_sample_size_by_clip.csv",
        index=False,
    )

    deciles.to_csv(
        output_dir
        / "observation_probability_deciles.csv",
        index=False,
    )

    covariate_balance.to_csv(
        output_dir
        / "mimic_covariate_balance_observed_vs_hidden.csv",
        index=False,
    )

    estimator_errors.to_csv(
        output_dir
        / "estimator_error_vs_overlap.csv",
        index=False,
    )

    make_plots(
        plot_records,
        output_dir,
    )

    if (
        not covariate_balance.empty
    ):
        validation_rows.extend(
            [
                {
                    "check": (
                        "mimic_covariate_balance_generated"
                    ),
                    "value": int(
                        len(
                            covariate_balance
                        )
                    ),
                    "expected": ">0",
                    "passed": True,
                },
                {
                    "check": (
                        "mimic_levels_abs_smd_gt_0p1"
                    ),
                    "value": int(
                        (
                            covariate_balance[
                                "abs_smd"
                            ]
                            > 0.1
                        ).sum()
                    ),
                    "expected": (
                        "diagnostic"
                    ),
                    "passed": True,
                },
                {
                    "check": (
                        "mimic_max_abs_smd"
                    ),
                    "value": float(
                        covariate_balance[
                            "abs_smd"
                        ].max()
                    ),
                    "expected": (
                        "diagnostic"
                    ),
                    "passed": True,
                },
            ]
        )

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    required = validation.loc[
        validation[
            "expected"
        ].ne(
            "diagnostic"
        )
    ]

    passed = bool(
        required[
            "passed"
        ].all()
    )

    status = (
        "phase7a_overlap_diagnostics_passed"
        if passed
        else (
            "phase7a_overlap_diagnostics_failed"
        )
    )

    (
        output_dir
        / "phase7a_status.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    print(
        "="
        * 100
    )

    print(
        "ACTIONAUDIT KDD REVISION — "
        "PHASE 7A OVERLAP / POSITIVITY DIAGNOSTICS"
    )

    print(
        "="
        * 100
    )

    print(
        "\nOVERLAP SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nIPCW ESS BY CLIP"
    )

    print(
        ess_table.to_string(
            index=False
        )
    )

    if (
        not covariate_balance.empty
    ):
        print(
            "\nTOP 20 MIMIC "
            "COVARIATE IMBALANCES"
        )

        print(
            covariate_balance
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    print(
        "\nVALIDATION"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        f"\nPHASE 7A STATUS: "
        f"{status}"
    )

    print(
        f"Outputs: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
