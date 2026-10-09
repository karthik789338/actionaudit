from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Reuse the exact Phase 5G implementation.
# ---------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import phase5g_nonlinear_estimators as base
import phase5g_random_forest_estimators as rf


CLIPS = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
)

REFERENCE_CLIP = 0.05

FROZEN_RF_SUMMARY = Path(
    "artifacts/"
    "phase5g_random_forest_estimators/"
    "random_forest_metric_summary.csv"
)

RICH_INPUT = Path(
    "artifacts/"
    "phase5g_baseline_features/"
    "rich_test_analysis.csv.gz"
)

NUISANCE_FILES = {
    "rich_logistic": Path(
        "artifacts/"
        "phase5g_rich_baseline_estimators/"
        "cross_fitted_rich_nuisance_predictions.csv.gz"
    ),
    "hgb": Path(
        "artifacts/"
        "phase5g_nonlinear_estimators/"
        "cross_fitted_hgb_nuisance_predictions.csv.gz"
    ),
    "rf": Path(
        "artifacts/"
        "phase5g_random_forest_estimators/"
        "cross_fitted_rf_nuisance_predictions.csv.gz"
    ),
}


def command_argument(
    name: str,
    default,
):
    if name not in sys.argv:
        return default

    index = sys.argv.index(name)

    if index + 1 >= len(sys.argv):
        raise ValueError(
            f"Missing value after {name}"
        )

    return sys.argv[index + 1]


def clip_label(
    value: float,
) -> str:
    return (
        f"{value:.3f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


def remedy_methods() -> list[str]:
    methods = []

    for family in (
        "rich_logistic",
        "hgb",
        "rf",
    ):
        for estimator in (
            "ipcw",
            "aipw",
        ):
            for clip in CLIPS:
                methods.append(
                    f"{family}_{estimator}_"
                    f"clip_{clip_label(clip)}"
                )

    return methods


def check_method_expansion() -> None:
    expected = set(
        remedy_methods()
    )

    actual = set(
        rf.method_columns()
    )

    missing = sorted(
        expected - actual
    )

    if missing:
        raise RuntimeError(
            "Expanded clipping methods were not "
            f"registered: {missing}"
        )

    print(
        "Expanded remedy methods:",
        len(expected),
        "VERIFIED",
    )


def reproduce_frozen_results(
    new_summary: pd.DataFrame,
) -> pd.DataFrame:
    frozen = pd.read_csv(
        FROZEN_RF_SUMMARY
    )

    metric_columns = [
        "estimated_event_rate",
        "reference_event_rate",
        "event_rate_error",
        "estimated_ici",
        "reference_ici",
        "ici_error",
        "bin_curve_mae",
        "signed_curve_error",
        "maximum_bin_error",
    ]

    rows = []

    indexed_new = (
        new_summary
        .set_index("method")
    )

    for _, old_row in frozen.iterrows():
        method = old_row[
            "method"
        ]

        if method not in indexed_new.index:
            rows.append(
                {
                    "method": method,
                    "present": False,
                    "maximum_absolute_difference": (
                        np.nan
                    ),
                }
            )
            continue

        differences = []

        for metric in metric_columns:
            new_value = float(
                indexed_new.loc[
                    method,
                    metric,
                ]
            )

            old_value = float(
                old_row[
                    metric
                ]
            )

            differences.append(
                abs(
                    new_value
                    - old_value
                )
            )

        rows.append(
            {
                "method": method,
                "present": True,
                "maximum_absolute_difference": (
                    max(differences)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_point_remedy_summary(
    metric_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    indexed = (
        metric_summary
        .set_index("method")
    )

    for family in (
        "rich_logistic",
        "hgb",
        "rf",
    ):
        for estimator in (
            "ipcw",
            "aipw",
        ):
            for clip in CLIPS:
                label = clip_label(
                    clip
                )

                method = (
                    f"{family}_"
                    f"{estimator}_"
                    f"clip_{label}"
                )

                if method not in indexed.index:
                    raise RuntimeError(
                        f"Missing method: {method}"
                    )

                row = indexed.loc[
                    method
                ]

                rows.append(
                    {
                        "family": family,
                        "estimator": estimator,
                        "clip": clip,
                        "method": method,
                        "estimated_event_rate": float(
                            row[
                                "estimated_event_rate"
                            ]
                        ),
                        "reference_event_rate": float(
                            row[
                                "reference_event_rate"
                            ]
                        ),
                        "event_rate_error": float(
                            row[
                                "event_rate_error"
                            ]
                        ),
                        "estimated_ici": float(
                            row[
                                "estimated_ici"
                            ]
                        ),
                        "reference_ici": float(
                            row[
                                "reference_ici"
                            ]
                        ),
                        "ici_error": float(
                            row[
                                "ici_error"
                            ]
                        ),
                        "bin_curve_mae": float(
                            row[
                                "bin_curve_mae"
                            ]
                        ),
                        "signed_curve_error": float(
                            row[
                                "signed_curve_error"
                            ]
                        ),
                        "maximum_bin_error": float(
                            row[
                                "maximum_bin_error"
                            ]
                        ),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    reference = (
        result.loc[
            np.isclose(
                result[
                    "clip"
                ],
                REFERENCE_CLIP,
            ),
            [
                "family",
                "estimator",
                "bin_curve_mae",
            ],
        ]
        .rename(
            columns={
                "bin_curve_mae":
                "clip0p05_bin_curve_mae"
            }
        )
    )

    result = result.merge(
        reference,
        on=[
            "family",
            "estimator",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "improvement_vs_clip0p05"
    ] = (
        result[
            "clip0p05_bin_curve_mae"
        ]
        - result[
            "bin_curve_mae"
        ]
    )

    return result


def detect_probability_column(
    frame: pd.DataFrame,
    family: str,
) -> str:
    """
    Select the exact observation-probability column associated
    with each frozen nuisance family.

    Some later Phase 5G nuisance artifacts carry forward earlier
    nuisance columns, so searching only for the substring
    'observation_probability' is intentionally not sufficient.
    """

    expected = {
        "rich_logistic": (
            "rich_observation_probability"
        ),
        "hgb": (
            "hgb_observation_probability"
        ),
        "rf": (
            "rf_observation_probability"
        ),
    }

    if family not in expected:
        raise KeyError(
            f"Unknown nuisance family: {family}"
        )

    column = expected[
        family
    ]

    if column not in frame.columns:
        candidates = [
            value
            for value in frame.columns
            if (
                "observation_probability"
                in value.lower()
            )
        ]

        raise RuntimeError(
            f"Expected {column!r} for {family}, "
            f"but it was absent. Available "
            f"observation-probability columns: "
            f"{candidates}"
        )

    return column


def build_positivity_diagnostics(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "subject_id",
        "hadm_id",
    ]

    observed = (
        analysis[
            keys
            + [
                "observed"
            ]
        ]
        .copy()
    )

    rows = []

    for family, path in (
        NUISANCE_FILES.items()
    ):
        nuisance = pd.read_csv(
            path
        )

        probability_column = (
            detect_probability_column(
                nuisance,
                family,
            )
        )

        merged = observed.merge(
            nuisance[
                keys
                + [
                    probability_column
                ]
            ],
            on=keys,
            how="left",
            validate="one_to_one",
        )

        if merged[
            probability_column
        ].isna().any():
            raise RuntimeError(
                f"Missing nuisance predictions "
                f"for {family}"
            )

        probability = (
            merged[
                probability_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        observed_mask = (
            merged[
                "observed"
            ]
            .to_numpy(
                dtype=int
            )
            == 1
        )

        for clip in CLIPS:
            clipped = np.clip(
                probability,
                clip,
                1.0 - clip,
            )

            observed_weights = (
                1.0
                / clipped[
                    observed_mask
                ]
            )

            ess = (
                observed_weights.sum()
                ** 2
                / np.square(
                    observed_weights
                ).sum()
            )

            rows.append(
                {
                    "family": family,
                    "clip": clip,
                    "n": len(
                        probability
                    ),
                    "n_observed": int(
                        observed_mask.sum()
                    ),
                    "raw_minimum_probability": (
                        float(
                            probability.min()
                        )
                    ),
                    "raw_q01_probability": float(
                        np.quantile(
                            probability,
                            0.01,
                        )
                    ),
                    "raw_q05_probability": float(
                        np.quantile(
                            probability,
                            0.05,
                        )
                    ),
                    "raw_median_probability": (
                        float(
                            np.median(
                                probability
                            )
                        )
                    ),
                    "fraction_clipped_low": float(
                        np.mean(
                            probability
                            < clip
                        )
                    ),
                    "fraction_clipped_high": float(
                        np.mean(
                            probability
                            > (
                                1.0
                                - clip
                            )
                        )
                    ),
                    "observed_fraction_clipped_low": (
                        float(
                            np.mean(
                                probability[
                                    observed_mask
                                ]
                                < clip
                            )
                        )
                    ),
                    "weighted_observed_ess": (
                        float(
                            ess
                        )
                    ),
                    "maximum_observed_weight": (
                        float(
                            observed_weights.max()
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def paired_bootstrap_intervals(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "bootstrap_id",
        "method",
        "bin_curve_mae",
    }

    missing = (
        required
        - set(
            bootstrap.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Bootstrap file missing columns: "
            f"{sorted(missing)}"
        )

    rows = []

    methods = set(
        bootstrap[
            "method"
        ]
    )

    for method in remedy_methods():
        if method not in methods:
            raise RuntimeError(
                f"Bootstrap missing {method}"
            )

        family = (
            method
            .split(
                "_ipcw_"
            )[0]
            if "_ipcw_" in method
            else method.split(
                "_aipw_"
            )[0]
        )

        estimator = (
            "ipcw"
            if "_ipcw_" in method
            else "aipw"
        )

        reference_method = (
            f"{family}_{estimator}_"
            "clip_0p05"
        )

        comparators = [
            reference_method,
            "restricted_complete_case",
            "grouped_km",
        ]

        left = (
            bootstrap.loc[
                bootstrap[
                    "method"
                ].eq(
                    method
                ),
                [
                    "bootstrap_id",
                    "bin_curve_mae",
                ],
            ]
            .rename(
                columns={
                    "bin_curve_mae":
                    "method_value"
                }
            )
        )

        for comparator in comparators:
            right = (
                bootstrap.loc[
                    bootstrap[
                        "method"
                    ].eq(
                        comparator
                    ),
                    [
                        "bootstrap_id",
                        "bin_curve_mae",
                    ],
                ]
                .rename(
                    columns={
                        "bin_curve_mae":
                        "comparator_value"
                    }
                )
            )

            paired = left.merge(
                right,
                on="bootstrap_id",
                how="inner",
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

            rows.append(
                {
                    "method": method,
                    "family": family,
                    "estimator": estimator,
                    "comparator": comparator,
                    "n_valid": len(
                        paired
                    ),
                    "mean_difference": float(
                        difference.mean()
                    ),
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
                }
            )

    return pd.DataFrame(
        rows
    )


def build_validation(
    *,
    output_dir: Path,
    reproduction: pd.DataFrame,
    points: pd.DataFrame,
    positivity: pd.DataFrame,
    intervals: pd.DataFrame,
    bootstrap_requested: int,
) -> pd.DataFrame:
    checks = []

    maximum_reproduction_error = float(
        reproduction[
            "maximum_absolute_difference"
        ]
        .max()
    )

    checks.append(
        {
            "check": (
                "all frozen Phase 5G "
                "methods reproduced"
            ),
            "value": bool(
                reproduction[
                    "present"
                ].all()
            ),
            "threshold": True,
            "comparison": "==",
            "passed": bool(
                reproduction[
                    "present"
                ].all()
            ),
        }
    )

    checks.append(
        {
            "check": (
                "frozen Phase 5G numerical "
                "reproduction"
            ),
            "value": (
                maximum_reproduction_error
            ),
            "threshold": 1e-10,
            "comparison": "<=",
            "passed": (
                maximum_reproduction_error
                <= 1e-10
            ),
        }
    )

    checks.append(
        {
            "check": (
                "all expanded remedy "
                "point estimates present"
            ),
            "value": len(
                points
            ),
            "threshold": 36,
            "comparison": "==",
            "passed": (
                len(points)
                == 36
            ),
        }
    )

    checks.append(
        {
            "check": (
                "all three nuisance families "
                "have positivity diagnostics"
            ),
            "value": (
                positivity[
                    "family"
                ]
                .nunique()
            ),
            "threshold": 3,
            "comparison": "==",
            "passed": (
                positivity[
                    "family"
                ]
                .nunique()
                == 3
            ),
        }
    )

    checks.append(
        {
            "check": (
                "MIMIC admission count "
                "preserved"
            ),
            "value": int(
                positivity[
                    "n"
                ].min()
            ),
            "threshold": 31012,
            "comparison": "==",
            "passed": bool(
                positivity[
                    "n"
                ].eq(
                    31012
                ).all()
            ),
        }
    )

    checks.append(
        {
            "check": (
                "all remedy bootstrap "
                "comparisons complete"
            ),
            "value": int(
                intervals[
                    "n_valid"
                ].min()
            ),
            "threshold": (
                bootstrap_requested
            ),
            "comparison": "==",
            "passed": bool(
                intervals[
                    "n_valid"
                ].eq(
                    bootstrap_requested
                ).all()
            ),
        }
    )

    finite_points = bool(
        np.isfinite(
            points.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy()
        ).all()
    )

    checks.append(
        {
            "check": (
                "all remedy point outputs finite"
            ),
            "value": finite_points,
            "threshold": True,
            "comparison": "==",
            "passed": finite_points,
        }
    )

    # Diagnostic only:
    # We deliberately do not require any clipping
    # threshold to improve calibration.
    best_aipw = (
        points.loc[
            points[
                "estimator"
            ].eq(
                "aipw"
            )
        ]
        .sort_values(
            [
                "family",
                "bin_curve_mae",
            ]
        )
        .groupby(
            "family",
            as_index=False,
        )
        .first()
    )

    checks.append(
        {
            "check": (
                "best empirical AIPW clips "
                "recorded without success gate"
            ),
            "value": (
                "; ".join(
                    f"{row.family}:{row.clip:g}"
                    for row in (
                        best_aipw
                        .itertuples()
                    )
                )
            ),
            "threshold": (
                "diagnostic_only"
            ),
            "comparison": "report",
            "passed": True,
        }
    )

    return pd.DataFrame(
        checks
    )


def main() -> None:
    # -------------------------------------------------------------
    # Extend only the clipping grid.
    # Everything else remains the frozen Phase 5G implementation.
    # -------------------------------------------------------------

    base.PROPENSITY_CLIPS = CLIPS
    rf.PROPENSITY_CLIPS = CLIPS

    check_method_expansion()

    output_dir = Path(
        command_argument(
            "--output-dir",
            (
                "revision_kdd2027/"
                "phase7fb_mimic_overlap_"
                "remedy_benchmark"
            ),
        )
    )

    bootstrap_requested = int(
        command_argument(
            "--bootstrap-replicates",
            1000,
        )
    )

    print("=" * 88)
    print(
        "ACTIONAUDIT PHASE 7F-B: "
        "REAL MIMIC-IV OVERLAP REMEDY BENCHMARK"
    )
    print("=" * 88)

    print(
        "Expanded clipping grid:",
        CLIPS,
    )

    print(
        "Reference/frozen clip:",
        REFERENCE_CLIP,
    )

    print(
        "Bootstrap replicates:",
        bootstrap_requested,
    )

    # ---------------------------------------------------------
    # Run the exact Phase 5G estimator/bootstrap machinery.
    #
    # The legacy Phase 5G validator assumes the historical
    # clipping grid {0.01, 0.025, 0.05}. Expanding that grid
    # causes its reproduction check to request new methods
    # from the old frozen summary (for example clip_0p005),
    # which correctly do not exist there.
    #
    # We therefore bypass ONLY that legacy validation function.
    # Phase 7F-B performs its own validation immediately after
    # the run, including exact reproduction of every original
    # Phase 5G point estimate to 1e-10.
    # ---------------------------------------------------------

    def expanded_grid_legacy_validation(
        *args,
        **kwargs,
    ):
        return pd.DataFrame(
            [
                {
                    "check": (
                        "legacy Phase 5G validator bypassed "
                        "for expanded clipping grid"
                    ),
                    "value": True,
                    "threshold": True,
                    "comparison": "==",
                    "passed": True,
                }
            ]
        )

    rf.build_validation_checks = (
        expanded_grid_legacy_validation
    )

    # Because its estimator implementation carries forward
    # logistic, HGB, and RF nuisance families, all three are
    # evaluated on the expanded clipping grid.
    metric_path_existing = (
        output_dir
        / "random_forest_metric_summary.csv"
    )
    
    bootstrap_path_existing = (
        output_dir
        / "bootstrap_metric_replicates.csv.gz"
    )
    
    if (
        metric_path_existing.exists()
        and bootstrap_path_existing.exists()
    ):
        print(
            "Reusing existing Phase 7F-B "
            "estimator/bootstrap outputs."
        )
    else:
        rf.main()

    metric_path = (
        output_dir
        / "random_forest_metric_summary.csv"
    )

    bootstrap_path = (
        output_dir
        / "bootstrap_metric_replicates.csv.gz"
    )

    if not metric_path.exists():
        raise FileNotFoundError(
            metric_path
        )

    if not bootstrap_path.exists():
        raise FileNotFoundError(
            bootstrap_path
        )

    metrics = pd.read_csv(
        metric_path
    )

    bootstrap = pd.read_csv(
        bootstrap_path
    )

    analysis = pd.read_csv(
        RICH_INPUT
    )

    reproduction = (
        reproduce_frozen_results(
            metrics
        )
    )

    points = (
        build_point_remedy_summary(
            metrics
        )
    )

    positivity = (
        build_positivity_diagnostics(
            analysis
        )
    )

    intervals = (
        paired_bootstrap_intervals(
            bootstrap
        )
    )

    reproduction.to_csv(
        output_dir
        / "frozen_phase5g_reproduction.csv",
        index=False,
    )

    points.to_csv(
        output_dir
        / "phase7fb_remedy_point_summary.csv",
        index=False,
    )

    positivity.to_csv(
        output_dir
        / "phase7fb_positivity_diagnostics.csv",
        index=False,
    )

    intervals.to_csv(
        output_dir
        / "phase7fb_paired_bootstrap_intervals.csv",
        index=False,
    )

    validation = (
        build_validation(
            output_dir=output_dir,
            reproduction=reproduction,
            points=points,
            positivity=positivity,
            intervals=intervals,
            bootstrap_requested=(
                bootstrap_requested
            ),
        )
    )

    validation.to_csv(
        output_dir
        / "phase7fb_validation_checks.csv",
        index=False,
    )

    passed = bool(
        validation[
            "passed"
        ].all()
    )

    status = (
        "phase7fb_mimic_overlap_remedy_benchmark_passed"
        if passed
        else
        "phase7fb_mimic_overlap_remedy_benchmark_failed"
    )

    (
        output_dir
        / "phase7fb_status.txt"
    ).write_text(
        status + "\n"
    )

    design = f"""Phase 7F-B design

Dataset:
MIMIC-IV frozen test cohort

Admissions:
31012

Purpose:
Extend the empirical Phase 5G overlap-remedy benchmark while
changing only the propensity clipping grid.

Clipping grid:
{list(CLIPS)}

Frozen reference clipping level:
{REFERENCE_CLIP}

Nuisance families:
rich logistic
histogram gradient boosting
random forest

Estimators:
IPCW
AIPW

Calibration bins:
5

Inference:
patient-cluster bootstrap conditional on repeated cross-fitted
nuisance predictions

Bootstrap replicates:
{bootstrap_requested}

Validation:
All original Phase 5G point estimates must reproduce within
1e-10. No validation gate requires any remedy to improve error.

Interpretation boundary:
This empirical analysis uses the Phase 5G binary
observation-probability weighting/augmentation construction.
It must not be described as survival risk-set IPCW.
"""

    (
        output_dir
        / "phase7fb_design.txt"
    ).write_text(
        design
    )

    print("\n" + "=" * 88)
    print("PHASE 7F-B VALIDATION")
    print("=" * 88)

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nAIPW remedy point estimates:"
    )

    print(
        points.loc[
            points[
                "estimator"
            ].eq(
                "aipw"
            ),
            [
                "family",
                "clip",
                "bin_curve_mae",
                "maximum_bin_error",
                "improvement_vs_clip0p05",
            ],
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nPositivity diagnostics:"
    )

    print(
        positivity[
            [
                "family",
                "clip",
                "raw_q05_probability",
                "fraction_clipped_low",
                "weighted_observed_ess",
                "maximum_observed_weight",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:",
        status,
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
