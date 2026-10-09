from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actionaudit import AuditConfig, binned_action_audit
from actionaudit.phase1b_audit_utils import (
    calculate_metrics,
    create_interval_table,
)


CORE_BINS = [3, 4, 5, 10, 15, 20, 30, 40]
MIN_BIN_SIZES = [10, 25, 50, 100]
STRESS_BINS = [60, 80]

FROZEN_N_BINS = 5
FROZEN_MIN_BIN_SIZE = 25

EXPECTED = {
    "SUPPORT2": {
        "n": 1821,
        "horizon": 60,
        "reference_ici": 0.020,
        "restricted_ici": 0.407,
    },
    "MIMIC-IV": {
        "n": 31012,
        "horizon": 30,
        "reference_ici": 0.001,
        "restricted_ici": 0.487,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 7E: ActionAudit audit-design robustness."
    )

    parser.add_argument(
        "--support2",
        default="artifacts/phase1b_logistic/test_predictions.csv",
    )

    parser.add_argument(
        "--mimic",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help=(
            "Maximum bootstrap replicates for frozen-config "
            "bootstrap-count sensitivity. Use 0 for smoke test."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default="revision_kdd2027/phase7e_audit_design_robustness",
    )

    return parser.parse_args()


def standardize_support2(path):
    frame = pd.read_csv(path)

    required = [
        "patient_id",
        "prediction",
        "outcome",
        "observed",
        "action",
        "exit_time",
    ]

    missing = [
        c for c in required
        if c not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"SUPPORT2 missing required columns: {missing}"
        )

    frame = frame[required].copy()

    frame["dataset"] = "SUPPORT2"

    return frame


def standardize_mimic(path):
    frame = pd.read_csv(path)

    required = [
        "subject_id",
        "hadm_id",
        "prediction",
        "outcome",
        "observed",
        "action",
        "exit_time",
    ]

    missing = [
        c for c in required
        if c not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"MIMIC-IV missing required columns: {missing}"
        )

    frame = frame[required].copy()

    frame["patient_id"] = (
        frame["subject_id"].astype(str)
        + "_"
        + frame["hadm_id"].astype(str)
    )

    # ActionAudit only requires action when observed == 0.
    # Keep an explicit label for observed rows so exports are simple.
    frame.loc[
        frame["observed"].astype(int).eq(1)
        & frame["action"].isna(),
        "action",
    ] = "outcome_observed"

    frame = frame[
        [
            "patient_id",
            "prediction",
            "outcome",
            "observed",
            "action",
            "exit_time",
        ]
    ]

    frame["dataset"] = "MIMIC-IV"

    return frame


def bin_occupancy(frame, result):
    """
    Reconstruct occupancy using the exact bin edges returned by
    binned_action_audit.

    This lets us explicitly report:
      - total reference count per bin,
      - number of observed outcomes per bin,
      - bins with zero observed outcomes,
      - bins smaller than the configured threshold.
    """

    work = frame.copy()

    work["_audit_bin"] = pd.cut(
        work["prediction"].astype(float),
        bins=result.bin_edges,
        labels=False,
        include_lowest=True,
    )

    n_actual_bins = len(result.bin_edges) - 1

    rows = []

    for bin_id in range(n_actual_bins):
        group = work.loc[
            work["_audit_bin"].eq(bin_id)
        ]

        n_reference = len(group)

        n_observed = int(
            group["observed"]
            .astype(int)
            .sum()
        )

        rows.append(
            {
                "bin": bin_id,
                "n_reference": n_reference,
                "n_observed": n_observed,
            }
        )

    return pd.DataFrame(rows)


def point_audit(
    dataset,
    frame,
    horizon,
    n_bins,
    min_bin_size,
    design_type,
):
    config = AuditConfig(
        horizon_days=horizon,
        boundary="hospital_discharge",
        n_bins=n_bins,
        min_bin_size=min_bin_size,
    )

    result = binned_action_audit(
        frame,
        config,
    )

    occupancy = bin_occupancy(
        frame,
        result,
    )

    bin_summary = result.bin_summary.copy()

    eligible = (
        bin_summary["eligible"]
        .astype(bool)
    )

    n_actual_bins = len(result.bin_edges) - 1

    n_eligible = int(eligible.sum())
    n_ineligible = int(
        len(bin_summary) - n_eligible
    )

    retained_reference_weight = float(
        bin_summary.loc[
            eligible,
            "reference_weight",
        ].sum()
    )

    empty_observed_bins = int(
        (occupancy["n_observed"] == 0).sum()
    )

    small_reference_bins = int(
        (
            occupancy["n_reference"]
            < min_bin_size
        ).sum()
    )

    min_reference_n = int(
        occupancy["n_reference"].min()
    )

    max_reference_n = int(
        occupancy["n_reference"].max()
    )

    min_observed_n = int(
        occupancy["n_observed"].min()
    )

    max_observed_n = int(
        occupancy["n_observed"].max()
    )

    row = {
        "dataset": dataset,
        "design_type": design_type,
        "requested_n_bins": n_bins,
        "actual_n_bins": n_actual_bins,
        "min_bin_size": min_bin_size,
        "n_eligible_bins": n_eligible,
        "n_ineligible_bins": n_ineligible,
        "empty_observed_bins": empty_observed_bins,
        "small_reference_bins": small_reference_bins,
        "retained_reference_weight": (
            retained_reference_weight
        ),
        "dropped_reference_weight": (
            1.0 - retained_reference_weight
        ),
        "min_reference_n_per_bin": min_reference_n,
        "max_reference_n_per_bin": max_reference_n,
        "min_observed_n_per_bin": min_observed_n,
        "max_observed_n_per_bin": max_observed_n,
    }

    if n_eligible == 0:
        row.update(
            {
                "estimable": False,
                "reference_ici": np.nan,
                "restricted_ici": np.nan,
                "delta_ici": np.nan,
                "reference_calerr": np.nan,
                "restricted_calerr": np.nan,
                "delta_calerr": np.nan,
                "mean_abs_curve_gap": np.nan,
                "max_abs_curve_gap": np.nan,
            }
        )

        return row, occupancy, result

    metrics = calculate_metrics(
        frame,
        result,
    )

    row.update(
        {
            "estimable": True,
            "reference_ici": (
                metrics["reference_ici"]
            ),
            "restricted_ici": (
                metrics["restricted_ici"]
            ),
            "delta_ici": (
                metrics["delta_ici"]
            ),
            "reference_calerr": (
                metrics["reference_calerr"]
            ),
            "restricted_calerr": (
                metrics["restricted_calerr"]
            ),
            "delta_calerr": (
                metrics["delta_calerr"]
            ),
            "mean_abs_curve_gap": (
                metrics["mean_abs_curve_gap"]
            ),
            "max_abs_curve_gap": (
                metrics["max_abs_curve_gap"]
            ),
        }
    )

    return row, occupancy, result


def run_point_grid(dataset, frame, horizon):
    rows = []
    occupancy_rows = []

    print("\n" + "=" * 84)
    print(f"{dataset}: POINT-ESTIMATE DESIGN GRID")
    print("=" * 84)

    # Main factorial grid.
    for min_bin_size in MIN_BIN_SIZES:
        for n_bins in CORE_BINS:
            row, occupancy, _ = point_audit(
                dataset=dataset,
                frame=frame,
                horizon=horizon,
                n_bins=n_bins,
                min_bin_size=min_bin_size,
                design_type="core",
            )

            rows.append(row)

            occupancy = occupancy.copy()
            occupancy.insert(
                0,
                "dataset",
                dataset,
            )
            occupancy.insert(
                1,
                "design_type",
                "core",
            )
            occupancy.insert(
                2,
                "requested_n_bins",
                n_bins,
            )
            occupancy.insert(
                3,
                "min_bin_size",
                min_bin_size,
            )

            occupancy_rows.append(
                occupancy
            )

            print(
                f"{dataset:10s} "
                f"bins={n_bins:2d} "
                f"min={min_bin_size:2d} "
                f"eligible="
                f"{row['n_eligible_bins']:2d}/"
                f"{row['actual_n_bins']:2d} "
                f"retained="
                f"{row['retained_reference_weight']:.3f} "
                f"refICI="
                f"{row['reference_ici']:.6f} "
                f"restrICI="
                f"{row['restricted_ici']:.6f}"
            )

    # Explicit stress tests.
    for n_bins in STRESS_BINS:
        row, occupancy, _ = point_audit(
            dataset=dataset,
            frame=frame,
            horizon=horizon,
            n_bins=n_bins,
            min_bin_size=FROZEN_MIN_BIN_SIZE,
            design_type="stress",
        )

        rows.append(row)

        occupancy = occupancy.copy()
        occupancy.insert(
            0,
            "dataset",
            dataset,
        )
        occupancy.insert(
            1,
            "design_type",
            "stress",
        )
        occupancy.insert(
            2,
            "requested_n_bins",
            n_bins,
        )
        occupancy.insert(
            3,
            "min_bin_size",
            FROZEN_MIN_BIN_SIZE,
        )

        occupancy_rows.append(
            occupancy
        )

        print(
            f"{dataset:10s} "
            f"STRESS bins={n_bins:2d} "
            f"empty observed="
            f"{row['empty_observed_bins']:2d} "
            f"ineligible="
            f"{row['n_ineligible_bins']:2d} "
            f"retained="
            f"{row['retained_reference_weight']:.3f} "
            f"estimable={row['estimable']}"
        )

    point = pd.DataFrame(rows)

    occupancy = pd.concat(
        occupancy_rows,
        ignore_index=True,
    )

    return point, occupancy


def frozen_point_metrics(dataset, point):
    mask = (
        point["dataset"].eq(dataset)
        & point["design_type"].eq("core")
        & point["requested_n_bins"].eq(
            FROZEN_N_BINS
        )
        & point["min_bin_size"].eq(
            FROZEN_MIN_BIN_SIZE
        )
    )

    part = point.loc[mask]

    if len(part) != 1:
        raise RuntimeError(
            f"Could not uniquely locate frozen "
            f"configuration for {dataset}."
        )

    return part.iloc[0]



def run_generic_bootstrap(
    audit_frame,
    config,
    replicates,
    seed,
):
    """
    Patient-level nonparametric bootstrap using the actions
    actually present in each dataset.

    Unlike the historical SUPPORT2 helper, this function does
    not hard-code alive_hospital_discharge, so it is valid for
    both SUPPORT2 and MIMIC-IV.
    """

    rng = np.random.default_rng(seed)
    n = len(audit_frame)

    rows = []
    invalid = 0

    for bootstrap_id in range(replicates):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = (
            audit_frame
            .iloc[indices]
            .copy()
            .reset_index(drop=True)
        )

        # Bootstrap resampling creates repeated original IDs.
        # Give the sampled rows unique audit IDs.
        sample["patient_id"] = np.arange(
            n,
            dtype=int,
        )

        try:
            result = binned_action_audit(
                sample,
                config,
            )

            metrics = calculate_metrics(
                sample,
                result,
            )

        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            invalid += 1
            continue

        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                **metrics,
            }
        )

        if (bootstrap_id + 1) % 100 == 0:
            print(
                f"Completed {bootstrap_id + 1}/"
                f"{replicates} bootstrap replicates"
            )

    scalar_frame = pd.DataFrame(rows)

    print(
        "Valid bootstrap replicates:",
        len(scalar_frame),
        "/",
        replicates,
    )

    print(
        "Invalid bootstrap replicates:",
        invalid,
    )

    if len(scalar_frame) != replicates:
        raise RuntimeError(
            "Not all requested bootstrap replicates were valid: "
            f"{len(scalar_frame)}/{replicates}"
        )

    return scalar_frame


def bootstrap_sensitivity(
    dataset,
    frame,
    horizon,
    replicates,
    seed,
):
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()

    config = AuditConfig(
        horizon_days=horizon,
        boundary="hospital_discharge",
        n_bins=FROZEN_N_BINS,
        min_bin_size=FROZEN_MIN_BIN_SIZE,
    )

    result = binned_action_audit(
        frame,
        config,
    )

    point_metrics = calculate_metrics(
        frame,
        result,
    )

    print("\n" + "=" * 84)
    print(
        f"{dataset}: FROZEN-CONFIG BOOTSTRAP "
        f"({replicates} replicates)"
    )
    print("=" * 84)

    scalars = run_generic_bootstrap(
        audit_frame=frame,
        config=config,
        replicates=replicates,
        seed=seed,
    )

    count_rows = []

    requested_counts = [
        x
        for x in [200, 500, 1000]
        if x <= replicates
    ]

    # Always include the actual maximum if it is not
    # one of the standard checkpoints.
    if (
        replicates not in requested_counts
        and replicates > 0
    ):
        requested_counts.append(replicates)

    requested_counts = sorted(
        set(requested_counts)
    )

    interval_rows = []

    for count in requested_counts:
        prefix = (
            scalars.iloc[:count]
            .copy()
            .reset_index(drop=True)
        )

        intervals = create_interval_table(
            point_metrics,
            prefix,
        )

        keep = intervals.loc[
            intervals["metric"].isin(
                [
                    "reference_ici",
                    "restricted_ici",
                    "delta_ici",
                    "mean_abs_curve_gap",
                ]
            )
        ].copy()

        keep.insert(
            0,
            "dataset",
            dataset,
        )

        keep.insert(
            1,
            "bootstrap_requested",
            count,
        )

        interval_rows.append(keep)

        count_rows.append(
            {
                "dataset": dataset,
                "bootstrap_requested": count,
                "bootstrap_available": len(prefix),
                "restricted_ici_q025": float(
                    prefix[
                        "restricted_ici"
                    ].quantile(0.025)
                ),
                "restricted_ici_median": float(
                    prefix[
                        "restricted_ici"
                    ].quantile(0.50)
                ),
                "restricted_ici_q975": float(
                    prefix[
                        "restricted_ici"
                    ].quantile(0.975)
                ),
                "delta_ici_q025": float(
                    prefix[
                        "delta_ici"
                    ].quantile(0.025)
                ),
                "delta_ici_median": float(
                    prefix[
                        "delta_ici"
                    ].quantile(0.50)
                ),
                "delta_ici_q975": float(
                    prefix[
                        "delta_ici"
                    ].quantile(0.975)
                ),
            }
        )

    sensitivity = pd.DataFrame(
        count_rows
    )

    interval_table = pd.concat(
        interval_rows,
        ignore_index=True,
    )

    return sensitivity, interval_table


def summarize_core_robustness(point):
    rows = []

    for dataset in [
        "SUPPORT2",
        "MIMIC-IV",
    ]:
        part = point.loc[
            point["dataset"].eq(dataset)
            & point["design_type"].eq("core")
            & point["estimable"].eq(True)
        ].copy()

        frozen_min = part.loc[
            part["min_bin_size"].eq(
                FROZEN_MIN_BIN_SIZE
            )
        ]

        rows.append(
            {
                "dataset": dataset,
                "n_estimable_core_designs": len(part),
                "reference_ici_min": float(
                    part["reference_ici"].min()
                ),
                "reference_ici_max": float(
                    part["reference_ici"].max()
                ),
                "restricted_ici_min": float(
                    part["restricted_ici"].min()
                ),
                "restricted_ici_max": float(
                    part["restricted_ici"].max()
                ),
                "delta_ici_min": float(
                    part["delta_ici"].min()
                ),
                "delta_ici_max": float(
                    part["delta_ici"].max()
                ),
                "retained_reference_weight_min": float(
                    part[
                        "retained_reference_weight"
                    ].min()
                ),
                "frozen_min25_restricted_ici_min": float(
                    frozen_min[
                        "restricted_ici"
                    ].min()
                ),
                "frozen_min25_restricted_ici_max": float(
                    frozen_min[
                        "restricted_ici"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_bin_plot(point, dataset, output_path):
    part = point.loc[
        point["dataset"].eq(dataset)
        & point["design_type"].eq("core")
        & point["estimable"].eq(True)
    ].copy()

    fig, ax = plt.subplots(
        figsize=(7.0, 4.8)
    )

    for min_size in MIN_BIN_SIZES:
        p = (
            part.loc[
                part["min_bin_size"].eq(
                    min_size
                )
            ]
            .sort_values(
                "requested_n_bins"
            )
        )

        if p.empty:
            continue

        ax.plot(
            p["requested_n_bins"],
            p["restricted_ici"],
            marker="o",
            label=f"min bin size={min_size}",
        )

    ax.set_xlabel(
        "Number of reference quantile bins"
    )

    ax.set_ylabel(
        "Restricted integrated calibration index"
    )

    ax.set_title(
        f"{dataset}: audit-design sensitivity"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def add_check(
    rows,
    check,
    value,
    expected,
    passed,
):
    rows.append(
        {
            "check": check,
            "value": value,
            "expected": expected,
            "passed": bool(passed),
        }
    )


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "ACTIONAUDIT PHASE 7E: "
        "AUDIT-DESIGN ROBUSTNESS"
    )
    print("=" * 88)

    support2 = standardize_support2(
        args.support2
    )

    mimic = standardize_mimic(
        args.mimic
    )

    datasets = {
        "SUPPORT2": support2,
        "MIMIC-IV": mimic,
    }

    all_points = []
    all_occupancy = []

    for dataset, frame in datasets.items():
        horizon = EXPECTED[
            dataset
        ]["horizon"]

        point, occupancy = run_point_grid(
            dataset,
            frame,
            horizon,
        )

        all_points.append(point)
        all_occupancy.append(
            occupancy
        )

    point = pd.concat(
        all_points,
        ignore_index=True,
    )

    occupancy = pd.concat(
        all_occupancy,
        ignore_index=True,
    )

    core_summary = (
        summarize_core_robustness(
            point
        )
    )

    # ----------------------------------------------------------
    # Bootstrap-count sensitivity
    # ----------------------------------------------------------

    bootstrap_tables = []
    interval_tables = []

    if args.bootstrap > 0:
        for dataset, frame in datasets.items():
            horizon = EXPECTED[
                dataset
            ]["horizon"]

            (
                boot_sensitivity,
                boot_intervals,
            ) = bootstrap_sensitivity(
                dataset=dataset,
                frame=frame,
                horizon=horizon,
                replicates=args.bootstrap,
                seed=args.seed,
            )

            bootstrap_tables.append(
                boot_sensitivity
            )

            interval_tables.append(
                boot_intervals
            )

    bootstrap_count = (
        pd.concat(
            bootstrap_tables,
            ignore_index=True,
        )
        if bootstrap_tables
        else pd.DataFrame()
    )

    bootstrap_intervals = (
        pd.concat(
            interval_tables,
            ignore_index=True,
        )
        if interval_tables
        else pd.DataFrame()
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    checks = []

    for dataset, frame in datasets.items():
        expected = EXPECTED[
            dataset
        ]

        add_check(
            checks,
            f"{dataset}_expected_n",
            len(frame),
            expected["n"],
            len(frame)
            == expected["n"],
        )

        frozen = frozen_point_metrics(
            dataset,
            point,
        )

        add_check(
            checks,
            (
                f"{dataset}_frozen_config_estimable"
            ),
            frozen["estimable"],
            True,
            bool(
                frozen["estimable"]
            ),
        )

        add_check(
            checks,
            (
                f"{dataset}_frozen_reference_ici"
            ),
            frozen["reference_ici"],
            expected["reference_ici"],
            abs(
                frozen["reference_ici"]
                - expected["reference_ici"]
            )
            <= 0.03,
        )

        add_check(
            checks,
            (
                f"{dataset}_frozen_restricted_ici"
            ),
            frozen["restricted_ici"],
            expected["restricted_ici"],
            abs(
                frozen["restricted_ici"]
                - expected["restricted_ici"]
            )
            <= 0.05,
        )

        core_frozen_min = point.loc[
            point["dataset"].eq(dataset)
            & point["design_type"].eq("core")
            & point["min_bin_size"].eq(
                FROZEN_MIN_BIN_SIZE
            )
        ].copy()

        all_core_estimable = bool(
            core_frozen_min[
                "estimable"
            ].all()
        )

        add_check(
            checks,
            (
                f"{dataset}_core_bin_counts_estimable"
            ),
            all_core_estimable,
            True,
            all_core_estimable,
        )

        if all_core_estimable:
            same_direction = bool(
                (
                    core_frozen_min[
                        "restricted_ici"
                    ]
                    >
                    core_frozen_min[
                        "reference_ici"
                    ]
                ).all()
            )

            add_check(
                checks,
                (
                    f"{dataset}_core_bin_counts_"
                    f"preserve_primary_direction"
                ),
                same_direction,
                True,
                same_direction,
            )

        estimable_core = point.loc[
            point["dataset"].eq(dataset)
            & point["design_type"].eq("core")
            & point["estimable"].eq(True)
        ]

        finite = bool(
            np.isfinite(
                estimable_core[
                    [
                        "reference_ici",
                        "restricted_ici",
                        "delta_ici",
                        "retained_reference_weight",
                    ]
                ].to_numpy(
                    dtype=float
                )
            ).all()
        )

        add_check(
            checks,
            (
                f"{dataset}_estimable_core_"
                f"outputs_finite"
            ),
            finite,
            True,
            finite,
        )

    if args.bootstrap > 0:
        for dataset in datasets:
            part = bootstrap_count.loc[
                bootstrap_count[
                    "dataset"
                ].eq(dataset)
            ]

            has_requested = set(
                part[
                    "bootstrap_requested"
                ].astype(int)
            )

            expected_counts = {
                x
                for x in [200, 500, 1000]
                if x <= args.bootstrap
            }

            complete = (
                expected_counts
                .issubset(
                    has_requested
                )
            )

            add_check(
                checks,
                (
                    f"{dataset}_bootstrap_"
                    f"checkpoints_present"
                ),
                sorted(
                    has_requested
                ),
                sorted(
                    expected_counts
                ),
                complete,
            )

    checks_df = pd.DataFrame(
        checks
    )

    passed = bool(
        checks_df["passed"].all()
    )

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    point.to_csv(
        output_dir
        / "point_sensitivity.csv",
        index=False,
    )

    occupancy.to_csv(
        output_dir
        / "bin_occupancy.csv.gz",
        index=False,
        compression="gzip",
    )

    core_summary.to_csv(
        output_dir
        / "core_robustness_summary.csv",
        index=False,
    )

    if not bootstrap_count.empty:
        bootstrap_count.to_csv(
            output_dir
            / "bootstrap_count_sensitivity.csv",
            index=False,
        )

    if not bootstrap_intervals.empty:
        bootstrap_intervals.to_csv(
            output_dir
            / "bootstrap_interval_details.csv",
            index=False,
        )

    checks_df.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    for dataset in datasets:
        safe_name = (
            dataset
            .lower()
            .replace("-", "_")
        )

        make_bin_plot(
            point,
            dataset,
            output_dir
            / (
                f"{safe_name}_"
                f"bin_sensitivity.png"
            ),
        )

    frozen_rows = []

    for dataset in datasets:
        frozen_rows.append(
            frozen_point_metrics(
                dataset,
                point,
            )
        )

    frozen = pd.DataFrame(
        frozen_rows
    )

    frozen.to_csv(
        output_dir
        / "frozen_config_metrics.csv",
        index=False,
    )

    status = (
        "phase7e_audit_design_robustness_passed"
        if passed
        else
        "phase7e_audit_design_robustness_failed"
    )

    (
        output_dir
        / "phase7e_status.txt"
    ).write_text(
        status + "\n"
    )

    notes = """Phase 7E interpretation

This experiment varies design choices in the ActionAudit
calibration machinery while leaving predictions, outcomes,
observation indicators, and evaluation populations unchanged.

Core bin-count grid:
5, 10, 15, 20, 30, 40 reference-cohort quantile bins.

Minimum-bin-size grid:
10, 25, 50.

Manuscript-primary harmonized configuration:
5 quantile bins.

Historical SUPPORT2 Phase 1B used 10 bins. Phase 5H later
introduced five harmonized score bins for the cross-dataset
manuscript analysis. The minimum-bin-size value of 25 is an
ActionAudit eligibility setting, not the historical Phase 1B
bin-count definition.

Stress tests:
60 and 80 quantile bins with minimum bin size 25.

Bins are marked ineligible by the existing ActionAudit
implementation when the reference bin is smaller than the
configured minimum or has no observed outcome. Metrics are
reported only when at least one eligible bin remains. The
retained reference weight and number of ineligible bins are
reported explicitly so that sparse-bin behavior cannot be
silently hidden.

Bootstrap-count sensitivity:
For the frozen configuration, one patient-level bootstrap run
is generated and percentile intervals are recomputed using the
first 200, 500, and 1000 replicates. This tests Monte Carlo
stability without changing the point estimate or selectively
rerunning favorable samples.
"""

    (
        output_dir
        / "phase7e_design.txt"
    ).write_text(notes)

    print("\n" + "=" * 88)
    print("PHASE 7E SUMMARY")
    print("=" * 88)

    print("\nFrozen configuration")
    print(
        frozen[
            [
                "dataset",
                "requested_n_bins",
                "min_bin_size",
                "n_eligible_bins",
                "n_ineligible_bins",
                "retained_reference_weight",
                "reference_ici",
                "restricted_ici",
                "delta_ici",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nCore robustness")
    print(
        core_summary.to_string(
            index=False
        )
    )

    print("\nStress tests")
    print(
        point.loc[
            point["design_type"]
            .eq("stress"),
            [
                "dataset",
                "requested_n_bins",
                "min_bin_size",
                "n_eligible_bins",
                "n_ineligible_bins",
                "empty_observed_bins",
                "small_reference_bins",
                "retained_reference_weight",
                "estimable",
                "restricted_ici",
            ],
        ].to_string(
            index=False
        )
    )

    if not bootstrap_count.empty:
        print(
            "\nBootstrap-count sensitivity"
        )
        print(
            bootstrap_count.to_string(
                index=False
            )
        )

    print("\nValidation")
    print(
        checks_df.to_string(
            index=False
        )
    )

    print("\nSTATUS:", status)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
