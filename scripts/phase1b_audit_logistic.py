"""Phase 1B Part 2: Audit frozen SUPPORT2 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from actionaudit import AuditConfig, binned_action_audit
from actionaudit.phase1b_audit_utils import (
    calculate_metrics,
    create_bin_bands,
    create_interval_table,
    determine_status,
    plot_calibration,
    plot_distortion,
    run_bootstrap,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen SUPPORT2 predictions under reference "
            "and hospital-only observation."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "artifacts/phase1b_logistic/"
            "test_predictions.csv"
        ),
        help="Frozen test-prediction CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase1b_logistic",
        help="Directory for audit artifacts.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Prediction horizon in days.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of score quantile bins.",
    )

    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=25,
        help="Minimum number of patients per bin.",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Number of patient-level bootstrap samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help="Bootstrap random seed.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete frozen-prediction audit."""

    args = parse_arguments()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Frozen predictions not found: {input_path}\n"
            "Run scripts/phase1b_train_logistic.py first."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading frozen predictions from {input_path}"
    )

    audit_frame = pd.read_csv(input_path)

    required_columns = {
        "patient_id",
        "prediction",
        "outcome",
        "observed",
        "action",
        "exit_time",
    }

    missing = required_columns.difference(
        audit_frame.columns
    )

    if missing:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing)}"
        )

    config = AuditConfig(
        horizon_days=args.horizon,
        boundary="hospital",
        n_bins=args.n_bins,
        min_bin_size=args.min_bin_size,
    )

    print("Calculating point-estimate audit...")

    audit_result = binned_action_audit(
        audit_frame,
        config,
        action_order=[
            "alive_hospital_discharge",
        ],
    )

    if (
        audit_result.max_decomposition_residual
        >= 1e-10
    ):
        raise RuntimeError(
            "Action decomposition failed numerical verification."
        )

    point_metrics = calculate_metrics(
        audit_frame,
        audit_result,
    )

    print(
        f"Running {args.bootstrap_replicates} "
        "bootstrap replicates..."
    )

    bootstrap_scalars, bootstrap_bins = run_bootstrap(
        audit_frame,
        config,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    interval_table = create_interval_table(
        point_metrics,
        bootstrap_scalars,
    )

    bin_bands = create_bin_bands(
        bootstrap_bins
    )

    phase_status = determine_status(
        interval_table
    )

    scalar_summary = pd.DataFrame(
        {
            "metric": list(point_metrics.keys()),
            "value": list(point_metrics.values()),
        }
    )

    audit_result.bin_summary.to_csv(
        output_dir / "calibration_bin_summary.csv",
        index=False,
    )

    audit_result.action_summary.to_csv(
        output_dir / "action_summary.csv",
        index=False,
    )

    scalar_summary.to_csv(
        output_dir / "scalar_summary.csv",
        index=False,
    )

    bootstrap_scalars.to_csv(
        output_dir
        / "bootstrap_scalar_replicates.csv",
        index=False,
    )

    bootstrap_bins.to_csv(
        output_dir
        / "bootstrap_bin_replicates.csv",
        index=False,
    )

    interval_table.to_csv(
        output_dir / "bootstrap_intervals.csv",
        index=False,
    )

    bin_bands.to_csv(
        output_dir / "bootstrap_bin_bands.csv",
        index=False,
    )

    plot_calibration(
        audit_result.bin_summary,
        bin_bands,
        output_dir
        / "calibration_reference_vs_restricted.png",
        args.horizon,
    )

    plot_distortion(
        audit_result.bin_summary,
        bin_bands,
        output_dir
        / "calibration_distortion_curve.png",
        args.horizon,
    )

    training_metadata_path = (
        output_dir / "training_metadata.json"
    )

    training_metadata = {}

    if training_metadata_path.exists():
        training_metadata = json.loads(
            training_metadata_path.read_text(
                encoding="utf-8"
            )
        )

    final_metadata = {
        "phase": "1B",
        "part": "audit",
        "horizon_days": args.horizon,
        "observation_boundary": "hospital",
        "test_predictions_frozen": True,
        "test_used_for_model_selection": False,
        "n_test_patients": int(len(audit_frame)),
        "n_calibration_bins": args.n_bins,
        "bootstrap_replicates_requested": (
            args.bootstrap_replicates
        ),
        "bootstrap_replicates_valid": int(
            bootstrap_scalars[
                "bootstrap_id"
            ].nunique()
        ),
        "max_decomposition_residual": float(
            audit_result.max_decomposition_residual
        ),
        "phase_status": phase_status,
        "training_metadata": training_metadata,
        "point_metrics": point_metrics,
    }

    (
        output_dir / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            final_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nObservation-boundary audit")
    print(
        interval_table.to_string(
            index=False
        )
    )

    print("\nAction summary")
    print(
        audit_result.action_summary.to_string(
            index=False
        )
    )

    print(
        "\nMaximum decomposition residual: "
        f"{audit_result.max_decomposition_residual:.3e}"
    )

    print(
        f"\nPhase 1B status: {phase_status}"
    )

    print(
        f"\nArtifacts saved under: {output_dir}"
    )

    print("\nPhase 1B Part 2 complete.")


if __name__ == "__main__":
    main()
