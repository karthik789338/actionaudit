"""Run the Phase 1A SUPPORT2 feasibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionaudit.support2 import (
    load_support2_csv,
    support2_feasibility_audit,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit SUPPORT2 follow-up, outcomes, "
            "discharge truncation, and missingness."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/support2.csv",
        help="Path to the SUPPORT2 CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase1_feasibility",
        help="Directory for Phase 1A output tables.",
    )

    return parser.parse_args()


def main() -> None:
    """Execute and export the SUPPORT2 feasibility audit."""

    args = parse_arguments()

    data = load_support2_csv(
        args.input
    )

    result = support2_feasibility_audit(
        data,
        horizons=(60, 180),
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.overview.to_csv(
        output_dir / "overview.csv",
        header=True,
    )

    result.horizon_summary.to_csv(
        output_dir / "horizon_summary.csv",
        index=False,
    )

    result.consistency_checks.to_csv(
        output_dir / "consistency_checks.csv",
        index=False,
    )

    result.missingness.to_csv(
        output_dir / "missingness.csv",
        index=False,
    )

    machine_summary = {
        "overview": result.overview.to_dict(),
        "horizons": result.horizon_summary.to_dict(
            orient="records"
        ),
        "consistency_checks": (
            result.consistency_checks.to_dict(
                orient="records"
            )
        ),
    }

    summary_path = (
        output_dir
        / "feasibility_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            machine_summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "SUPPORT2 Phase 1A feasibility audit complete"
    )

    print("\nOverview")
    print(
        result.overview.to_string()
    )

    print("\nHorizon summary")
    print(
        result.horizon_summary.to_string(
            index=False
        )
    )

    failed_checks = result.consistency_checks[
        ~result.consistency_checks["passed"]
    ]

    print("\nFailed consistency checks")

    if failed_checks.empty:
        print("None")
    else:
        print(
            failed_checks.to_string(
                index=False
            )
        )

    print(
        f"\nArtifacts saved under: {output_dir}"
    )


if __name__ == "__main__":
    main()
