"""Run the Phase 0 synthetic smoke test and export auditable tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actionaudit import (
    AuditConfig,
    binned_action_audit,
)


def synthetic_reference_cohort(
    seed: int = 2027,
    n: int = 5000,
) -> pd.DataFrame:
    """Generate a synthetic reference-follow-up cohort."""

    rng = np.random.default_rng(seed)

    severity = rng.normal(
        size=n
    )

    prediction = 1.0 / (
        1.0
        + np.exp(
            -(
                -0.4
                + 0.9 * severity
            )
        )
    )

    outcome = rng.binomial(
        1,
        prediction,
    )

    home_probability = 1.0 / (
        1.0
        + np.exp(
            -(
                -1.3
                - 1.1 * severity
                - 0.8 * outcome
            )
        )
    )

    transfer_probability = 1.0 / (
        1.0
        + np.exp(
            -(
                -2.0
                + 0.4 * severity
                + 1.2 * outcome
            )
        )
    )

    home = (
        rng.uniform(size=n)
        < home_probability
    )

    transfer = (
        ~home
        & (
            rng.uniform(size=n)
            < transfer_probability
        )
    )

    observed = ~(home | transfer)

    action = np.full(
        n,
        None,
        dtype=object,
    )

    action[home] = "home"
    action[transfer] = "external_transfer"

    exit_time = np.where(
        observed,
        np.nan,
        rng.uniform(
            1,
            59,
            size=n,
        ),
    )

    return pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "prediction": prediction,
            "outcome": outcome,
            "observed": observed.astype(int),
            "action": action,
            "exit_time": exit_time,
        }
    )


def main() -> None:
    """Run the synthetic ActionAudit verification."""

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_dir = (
        project_root
        / "artifacts"
        / "phase0"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort = synthetic_reference_cohort()

    result = binned_action_audit(
        cohort,
        AuditConfig(
            horizon_days=60,
            boundary="hospital",
            n_bins=10,
            min_bin_size=50,
        ),
    )

    result.bin_summary.to_csv(
        output_dir / "bin_summary.csv",
        index=False,
    )

    result.action_summary.to_csv(
        output_dir / "action_summary.csv",
        index=False,
    )

    result.scalar_summary.to_csv(
        output_dir / "scalar_summary.csv",
        header=True,
    )

    print("Phase 0 executable checks passed")

    print(
        "Maximum decomposition residual: "
        f"{result.max_decomposition_residual:.3e}"
    )

    print("\nScalar summary")
    print(
        result.scalar_summary.to_string()
    )

    print("\nAction summary")
    print(
        result.action_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
