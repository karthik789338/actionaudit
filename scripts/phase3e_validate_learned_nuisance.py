"""Create manuscript-ready validation checks for Phase 3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "artifacts/phase3d_learned_nuisance/"
    "replication_results.csv"
)

OUTPUT_DIR = Path(
    "artifacts/phase3e_learned_nuisance_validation"
)

RESIDUAL_SCENARIOS = {
    "residual_same_direction",
    "residual_opposite_direction",
}

NO_LATENT_CONFOUNDING_SCENARIOS = {
    "independent",
    "observed_dependent",
    "positivity_stress",
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 3D results: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.read_csv(
        INPUT_PATH
    )

    required = {
        "scenario",
        "clip",
        "complete_case_mae",
        "kaplan_meier_mae",
        "true_oracle_or_mae",
        "true_oracle_ipcw_mae",
        "learned_oracle_or_mae",
        "learned_oracle_ipcw_mae",
        "learned_working_or_mae",
        "learned_working_ipcw_mae",
        "learned_oracle_or_signed_gap",
        "learned_oracle_ipcw_signed_gap",
        "learned_working_or_signed_gap",
        "learned_working_ipcw_signed_gap",
        "learned_oracle_ipcw_ess",
        "learned_working_ipcw_ess",
    }

    missing = sorted(
        required.difference(
            results.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing Phase 3D columns: {missing}"
        )

    numeric_columns = [
        column
        for column in required
        if column not in {
            "scenario",
            "clip",
        }
    ]

    summary = (
        results.groupby(
            [
                "scenario",
                "clip",
            ],
            as_index=False,
            sort=True,
        )[numeric_columns]
        .mean()
    )

    summary[
        "learned_oracle_or_estimation_penalty"
    ] = (
        summary[
            "learned_oracle_or_mae"
        ]
        - summary[
            "true_oracle_or_mae"
        ]
    )

    summary[
        "learned_oracle_ipcw_estimation_penalty"
    ] = (
        summary[
            "learned_oracle_ipcw_mae"
        ]
        - summary[
            "true_oracle_ipcw_mae"
        ]
    )

    summary[
        "latent_omission_penalty_or"
    ] = (
        summary[
            "learned_working_or_mae"
        ]
        - summary[
            "learned_oracle_or_mae"
        ]
    )

    summary[
        "latent_omission_penalty_ipcw"
    ] = (
        summary[
            "learned_working_ipcw_mae"
        ]
        - summary[
            "learned_oracle_ipcw_mae"
        ]
    )

    summary[
        "or_improvement_over_km"
    ] = (
        summary[
            "kaplan_meier_mae"
        ]
        - summary[
            "learned_oracle_or_mae"
        ]
    )

    summary[
        "ipcw_improvement_over_km"
    ] = (
        summary[
            "kaplan_meier_mae"
        ]
        - summary[
            "learned_oracle_ipcw_mae"
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "learned_nuisance_validation_summary.csv",
        index=False,
    )

    no_latent_rows = summary.loc[
        summary[
            "scenario"
        ].isin(
            NO_LATENT_CONFOUNDING_SCENARIOS
        )
    ]

    residual_rows = summary.loc[
        summary[
            "scenario"
        ].isin(
            RESIDUAL_SCENARIOS
        )
    ]

    positivity_rows = summary.loc[
        summary[
            "scenario"
        ].eq(
            "positivity_stress"
        )
    ]

    cancellation_rows = summary.loc[
        summary[
            "scenario"
        ].eq(
            "residual_same_direction"
        )
    ]

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "learned oracle OR remains close "
                    "to true oracle OR"
                ),
                "value": float(
                    summary[
                        "learned_oracle_or_estimation_penalty"
                    ].abs().max()
                ),
                "threshold": 0.02,
                "comparison": "<=",
                "passed": bool(
                    summary[
                        "learned_oracle_or_estimation_penalty"
                    ].abs().max()
                    <= 0.02
                ),
            },
            {
                "check": (
                    "learned oracle IPCW remains close "
                    "to true oracle IPCW"
                ),
                "value": float(
                    summary[
                        "learned_oracle_ipcw_estimation_penalty"
                    ].abs().max()
                ),
                "threshold": 0.02,
                "comparison": "<=",
                "passed": bool(
                    summary[
                        "learned_oracle_ipcw_estimation_penalty"
                    ].abs().max()
                    <= 0.02
                ),
            },
            {
                "check": (
                    "working OR matches oracle OR when "
                    "latent U is not required"
                ),
                "value": float(
                    no_latent_rows[
                        "latent_omission_penalty_or"
                    ].abs().max()
                ),
                "threshold": 0.01,
                "comparison": "<=",
                "passed": bool(
                    no_latent_rows[
                        "latent_omission_penalty_or"
                    ].abs().max()
                    <= 0.01
                ),
            },
            {
                "check": (
                    "omitting latent U materially harms OR "
                    "under residual dependence"
                ),
                "value": float(
                    residual_rows[
                        "latent_omission_penalty_or"
                    ].min()
                ),
                "threshold": 0.05,
                "comparison": ">",
                "passed": bool(
                    residual_rows[
                        "latent_omission_penalty_or"
                    ].min()
                    > 0.05
                ),
            },
            {
                "check": (
                    "outcome regression outperforms IPCW "
                    "under positivity stress"
                ),
                "value": float(
                    (
                        positivity_rows[
                            "learned_oracle_ipcw_mae"
                        ]
                        - positivity_rows[
                            "learned_oracle_or_mae"
                        ]
                    ).min()
                ),
                "threshold": 0.05,
                "comparison": ">",
                "passed": bool(
                    (
                        positivity_rows[
                            "learned_oracle_ipcw_mae"
                        ]
                        - positivity_rows[
                            "learned_oracle_or_mae"
                        ]
                    ).min()
                    > 0.05
                ),
            },
            {
                "check": (
                    "complete-case error can appear smaller "
                    "than Kaplan-Meier through cancellation"
                ),
                "value": float(
                    (
                        cancellation_rows[
                            "kaplan_meier_mae"
                        ]
                        - cancellation_rows[
                            "complete_case_mae"
                        ]
                    ).min()
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    (
                        cancellation_rows[
                            "kaplan_meier_mae"
                        ]
                        - cancellation_rows[
                            "complete_case_mae"
                        ]
                    ).min()
                    > 0.0
                ),
            },
        ]
    )

    checks.to_csv(
        OUTPUT_DIR
        / "phase3d_validation_checks.csv",
        index=False,
    )

    overall_pass = bool(
        checks["passed"].all()
    )

    status = (
        "phase3d_passed"
        if overall_pass
        else "phase3d_requires_review"
    )

    (
        OUTPUT_DIR
        / "phase3d_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "scenario",
        "clip",
        "kaplan_meier_mae",
        "learned_oracle_or_mae",
        "learned_working_or_mae",
        "learned_oracle_ipcw_mae",
        "learned_working_ipcw_mae",
        "latent_omission_penalty_or",
        "latent_omission_penalty_ipcw",
    ]

    print(
        "Phase 3D learned-nuisance validation"
    )

    print(
        "\nEstimator comparison"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nValidation checks"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )

    print(
        f"\nSaved under: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
