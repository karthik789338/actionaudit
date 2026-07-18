"""Freeze validated Phase 5D sensitivity results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(
    "artifacts/phase5d_frozen_for_manuscript"
)

REQUIRED_STATUSES = {
    (
        "artifacts/phase5d_action_specific_sensitivity/"
        "phase5d_status.txt"
    ): "phase5d_action_specific_sensitivity_informative",
    (
        "artifacts/phase5d_action_specific_sensitivity_extended/"
        "phase5d_status.txt"
    ): "phase5d_action_specific_sensitivity_informative",
    (
        "artifacts/phase5d_bootstrap_tipping_points/"
        "phase5d_tipping_status.txt"
    ): (
        "phase5d_exact_tipping_and_sampling_"
        "uncertainty_validated"
    ),
    (
        "artifacts/phase5d_practical_smallness/"
        "phase5d_practical_status.txt"
    ): "phase5d_practical_smallness_not_reached",
}

HASHED_FILES = [
    "scripts/phase5d_action_specific_sensitivity.py",
    "scripts/phase5d_bootstrap_tipping_points.py",
    "scripts/phase5d_practical_smallness.py",
    (
        "artifacts/phase5d_action_specific_sensitivity/"
        "oracle_matching_action_deltas.csv"
    ),
    (
        "artifacts/phase5d_bootstrap_tipping_points/"
        "exact_action_tipping_points.csv"
    ),
    (
        "artifacts/phase5d_bootstrap_tipping_points/"
        "action_bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5d_bootstrap_tipping_points/"
        "global_bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5d_practical_smallness/"
        "bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5d_practical_smallness/"
        "practical_smallness_evidence.json"
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_rows = []

    for path_text, expected in REQUIRED_STATUSES.items():
        path = Path(path_text)

        actual = (
            path.read_text(
                encoding="utf-8"
            ).strip()
            if path.exists()
            else "missing"
        )

        status_rows.append(
            {
                "artifact": path_text,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )

    status_checks = pd.DataFrame(status_rows)

    tipping = pd.read_csv(
        "artifacts/phase5d_bootstrap_tipping_points/"
        "action_bootstrap_intervals.csv"
    )

    distances = tipping.loc[
        tipping["metric"].eq(
            "distance_oracle_to_tipping"
        )
    ]

    practical = json.loads(
        Path(
            "artifacts/phase5d_practical_smallness/"
            "practical_smallness_evidence.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    validation = pd.DataFrame(
        [
            {
                "check": "all required statuses passed",
                "value": int(
                    status_checks["passed"].sum()
                ),
                "threshold": len(status_checks),
                "comparison": "==",
                "passed": bool(
                    status_checks["passed"].all()
                ),
            },
            {
                "check": (
                    "all action oracle-to-tipping "
                    "lower bounds are positive"
                ),
                "value": float(
                    distances["q025"].min()
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    distances["q025"].min() > 0
                ),
            },
            {
                "check": (
                    "minimum-distortion lower bound "
                    "exceeds practical threshold"
                ),
                "value": practical[
                    "minimum_distortion_bootstrap_q025"
                ],
                "threshold": practical[
                    "practical_threshold"
                ],
                "comparison": ">",
                "passed": (
                    practical[
                        "minimum_distortion_bootstrap_q025"
                    ]
                    > practical[
                        "practical_threshold"
                    ]
                ),
            },
            {
                "check": (
                    "practical-smallness probability "
                    "upper bound below one percent"
                ),
                "value": practical[
                    "practical_smallness_probability_q975"
                ],
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    practical[
                        "practical_smallness_probability_q975"
                    ]
                    < 0.01
                ),
            },
        ]
    )

    status = (
        "phase5d_frozen_for_manuscript"
        if validation["passed"].all()
        else "phase5d_freeze_requires_review"
    )

    hashes = {
        path_text: sha256_file(
            Path(path_text)
        )
        for path_text in HASHED_FILES
    }

    manifest = {
        "phase": "5D",
        "status": status,
        "hypothesis": (
            "Action-specific sensitivity is more "
            "informative than global sensitivity."
        ),
        "core_claim": (
            "Action-specific residual-dependence shifts "
            "are heterogeneous and directionally mixed, "
            "but all integrated action conclusions remain "
            "below their exact reversal thresholds."
        ),
        "model_ranking_tipping_points": (
            "Deferred until multiple frozen prediction "
            "families are available."
        ),
        "minimum_oracle_to_tipping_q025": float(
            distances["q025"].min()
        ),
        "minimum_distortion_q025": practical[
            "minimum_distortion_bootstrap_q025"
        ],
        "practical_smallness_probability_q975": practical[
            "practical_smallness_probability_q975"
        ],
        "sha256": hashes,
    }

    status_checks.to_csv(
        OUTPUT_DIR
        / "required_status_checks.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "freeze_validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5d_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5d_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print("Phase 5D freeze validation")
    print("\nRequired statuses")
    print(status_checks.to_string(index=False))
    print("\nFreeze checks")
    print(validation.to_string(index=False))
    print(f"\nStatus: {status}")


if __name__ == "__main__":
    main()
