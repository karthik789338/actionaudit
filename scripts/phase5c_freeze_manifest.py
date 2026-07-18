"""Freeze the validated Phase 5C empirical results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(
    "artifacts/phase5c_frozen_for_manuscript"
)

REQUIRED_STATUSES = {
    (
        "artifacts/phase5c_frozen_cohort/"
        "phase5c_cohort_status.txt"
    ): "phase5c_primary_and_icu_cohorts_frozen",
    (
        "artifacts/phase5c_admission_model/"
        "phase5c_model_status.txt"
    ): "phase5c_admission_prediction_model_frozen",
    (
        "artifacts/phase5c_empirical_action_decomposition/"
        "phase5c_empirical_distortion_status.txt"
    ): (
        "phase5c_empirical_unidirectional_"
        "action_distortion_validated"
    ),
    (
        "artifacts/phase5c_binning_robustness/"
        "phase5c_binning_status.txt"
    ): "phase5c_empirical_distortion_robust_to_binning",
}

HASHED_FILES = [
    "scripts/phase5c_mimic_readiness.py",
    "scripts/phase5c_compare_cohort_definitions.py",
    "scripts/phase5c_freeze_primary_cohort.py",
    "scripts/phase5c_train_admission_model.py",
    "scripts/phase5c_empirical_action_cancellation.py",
    "scripts/phase5c_validate_empirical_distortion.py",
    "scripts/phase5c_binning_robustness.py",
    (
        "artifacts/phase5c_admission_model/"
        "test_predictions.csv.gz"
    ),
    (
        "artifacts/phase5c_empirical_action_decomposition/"
        "primary/action_summary.csv"
    ),
    (
        "artifacts/phase5c_empirical_action_decomposition/"
        "primary/scalar_bootstrap_intervals.csv"
    ),
    (
        "artifacts/phase5c_binning_robustness/"
        "point_estimate_summary.csv"
    ),
    (
        "artifacts/phase5c_binning_robustness/"
        "action_bootstrap_intervals.csv"
    ),
]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def read_json(
    path: str,
) -> dict:
    return json.loads(
        Path(
            path
        ).read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_checks = []

    for path_text, expected in (
        REQUIRED_STATUSES.items()
    ):
        path = Path(
            path_text
        )

        actual = (
            path.read_text(
                encoding="utf-8"
            ).strip()
            if path.exists()
            else "missing"
        )

        status_checks.append(
            {
                "artifact": path_text,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )

    status_frame = pd.DataFrame(
        status_checks
    )

    model_metadata = read_json(
        "artifacts/phase5c_admission_model/"
        "experiment_metadata.json"
    )

    distortion_evidence = read_json(
        "artifacts/phase5c_empirical_action_decomposition/"
        "empirical_distortion_evidence.json"
    )

    binning_metadata = read_json(
        "artifacts/phase5c_binning_robustness/"
        "experiment_metadata.json"
    )

    robustness = binning_metadata[
        "validation_evidence"
    ]

    action_intervals = pd.read_csv(
        "artifacts/phase5c_binning_robustness/"
        "action_bootstrap_intervals.csv"
    )

    all_action_intervals_positive = bool(
        (
            action_intervals[
                "q025"
            ]
            > 0
        ).all()
    )

    hashes = {}

    for path_text in HASHED_FILES:
        path = Path(
            path_text
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing freeze artifact: {path}"
            )

        hashes[
            path_text
        ] = sha256_file(
            path
        )

    validation_checks = pd.DataFrame(
        [
            {
                "check": (
                    "all required phase statuses passed"
                ),
                "value": int(
                    status_frame[
                        "passed"
                    ].sum()
                ),
                "threshold": len(
                    status_frame
                ),
                "comparison": "==",
                "passed": bool(
                    status_frame[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "all 48 binning-action intervals positive"
                ),
                "value": int(
                    (
                        action_intervals[
                            "q025"
                        ]
                        > 0
                    ).sum()
                ),
                "threshold": 48,
                "comparison": "==",
                "passed": (
                    all_action_intervals_positive
                    and len(
                        action_intervals
                    )
                    == 48
                ),
            },
            {
                "check": (
                    "primary reference ICI below 0.01"
                ),
                "value": (
                    distortion_evidence[
                        "primary_reference_ici"
                    ]
                ),
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    distortion_evidence[
                        "primary_reference_ici"
                    ]
                    < 0.01
                ),
            },
            {
                "check": (
                    "primary restricted ICI above 0.10"
                ),
                "value": (
                    distortion_evidence[
                        "primary_restricted_ici"
                    ]
                ),
                "threshold": 0.10,
                "comparison": ">",
                "passed": (
                    distortion_evidence[
                        "primary_restricted_ici"
                    ]
                    > 0.10
                ),
            },
            {
                "check": (
                    "maximum primary cancellation upper "
                    "bound below one percent"
                ),
                "value": (
                    robustness[
                        "maximum_primary_cancellation_q975"
                    ]
                ),
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    robustness[
                        "maximum_primary_cancellation_q975"
                    ]
                    < 0.01
                ),
            },
            {
                "check": (
                    "maximum ICU cancellation upper "
                    "bound below one percent"
                ),
                "value": (
                    robustness[
                        "maximum_icu_cancellation_q975"
                    ]
                ),
                "threshold": 0.01,
                "comparison": "<",
                "passed": (
                    robustness[
                        "maximum_icu_cancellation_q975"
                    ]
                    < 0.01
                ),
            },
        ]
    )

    status = (
        "phase5c_frozen_for_manuscript"
        if validation_checks[
            "passed"
        ].all()
        else "phase5c_freeze_requires_review"
    )

    manifest = {
        "phase": "5C",
        "status": status,
        "primary_cohort_n": 155059,
        "test_cohort_n": (
            model_metadata[
                "test_n"
            ]
        ),
        "selected_prediction_model": {
            "family": (
                model_metadata[
                    "selected_family"
                ]
            ),
            "parameters": (
                model_metadata[
                    "selected_parameters"
                ]
            ),
            "test_metrics": (
                model_metadata[
                    "test_metrics"
                ]
            ),
        },
        "empirical_distortion": (
            distortion_evidence
        ),
        "binning_robustness": (
            robustness
        ),
        "claim": (
            "Hospital-boundary restriction creates "
            "large, unidirectional action-attributed "
            "calibration distortion with negligible "
            "empirical cancellation."
        ),
        "icu_interpretation": (
            "The ICU cohort supports robustness of the "
            "restriction-induced distortion and action "
            "direction, but not excellent subgroup "
            "reference calibration."
        ),
        "sha256": hashes,
    }

    status_frame.to_csv(
        OUTPUT_DIR
        / "required_status_checks.csv",
        index=False,
    )

    validation_checks.to_csv(
        OUTPUT_DIR
        / "freeze_validation_checks.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "phase5c_freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5c_freeze_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5C freeze validation"
    )

    print(
        "\nRequired statuses"
    )

    print(
        status_frame.to_string(
            index=False
        )
    )

    print(
        "\nFreeze checks"
    )

    print(
        validation_checks.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
