"""Freeze H7 with MIMIC support and SUPPORT2 non-replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_SUPPORT2_STATUS = (
    "phase5j_support2_recalibration_benchmarked"
)

EXPECTED_MIMIC_STATUS = (
    "phase5j_mimic_recalibration_benchmarked"
)

EXPECTED_SYNTHETIC_STATUS = (
    "phase5f_frozen_for_manuscript"
)

EXPECTED_H6_STATUS = (
    "phase5j_cross_dataset_h6_"
    "frozen_for_manuscript"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the mixed cross-dataset "
            "evidence for H7."
        )
    )

    parser.add_argument(
        "--support2-dir",
        default=(
            "artifacts/"
            "phase5j_support2_recalibration"
        ),
    )

    parser.add_argument(
        "--mimic-dir",
        default=(
            "artifacts/"
            "phase5j_mimic_recalibration"
        ),
    )

    parser.add_argument(
        "--synthetic-dir",
        default=(
            "artifacts/"
            "phase5f_frozen_for_manuscript"
        ),
    )

    parser.add_argument(
        "--h6-dir",
        default=(
            "artifacts/"
            "phase5j_cross_dataset_h6_"
            "frozen_for_manuscript"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase5j_h7_mixed_evidence_"
            "frozen_for_manuscript"
        ),
    )

    return parser.parse_args()


def read_status(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing status file: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


def require_single_row(
    data: pd.DataFrame,
    mask: pd.Series,
    *,
    label: str,
) -> pd.Series:
    selected = data.loc[
        mask
    ]

    if len(
        selected
    ) != 1:
        raise ValueError(
            f"{label}: expected one row, "
            f"found {len(selected)}"
        )

    return selected.iloc[
        0
    ]


def aggressive_logistic_point(
    directory: Path,
) -> pd.Series:
    data = pd.read_csv(
        directory
        / "point_contrasts.csv"
    )

    return require_single_row(
        data,
        data[
            "fit_policy"
        ].eq(
            "aggressive_early"
        )
        & data[
            "method"
        ].eq(
            "logistic"
        ),
        label=(
            f"{directory} aggressive logistic"
        ),
    )


def read_decision_checks(
    directory: Path,
    filename: str,
) -> pd.DataFrame:
    data = pd.read_csv(
        directory
        / filename
    )

    data[
        "passed"
    ] = (
        data[
            "passed"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            "true"
        )
    )

    return data


def main() -> None:
    args = parse_arguments()

    support2_dir = Path(
        args.support2_dir
    )

    mimic_dir = Path(
        args.mimic_dir
    )

    synthetic_dir = Path(
        args.synthetic_dir
    )

    h6_dir = Path(
        args.h6_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    support2_status = read_status(
        support2_dir
        / (
            "phase5j_support2_"
            "recalibration_status.txt"
        )
    )

    mimic_status = read_status(
        mimic_dir
        / (
            "phase5j_mimic_"
            "recalibration_status.txt"
        )
    )

    synthetic_status = read_status(
        synthetic_dir
        / "phase5f_freeze_status.txt"
    )

    h6_status = read_status(
        h6_dir
        / (
            "phase5j_cross_dataset_"
            "h6_status.txt"
        )
    )

    support2_core = read_decision_checks(
        support2_dir,
        "h7_core_decision_checks.csv",
    )

    mimic_core = read_decision_checks(
        mimic_dir,
        "h7_core_decision_checks.csv",
    )

    support2_secondary = (
        read_decision_checks(
            support2_dir,
            "h7_secondary_checks.csv",
        )
    )

    mimic_secondary = (
        read_decision_checks(
            mimic_dir,
            "h7_secondary_checks.csv",
        )
    )

    support2_point = (
        aggressive_logistic_point(
            support2_dir
        )
    )

    mimic_point = (
        aggressive_logistic_point(
            mimic_dir
        )
    )

    evidence = pd.DataFrame(
        [
            {
                "dataset": "SUPPORT2",
                "horizon_days": 60,
                "core_checks_passed": int(
                    support2_core[
                        "passed"
                    ].sum()
                ),
                "core_checks_total": int(
                    len(
                        support2_core
                    )
                ),
                "secondary_checks_passed": int(
                    support2_secondary[
                        "passed"
                    ].sum()
                ),
                "secondary_checks_total": int(
                    len(
                        support2_secondary
                    )
                ),
                "aggressive_restricted_ici_change": float(
                    support2_point[
                        "restricted_ici_change"
                    ]
                ),
                "aggressive_reference_ici_change": float(
                    support2_point[
                        "reference_ici_change"
                    ]
                ),
                "aggressive_restricted_log_loss_change": float(
                    support2_point[
                        "restricted_log_loss_change"
                    ]
                ),
                "aggressive_reference_log_loss_change": float(
                    support2_point[
                        "reference_log_loss_change"
                    ]
                ),
                "h7_supported": False,
                "evidence_classification": (
                    "non-replication"
                ),
            },
            {
                "dataset": "MIMIC-IV",
                "horizon_days": 30,
                "core_checks_passed": int(
                    mimic_core[
                        "passed"
                    ].sum()
                ),
                "core_checks_total": int(
                    len(
                        mimic_core
                    )
                ),
                "secondary_checks_passed": int(
                    mimic_secondary[
                        "passed"
                    ].sum()
                ),
                "secondary_checks_total": int(
                    len(
                        mimic_secondary
                    )
                ),
                "aggressive_restricted_ici_change": float(
                    mimic_point[
                        "restricted_ici_change"
                    ]
                ),
                "aggressive_reference_ici_change": float(
                    mimic_point[
                        "reference_ici_change"
                    ]
                ),
                "aggressive_restricted_log_loss_change": float(
                    mimic_point[
                        "restricted_log_loss_change"
                    ]
                ),
                "aggressive_reference_log_loss_change": float(
                    mimic_point[
                        "reference_log_loss_change"
                    ]
                ),
                "h7_supported": True,
                "evidence_classification": (
                    "strong real-data-grounded support"
                ),
            },
        ]
    )

    validations = pd.DataFrame(
        [
            {
                "check": (
                    "SUPPORT2 recalibration "
                    "benchmark passed structurally"
                ),
                "value": support2_status,
                "threshold": (
                    EXPECTED_SUPPORT2_STATUS
                ),
                "comparison": "==",
                "passed": (
                    support2_status
                    == EXPECTED_SUPPORT2_STATUS
                ),
            },
            {
                "check": (
                    "MIMIC recalibration "
                    "benchmark passed structurally"
                ),
                "value": mimic_status,
                "threshold": (
                    EXPECTED_MIMIC_STATUS
                ),
                "comparison": "==",
                "passed": (
                    mimic_status
                    == EXPECTED_MIMIC_STATUS
                ),
            },
            {
                "check": (
                    "synthetic H7 mechanism "
                    "already frozen"
                ),
                "value": synthetic_status,
                "threshold": (
                    EXPECTED_SYNTHETIC_STATUS
                ),
                "comparison": "==",
                "passed": (
                    synthetic_status
                    == EXPECTED_SYNTHETIC_STATUS
                ),
            },
            {
                "check": (
                    "H6 package already frozen"
                ),
                "value": h6_status,
                "threshold": (
                    EXPECTED_H6_STATUS
                ),
                "comparison": "==",
                "passed": (
                    h6_status
                    == EXPECTED_H6_STATUS
                ),
            },
            {
                "check": (
                    "MIMIC passes all four "
                    "primary H7 checks"
                ),
                "value": int(
                    mimic_core[
                        "passed"
                    ].sum()
                ),
                "threshold": 4,
                "comparison": "==",
                "passed": bool(
                    mimic_core[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "MIMIC passes all three "
                    "secondary H7 checks"
                ),
                "value": int(
                    mimic_secondary[
                        "passed"
                    ].sum()
                ),
                "threshold": 3,
                "comparison": "==",
                "passed": bool(
                    mimic_secondary[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "SUPPORT2 does not replicate "
                    "the four-part primary H7 pattern"
                ),
                "value": int(
                    support2_core[
                        "passed"
                    ].sum()
                ),
                "threshold": 4,
                "comparison": "<",
                "passed": bool(
                    not support2_core[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "SUPPORT2 aggressive logistic "
                    "recalibration does not improve ICI"
                ),
                "value": float(
                    support2_point[
                        "restricted_ici_change"
                    ]
                ),
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(
                    float(
                        support2_point[
                            "restricted_ici_change"
                        ]
                    )
                    > 0
                ),
            },
            {
                "check": (
                    "SUPPORT2 non-replication "
                    "is retained in final claim"
                ),
                "value": (
                    "explicitly retained"
                ),
                "threshold": (
                    "explicitly retained"
                ),
                "comparison": "==",
                "passed": True,
            },
        ]
    )

    claim = pd.DataFrame(
        [
            {
                "claim": "H7",
                "original_hypothesis": (
                    "Standard recalibration may be "
                    "assumption-fragile under "
                    "restricted follow-up."
                ),
                "frozen_claim": (
                    "Recalibration fitted on "
                    "policy-selected observed outcomes "
                    "can improve the restricted target "
                    "while harming the unchanged "
                    "complete-population target, but "
                    "this behavior is dataset- and "
                    "policy-dependent rather than "
                    "universal."
                ),
                "synthetic_evidence": (
                    "Strong controlled support from "
                    "the frozen Phase 5F experiment."
                ),
                "mimic_evidence": (
                    "Strong real-data-grounded support: "
                    "aggressive logistic recalibration "
                    f"changed restricted ICI by "
                    f"{float(mimic_point['restricted_ici_change']):.3f} "
                    f"and reference ICI by "
                    f"+{float(mimic_point['reference_ici_change']):.3f}."
                ),
                "support2_evidence": (
                    "Not replicated: aggressive "
                    "logistic recalibration changed "
                    f"restricted ICI by "
                    f"+{float(support2_point['restricted_ici_change']):.3f} "
                    f"and reference ICI by "
                    f"+{float(support2_point['reference_ici_change']):.3f}."
                ),
                "final_status": (
                    "bounded support with explicit "
                    "dataset-specific non-replication"
                ),
                "manuscript_role": (
                    "secondary methodological and "
                    "scope-condition result"
                ),
                "not_permitted_wording": (
                    "Do not describe H7 as strong "
                    "cross-dataset support or as a "
                    "universal failure of recalibration."
                ),
            }
        ]
    )

    status = (
        "phase5j_h7_mixed_evidence_"
        "frozen_for_manuscript"
        if validations[
            "passed"
        ].all()
        else (
            "phase5j_h7_mixed_evidence_"
            "freeze_requires_review"
        )
    )

    evidence.to_csv(
        output_dir
        / "h7_dataset_evidence.csv",
        index=False,
    )

    claim.to_csv(
        output_dir
        / "h7_frozen_claim.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "freeze_validation_checks.csv",
        index=False,
    )

    manifest = {
        "phase": "5J-F",
        "status": status,
        "claim": "H7",
        "final_status": (
            "bounded support with explicit "
            "dataset-specific non-replication"
        ),
        "evidence_layers": {
            "synthetic": "strong support",
            "MIMIC-IV": (
                "strong real-data-grounded support"
            ),
            "SUPPORT2": "not supported",
        },
        "manuscript_role": (
            "secondary methodological and "
            "scope-condition result"
        ),
        "interpretation": [
            (
                "The MIMIC result demonstrates that "
                "policy-selected recalibration can "
                "optimize the restricted target while "
                "harming the linked reference target."
            ),
            (
                "The SUPPORT2 result shows that this "
                "pattern is not inevitable."
            ),
            (
                "Recalibration performance depends on "
                "the dataset, observation policy, "
                "sample size, predictor, and selected "
                "target."
            ),
            (
                "The negative SUPPORT2 replication is "
                "retained rather than treated as a "
                "failed experiment."
            ),
        ],
    }

    (
        output_dir
        / "phase5j_h7_mixed_evidence_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase5j_h7_mixed_evidence_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5J-F H7 mixed-evidence freeze"
    )

    print(
        "\nDataset evidence"
    )

    print(
        evidence.to_string(
            index=False
        )
    )

    print(
        "\nFrozen H7 claim"
    )

    print(
        claim.to_string(
            index=False
        )
    )

    print(
        "\nFreeze validation"
    )

    print(
        validations.to_string(
            index=False
        )
    )

    print(
        f"\nStatus: {status}"
    )


if __name__ == "__main__":
    main()
