"""Phase 6A: consolidate all frozen experimental claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


STATUS_SPECS = [
    {
        "phase": "5C",
        "component": "MIMIC empirical benchmark",
        "path": (
            "artifacts/"
            "phase5c_frozen_for_manuscript/"
            "phase5c_freeze_status.txt"
        ),
        "expected": (
            "phase5c_frozen_for_manuscript"
        ),
        "required": True,
    },
    {
        "phase": "5D",
        "component": (
            "MIMIC action-specific sensitivity"
        ),
        "path": (
            "artifacts/"
            "phase5d_bootstrap_tipping_points/"
            "phase5d_tipping_status.txt"
        ),
        "expected": (
            "phase5d_exact_tipping_and_"
            "sampling_uncertainty_validated"
        ),
        "required": True,
    },
    {
        "phase": "5E",
        "component": (
            "Synthetic observation-policy shift"
        ),
        "path": (
            "artifacts/"
            "phase5e_frozen_for_manuscript/"
            "phase5e_freeze_status.txt"
        ),
        "expected": (
            "phase5e_frozen_for_manuscript"
        ),
        "required": True,
    },
    {
        "phase": "5F",
        "component": (
            "Synthetic recalibration fragility"
        ),
        "path": (
            "artifacts/"
            "phase5f_frozen_for_manuscript/"
            "phase5f_freeze_status.txt"
        ),
        "expected": (
            "phase5f_frozen_for_manuscript"
        ),
        "required": True,
    },
    {
        "phase": "5G",
        "component": (
            "MIMIC nonlinear and augmented "
            "adjustment benchmark"
        ),
        "path": (
            "artifacts/"
            "phase5g_rf_frozen_for_manuscript/"
            "phase5g_rf_freeze_status.txt"
        ),
        "expected": None,
        "required": True,
    },
    {
        "phase": "5H",
        "component": (
            "Cross-dataset H1 H2 H4 freeze"
        ),
        "path": (
            "artifacts/"
            "phase5h_cross_dataset_"
            "frozen_for_manuscript/"
            "phase5h_cross_dataset_"
            "freeze_status.txt"
        ),
        "expected": (
            "phase5h_cross_dataset_h1_h2_h4_"
            "frozen_for_manuscript"
        ),
        "required": True,
    },
    {
        "phase": "5I",
        "component": (
            "Cross-dataset H5 freeze"
        ),
        "path": (
            "artifacts/"
            "phase5i_cross_dataset_h5_"
            "frozen_for_manuscript/"
            "phase5i_cross_dataset_h5_status.txt"
        ),
        "expected": (
            "phase5i_cross_dataset_h5_"
            "frozen_for_manuscript"
        ),
        "required": True,
    },
    {
        "phase": "5J-D",
        "component": (
            "Cross-dataset H6 freeze"
        ),
        "path": (
            "artifacts/"
            "phase5j_cross_dataset_h6_"
            "frozen_for_manuscript/"
            "phase5j_cross_dataset_h6_status.txt"
        ),
        "expected": (
            "phase5j_cross_dataset_h6_"
            "frozen_for_manuscript"
        ),
        "required": True,
    },
    {
        "phase": "5J-F",
        "component": (
            "Mixed-evidence H7 freeze"
        ),
        "path": (
            "artifacts/"
            "phase5j_h7_mixed_evidence_"
            "frozen_for_manuscript/"
            "phase5j_h7_mixed_evidence_status.txt"
        ),
        "expected": (
            "phase5j_h7_mixed_evidence_"
            "frozen_for_manuscript"
        ),
        "required": True,
    },
]


CLAIM_SOURCE_FILES = [
    (
        "artifacts/"
        "phase5h_cross_dataset_"
        "frozen_for_manuscript/"
        "cross_dataset_claim_matrix.csv"
    ),
    (
        "artifacts/"
        "phase5i_cross_dataset_h5_"
        "frozen_for_manuscript/"
        "h5_cross_dataset_claim.csv"
    ),
    (
        "artifacts/"
        "phase5j_cross_dataset_h6_"
        "frozen_for_manuscript/"
        "h6_cross_dataset_claim.csv"
    ),
    (
        "artifacts/"
        "phase5j_h7_mixed_evidence_"
        "frozen_for_manuscript/"
        "h7_frozen_claim.csv"
    ),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the authoritative final "
            "experiment and claim manifest."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase6a_final_claim_manifest"
        ),
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_status_registry() -> pd.DataFrame:
    rows = []

    for specification in STATUS_SPECS:
        path = Path(
            specification[
                "path"
            ]
        )

        exists = path.exists()

        actual = (
            path.read_text(
                encoding="utf-8"
            ).strip()
            if exists
            else ""
        )

        expected = specification[
            "expected"
        ]

        if not exists:
            passed = not specification[
                "required"
            ]
        elif expected is None:
            passed = bool(
                actual
            )
        else:
            passed = (
                actual
                == expected
            )

        rows.append(
            {
                "phase": (
                    specification[
                        "phase"
                    ]
                ),
                "component": (
                    specification[
                        "component"
                    ]
                ),
                "status_path": str(
                    path
                ),
                "file_exists": bool(
                    exists
                ),
                "expected_status": (
                    expected
                    if expected is not None
                    else (
                        "nonempty frozen status"
                    )
                ),
                "actual_status": actual,
                "required": bool(
                    specification[
                        "required"
                    ]
                ),
                "passed": bool(
                    passed
                ),
                "sha256": (
                    sha256_file(
                        path
                    )
                    if exists
                    else ""
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def source_file_registry() -> pd.DataFrame:
    rows = []

    for source in CLAIM_SOURCE_FILES:
        path = Path(
            source
        )

        exists = path.exists()

        if exists:
            frame = pd.read_csv(
                path,
                low_memory=False,
            )

            row_count = int(
                len(
                    frame
                )
            )

            columns = json.dumps(
                list(
                    frame.columns
                )
            )
        else:
            row_count = 0
            columns = "[]"

        rows.append(
            {
                "source_path": str(
                    path
                ),
                "file_exists": bool(
                    exists
                ),
                "row_count": (
                    row_count
                ),
                "columns": columns,
                "sha256": (
                    sha256_file(
                        path
                    )
                    if exists
                    else ""
                ),
                "passed": bool(
                    exists
                    and row_count > 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def final_claim_matrix() -> pd.DataFrame:
    rows = [
        {
            "claim": "H1",
            "manuscript_claim": (
                "Outcome truncation induced by "
                "action-dependent observation can "
                "substantially distort measured "
                "calibration."
            ),
            "synthetic_evidence": (
                "Strong controlled support."
            ),
            "support2_evidence": (
                "Strong empirical support using "
                "linked 60-day outcomes."
            ),
            "mimic_evidence": (
                "Strong empirical support using "
                "linked 30-day outcomes."
            ),
            "final_status": (
                "strong cross-dataset support"
            ),
            "manuscript_role": (
                "primary empirical claim"
            ),
            "permitted_language": (
                "Action-dependent outcome truncation "
                "substantially changed measured "
                "calibration in both clinical datasets."
            ),
            "prohibited_language": (
                "Do not describe the distortion as "
                "a causal treatment effect."
            ),
            "source_artifacts": (
                "phase5h_cross_dataset_"
                "frozen_for_manuscript"
            ),
        },
        {
            "claim": "H2",
            "manuscript_claim": (
                "Calibration distortion is "
                "heterogeneous across risk, "
                "observation pathways, clinical "
                "subgroups, care settings, and time."
            ),
            "synthetic_evidence": (
                "Strong controlled support."
            ),
            "support2_evidence": (
                "Supported across DNR, disease, "
                "and hospital-timing partitions."
            ),
            "mimic_evidence": (
                "Supported across ICU status, era, "
                "time of admission, disease chapter, "
                "service, admission type, and "
                "admission location; weekend effects "
                "were not supported."
            ),
            "final_status": (
                "strong bounded cross-dataset support"
            ),
            "manuscript_role": (
                "primary heterogeneity claim"
            ),
            "permitted_language": (
                "The magnitude of distortion varied "
                "meaningfully across prespecified and "
                "descriptive strata."
            ),
            "prohibited_language": (
                "Do not interpret subgroup contrasts "
                "as causal effects. Do not claim an "
                "institution effect because "
                "institution identifiers were "
                "unavailable."
            ),
            "source_artifacts": (
                "phase5h_cross_dataset_"
                "frozen_for_manuscript"
            ),
        },
        {
            "claim": "H3",
            "manuscript_claim": (
                "Opposing action-specific "
                "distortions can cancel in aggregate, "
                "but meaningful cancellation was not "
                "observed in the two clinical "
                "datasets."
            ),
            "synthetic_evidence": (
                "Mechanism strongly demonstrated."
            ),
            "support2_evidence": (
                "No meaningful empirical "
                "cancellation; the native boundary "
                "contains one action."
            ),
            "mimic_evidence": (
                "Action contributions were almost "
                "uniformly signed, producing "
                "negligible cancellation."
            ),
            "final_status": (
                "synthetic mechanism supported; "
                "empirical negative result"
            ),
            "manuscript_role": (
                "secondary diagnostic and "
                "scope-condition result"
            ),
            "permitted_language": (
                "The decomposition can detect "
                "cancellation, although the clinical "
                "datasets showed little evidence of it."
            ),
            "prohibited_language": (
                "Do not present cancellation as a "
                "replicated empirical phenomenon or "
                "as a primary confirmed hypothesis."
            ),
            "source_artifacts": (
                "phase4b; phase5c; phase5h"
            ),
        },
        {
            "claim": "H4",
            "manuscript_claim": (
                "Adjustment performance depends on "
                "identification and overlap; greater "
                "model flexibility alone does not "
                "guarantee recovery of linked "
                "calibration."
            ),
            "synthetic_evidence": (
                "Strong support under controlled "
                "identified and non-identified regimes."
            ),
            "support2_evidence": (
                "Survival outcome regression "
                "substantially reduced error and "
                "performed best."
            ),
            "mimic_evidence": (
                "Score-only, rich logistic, HGB, RF, "
                "IPCW, and augmented estimators "
                "remained materially separated from "
                "the linked reference; RF-AIPW gave "
                "partial reduction only."
            ),
            "final_status": (
                "strong bounded cross-dataset support"
            ),
            "manuscript_role": (
                "primary adjustment-regime claim"
            ),
            "permitted_language": (
                "Adjustment succeeded in the better-"
                "identified SUPPORT2 setting but not "
                "under the severe MIMIC positivity "
                "limitations."
            ),
            "prohibited_language": (
                "Do not claim that all adjustment "
                "methods fail universally or that "
                "survival regression always succeeds."
            ),
            "source_artifacts": (
                "phase5g; phase5h_cross_dataset_"
                "frozen_for_manuscript"
            ),
        },
        {
            "claim": "H5",
            "manuscript_claim": (
                "Localized action- or stratum-specific "
                "sensitivity analysis reveals "
                "heterogeneous robustness that can be "
                "obscured by a common global "
                "perturbation."
            ),
            "synthetic_evidence": (
                "Sensitivity model validated under "
                "controlled conditions."
            ),
            "support2_evidence": (
                "Descriptive discharge-linked "
                "partitions reduced the closest "
                "robustness margin by up to 41.8%."
            ),
            "mimic_evidence": (
                "Native discharge-action analysis "
                "reduced the closest robustness "
                "margin by 76.3%; hospice was closest "
                "to tipping."
            ),
            "final_status": (
                "strong bounded cross-dataset support"
            ),
            "manuscript_role": (
                "primary methodological claim"
            ),
            "permitted_language": (
                "Localized sensitivity provided "
                "information hidden by a single "
                "global perturbation."
            ),
            "prohibited_language": (
                "Do not describe SUPPORT2 DNR, "
                "disease, or timing partitions as "
                "native causal actions."
            ),
            "source_artifacts": (
                "phase5i_cross_dataset_h5_"
                "frozen_for_manuscript"
            ),
        },
        {
            "claim": "H6",
            "manuscript_claim": (
                "Changes in observation policy can "
                "generate substantial apparent "
                "calibration drift despite an "
                "unchanged predictor, population, "
                "and complete-outcome target."
            ),
            "synthetic_evidence": (
                "Strong controlled mechanism "
                "demonstration."
            ),
            "support2_evidence": (
                "Aggressive policy increased "
                "restricted ICI while the reference "
                "target remained fixed; the "
                "conservative contrast was "
                "inconclusive."
            ),
            "mimic_evidence": (
                "Aggressive policy increased "
                "restricted ICI and conservative "
                "policy decreased it while reference "
                "quantities remained fixed."
            ),
            "final_status": (
                "strong bounded cross-dataset support"
            ),
            "manuscript_role": (
                "primary methodological claim"
            ),
            "permitted_language": (
                "Observation-policy shifts produced "
                "apparent drift in controlled "
                "real-data-grounded emulations."
            ),
            "prohibited_language": (
                "Do not describe the emulations as "
                "historical or causal estimates of "
                "actual hospital policy changes."
            ),
            "source_artifacts": (
                "phase5j_cross_dataset_h6_"
                "frozen_for_manuscript"
            ),
        },
        {
            "claim": "H7",
            "manuscript_claim": (
                "Recalibration fitted on "
                "policy-selected observed outcomes "
                "can improve the restricted target "
                "while harming the unchanged "
                "complete-population target, but this "
                "behavior is dataset- and "
                "policy-dependent rather than "
                "universal."
            ),
            "synthetic_evidence": (
                "Strong controlled support."
            ),
            "support2_evidence": (
                "Not replicated: aggressive logistic "
                "recalibration worsened ICI on both "
                "the restricted and reference targets."
            ),
            "mimic_evidence": (
                "Strong support: aggressive logistic "
                "recalibration improved restricted "
                "ICI and log loss while worsening "
                "both reference metrics."
            ),
            "final_status": (
                "bounded support with explicit "
                "dataset-specific non-replication"
            ),
            "manuscript_role": (
                "secondary methodological and "
                "scope-condition result"
            ),
            "permitted_language": (
                "Policy-selected recalibration can "
                "create target mismatch, as observed "
                "in synthetic data and MIMIC."
            ),
            "prohibited_language": (
                "Do not describe H7 as strong "
                "cross-dataset support or as a "
                "universal failure of recalibration."
            ),
            "source_artifacts": (
                "phase5j_h7_mixed_evidence_"
                "frozen_for_manuscript"
            ),
        },
    ]

    claims = pd.DataFrame(
        rows
    )

    # H3 remains only in internal experimental archives.
    # It is excluded from every manuscript-facing output.
    claims = claims.loc[
        ~claims[
            "claim"
        ].eq(
            "H3"
        )
    ].reset_index(
        drop=True
    )

    return claims


def build_validation_checks(
    *,
    statuses: pd.DataFrame,
    sources: pd.DataFrame,
    claims: pd.DataFrame,
) -> pd.DataFrame:
    expected_claims = {
        "H1",
        "H2",
        "H4",
        "H5",
        "H6",
        "H7",
    }

    expected_order = [
        "H1",
        "H2",
        "H4",
        "H5",
        "H6",
        "H7",
    ]

    actual_claims = set(
        claims[
            "claim"
        ]
    )

    actual_order = claims[
        "claim"
    ].tolist()

    primary_claims = claims.loc[
        claims[
            "manuscript_role"
        ].str.startswith(
            "primary",
            na=False,
        ),
        "claim",
    ].tolist()

    secondary_claims = claims.loc[
        ~claims[
            "manuscript_role"
        ].str.startswith(
            "primary",
            na=False,
        ),
        "claim",
    ].tolist()

    h7 = claims.loc[
        claims[
            "claim"
        ].eq(
            "H7"
        )
    ].iloc[
        0
    ]

    return pd.DataFrame(
        [
            {
                "check": (
                    "all required frozen status "
                    "files pass"
                ),
                "value": int(
                    statuses[
                        "passed"
                    ].sum()
                ),
                "threshold": int(
                    len(
                        statuses
                    )
                ),
                "comparison": "==",
                "passed": bool(
                    statuses[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "all frozen claim source "
                    "files are present and nonempty"
                ),
                "value": int(
                    sources[
                        "passed"
                    ].sum()
                ),
                "threshold": int(
                    len(
                        sources
                    )
                ),
                "comparison": "==",
                "passed": bool(
                    sources[
                        "passed"
                    ].all()
                ),
            },
            {
                "check": (
                    "exactly six manuscript "
                    "claims are represented"
                ),
                "value": ",".join(
                    actual_order
                ),
                "threshold": (
                    "H1,H2,H4,H5,H6,H7"
                ),
                "comparison": "==",
                "passed": bool(
                    actual_claims
                    == expected_claims
                    and actual_order
                    == expected_order
                    and len(
                        claims
                    )
                    == 6
                ),
            },
            {
                "check": (
                    "primary manuscript claims "
                    "are H1 H2 H4 H5 H6"
                ),
                "value": ",".join(
                    primary_claims
                ),
                "threshold": (
                    "H1,H2,H4,H5,H6"
                ),
                "comparison": "==",
                "passed": (
                    primary_claims
                    == [
                        "H1",
                        "H2",
                        "H4",
                        "H5",
                        "H6",
                    ]
                ),
            },
            {
                "check": (
                    "H7 is the only secondary "
                    "manuscript claim"
                ),
                "value": ",".join(
                    secondary_claims
                ),
                "threshold": "H7",
                "comparison": "==",
                "passed": (
                    secondary_claims
                    == [
                        "H7",
                    ]
                ),
            },
            {
                "check": (
                    "H7 explicitly retains "
                    "SUPPORT2 non-replication"
                ),
                "value": h7[
                    "final_status"
                ],
                "threshold": (
                    "bounded support with explicit "
                    "dataset-specific non-replication"
                ),
                "comparison": "==",
                "passed": (
                    h7[
                        "final_status"
                    ]
                    == (
                        "bounded support with explicit "
                        "dataset-specific non-replication"
                    )
                ),
            },
            {
                "check": (
                    "every manuscript claim has "
                    "permitted and prohibited language"
                ),
                "value": int(
                    (
                        claims[
                            "permitted_language"
                        ].str.len()
                        > 0
                    ).sum()
                    + (
                        claims[
                            "prohibited_language"
                        ].str.len()
                        > 0
                    ).sum()
                ),
                "threshold": 12,
                "comparison": "==",
                "passed": bool(
                    (
                        claims[
                            "permitted_language"
                        ].str.len()
                        > 0
                    ).all()
                    and (
                        claims[
                            "prohibited_language"
                        ].str.len()
                        > 0
                    ).all()
                ),
            },
        ]
    )

def main() -> None:
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statuses = read_status_registry()

    sources = source_file_registry()

    claims = final_claim_matrix()

    validations = build_validation_checks(
        statuses=statuses,
        sources=sources,
        claims=claims,
    )

    status = (
        "phase6a_final_claim_manifest_frozen"
        if validations[
            "passed"
        ].all()
        else (
            "phase6a_final_claim_manifest_"
            "requires_review"
        )
    )

    statuses.to_csv(
        output_dir
        / "experiment_registry.csv",
        index=False,
    )

    sources.to_csv(
        output_dir
        / "claim_source_registry.csv",
        index=False,
    )

    claims.to_csv(
        output_dir
        / "final_claim_matrix.csv",
        index=False,
    )

    claims[
        [
            "claim",
            "permitted_language",
            "prohibited_language",
        ]
    ].to_csv(
        output_dir
        / "manuscript_language_boundaries.csv",
        index=False,
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    manifest = {
        "phase": "6A",
        "status": status,
        "purpose": (
            "Authoritative experiment, claim, "
            "and language-boundary registry for "
            "manuscript preparation."
        ),
        "primary_claims": [
            "H1",
            "H2",
            "H4",
            "H5",
            "H6",
        ],
        "secondary_claims": [
            "H7",
        ],
        "claim_count": int(
            len(
                claims
            )
        ),
        "experiment_registry_count": int(
            len(
                statuses
            )
        ),
        "source_claim_file_count": int(
            len(
                sources
            )
        ),
        "rules": [
            (
                "Only frozen artifacts may be "
                "used for manuscript tables, "
                "figures, or numerical claims."
            ),
            (
                "H7 must retain the SUPPORT2 "
                "non-replication."
            ),
            (
                "Real-data policy experiments "
                "must be called controlled "
                "real-data-grounded emulations."
            ),
            (
                "SUPPORT2 descriptive partitions "
                "must not be described as native "
                "causal actions."
            ),
            (
                "SUPPORT2 and MIMIC estimates "
                "must not be statistically pooled."
            ),
        ],
        "next_stage": (
            "Phase 6B manuscript tables "
            "and figures."
        ),
    }

    (
        output_dir
        / "final_claim_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase6a_final_claim_manifest_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 6A final claim manifest"
    )

    print(
        "\nExperiment registry"
    )

    print(
        statuses.to_string(
            index=False
        )
    )

    print(
        "\nFinal claim matrix"
    )

    print(
        claims[
            [
                "claim",
                "final_status",
                "manuscript_role",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nValidation"
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
