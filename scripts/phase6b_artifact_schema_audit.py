"""Audit frozen artifacts needed for manuscript tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOTS = [
    Path(
        "artifacts/"
        "phase5c_frozen_for_manuscript"
    ),
    Path(
        "artifacts/"
        "phase5d_bootstrap_tipping_points"
    ),
    Path(
        "artifacts/"
        "phase5g_rf_frozen_for_manuscript"
    ),
    Path(
        "artifacts/"
        "phase5h_cross_dataset_frozen_for_manuscript"
    ),
    Path(
        "artifacts/"
        "phase5i_cross_dataset_h5_frozen_for_manuscript"
    ),
    Path(
        "artifacts/"
        "phase5j_cross_dataset_h6_frozen_for_manuscript"
    ),
    Path(
        "artifacts/"
        "phase5j_h7_mixed_evidence_frozen_for_manuscript"
    ),
    Path(
        "artifacts/"
        "phase5j_support2_policy_shift"
    ),
    Path(
        "artifacts/"
        "phase5j_mimic_policy_shift"
    ),
    Path(
        "artifacts/"
        "phase5j_support2_recalibration"
    ),
    Path(
        "artifacts/"
        "phase5j_mimic_recalibration"
    ),
    Path(
        "artifacts/"
        "phase6a_final_claim_manifest"
    ),
]

OUTPUT_DIR = Path(
    "artifacts/"
    "phase6b_artifact_schema_audit"
)


def inspect_csv(
    path: Path,
) -> dict[str, object]:
    try:
        data = pd.read_csv(
            path,
            nrows=8,
            low_memory=False,
        )

        return {
            "path": str(
                path
            ),
            "suffix": "".join(
                path.suffixes
            ),
            "readable": True,
            "column_count": int(
                len(
                    data.columns
                )
            ),
            "columns": list(
                data.columns
            ),
            "sample_rows": (
                data.fillna(
                    ""
                ).to_dict(
                    orient="records"
                )
            ),
        }
    except Exception as error:
        return {
            "path": str(
                path
            ),
            "suffix": "".join(
                path.suffixes
            ),
            "readable": False,
            "error": repr(
                error
            ),
        }


def inspect_json(
    path: Path,
) -> dict[str, object]:
    try:
        content = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return {
            "path": str(
                path
            ),
            "suffix": ".json",
            "readable": True,
            "top_level_type": type(
                content
            ).__name__,
            "top_level_keys": (
                list(
                    content.keys()
                )
                if isinstance(
                    content,
                    dict,
                )
                else []
            ),
        }
    except Exception as error:
        return {
            "path": str(
                path
            ),
            "suffix": ".json",
            "readable": False,
            "error": repr(
                error
            ),
        }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_rows = []
    csv_details = []
    json_details = []

    for root in ROOTS:
        exists = root.exists()

        if not exists:
            file_rows.append(
                {
                    "root": str(
                        root
                    ),
                    "root_exists": False,
                    "path": "",
                    "filename": "",
                    "suffix": "",
                    "size_bytes": 0,
                }
            )
            continue

        files = sorted(
            path
            for path in root.rglob(
                "*"
            )
            if path.is_file()
        )

        if not files:
            file_rows.append(
                {
                    "root": str(
                        root
                    ),
                    "root_exists": True,
                    "path": "",
                    "filename": "",
                    "suffix": "",
                    "size_bytes": 0,
                }
            )

        for path in files:
            suffix = "".join(
                path.suffixes
            )

            file_rows.append(
                {
                    "root": str(
                        root
                    ),
                    "root_exists": True,
                    "path": str(
                        path
                    ),
                    "filename": path.name,
                    "suffix": suffix,
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
            )

            if (
                path.name.endswith(
                    ".csv"
                )
                or path.name.endswith(
                    ".csv.gz"
                )
            ):
                csv_details.append(
                    inspect_csv(
                        path
                    )
                )

            elif path.suffix == ".json":
                json_details.append(
                    inspect_json(
                        path
                    )
                )

    inventory = pd.DataFrame(
        file_rows
    )

    inventory.to_csv(
        OUTPUT_DIR
        / "frozen_file_inventory.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "csv_schema_samples.json"
    ).write_text(
        json.dumps(
            csv_details,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "json_schema_inventory.json"
    ).write_text(
        json.dumps(
            json_details,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    missing_roots = [
        str(
            root
        )
        for root in ROOTS
        if not root.exists()
    ]

    summary = {
        "phase": "6B-A",
        "status": (
            "phase6b_artifact_schema_audit_completed"
            if not missing_roots
            else (
                "phase6b_artifact_schema_audit_"
                "has_missing_roots"
            )
        ),
        "root_count": len(
            ROOTS
        ),
        "missing_roots": (
            missing_roots
        ),
        "file_count": int(
            (
                inventory[
                    "path"
                ]
                != ""
            ).sum()
        ),
        "csv_file_count": len(
            csv_details
        ),
        "json_file_count": len(
            json_details
        ),
        "manuscript_claims": [
            "H1",
            "H2",
            "H4",
            "H5",
            "H6",
            "H7",
        ],
        "excluded_topics": [
            "H3",
            "cancellation",
        ],
    }

    (
        OUTPUT_DIR
        / "artifact_schema_audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase6b_artifact_schema_audit_status.txt"
    ).write_text(
        summary[
            "status"
        ]
        + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 6B-A frozen artifact audit"
    )

    print(
        f"\nRoots inspected: {len(ROOTS)}"
    )

    print(
        "Missing roots:",
        (
            missing_roots
            if missing_roots
            else "none"
        ),
    )

    print(
        "\nFiles by root"
    )

    root_summary = (
        inventory.loc[
            inventory[
                "path"
            ].ne(
                ""
            )
        ]
        .groupby(
            "root",
            as_index=False,
        )
        .agg(
            file_count=(
                "path",
                "count",
            ),
            total_bytes=(
                "size_bytes",
                "sum",
            ),
        )
    )

    print(
        root_summary.to_string(
            index=False
        )
    )

    print(
        "\nCSV files"
    )

    csv_table = inventory.loc[
        inventory[
            "suffix"
        ].isin(
            [
                ".csv",
                ".csv.gz",
            ]
        ),
        [
            "root",
            "filename",
            "size_bytes",
        ],
    ]

    print(
        csv_table.to_string(
            index=False
        )
    )

    print(
        "\nStatus:",
        summary[
            "status"
        ],
    )


if __name__ == "__main__":
    main()
