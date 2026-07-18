"""Locate practitioner-available baseline covariates for Phase 5G-B."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT = Path(
    "artifacts/phase5c_admission_model/"
    "test_predictions.csv.gz"
)

OUTPUT_DIR = Path(
    "artifacts/phase5g_covariate_input_audit"
)

SEARCH_ROOT = Path(
    "artifacts"
)

LEAKAGE_TOKENS = {
    "outcome",
    "observed",
    "event",
    "restricted",
    "exit",
    "action",
    "death",
    "dod",
    "mortality",
    "discharge",
    "censor",
    "followup",
    "follow_up",
    "reference",
    "label",
}

IDENTIFIER_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
    "admission_id",
}


def classify_column(
    column: str,
) -> str:
    lower = column.lower()

    if lower in IDENTIFIER_COLUMNS:
        return "identifier"

    if lower == "prediction":
        return "frozen_prediction"

    if any(
        token in lower
        for token in LEAKAGE_TOKENS
    ):
        return "exclude_postbaseline_or_label"

    return "candidate_baseline_covariate"


def find_feature_hints(
    value,
    *,
    path: str = "",
) -> list[dict[str, str]]:
    rows = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():
            next_path = (
                f"{path}.{key}"
                if path
                else key
            )

            key_lower = key.lower()

            if (
                "feature" in key_lower
                or "numeric" in key_lower
                or "categorical" in key_lower
                or "covariate" in key_lower
            ):
                rows.append(
                    {
                        "json_key": next_path,
                        "value": json.dumps(
                            item,
                            default=str,
                        )[:4000],
                    }
                )

            rows.extend(
                find_feature_hints(
                    item,
                    path=next_path,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value
        ):
            rows.extend(
                find_feature_hints(
                    item,
                    path=f"{path}[{index}]",
                )
            )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    sample = pd.read_csv(
        INPUT,
        nrows=5000,
    )

    column_rows = []

    for column in sample.columns:
        series = sample[
            column
        ]

        column_rows.append(
            {
                "column": column,
                "classification": (
                    classify_column(
                        column
                    )
                ),
                "dtype": str(
                    series.dtype
                ),
                "sample_missing_fraction": float(
                    series.isna().mean()
                ),
                "sample_unique_values": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    column_audit = pd.DataFrame(
        column_rows
    )

    candidate_files = []

    patterns = [
        "phase5c*/**/*.csv",
        "phase5c*/**/*.csv.gz",
        "phase5a*/**/*.csv",
        "phase5a*/**/*.csv.gz",
    ]

    paths = set()

    for pattern in patterns:
        paths.update(
            SEARCH_ROOT.glob(
                pattern
            )
        )

    for path in sorted(
        paths
    ):
        try:
            header = pd.read_csv(
                path,
                nrows=3,
            )
        except Exception:
            continue

        columns = list(
            header.columns
        )

        has_identifier = bool(
            IDENTIFIER_COLUMNS.intersection(
                columns
            )
        )

        if not has_identifier:
            continue

        candidate_baseline = [
            column
            for column in columns
            if classify_column(
                column
            )
            == "candidate_baseline_covariate"
        ]

        candidate_files.append(
            {
                "path": str(
                    path
                ),
                "size_mb": round(
                    path.stat().st_size
                    / 1024**2,
                    3,
                ),
                "n_columns": len(
                    columns
                ),
                "has_subject_id": (
                    "subject_id"
                    in columns
                ),
                "has_hadm_id": (
                    "hadm_id"
                    in columns
                ),
                "candidate_baseline_count": len(
                    candidate_baseline
                ),
                "candidate_baseline_columns": (
                    "|".join(
                        candidate_baseline
                    )
                ),
            }
        )

    json_rows = []

    for path in sorted(
        SEARCH_ROOT.glob(
            "phase5*/**/*.json"
        )
    ):
        try:
            content = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        for row in find_feature_hints(
            content
        ):
            json_rows.append(
                {
                    "path": str(
                        path
                    ),
                    **row,
                }
            )

    candidate_table = pd.DataFrame(
        candidate_files
    )

    if not candidate_table.empty:
        candidate_table = (
            candidate_table.sort_values(
                [
                    "candidate_baseline_count",
                    "n_columns",
                ],
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    json_table = pd.DataFrame(
        json_rows
    )

    column_audit.to_csv(
        OUTPUT_DIR
        / "test_prediction_column_audit.csv",
        index=False,
    )

    candidate_table.to_csv(
        OUTPUT_DIR
        / "candidate_feature_files.csv",
        index=False,
    )

    json_table.to_csv(
        OUTPUT_DIR
        / "json_feature_hints.csv",
        index=False,
    )

    print(
        "Phase 5G covariate input audit"
    )

    print(
        "\nCurrent test-prediction columns"
    )

    print(
        column_audit.to_string(
            index=False
        )
    )

    print(
        "\nCandidate feature files"
    )

    if candidate_table.empty:
        print(
            "No candidate feature files found."
        )
    else:
        print(
            candidate_table.head(
                20
            ).to_string(
                index=False
            )
        )

    print(
        "\nFeature hints from JSON artifacts"
    )

    if json_table.empty:
        print(
            "No JSON feature hints found."
        )
    else:
        print(
            json_table.head(
                30
            ).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
