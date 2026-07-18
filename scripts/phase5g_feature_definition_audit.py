"""Extract exact Phase 5C baseline-feature construction definitions."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


SCRIPT_CANDIDATES = [
    Path(
        "scripts/phase5c_admission_prediction_model.py"
    ),
    Path(
        "scripts/phase5c_admission_model.py"
    ),
    Path(
        "scripts/phase5c_build_admission_model.py"
    ),
]

METADATA_PATH = Path(
    "artifacts/phase5c_admission_model/"
    "experiment_metadata.json"
)

OUTPUT_DIR = Path(
    "artifacts/phase5g_feature_definition_audit"
)

FEATURE_TOKENS = [
    "age_at_admission",
    "admission_hour_sin",
    "admission_hour_cos",
    "admission_weekday_sin",
    "admission_weekday_cos",
    "prior_admission_count",
    "prior_emergency_or_urgent_count",
    "prior_total_los_days",
    "days_since_previous_discharge",
    "prior_icu_stay_count",
    "prior_icu_los_days",
    "has_prior_admission",
    "has_prior_icu",
]


def find_source_script() -> Path:
    for path in SCRIPT_CANDIDATES:
        if path.exists():
            return path

    matches = sorted(
        Path(
            "scripts"
        ).glob(
            "*phase5c*admission*.py"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find the Phase 5C "
            "admission-model script."
        )

    return matches[0]


def function_source_rows(
    source: str,
    tree: ast.AST,
) -> list[dict[str, object]]:
    lines = source.splitlines()

    rows = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno,
            )
        )

        text = "\n".join(
            lines[
                start - 1:end
            ]
        )

        matched_tokens = [
            token
            for token in FEATURE_TOKENS
            if token in text
        ]

        if not matched_tokens:
            continue

        rows.append(
            {
                "function": node.name,
                "start_line": start,
                "end_line": end,
                "matched_features": "|".join(
                    matched_tokens
                ),
                "source": text,
            }
        )

    return rows


def contextual_matches(
    source: str,
    *,
    radius: int = 8,
) -> list[dict[str, object]]:
    lines = source.splitlines()

    rows = []

    seen = set()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        matches = [
            token
            for token in FEATURE_TOKENS
            if token in line
        ]

        if not matches:
            continue

        start = max(
            1,
            line_number - radius,
        )

        end = min(
            len(
                lines
            ),
            line_number + radius,
        )

        key = (
            start,
            end,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        rows.append(
            {
                "start_line": start,
                "end_line": end,
                "matched_features": "|".join(
                    matches
                ),
                "source": "\n".join(
                    lines[
                        start - 1:end
                    ]
                ),
            }
        )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    script_path = find_source_script()

    source = script_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    function_rows = function_source_rows(
        source,
        tree,
    )

    context_rows = contextual_matches(
        source
    )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    manifest = pd.DataFrame(
        [
            {
                "feature_type": (
                    "numeric"
                ),
                "feature": feature,
            }
            for feature in metadata[
                "numeric_features"
            ]
        ]
        + [
            {
                "feature_type": (
                    "categorical"
                ),
                "feature": feature,
            }
            for feature in metadata[
                "categorical_features"
            ]
        ]
        + [
            {
                "feature_type": (
                    "excluded_post_admission"
                ),
                "feature": feature,
            }
            for feature in metadata[
                "post_admission_features_excluded"
            ]
        ]
    )

    functions = pd.DataFrame(
        function_rows
    )

    contexts = pd.DataFrame(
        context_rows
    )

    manifest.to_csv(
        OUTPUT_DIR
        / "frozen_feature_manifest.csv",
        index=False,
    )

    functions.to_csv(
        OUTPUT_DIR
        / "matching_feature_functions.csv",
        index=False,
    )

    contexts.to_csv(
        OUTPUT_DIR
        / "feature_source_contexts.csv",
        index=False,
    )

    evidence = {
        "phase": "5G feature-definition audit",
        "source_script": str(
            script_path
        ),
        "n_matching_functions": len(
            functions
        ),
        "n_context_blocks": len(
            contexts
        ),
        "feature_tokens": (
            FEATURE_TOKENS
        ),
        "status": (
            "phase5g_feature_definitions_located"
            if (
                len(
                    functions
                )
                > 0
                or len(
                    contexts
                )
                > 0
            )
            else (
                "phase5g_feature_definitions_"
                "require_manual_review"
            )
        ),
    }

    (
        OUTPUT_DIR
        / "feature_definition_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "phase5g_feature_definition_status.txt"
    ).write_text(
        evidence[
            "status"
        ]
        + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5G feature-definition audit"
    )

    print(
        f"\nSource script: {script_path}"
    )

    print(
        "\nFrozen feature manifest"
    )

    print(
        manifest.to_string(
            index=False
        )
    )

    print(
        "\nFunctions containing feature construction"
    )

    if functions.empty:
        print(
            "No complete function matched."
        )
    else:
        print(
            functions[
                [
                    "function",
                    "start_line",
                    "end_line",
                    "matched_features",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nStatus: "
        f"{evidence['status']}"
    )


if __name__ == "__main__":
    main()
