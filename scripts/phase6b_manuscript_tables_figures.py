"""Phase 6B: generate manuscript-ready tables and figures."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


EXPECTED_PHASE6A_STATUS = (
    "phase6a_final_claim_manifest_frozen"
)

CLAIM_ORDER = [
    "H1",
    "H2",
    "H4",
    "H5",
    "H6",
    "H7",
]

POLICY_ORDER = [
    "conservative_late",
    "baseline",
    "aggressive_early",
]

POLICY_LABELS = {
    "conservative_late": "Conservative",
    "baseline": "Baseline",
    "aggressive_early": "Aggressive",
}

DATASET_ORDER = [
    "SUPPORT2",
    "MIMIC-IV",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manuscript tables, figures, "
            "captions, and validation artifacts from "
            "the frozen ActionAudit evidence."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "phase6b_manuscript_tables_figures"
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


def require_columns(
    data: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing columns: {missing}"
        )


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


def read_csv(
    path: str,
    required: list[str],
) -> pd.DataFrame:
    file_path = Path(
        path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing frozen artifact: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        low_memory=False,
    )

    require_columns(
        data,
        required,
        label=str(
            file_path
        ),
    )

    return data


def extract_number(
    text: str,
    pattern: str,
    *,
    label: str,
) -> float:
    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not extract {label} from: {text}"
        )

    number_text = (
        match.group(
            1
        )
        .strip()
        .rstrip(
            ".,;:"
        )
    )

    return float(
        number_text
    )


def format_float(
    value: object,
    digits: int = 3,
) -> str:
    if value is None:
        return ""

    try:
        number = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    if not math.isfinite(
        number
    ):
        return ""

    return f"{number:.{digits}f}"


def dataframe_to_markdown(
    data: pd.DataFrame,
) -> str:
    columns = [
        str(
            column
        )
        for column in data.columns
    ]

    rows = [
        [
            str(
                value
            ).replace(
                "\n",
                " ",
            )
            for value in row
        ]
        for row in data.astype(
            object
        ).where(
            pd.notna(
                data
            ),
            "",
        ).values.tolist()
    ]

    header = (
        "| "
        + " | ".join(
            columns
        )
        + " |"
    )

    separator = (
        "| "
        + " | ".join(
            [
                "---"
                for _ in columns
            ]
        )
        + " |"
    )

    body = [
        (
            "| "
            + " | ".join(
                row
            )
            + " |"
        )
        for row in rows
    ]

    return "\n".join(
        [
            header,
            separator,
            *body,
        ]
    )


def save_table(
    data: pd.DataFrame,
    *,
    stem: str,
    tables_dir: Path,
) -> None:
    data.to_csv(
        tables_dir
        / f"{stem}.csv",
        index=False,
    )

    latex = data.to_latex(
        index=False,
        escape=True,
        na_rep="",
    )

    (
        tables_dir
        / f"{stem}.tex"
    ).write_text(
        latex,
        encoding="utf-8",
    )

    (
        tables_dir
        / f"{stem}.md"
    ).write_text(
        dataframe_to_markdown(
            data
        )
        + "\n",
        encoding="utf-8",
    )


def save_figure(
    figure: plt.Figure,
    *,
    stem: str,
    figures_dir: Path,
) -> None:
    figure.savefig(
        figures_dir
        / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        figures_dir
        / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def load_sources() -> dict[str, pd.DataFrame]:
    sources = {}

    sources[
        "claims"
    ] = read_csv(
        (
            "artifacts/"
            "phase6a_final_claim_manifest/"
            "final_claim_matrix.csv"
        ),
        [
            "claim",
            "manuscript_claim",
            "final_status",
            "manuscript_role",
        ],
    )

    actual_claims = sources[
        "claims"
    ][
        "claim"
    ].tolist()

    if actual_claims != CLAIM_ORDER:
        raise ValueError(
            "Unexpected manuscript claim order: "
            f"{actual_claims}"
        )

    sources[
        "cross_dataset"
    ] = read_csv(
        (
            "artifacts/"
            "phase5h_cross_dataset_"
            "frozen_for_manuscript/"
            "cross_dataset_claim_matrix.csv"
        ),
        [
            "claim",
            "claim_text",
            "support2_evidence",
            "mimic_evidence",
            "final_status",
        ],
    )

    sources[
        "mimic_h2"
    ] = read_csv(
        (
            "artifacts/"
            "phase5h_cross_dataset_"
            "frozen_for_manuscript/"
            "mimic_h2_prespecified_contrasts.csv"
        ),
        [
            "contrast",
            "dimension",
            "point_difference",
            "q025",
            "q975",
            "interval_excludes_zero",
        ],
    )

    sources[
        "support2_h2"
    ] = read_csv(
        (
            "artifacts/"
            "phase5h_cross_dataset_"
            "frozen_for_manuscript/"
            "support2_h2_partition_evidence.csv"
        ),
        [
            "dataset",
            "dimension",
            "point_range",
            "q025",
            "q975",
            "evidence_type",
        ],
    )

    sources[
        "mimic_h5"
    ] = read_csv(
        (
            "artifacts/"
            "phase5i_cross_dataset_h5_"
            "frozen_for_manuscript/"
            "mimic_localized_sensitivity_summary.csv"
        ),
        [
            "dataset",
            "view",
            "common_global_margin",
            "minimum_local_margin",
            "closest_local_action",
            "relative_margin_reduction",
            "advantage_q025",
            "advantage_q975",
            "action_tipping_delta_range",
        ],
    )

    sources[
        "support2_h5"
    ] = read_csv(
        (
            "artifacts/"
            "phase5i_cross_dataset_h5_"
            "frozen_for_manuscript/"
            "support2_localized_sensitivity_summary.csv"
        ),
        [
            "dataset",
            "view",
            "common_global_margin",
            "minimum_local_margin",
            "closest_local_action",
            "relative_margin_reduction",
            "advantage_q025",
            "advantage_q975",
            "action_tipping_delta_range",
        ],
    )

    sources[
        "h6"
    ] = read_csv(
        (
            "artifacts/"
            "phase5j_cross_dataset_h6_"
            "frozen_for_manuscript/"
            "h6_cross_dataset_evidence.csv"
        ),
        [
            "dataset",
            "horizon_days",
            "n",
            "reference_event_rate",
            "reference_ici",
            "observation_fraction_range",
            "restricted_ici_range",
            "baseline_observed_fraction",
            "aggressive_observed_fraction",
            "conservative_observed_fraction",
            "baseline_restricted_ici",
            "aggressive_restricted_ici",
            "conservative_restricted_ici",
            "aggressive_ici_difference",
            "aggressive_ici_q025",
            "aggressive_ici_q975",
            "conservative_ici_difference",
            "conservative_ici_q025",
            "conservative_ici_q975",
        ],
    )

    sources[
        "h7"
    ] = read_csv(
        (
            "artifacts/"
            "phase5j_h7_mixed_evidence_"
            "frozen_for_manuscript/"
            "h7_dataset_evidence.csv"
        ),
        [
            "dataset",
            "horizon_days",
            "core_checks_passed",
            "core_checks_total",
            "aggressive_restricted_ici_change",
            "aggressive_reference_ici_change",
            "aggressive_restricted_log_loss_change",
            "aggressive_reference_log_loss_change",
            "h7_supported",
            "evidence_classification",
        ],
    )

    sources[
        "support2_policy"
    ] = read_csv(
        (
            "artifacts/"
            "phase5j_support2_policy_shift/"
            "policy_scalar_summary.csv"
        ),
        [
            "policy",
            "observed_fraction",
            "reference_ici",
            "restricted_ici",
        ],
    )

    sources[
        "mimic_policy"
    ] = read_csv(
        (
            "artifacts/"
            "phase5j_mimic_policy_shift/"
            "policy_scalar_summary.csv"
        ),
        [
            "policy",
            "observed_fraction",
            "reference_ici",
            "restricted_ici",
        ],
    )

    return sources


def build_table_01(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    h1 = require_single_row(
        sources[
            "cross_dataset"
        ],
        sources[
            "cross_dataset"
        ][
            "claim"
        ].eq(
            "H1"
        ),
        label="H1 claim",
    )

    support2_text = str(
        h1[
            "support2_evidence"
        ]
    )

    mimic_text = str(
        h1[
            "mimic_evidence"
        ]
    )

    h6 = sources[
        "h6"
    ].set_index(
        "dataset"
    )

    rows = []

    for dataset, evidence_text in [
        (
            "SUPPORT2",
            support2_text,
        ),
        (
            "MIMIC-IV",
            mimic_text,
        ),
    ]:
        reference_ici = extract_number(
            evidence_text,
            r"reference ICI\s+([0-9.]+)",
            label=(
                f"{dataset} reference ICI"
            ),
        )

        restricted_ici = extract_number(
            evidence_text,
            r"restricted ICI\s+([0-9.]+)",
            label=(
                f"{dataset} restricted ICI"
            ),
        )

        distortion = extract_number(
            evidence_text,
            (
                r"(?:standardized )?"
                r"signed distortion\s+([0-9.]+)"
            ),
            label=(
                f"{dataset} signed distortion"
            ),
        )

        row = h6.loc[
            dataset
        ]

        rows.append(
            {
                "Dataset": dataset,
                "Outcome horizon (days)": int(
                    row[
                        "horizon_days"
                    ]
                ),
                "Evaluation sample": int(
                    row[
                        "n"
                    ]
                ),
                "Linked event rate": format_float(
                    row[
                        "reference_event_rate"
                    ],
                    3,
                ),
                "Reference ICI": format_float(
                    reference_ici,
                    3,
                ),
                "Hospital-restricted ICI": (
                    format_float(
                        restricted_ici,
                        3,
                    )
                ),
                "Standardized signed distortion": (
                    format_float(
                        distortion,
                        3,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_table_02(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    h1 = require_single_row(
        sources[
            "cross_dataset"
        ],
        sources[
            "cross_dataset"
        ][
            "claim"
        ].eq(
            "H1"
        ),
        label="H1 claim",
    )

    for dataset, column in [
        (
            "SUPPORT2",
            "support2_evidence",
        ),
        (
            "MIMIC-IV",
            "mimic_evidence",
        ),
    ]:
        rows.append(
            {
                "Claim": "H1",
                "Dataset": dataset,
                "Dimension": (
                    "Hospital-boundary outcome "
                    "restriction"
                ),
                "Estimate": str(
                    h1[
                        column
                    ]
                ),
                "95% interval": "",
                "Interpretation": (
                    "Measured calibration changed "
                    "materially when evaluation was "
                    "restricted to outcomes observed "
                    "before hospital exit."
                ),
            }
        )

    for _, row in sources[
        "support2_h2"
    ].iterrows():
        rows.append(
            {
                "Claim": "H2",
                "Dataset": "SUPPORT2",
                "Dimension": row[
                    "dimension"
                ],
                "Estimate": (
                    "Range = "
                    + format_float(
                        row[
                            "point_range"
                        ],
                        3,
                    )
                ),
                "95% interval": (
                    "["
                    + format_float(
                        row[
                            "q025"
                        ],
                        3,
                    )
                    + ", "
                    + format_float(
                        row[
                            "q975"
                        ],
                        3,
                    )
                    + "]"
                ),
                "Interpretation": (
                    "Descriptive discharge-linked "
                    "partition; not a causal action "
                    "contrast."
                ),
            }
        )

    for _, row in sources[
        "mimic_h2"
    ].iterrows():
        supported = bool(
            row[
                "interval_excludes_zero"
            ]
        )

        rows.append(
            {
                "Claim": "H2",
                "Dataset": "MIMIC-IV",
                "Dimension": row[
                    "contrast"
                ],
                "Estimate": (
                    "Difference = "
                    + format_float(
                        row[
                            "point_difference"
                        ],
                        3,
                    )
                ),
                "95% interval": (
                    "["
                    + format_float(
                        row[
                            "q025"
                        ],
                        3,
                    )
                    + ", "
                    + format_float(
                        row[
                            "q975"
                        ],
                        3,
                    )
                    + "]"
                ),
                "Interpretation": (
                    "Interval excludes zero."
                    if supported
                    else (
                        "Interval includes zero; "
                        "retained as a dimension-"
                        "specific null result."
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_table_03() -> pd.DataFrame:
    rows = [
        {
            "Dataset": "SUPPORT2",
            "Estimator": "Complete case",
            "Reference-calibration MAE": "0.407",
            "Empirical reading": (
                "Large residual error when analysis "
                "used only directly observed outcomes."
            ),
        },
        {
            "Dataset": "SUPPORT2",
            "Estimator": "Kaplan–Meier",
            "Reference-calibration MAE": "0.087",
            "Empirical reading": (
                "Substantial improvement, but not the "
                "best recovery."
            ),
        },
        {
            "Dataset": "SUPPORT2",
            "Estimator": "Survival outcome regression",
            "Reference-calibration MAE": "0.056",
            "Empirical reading": (
                "Best SUPPORT2 recovery under the "
                "available identification regime."
            ),
        },
        {
            "Dataset": "SUPPORT2",
            "Estimator": "IPCW",
            "Reference-calibration MAE": "0.116",
            "Empirical reading": (
                "Improved over complete case but "
                "remained less accurate than outcome "
                "regression."
            ),
        },
        {
            "Dataset": "MIMIC-IV",
            "Estimator": "Score-only outcome regression",
            "Reference-calibration MAE": "0.149",
            "Empirical reading": (
                "Best unaugmented MIMIC estimator, "
                "but materially separated from the "
                "linked reference."
            ),
        },
        {
            "Dataset": "MIMIC-IV",
            "Estimator": "Rich logistic outcome model",
            "Reference-calibration MAE": "0.152",
            "Empirical reading": (
                "Additional covariates did not restore "
                "the linked calibration target."
            ),
        },
        {
            "Dataset": "MIMIC-IV",
            "Estimator": "Kaplan–Meier",
            "Reference-calibration MAE": "0.157",
            "Empirical reading": (
                "Limited recovery under the severe "
                "hospital-boundary observation pattern."
            ),
        },
        {
            "Dataset": "MIMIC-IV",
            "Estimator": "Histogram gradient boosting",
            "Reference-calibration MAE": "0.168",
            "Empirical reading": (
                "Greater model flexibility did not "
                "improve recovery."
            ),
        },
        {
            "Dataset": "MIMIC-IV",
            "Estimator": "Random forest",
            "Reference-calibration MAE": "0.168",
            "Empirical reading": (
                "Nonlinear nuisance estimation alone "
                "did not resolve the identification "
                "and overlap limitations."
            ),
        },
        {
            "Dataset": "MIMIC-IV",
            "Estimator": "RF-AIPW, clipping 0.05",
            "Reference-calibration MAE": "0.420",
            "Empirical reading": (
                "Augmentation was unstable under "
                "severe positivity limitations."
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def build_table_04(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    mimic = sources[
        "mimic_h5"
    ].iloc[
        0
    ]

    support2_candidates = sources[
        "support2_h5"
    ].loc[
        sources[
            "support2_h5"
        ][
            "view"
        ].ne(
            "original_boundary_action"
        )
    ]

    support2 = support2_candidates.sort_values(
        "relative_margin_reduction",
        ascending=False,
    ).iloc[
        0
    ]

    rows = []

    for row, evidence_type in [
        (
            support2,
            (
                "Descriptive discharge-linked "
                "partition"
            ),
        ),
        (
            mimic,
            "Native discharge actions",
        ),
    ]:
        dataset = str(
            row[
                "dataset"
            ]
        )

        rows.append(
            {
                "Dataset": dataset,
                "Sensitivity view": row[
                    "view"
                ],
                "Evidence type": evidence_type,
                "Global robustness margin": (
                    format_float(
                        row[
                            "common_global_margin"
                        ],
                        3,
                    )
                ),
                "Closest localized margin": (
                    format_float(
                        row[
                            "minimum_local_margin"
                        ],
                        3,
                    )
                ),
                "Closest localized group": (
                    row[
                        "closest_local_action"
                    ]
                ),
                "Relative margin reduction": (
                    format_float(
                        100
                        * float(
                            row[
                                "relative_margin_reduction"
                            ]
                        ),
                        1,
                    )
                    + "%"
                ),
                "Localized advantage 95% interval": (
                    "["
                    + format_float(
                        row[
                            "advantage_q025"
                        ],
                        3,
                    )
                    + ", "
                    + format_float(
                        row[
                            "advantage_q975"
                        ],
                        3,
                    )
                    + "]"
                ),
                "Tipping-threshold range": (
                    format_float(
                        row[
                            "action_tipping_delta_range"
                        ],
                        3,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_table_05(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for _, row in sources[
        "h6"
    ].iterrows():
        conservative_interval = (
            "["
            + format_float(
                row[
                    "conservative_ici_q025"
                ],
                3,
            )
            + ", "
            + format_float(
                row[
                    "conservative_ici_q975"
                ],
                3,
            )
            + "]"
        )

        conservative_supported = bool(
            float(
                row[
                    "conservative_ici_q975"
                ]
            )
            < 0
        )

        rows.append(
            {
                "Dataset": row[
                    "dataset"
                ],
                "Outcome horizon (days)": int(
                    row[
                        "horizon_days"
                    ]
                ),
                "Observation-fraction range": (
                    format_float(
                        row[
                            "observation_fraction_range"
                        ],
                        3,
                    )
                ),
                "Restricted-ICI range": (
                    format_float(
                        row[
                            "restricted_ici_range"
                        ],
                        3,
                    )
                ),
                "Aggressive minus baseline ICI": (
                    format_float(
                        row[
                            "aggressive_ici_difference"
                        ],
                        3,
                    )
                ),
                "Aggressive contrast 95% interval": (
                    "["
                    + format_float(
                        row[
                            "aggressive_ici_q025"
                        ],
                        3,
                    )
                    + ", "
                    + format_float(
                        row[
                            "aggressive_ici_q975"
                        ],
                        3,
                    )
                    + "]"
                ),
                "Conservative minus baseline ICI": (
                    format_float(
                        row[
                            "conservative_ici_difference"
                        ],
                        3,
                    )
                ),
                "Conservative contrast 95% interval": (
                    conservative_interval
                ),
                "Conservative result": (
                    "Decrease supported"
                    if conservative_supported
                    else "Inconclusive"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_table_06(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for _, row in sources[
        "h7"
    ].iterrows():
        rows.append(
            {
                "Dataset": row[
                    "dataset"
                ],
                "Outcome horizon (days)": int(
                    row[
                        "horizon_days"
                    ]
                ),
                "Primary checks": (
                    f"{int(row['core_checks_passed'])}/"
                    f"{int(row['core_checks_total'])}"
                ),
                "Restricted ICI change": (
                    format_float(
                        row[
                            "aggressive_restricted_ici_change"
                        ],
                        3,
                    )
                ),
                "Reference ICI change": (
                    format_float(
                        row[
                            "aggressive_reference_ici_change"
                        ],
                        3,
                    )
                ),
                "Restricted log-loss change": (
                    format_float(
                        row[
                            "aggressive_restricted_log_loss_change"
                        ],
                        3,
                    )
                ),
                "Reference log-loss change": (
                    format_float(
                        row[
                            "aggressive_reference_log_loss_change"
                        ],
                        3,
                    )
                ),
                "Evidence classification": (
                    row[
                        "evidence_classification"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def figure_01_framework(
    figures_dir: Path,
) -> None:
    figure = plt.figure(
        figsize=(
            12,
            6.5,
        )
    )

    axis = figure.add_axes(
        [
            0,
            0,
            1,
            1,
        ]
    )

    axis.set_xlim(
        0,
        12,
    )

    axis.set_ylim(
        0,
        7,
    )

    axis.axis(
        "off"
    )

    boxes = [
        (
            0.5,
            4.7,
            2.3,
            1.1,
            (
                "Fixed admission-time\n"
                "risk predictions"
            ),
        ),
        (
            3.5,
            4.7,
            2.3,
            1.1,
            (
                "Hospital exit and\n"
                "outcome observation"
            ),
        ),
        (
            6.5,
            4.7,
            2.3,
            1.1,
            (
                "Restricted\n"
                "calibration audit"
            ),
        ),
        (
            9.5,
            4.7,
            2.0,
            1.1,
            (
                "Apparent\n"
                "model behavior"
            ),
        ),
        (
            3.5,
            1.5,
            2.3,
            1.1,
            (
                "Linked complete\n"
                "outcome target"
            ),
        ),
        (
            6.5,
            1.5,
            2.3,
            1.1,
            (
                "Reference\n"
                "calibration audit"
            ),
        ),
        (
            9.5,
            1.5,
            2.0,
            1.1,
            (
                "Sensitivity,\n"
                "policy shift, and\n"
                "recalibration tests"
            ),
        ),
    ]

    for x, y, width, height, text in boxes:
        patch = FancyBboxPatch(
            (
                x,
                y,
            ),
            width,
            height,
            boxstyle=(
                "round,pad=0.04,"
                "rounding_size=0.08"
            ),
            linewidth=1.4,
            fill=False,
        )

        axis.add_patch(
            patch
        )

        axis.text(
            x
            + width / 2,
            y
            + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=11,
        )

    arrows = [
        (
            (
                2.8,
                5.25,
            ),
            (
                3.5,
                5.25,
            ),
        ),
        (
            (
                5.8,
                5.25,
            ),
            (
                6.5,
                5.25,
            ),
        ),
        (
            (
                8.8,
                5.25,
            ),
            (
                9.5,
                5.25,
            ),
        ),
        (
            (
                4.65,
                4.7,
            ),
            (
                4.65,
                2.6,
            ),
        ),
        (
            (
                5.8,
                2.05,
            ),
            (
                6.5,
                2.05,
            ),
        ),
        (
            (
                8.8,
                2.05,
            ),
            (
                9.5,
                2.05,
            ),
        ),
        (
            (
                10.5,
                2.6,
            ),
            (
                10.5,
                4.7,
            ),
        ),
    ]

    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.5,
            },
        )

    axis.text(
        6,
        6.45,
        (
            "ActionAudit evaluation framework"
        ),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    axis.text(
        6,
        0.55,
        (
            "Predictions and linked outcomes define "
            "the reference target; hospital-boundary "
            "observation determines what a restricted "
            "audit can see."
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    save_figure(
        figure,
        stem="figure_01_audit_framework",
        figures_dir=figures_dir,
    )


def figure_02_h1_distortion(
    table_01: pd.DataFrame,
    figures_dir: Path,
    figure_data_dir: Path,
) -> None:
    data = table_01[
        [
            "Dataset",
            "Reference ICI",
            "Hospital-restricted ICI",
        ]
    ].copy()

    data[
        "Reference ICI"
    ] = pd.to_numeric(
        data[
            "Reference ICI"
        ]
    )

    data[
        "Hospital-restricted ICI"
    ] = pd.to_numeric(
        data[
            "Hospital-restricted ICI"
        ]
    )

    data.to_csv(
        figure_data_dir
        / "figure_02_data.csv",
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            5.5,
        )
    )

    x = np.arange(
        len(
            data
        )
    )

    width = 0.34

    axis.bar(
        x
        - width / 2,
        data[
            "Reference ICI"
        ],
        width,
        label="Linked reference",
    )

    axis.bar(
        x
        + width / 2,
        data[
            "Hospital-restricted ICI"
        ],
        width,
        label="Hospital restricted",
    )

    axis.set_xticks(
        x,
        data[
            "Dataset"
        ],
    )

    axis.set_ylabel(
        "Integrated calibration index"
    )

    axis.set_title(
        "Hospital-boundary restriction changes measured calibration"
    )

    axis.legend(
        frameon=False
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        figure,
        stem="figure_02_reference_vs_restricted_ici",
        figures_dir=figures_dir,
    )


def figure_03_h5_sensitivity(
    table_04: pd.DataFrame,
    figures_dir: Path,
    figure_data_dir: Path,
) -> None:
    data = table_04[
        [
            "Dataset",
            "Global robustness margin",
            "Closest localized margin",
        ]
    ].copy()

    for column in [
        "Global robustness margin",
        "Closest localized margin",
    ]:
        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ]
        )

    data.to_csv(
        figure_data_dir
        / "figure_03_data.csv",
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            5.5,
        )
    )

    x = np.arange(
        len(
            data
        )
    )

    width = 0.34

    axis.bar(
        x
        - width / 2,
        data[
            "Global robustness margin"
        ],
        width,
        label="Common global margin",
    )

    axis.bar(
        x
        + width / 2,
        data[
            "Closest localized margin"
        ],
        width,
        label="Closest localized margin",
    )

    axis.set_xticks(
        x,
        data[
            "Dataset"
        ],
    )

    axis.set_ylabel(
        "Distance from oracle shift to tipping point"
    )

    axis.set_title(
        "Localized sensitivity reveals weaker robustness"
    )

    axis.legend(
        frameon=False
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        figure,
        stem="figure_03_localized_sensitivity_margins",
        figures_dir=figures_dir,
    )


def figure_04_h6_policy_shift(
    sources: dict[str, pd.DataFrame],
    figures_dir: Path,
    figure_data_dir: Path,
) -> None:
    rows = []

    for dataset, source_name in [
        (
            "SUPPORT2",
            "support2_policy",
        ),
        (
            "MIMIC-IV",
            "mimic_policy",
        ),
    ]:
        frame = sources[
            source_name
        ].loc[
            sources[
                source_name
            ][
                "policy"
            ].isin(
                POLICY_ORDER
            )
        ].copy()

        for _, row in frame.iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "Policy": POLICY_LABELS[
                        row[
                            "policy"
                        ]
                    ],
                    "Observed fraction": float(
                        row[
                            "observed_fraction"
                        ]
                    ),
                    "Restricted ICI": float(
                        row[
                            "restricted_ici"
                        ]
                    ),
                    "Reference ICI": float(
                        row[
                            "reference_ici"
                        ]
                    ),
                }
            )

    data = pd.DataFrame(
        rows
    )

    data.to_csv(
        figure_data_dir
        / "figure_04_data.csv",
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    markers = {
        "SUPPORT2": "o",
        "MIMIC-IV": "s",
    }

    for dataset in DATASET_ORDER:
        subset = data.loc[
            data[
                "Dataset"
            ].eq(
                dataset
            )
        ].copy()

        order = {
            label: index
            for index, label in enumerate(
                [
                    "Conservative",
                    "Baseline",
                    "Aggressive",
                ]
            )
        }

        subset[
            "_order"
        ] = subset[
            "Policy"
        ].map(
            order
        )

        subset = subset.sort_values(
            "_order"
        )

        axis.plot(
            subset[
                "Observed fraction"
            ],
            subset[
                "Restricted ICI"
            ],
            marker=markers[
                dataset
            ],
            linewidth=1.8,
            markersize=7,
            label=dataset,
        )

        for _, row in subset.iterrows():
            axis.annotate(
                row[
                    "Policy"
                ],
                (
                    row[
                        "Observed fraction"
                    ],
                    row[
                        "Restricted ICI"
                    ],
                ),
                xytext=(
                    5,
                    5,
                ),
                textcoords=(
                    "offset points"
                ),
                fontsize=8,
            )

    axis.set_xlabel(
        "Fraction with outcome observed under the emulated policy"
    )

    axis.set_ylabel(
        "Restricted integrated calibration index"
    )

    axis.set_title(
        "Observation-policy shifts create apparent calibration drift"
    )

    axis.legend(
        frameon=False
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        figure,
        stem="figure_04_observation_policy_drift",
        figures_dir=figures_dir,
    )


def figure_05_h7_recalibration(
    table_06: pd.DataFrame,
    figures_dir: Path,
    figure_data_dir: Path,
) -> None:
    data = table_06[
        [
            "Dataset",
            "Restricted ICI change",
            "Reference ICI change",
        ]
    ].copy()

    data[
        "Restricted ICI change"
    ] = pd.to_numeric(
        data[
            "Restricted ICI change"
        ]
    )

    data[
        "Reference ICI change"
    ] = pd.to_numeric(
        data[
            "Reference ICI change"
        ]
    )

    data.to_csv(
        figure_data_dir
        / "figure_05_data.csv",
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            5.5,
        )
    )

    x = np.arange(
        len(
            data
        )
    )

    width = 0.34

    axis.bar(
        x
        - width / 2,
        data[
            "Restricted ICI change"
        ],
        width,
        label="Restricted target",
    )

    axis.bar(
        x
        + width / 2,
        data[
            "Reference ICI change"
        ],
        width,
        label="Linked reference target",
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_xticks(
        x,
        data[
            "Dataset"
        ],
    )

    axis.set_ylabel(
        "ICI change after aggressive-policy logistic recalibration"
    )

    axis.set_title(
        "Recalibration target mismatch is dataset dependent"
    )

    axis.legend(
        frameon=False
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        figure,
        stem="figure_05_recalibration_target_mismatch",
        figures_dir=figures_dir,
    )


def build_table_captions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table": "Table 1",
                "file": "table_01_dataset_summary",
                "caption": (
                    "Evaluation cohorts and the "
                    "difference between calibration "
                    "measured against linked outcomes "
                    "and calibration measured only "
                    "within hospital-observed follow-up."
                ),
            },
            {
                "table": "Table 2",
                "file": "table_02_h1_h2_evidence",
                "caption": (
                    "Cross-dataset evidence for "
                    "hospital-boundary calibration "
                    "distortion and its heterogeneity "
                    "across clinically meaningful "
                    "strata."
                ),
            },
            {
                "table": "Table 3",
                "file": (
                    "table_03_h4_adjustment_benchmark"
                ),
                "caption": (
                    "Adjustment benchmarks under "
                    "different identification and "
                    "overlap regimes. Lower "
                    "reference-calibration MAE is "
                    "better."
                ),
            },
            {
                "table": "Table 4",
                "file": (
                    "table_04_h5_localized_sensitivity"
                ),
                "caption": (
                    "Comparison of a common global "
                    "sensitivity margin with the "
                    "closest localized action- or "
                    "stratum-specific margin."
                ),
            },
            {
                "table": "Table 5",
                "file": "table_05_h6_policy_shift",
                "caption": (
                    "Apparent calibration drift under "
                    "controlled real-data-grounded "
                    "observation-policy emulations."
                ),
            },
            {
                "table": "Table 6",
                "file": "table_06_h7_recalibration",
                "caption": (
                    "Effect of logistic recalibration "
                    "fitted under aggressive outcome "
                    "restriction on the restricted and "
                    "linked reference targets."
                ),
            },
        ]
    )


def build_figure_captions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure": "Figure 1",
                "file": "figure_01_audit_framework",
                "caption": (
                    "ActionAudit separates the fixed "
                    "prediction and linked-outcome "
                    "target from the hospital-boundary "
                    "process that determines which "
                    "outcomes enter a restricted audit."
                ),
            },
            {
                "figure": "Figure 2",
                "file": (
                    "figure_02_reference_vs_"
                    "restricted_ici"
                ),
                "caption": (
                    "Calibration measured only among "
                    "hospital-observed outcomes was "
                    "substantially worse than "
                    "calibration measured against the "
                    "linked complete-outcome target in "
                    "both datasets."
                ),
            },
            {
                "figure": "Figure 3",
                "file": (
                    "figure_03_localized_"
                    "sensitivity_margins"
                ),
                "caption": (
                    "Localized analysis identified "
                    "strata or discharge actions with "
                    "smaller robustness margins than "
                    "the common global perturbation."
                ),
            },
            {
                "figure": "Figure 4",
                "file": (
                    "figure_04_observation_"
                    "policy_drift"
                ),
                "caption": (
                    "Changing only the emulated "
                    "observation policy altered the "
                    "restricted calibration estimate "
                    "while predictions and linked "
                    "outcomes remained fixed."
                ),
            },
            {
                "figure": "Figure 5",
                "file": (
                    "figure_05_recalibration_"
                    "target_mismatch"
                ),
                "caption": (
                    "In MIMIC-IV, aggressive-policy "
                    "recalibration improved the "
                    "restricted target while harming "
                    "the linked target. SUPPORT2 did "
                    "not reproduce that pattern."
                ),
            },
        ]
    )


def build_validation_checks(
    *,
    phase6a_status: str,
    tables: dict[str, pd.DataFrame],
    tables_dir: Path,
    figures_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    expected_table_stems = [
        "table_01_dataset_summary",
        "table_02_h1_h2_evidence",
        "table_03_h4_adjustment_benchmark",
        "table_04_h5_localized_sensitivity",
        "table_05_h6_policy_shift",
        "table_06_h7_recalibration",
    ]

    expected_figure_stems = [
        "figure_01_audit_framework",
        "figure_02_reference_vs_restricted_ici",
        "figure_03_localized_sensitivity_margins",
        "figure_04_observation_policy_drift",
        "figure_05_recalibration_target_mismatch",
    ]

    table_files_exist = all(
        (
            tables_dir
            / f"{stem}.{suffix}"
        ).exists()
        and (
            tables_dir
            / f"{stem}.{suffix}"
        ).stat().st_size
        > 0
        for stem in expected_table_stems
        for suffix in [
            "csv",
            "tex",
            "md",
        ]
    )

    figure_files_exist = all(
        (
            figures_dir
            / f"{stem}.{suffix}"
        ).exists()
        and (
            figures_dir
            / f"{stem}.{suffix}"
        ).stat().st_size
        > 0
        for stem in expected_figure_stems
        for suffix in [
            "png",
            "pdf",
        ]
    )

    h7 = tables[
        "table_06_h7_recalibration"
    ].set_index(
        "Dataset"
    )

    h7_direction_preserved = bool(
        float(
            h7.loc[
                "SUPPORT2",
                "Restricted ICI change",
            ]
        )
        > 0
        and float(
            h7.loc[
                "SUPPORT2",
                "Reference ICI change",
            ]
        )
        > 0
        and float(
            h7.loc[
                "MIMIC-IV",
                "Restricted ICI change",
            ]
        )
        < 0
        and float(
            h7.loc[
                "MIMIC-IV",
                "Reference ICI change",
            ]
        )
        > 0
    )

    h6 = tables[
        "table_05_h6_policy_shift"
    ]

    h6_supported = all(
        float(
            interval.strip(
                "[]"
            ).split(
                ","
            )[
                0
            ]
        )
        > 0
        for interval in h6[
            "Aggressive contrast 95% interval"
        ]
    )

    h5 = tables[
        "table_04_h5_localized_sensitivity"
    ]

    h5_reductions = sorted(
        float(
            value.rstrip(
                "%"
            )
        )
        for value in h5[
            "Relative margin reduction"
        ]
    )

    h5_expected = bool(
        np.allclose(
            h5_reductions,
            [
                41.8,
                76.3,
            ],
            atol=0.1,
        )
    )

    manuscript_files = [
        path
        for path in output_dir.rglob(
            "*"
        )
        if path.is_file()
        and path.suffix.lower()
        in {
            ".csv",
            ".tex",
            ".md",
            ".json",
            ".txt",
        }
    ]

    excluded_pattern = re.compile(
        (
            r"\bH3\b|"
            r"cancellation|"
            r"cancel in aggregate|"
            r"opposing action-specific"
        ),
        flags=re.IGNORECASE,
    )

    violations = []

    for path in manuscript_files:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if excluded_pattern.search(
            text
        ):
            violations.append(
                str(
                    path
                )
            )

    return pd.DataFrame(
        [
            {
                "check": (
                    "Phase 6A claim manifest "
                    "already frozen"
                ),
                "value": phase6a_status,
                "threshold": (
                    EXPECTED_PHASE6A_STATUS
                ),
                "comparison": "==",
                "passed": (
                    phase6a_status
                    == EXPECTED_PHASE6A_STATUS
                ),
            },
            {
                "check": (
                    "six manuscript tables "
                    "generated in three formats"
                ),
                "value": (
                    18
                    if table_files_exist
                    else 0
                ),
                "threshold": 18,
                "comparison": "==",
                "passed": (
                    table_files_exist
                ),
            },
            {
                "check": (
                    "five manuscript figures "
                    "generated in PNG and PDF"
                ),
                "value": (
                    10
                    if figure_files_exist
                    else 0
                ),
                "threshold": 10,
                "comparison": "==",
                "passed": (
                    figure_files_exist
                ),
            },
            {
                "check": (
                    "H6 aggressive-policy "
                    "interval excludes zero "
                    "in both datasets"
                ),
                "value": int(
                    h6_supported
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    h6_supported
                ),
            },
            {
                "check": (
                    "H5 localized margin "
                    "reductions match frozen evidence"
                ),
                "value": ",".join(
                    f"{value:.1f}"
                    for value in h5_reductions
                ),
                "threshold": "41.8,76.3",
                "comparison": "==",
                "passed": (
                    h5_expected
                ),
            },
            {
                "check": (
                    "H7 dataset-specific "
                    "directions are preserved"
                ),
                "value": int(
                    h7_direction_preserved
                ),
                "threshold": 1,
                "comparison": "==",
                "passed": (
                    h7_direction_preserved
                ),
            },
            {
                "check": (
                    "excluded manuscript topic "
                    "is absent from package"
                ),
                "value": len(
                    violations
                ),
                "threshold": 0,
                "comparison": "==",
                "passed": (
                    len(
                        violations
                    )
                    == 0
                ),
            },
        ]
    )


def main() -> None:
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    tables_dir = (
        output_dir
        / "tables"
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    figure_data_dir = (
        output_dir
        / "figure_data"
    )

    for directory in [
        output_dir,
        tables_dir,
        figures_dir,
        figure_data_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    phase6a_status = read_status(
        Path(
            "artifacts/"
            "phase6a_final_claim_manifest/"
            "phase6a_final_claim_manifest_status.txt"
        )
    )

    sources = load_sources()

    tables = {
        "table_01_dataset_summary": (
            build_table_01(
                sources
            )
        ),
        "table_02_h1_h2_evidence": (
            build_table_02(
                sources
            )
        ),
        "table_03_h4_adjustment_benchmark": (
            build_table_03()
        ),
        "table_04_h5_localized_sensitivity": (
            build_table_04(
                sources
            )
        ),
        "table_05_h6_policy_shift": (
            build_table_05(
                sources
            )
        ),
        "table_06_h7_recalibration": (
            build_table_06(
                sources
            )
        ),
    }

    for stem, table in tables.items():
        save_table(
            table,
            stem=stem,
            tables_dir=tables_dir,
        )

    table_captions = (
        build_table_captions()
    )

    figure_captions = (
        build_figure_captions()
    )

    table_captions.to_csv(
        output_dir
        / "table_captions.csv",
        index=False,
    )

    figure_captions.to_csv(
        output_dir
        / "figure_captions.csv",
        index=False,
    )

    figure_01_framework(
        figures_dir
    )

    figure_02_h1_distortion(
        tables[
            "table_01_dataset_summary"
        ],
        figures_dir,
        figure_data_dir,
    )

    figure_03_h5_sensitivity(
        tables[
            "table_04_h5_localized_sensitivity"
        ],
        figures_dir,
        figure_data_dir,
    )

    figure_04_h6_policy_shift(
        sources,
        figures_dir,
        figure_data_dir,
    )

    figure_05_h7_recalibration(
        tables[
            "table_06_h7_recalibration"
        ],
        figures_dir,
        figure_data_dir,
    )

    validations = build_validation_checks(
        phase6a_status=(
            phase6a_status
        ),
        tables=tables,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        output_dir=output_dir,
    )

    status = (
        "phase6b_manuscript_tables_figures_frozen"
        if validations[
            "passed"
        ].all()
        else (
            "phase6b_manuscript_tables_figures_"
            "requires_review"
        )
    )

    validations.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
    )

    manifest = {
        "phase": "6B",
        "status": status,
        "claim_set": CLAIM_ORDER,
        "table_count": len(
            tables
        ),
        "figure_count": 5,
        "table_formats": [
            "CSV",
            "LaTeX",
            "Markdown",
        ],
        "figure_formats": [
            "PNG",
            "PDF",
        ],
        "table_files": [
            {
                "stem": stem,
                "rows": int(
                    len(
                        table
                    )
                ),
                "columns": list(
                    table.columns
                ),
            }
            for stem, table in tables.items()
        ],
        "figure_files": (
            figure_captions[
                "file"
            ].tolist()
        ),
        "interpretation_boundaries": [
            (
                "All numerical claims are drawn "
                "from frozen experimental evidence."
            ),
            (
                "Real-data policy analyses are "
                "controlled emulations rather than "
                "causal or historical policy effects."
            ),
            (
                "SUPPORT2 sensitivity partitions are "
                "descriptive discharge-linked strata."
            ),
            (
                "H7 retains the SUPPORT2 "
                "non-replication."
            ),
            (
                "SUPPORT2 and MIMIC-IV estimates "
                "are displayed separately and are "
                "not statistically pooled."
            ),
        ],
        "next_stage": (
            "Phase 6C manuscript methods, results, "
            "limitations, and claim-boundary drafts."
        ),
    }

    (
        output_dir
        / "phase6b_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "phase6b_manuscript_tables_figures_status.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 6B manuscript tables and figures"
    )

    print(
        "\nGenerated tables"
    )

    for stem, table in tables.items():
        print(
            f"  {stem}: "
            f"{len(table)} rows"
        )

    print(
        "\nGenerated figures"
    )

    for figure_name in figure_captions[
        "file"
    ]:
        print(
            f"  {figure_name}"
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
