#!/usr/bin/env python3
"""
Phase 8A addendum — recalibration-rule robustness and provenance interpretation.

AUDIT ONLY. No model fitting and no new scientific experiment.

Purpose:
  1) show whether the SUPPORT2/MIMIC classification depends on choosing 4/4
     instead of 1/4, 2/4, or 3/4 bootstrap-check thresholds;
  2) distinguish point-estimate sign agreement from bootstrap decision checks;
  3) record the exact logical interpretation of the released 4-part rule;
  4) explicitly state what the public one-commit Git history can/cannot prove.

Run from the ActionAudit repository root after Phase 8A.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "revision_kdd2027" / "phase8a_freeze_provenance"
OUT.mkdir(parents=True, exist_ok=True)

table_path = ROOT / "results" / "tables" / "table_06_h7_recalibration.csv"
if not table_path.exists():
    raise FileNotFoundError(f"Missing {table_path}")

tab = pd.read_csv(table_path)

required = {
    "Dataset",
    "Primary checks",
    "Restricted ICI change",
    "Reference ICI change",
    "Restricted log-loss change",
    "Reference log-loss change",
}
missing = required.difference(tab.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


def parse_passes(s: str) -> tuple[int, int]:
    left, right = str(s).split("/")
    return int(left), int(right)


rows = []
component_rows = []

for _, r in tab.iterrows():
    dataset = str(r["Dataset"])
    passed, total = parse_passes(r["Primary checks"])

    ric = float(r["Restricted ICI change"])
    refi = float(r["Reference ICI change"])
    rll = float(r["Restricted log-loss change"])
    refll = float(r["Reference log-loss change"])

    point_conditions = {
        "restricted_ici_improves": ric < 0,
        "reference_ici_worsens": refi > 0,
        "restricted_logloss_improves": rll < 0,
        "reference_logloss_worsens": refll > 0,
    }

    for name, value in point_conditions.items():
        component_rows.append({
            "dataset": dataset,
            "component": name,
            "point_direction_passes": bool(value),
        })

    point_signs = sum(point_conditions.values())

    for k in range(1, 5):
        rows.append({
            "dataset": dataset,
            "rule_family": "k_of_4_bootstrap_checks",
            "threshold_k": k,
            "bootstrap_checks_passed": passed,
            "classification_positive": passed >= k,
        })

    rows.append({
        "dataset": dataset,
        "rule_family": "all_four_point_estimate_directions",
        "threshold_k": 4,
        "bootstrap_checks_passed": passed,
        "classification_positive": point_signs == 4,
    })
    rows.append({
        "dataset": dataset,
        "rule_family": "three_of_four_point_estimate_directions",
        "threshold_k": 3,
        "bootstrap_checks_passed": passed,
        "classification_positive": point_signs >= 3,
    })

rule_df = pd.DataFrame(rows)
comp_df = pd.DataFrame(component_rows)

rule_df.to_csv(OUT / "h7_rule_threshold_robustness.csv", index=False)
comp_df.to_csv(OUT / "h7_point_direction_components.csv", index=False)

# Dataset-level summary.
summary_rows = []
for _, r in tab.iterrows():
    dataset = str(r["Dataset"])
    passed, total = parse_passes(r["Primary checks"])

    point_flags = [
        float(r["Restricted ICI change"]) < 0,
        float(r["Reference ICI change"]) > 0,
        float(r["Restricted log-loss change"]) < 0,
        float(r["Reference log-loss change"]) > 0,
    ]
    summary_rows.append({
        "dataset": dataset,
        "bootstrap_checks_passed": passed,
        "bootstrap_checks_total": total,
        "point_directions_aligned": int(sum(point_flags)),
        "positive_under_any_k_1_to_4_bootstrap_rule": bool(passed >= 1),
        "positive_under_all_four_bootstrap_rule": bool(passed == 4),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "h7_rule_robustness_summary.csv", index=False)

# Verify the central robustness statement.
support = summary.loc[summary["dataset"].eq("SUPPORT2")].iloc[0]
mimic = summary.loc[summary["dataset"].eq("MIMIC-IV")].iloc[0]

classification_invariant_k1_to_k4 = (
    int(support["bootstrap_checks_passed"]) == 0
    and int(mimic["bootstrap_checks_passed"]) == 4
)

# Read Phase 8A metadata if available.
meta_path = OUT / "phase8a_metadata.json"
metadata = {}
if meta_path.exists():
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

visible_commits = metadata.get("total_git_commits_visible", None)

interpretation = {
    "classification_invariant_for_bootstrap_thresholds_k_1_to_4":
        bool(classification_invariant_k1_to_k4),
    "support2_bootstrap_checks": "0/4",
    "mimic_bootstrap_checks": "4/4",
    "support2_point_estimate_directions": "3/4",
    "mimic_point_estimate_directions": "4/4",
    "four_part_rule_interpretation": (
        "The released rule is best interpreted as a conjunction, not a vote: "
        "restricted ICI and restricted log loss must both improve, while linked "
        "ICI and linked log loss must both worsen. A 3-of-4 point-sign rule would "
        "label SUPPORT2 despite its restricted ICI worsening, which is not the "
        "target-mismatch pattern being defined."
    ),
    "provenance_interpretation": (
        "The public repository contains a single consolidated release commit. "
        "It verifies the exact released configuration and rule, but does not "
        "independently establish when the authors first inspected linked-outcome "
        "results. No new experiment can retroactively prove that chronology; "
        "only contemporaneous dated records could."
    ),
    "visible_git_commits": visible_commits,
}

(OUT / "phase8a_addendum_interpretation.json").write_text(
    json.dumps(interpretation, indent=2), encoding="utf-8"
)

validation = pd.DataFrame([
    {
        "check": "support2_has_zero_of_four_bootstrap_checks",
        "passed": int(support["bootstrap_checks_passed"]) == 0,
    },
    {
        "check": "mimic_has_four_of_four_bootstrap_checks",
        "passed": int(mimic["bootstrap_checks_passed"]) == 4,
    },
    {
        "check": "classification_invariant_for_k_1_to_4_bootstrap_thresholds",
        "passed": classification_invariant_k1_to_k4,
    },
    {
        "check": "support2_three_of_four_point_directions",
        "passed": int(support["point_directions_aligned"]) == 3,
    },
    {
        "check": "mimic_four_of_four_point_directions",
        "passed": int(mimic["point_directions_aligned"]) == 4,
    },
])

validation.to_csv(OUT / "phase8a_addendum_validation.csv", index=False)

status = (
    "phase8a_rule_robustness_verified"
    if validation["passed"].all()
    else "phase8a_rule_robustness_requires_review"
)
(OUT / "phase8a_addendum_status.txt").write_text(status + "\n", encoding="utf-8")

print(status)
print("\n===== RULE ROBUSTNESS SUMMARY =====")
print(summary.to_string(index=False))
print("\n===== K-OF-4 ROBUSTNESS =====")
print(rule_df.to_string(index=False))
print("\n===== INTERPRETATION =====")
print(json.dumps(interpretation, indent=2))
print("\n===== VALIDATION =====")
print(validation.to_string(index=False))
