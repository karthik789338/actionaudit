#!/usr/bin/env python3
"""
Phase 8B repair — recover localized-sensitivity subgroup counts directly
from the frozen analysis inputs.

AUDIT ONLY. No model fitting.

This repairs only the two failed subgroup-count checks from Phase 8B.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "revision_kdd2027" / "phase8b_method_specification"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

# ------------------------------------------------------------------
# SUPPORT2: exact hidden_n definition used in Phase 5I
# ------------------------------------------------------------------
s2_path = (
    ROOT / "artifacts" / "phase5h_support2_definitions"
    / "support2_harmonized_test.csv.gz"
)
if not s2_path.exists():
    raise FileNotFoundError(
        "Missing frozen SUPPORT2 sensitivity input: "
        f"{s2_path}"
    )

s2 = pd.read_csv(s2_path, low_memory=False)

views = {
    "original_boundary_action": "boundary_action",
    "dnr_policy_partition": "dnr_partition_action",
    "disease_partition": "disease_partition_action",
    "hospital_timing_partition": "timing_partition_action",
}

required_s2 = {"observed", *views.values()}
missing_s2 = required_s2.difference(s2.columns)
if missing_s2:
    raise ValueError(f"SUPPORT2 missing columns: {sorted(missing_s2)}")

for view, col in views.items():
    hidden = s2.loc[s2["observed"].astype(int).eq(0)].copy()
    counts = hidden.groupby(col, dropna=False).size().reset_index(name="hidden_n")
    for _, r in counts.iterrows():
        rows.append({
            "dataset": "SUPPORT2",
            "view": view,
            "group": str(r[col]),
            "hidden_n": int(r["hidden_n"]),
            "eligible_for_inference": int(r["hidden_n"]) >= 20,
            "count_source": str(s2_path.relative_to(ROOT)),
        })

# ------------------------------------------------------------------
# MIMIC-IV: exact hidden action count definition used by Phase 5D.
# Sum over observed==0 and analysis_action.
# ------------------------------------------------------------------
mimic_path = (
    ROOT / "artifacts" / "phase5c_admission_model"
    / "test_predictions.csv.gz"
)
if not mimic_path.exists():
    raise FileNotFoundError(
        "Missing frozen MIMIC sensitivity input: "
        f"{mimic_path}"
    )

mimic = pd.read_csv(mimic_path, low_memory=False)

required_m = {"observed", "analysis_action"}
missing_m = required_m.difference(mimic.columns)
if missing_m:
    raise ValueError(f"MIMIC missing columns: {sorted(missing_m)}")

hidden_m = mimic.loc[mimic["observed"].astype(int).eq(0)].copy()
m_counts = (
    hidden_m.groupby("analysis_action", dropna=False)
    .size()
    .reset_index(name="hidden_n")
)

for _, r in m_counts.iterrows():
    rows.append({
        "dataset": "MIMIC-IV",
        "view": "native_discharge_action",
        "group": str(r["analysis_action"]),
        "hidden_n": int(r["hidden_n"]),
        # Phase 5D does not use the SUPPORT2 >=20 eligibility gate;
        # retain all native actions present in the frozen hidden cohort.
        "eligible_for_inference": True,
        "count_source": str(mimic_path.relative_to(ROOT)),
    })

df = pd.DataFrame(rows)
df = df.sort_values(["dataset", "view", "hidden_n", "group"],
                    ascending=[True, True, False, True]).reset_index(drop=True)
df.to_csv(OUT / "localized_sensitivity_subgroup_counts.csv", index=False)

validation = pd.DataFrame([
    {
        "check": "support2_subgroup_counts_recovered",
        "passed": bool((df["dataset"] == "SUPPORT2").any()),
    },
    {
        "check": "mimic_subgroup_counts_recovered",
        "passed": bool((df["dataset"] == "MIMIC-IV").any()),
    },
    {
        "check": "support2_hidden_total_original_boundary_equals_1270",
        "passed": int(
            df.loc[
                (df["dataset"] == "SUPPORT2")
                & (df["view"] == "original_boundary_action"),
                "hidden_n",
            ].sum()
        ) == 1270,
    },
    {
        "check": "mimic_hidden_total_equals_28587",
        "passed": int(
            df.loc[df["dataset"] == "MIMIC-IV", "hidden_n"].sum()
        ) == 28587,
    },
])

validation.to_csv(OUT / "subgroup_count_repair_validation.csv", index=False)

status = (
    "phase8b_subgroup_counts_recovered"
    if validation["passed"].all()
    else "phase8b_subgroup_counts_require_review"
)
(OUT / "phase8b_subgroup_repair_status.txt").write_text(
    status + "\n", encoding="utf-8"
)

print(status)
print("\n===== SUBGROUP COUNTS =====")
print(df.to_string(index=False))
print("\n===== VALIDATION =====")
print(validation.to_string(index=False))
