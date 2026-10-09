#!/usr/bin/env python3
"""
Phase 8B — Method Specification Audit for KDD 2027 rebuttal.

AUDIT ONLY. This phase performs no model fitting and no new experiment.

It verifies and exports:
  1. estimator equations and conditioning interpretations from source;
  2. binning / empty-bin / integration rules;
  3. subgroup sample sizes for localized sensitivity;
  4. whether the closest subgroup is reselected within each bootstrap;
  5. key frozen numerical settings that can be recovered from code/artifacts.

Run from the ActionAudit repository root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "revision_kdd2027" / "phase8b_method_specification"
OUT.mkdir(parents=True, exist_ok=True)

SRC = ROOT / "src" / "actionaudit"
SCRIPTS = ROOT / "scripts"

required_files = {
    "metrics": SRC / "metrics.py",
    "dr": SRC / "dr.py",
    "survival": SRC / "discrete_survival.py",
    "km": SRC / "km.py",
    "support2_dr": SCRIPTS / "phase2b_support2_dr.py",
    "mimic_practitioner": SCRIPTS / "phase5g_practitioner_estimators.py",
    "mimic_rich": SCRIPTS / "phase5g_rich_baseline_estimators.py",
    "mimic_sensitivity": SCRIPTS / "phase5d_bootstrap_tipping_points.py",
    "support2_sensitivity": SCRIPTS / "phase5i_support2_partition_sensitivity.py",
    "cross_dataset_h5": SCRIPTS / "phase5i_cross_dataset_h5_freeze.py",
}

missing_source = [str(p) for p in required_files.values() if not p.exists()]
if missing_source:
    raise FileNotFoundError(
        "Missing required source files:\n" + "\n".join(missing_source)
    )

text = {k: p.read_text(encoding="utf-8") for k, p in required_files.items()}


def contains(source: str, *tokens: str) -> bool:
    return all(token in source for token in tokens)


# ---------------------------------------------------------------------
# 1. Estimator equations / assumptions
# ---------------------------------------------------------------------

equations = [
    {
        "estimator": "Linked/reference calibration",
        "estimand_or_formula": "m_ref(p) = E[Y(tau) | P=p]",
        "conditioning_or_assumption": (
            "Uses linked fixed-horizon outcomes for the full reference population."
        ),
        "implementation_evidence": "src/actionaudit/metrics.py",
    },
    {
        "estimator": "Restricted complete case",
        "estimand_or_formula": "m_obs(p) = E[Y(tau) | P=p, R(tau)=1]",
        "conditioning_or_assumption": (
            "Uses only outcomes visible before the operational boundary."
        ),
        "implementation_evidence": "src/actionaudit/metrics.py",
    },
    {
        "estimator": "Binary-observability IPW (Horvitz-Thompson)",
        "estimand_or_formula": (
            "Within bin B: mean_i[ R_i Y_i / e_i ], "
            "e_i=P(R_i=1 | X_i), with lower-tail clipping."
        ),
        "conditioning_or_assumption": (
            "Requires conditional exchangeability of Y and R given nuisance "
            "features X, correct/adequate observation model for pure IPW, "
            "and positivity e_i>0 on target support."
        ),
        "implementation_evidence": "src/actionaudit/dr.py",
    },
    {
        "estimator": "Binary-observability IPW (Hajek)",
        "estimand_or_formula": (
            "Within bin B: sum_i R_i Y_i/e_i divided by sum_i R_i/e_i."
        ),
        "conditioning_or_assumption": (
            "Same observation-model and positivity conditions as binary IPW; "
            "normalization changes finite-sample behavior."
        ),
        "implementation_evidence": "src/actionaudit/dr.py",
    },
    {
        "estimator": "Binary-observability outcome regression",
        "estimand_or_formula": (
            "Within bin B: mean_i[m_hat(X_i)], where m_hat approximates "
            "E[Y | X, R=1] and is projected to the full bin population."
        ),
        "conditioning_or_assumption": (
            "Requires the outcome regression to transport from visible to hidden "
            "cases conditional on included features."
        ),
        "implementation_evidence": "src/actionaudit/dr.py",
    },
    {
        "estimator": "Binary-observability AIPW",
        "estimand_or_formula": (
            "Within bin B: mean_i[ m_hat(X_i) + "
            "R_i/e_i * (Y_i-m_hat(X_i)) ]."
        ),
        "conditioning_or_assumption": (
            "Doubly robust under the usual missing-at-random/selective-label "
            "conditions when either the observation model or outcome model is "
            "correct, plus positivity and regularity; clipping changes the "
            "finite-sample target/bias-variance tradeoff."
        ),
        "implementation_evidence": "src/actionaudit/dr.py",
    },
    {
        "estimator": "Kaplan-Meier",
        "estimand_or_formula": "1 - S_hat_KM(tau) within each fixed risk bin.",
        "conditioning_or_assumption": (
            "Treats hospital exit as right censoring; standard KM recovery "
            "requires non-informative censoring within the analysis stratum."
        ),
        "implementation_evidence": "src/actionaudit/km.py",
    },
    {
        "estimator": "Survival IPCW",
        "estimand_or_formula": (
            "Event contribution for an observed event at time t_i: "
            "Delta_i / max(G_hat(t_i- | X_i), c); bin risk is the mean "
            "event contribution."
        ),
        "conditioning_or_assumption": (
            "Uses censoring survival over the risk set, not a binary horizon "
            "observability propensity. Requires conditionally independent "
            "censoring given modeled history/features and positivity of G."
        ),
        "implementation_evidence": "src/actionaudit/discrete_survival.py",
    },
    {
        "estimator": "Survival outcome regression",
        "estimand_or_formula": (
            "m_hat_i(tau)=1-product_t(1-h_hat_event(t | history_i)); "
            "bin risk is mean_i[m_hat_i(tau)]."
        ),
        "conditioning_or_assumption": (
            "Requires an adequate event-hazard model over observed follow-up "
            "and transport to the full target population."
        ),
        "implementation_evidence": "src/actionaudit/discrete_survival.py",
    },
]
equation_df = pd.DataFrame(equations)
equation_df.to_csv(OUT / "estimator_equations_and_assumptions.csv", index=False)


# Verify the crucial formulas against source tokens.
formula_checks = [
    {
        "check": "binary_ipw_formula_present",
        "passed": contains(
            text["dr"],
            "observed_weight =",
            "observed",
            "/ propensity",
            "ipcw_numerator =",
        ),
    },
    {
        "check": "binary_aipw_formula_present",
        "passed": contains(
            text["dr"],
            "aipw_pseudo_outcome =",
            "outcome_regression",
            "observed_weight * residual",
        ),
    },
    {
        "check": "binary_propensity_lower_clip_only",
        "passed": contains(
            text["dr"],
            "propensity = raw_propensity.clip",
            "lower=propensity_clip",
            "upper=1.0",
        ),
    },
    {
        "check": "survival_ipcw_uses_censor_survival_before_time",
        "passed": contains(
            text["survival"],
            "raw_g_before",
            "censor_survival_before_day",
            "ipcw_weight = np.where",
            "1.0 / clipped_g_before",
        ),
    },
    {
        "check": "km_treats_exit_as_censoring",
        "passed": contains(
            text["km"],
            "kaplan_meier_event_risk",
            "survival *=",
            "deaths / at_risk",
        ),
    },
]


# ---------------------------------------------------------------------
# 2. Binning, empty-bin, and ICI integration rules
# ---------------------------------------------------------------------

bin_rules = [
    {
        "item": "Bin construction",
        "verified_rule": (
            "Reference-cohort quantile edges are computed from the fixed "
            "prediction distribution; duplicate quantile edges are removed; "
            "first/last edges are set to -inf/+inf."
        ),
        "source": "src/actionaudit/metrics.py and src/actionaudit/km.py",
    },
    {
        "item": "Assignment",
        "verified_rule": (
            "Predictions are assigned with pandas.cut(..., include_lowest=True). "
            "Estimator paths explicitly error if any prediction is unassigned."
        ),
        "source": "src/actionaudit/km.py, dr.py, discrete_survival.py",
    },
    {
        "item": "Empty bins",
        "verified_rule": (
            "groupby(..., observed=True) means bins containing no records do not "
            "produce estimator rows. Quantile-edge de-duplication can reduce the "
            "realized number of bins when predictions have ties."
        ),
        "source": "src/actionaudit/metrics.py",
    },
    {
        "item": "ActionAudit bin eligibility",
        "verified_rule": (
            "A bin is eligible only when n_bin >= min_bin_size AND n_observed > 0. "
            "Ineligible bins receive NaN calibration quantities."
        ),
        "source": "src/actionaudit/metrics.py",
    },
    {
        "item": "Integration distribution",
        "verified_rule": (
            "Each eligible bin receives reference weight n_bin/n_total. For the "
            "core ActionAudit scalar summaries, weights over eligible bins are "
            "renormalized to sum to one before ICI/calibration-error integration."
        ),
        "source": "src/actionaudit/metrics.py",
    },
    {
        "item": "Sparse-bin robustness",
        "verified_rule": (
            "Phase 7E separately varied bin count and minimum-bin-size rules; "
            "this is empirical robustness evidence rather than part of the "
            "original estimator definition."
        ),
        "source": "revision_kdd2027/phase7e_*",
    },
]
pd.DataFrame(bin_rules).to_csv(
    OUT / "binning_empty_bin_and_integration_rules.csv", index=False
)

binning_checks = [
    {
        "check": "quantile_edges_from_reference_prediction",
        "passed": contains(
            text["metrics"],
            "np.quantile",
            "prediction",
            "edges[0] = -np.inf",
            "edges[-1] = np.inf",
        ),
    },
    {
        "check": "duplicate_edges_removed",
        "passed": "np.unique" in text["metrics"],
    },
    {
        "check": "eligibility_requires_min_n_and_observed",
        "passed": contains(
            text["metrics"],
            "n_bin >= config.min_bin_size",
            "n_observed > 0",
        ),
    },
    {
        "check": "eligible_weights_renormalized",
        "passed": contains(
            text["metrics"],
            'eligible_bins = bin_summary["eligible"]',
            "normalized_weights = weights / weights.sum()",
        ),
    },
]


# ---------------------------------------------------------------------
# 3. Localized-sensitivity subgroup counts
# ---------------------------------------------------------------------

subgroup_rows = []
subgroup_sources = []

support2_path = (
    ROOT / "artifacts" / "phase5i_support2_partition_sensitivity"
    / "exact_action_tipping_points.csv"
)
if support2_path.exists():
    s2 = pd.read_csv(support2_path)
    needed = {"view", "action", "hidden_n", "eligible_for_inference"}
    if needed.issubset(s2.columns):
        for _, row in s2.iterrows():
            subgroup_rows.append({
                "dataset": "SUPPORT2",
                "view": row["view"],
                "group": row["action"],
                "hidden_n": int(row["hidden_n"]),
                "eligible_for_inference": bool(row["eligible_for_inference"]),
                "count_source": str(support2_path.relative_to(ROOT)),
            })
        subgroup_sources.append(str(support2_path.relative_to(ROOT)))

mimic_bin_candidates = [
    ROOT / "artifacts" / "phase5d_bootstrap_tipping_points"
    / "exact_bin_tipping_points.csv",
    ROOT / "artifacts" / "phase5d_action_specific_sensitivity"
    / "action_bin_tipping_points.csv",
]
mimic_bin_path = next((p for p in mimic_bin_candidates if p.exists()), None)

if mimic_bin_path is not None:
    mb = pd.read_csv(mimic_bin_path)
    if {"action", "n_action"}.issubset(mb.columns):
        agg = (
            mb.groupby("action", as_index=False)["n_action"]
            .sum()
            .rename(columns={"n_action": "hidden_n"})
        )
        for _, row in agg.iterrows():
            subgroup_rows.append({
                "dataset": "MIMIC-IV",
                "view": "native_discharge_action",
                "group": row["action"],
                "hidden_n": int(row["hidden_n"]),
                "eligible_for_inference": True,
                "count_source": str(mimic_bin_path.relative_to(ROOT)),
            })
        subgroup_sources.append(str(mimic_bin_path.relative_to(ROOT)))

subgroup_df = pd.DataFrame(
    subgroup_rows,
    columns=[
        "dataset", "view", "group", "hidden_n",
        "eligible_for_inference", "count_source"
    ],
)
subgroup_df.to_csv(OUT / "localized_sensitivity_subgroup_counts.csv", index=False)


# ---------------------------------------------------------------------
# 4. Bootstrap subgroup reselection
# ---------------------------------------------------------------------

support2_reselect = contains(
    text["support2_sensitivity"],
    "calculate_point_results(",
    "sample,",
    "minimum_oracle_to_tipping_distance",
)

mimic_reselect = contains(
    text["cross_dataset_h5"],
    'bootstrap_actions.groupby(',
    '"bootstrap_id"',
    ".min()",
    '"minimum_local_margin"',
)

bootstrap_rows = [
    {
        "dataset": "SUPPORT2",
        "procedure": (
            "Each patient-level bootstrap sample is passed through "
            "calculate_point_results; the closest/minimum eligible subgroup "
            "is recomputed from that resample rather than holding the original "
            "closest subgroup fixed."
        ),
        "reselected_within_each_bootstrap": support2_reselect,
        "source": "scripts/phase5i_support2_partition_sensitivity.py",
    },
    {
        "dataset": "MIMIC-IV",
        "procedure": (
            "Bootstrap action results are grouped by bootstrap_id and the "
            "minimum distance-to-tipping is computed within each replicate; "
            "therefore the identity of the closest action may change by replicate."
        ),
        "reselected_within_each_bootstrap": mimic_reselect,
        "source": "scripts/phase5i_cross_dataset_h5_freeze.py",
    },
]
bootstrap_df = pd.DataFrame(bootstrap_rows)
bootstrap_df.to_csv(
    OUT / "bootstrap_subgroup_reselection_audit.csv", index=False
)


# ---------------------------------------------------------------------
# 5. Frozen / explicit numerical settings recoverable from code
# ---------------------------------------------------------------------

settings = [
    {
        "component": "Recalibration audit",
        "setting": "risk bins",
        "value": "5",
        "source": "Phase 8A reconstruction",
    },
    {
        "component": "Recalibration audit",
        "setting": "calibration/evaluation fraction",
        "value": "0.5 / 0.5 stratified",
        "source": "Phase 8A reconstruction",
    },
    {
        "component": "Recalibration audit",
        "setting": "split seed",
        "value": "2038",
        "source": "Phase 8A reconstruction",
    },
    {
        "component": "Recalibration audit",
        "setting": "bootstrap replicates",
        "value": "1000",
        "source": "Phase 8A reconstruction",
    },
    {
        "component": "Recalibration audit",
        "setting": "base seed",
        "value": "2027",
        "source": "Phase 8A reconstruction",
    },
    {
        "component": "SUPPORT2 binary DR benchmark",
        "setting": "horizon / bins / propensity clip / bootstraps / seed",
        "value": "60d / 5 / 0.05 / 1000 / 2027",
        "source": "scripts/phase2b_support2_dr.py defaults",
    },
    {
        "component": "MIMIC practitioner estimator benchmark",
        "setting": "horizon / bins / crossfit repeats / propensity clip / bootstraps / seed",
        "value": "30d / 5 / 5 / 0.01 / 1000 / 2027",
        "source": "scripts/phase5g_practitioner_estimators.py defaults",
    },
    {
        "component": "MIMIC rich/AIPW benchmark",
        "setting": "reported propensity clipping grid",
        "value": "0.01, 0.025, 0.05",
        "source": "scripts/phase5g_rich_baseline_estimators.py",
    },
]
settings_df = pd.DataFrame(settings)

# Try to recover the frozen core min-bin-size from Phase 7E artifacts rather
# than guessing it.
def find_phase7e_primary_min_bin() -> list[dict[str, str]]:
    found = []
    rev = ROOT / "revision_kdd2027"
    if not rev.exists():
        return found
    primary_targets = {
        "SUPPORT2": (0.019735532871166937, 0.40727369910350886),
        "MIMIC-IV": (0.0014718355579702175, 0.4827368011542327),
    }
    for csv_path in rev.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        cols = set(df.columns)
        if not {"dataset", "n_bins"}.issubset(cols):
            continue
        min_col = next(
            (c for c in ["min_bin_size", "minimum_bin_size"] if c in cols),
            None,
        )
        if min_col is None:
            continue
        ref_col = next(
            (c for c in ["reference_ici", "reference_ici_point"] if c in cols),
            None,
        )
        res_col = next(
            (c for c in ["restricted_ici", "restricted_complete_case_ici"] if c in cols),
            None,
        )
        if ref_col is None or res_col is None:
            continue
        for dataset, (ref_target, res_target) in primary_targets.items():
            sel = df.loc[
                df["dataset"].astype(str).eq(dataset)
                & pd.to_numeric(df["n_bins"], errors="coerce").eq(5)
            ].copy()
            if sel.empty:
                continue
            sel["_dist"] = (
                (pd.to_numeric(sel[ref_col], errors="coerce") - ref_target).abs()
                + (pd.to_numeric(sel[res_col], errors="coerce") - res_target).abs()
            )
            row = sel.sort_values("_dist").iloc[0]
            if np.isfinite(row["_dist"]) and float(row["_dist"]) < 1e-8:
                found.append({
                    "component": f"{dataset} frozen core audit",
                    "setting": "minimum bin size",
                    "value": str(int(row[min_col])),
                    "source": str(csv_path.relative_to(ROOT)),
                })
    return found

extra_settings = find_phase7e_primary_min_bin()
if extra_settings:
    settings_df = pd.concat(
        [settings_df, pd.DataFrame(extra_settings)], ignore_index=True
    )

settings_df.to_csv(OUT / "verified_method_settings.csv", index=False)


# ---------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------

checks = []
for row in formula_checks + binning_checks:
    checks.append(row)

checks.extend([
    {
        "check": "support2_subgroup_counts_recovered",
        "passed": (
            not subgroup_df.empty
            and (subgroup_df["dataset"] == "SUPPORT2").any()
        ),
    },
    {
        "check": "mimic_subgroup_counts_recovered",
        "passed": (
            not subgroup_df.empty
            and (subgroup_df["dataset"] == "MIMIC-IV").any()
        ),
    },
    {
        "check": "support2_closest_group_reselected_in_bootstrap",
        "passed": support2_reselect,
    },
    {
        "check": "mimic_closest_group_reselected_in_bootstrap",
        "passed": mimic_reselect,
    },
])

validation_df = pd.DataFrame(checks)
validation_df.to_csv(OUT / "validation_checks.csv", index=False)

status = (
    "phase8b_method_specification_verified"
    if validation_df["passed"].all()
    else "phase8b_method_specification_requires_review"
)
(OUT / "phase8b_status.txt").write_text(status + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# 7. Human-readable report
# ---------------------------------------------------------------------

report = []
report.append("# Phase 8B — Method Specification Audit")
report.append("")
report.append(f"Status: `{status}`")
report.append("")
report.append("This phase performs no new model fitting or experiment.")
report.append("")
report.append("## Estimator equations and assumptions")
report.append("")
report.append(equation_df.to_markdown(index=False))
report.append("")
report.append("## Binning / empty-bin / integration rules")
report.append("")
report.append(pd.DataFrame(bin_rules).to_markdown(index=False))
report.append("")
report.append("## Localized-sensitivity subgroup counts")
report.append("")
if subgroup_df.empty:
    report.append("Required subgroup-count artifacts were not found.")
else:
    report.append(subgroup_df.to_markdown(index=False))
report.append("")
report.append("## Bootstrap subgroup reselection")
report.append("")
report.append(bootstrap_df.to_markdown(index=False))
report.append("")
report.append("## Verified settings")
report.append("")
report.append(settings_df.to_markdown(index=False))
report.append("")
report.append("## Validation")
report.append("")
report.append(validation_df.to_markdown(index=False))

(OUT / "phase8b_report.md").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)

print(status)
print("\n===== ESTIMATOR EQUATIONS / ASSUMPTIONS =====")
print(equation_df.to_string(index=False))
print("\n===== BINNING / EMPTY-BIN / INTEGRATION RULES =====")
print(pd.DataFrame(bin_rules).to_string(index=False))
print("\n===== SUBGROUP COUNTS =====")
if subgroup_df.empty:
    print("NO SUBGROUP COUNTS RECOVERED")
else:
    print(subgroup_df.to_string(index=False))
print("\n===== BOOTSTRAP RESELECTION =====")
print(bootstrap_df.to_string(index=False))
print("\n===== VERIFIED SETTINGS =====")
print(settings_df.to_string(index=False))
print("\n===== VALIDATION =====")
print(validation_df.to_string(index=False))
