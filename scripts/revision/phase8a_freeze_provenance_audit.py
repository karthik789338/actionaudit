#!/usr/bin/env python3
"""
Phase 8A — Freeze and provenance audit for KDD 2027 rebuttal.

This phase performs NO new experiment. It reconstructs and records:
  1. exact frozen risk-bin boundaries;
  2. exact calibration/evaluation split sizes;
  3. exact recalibration decision rule;
  4. Git provenance for the rule and relevant files;
  5. what chronology Git can and cannot establish.

Run from the ActionAudit repository root.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SCRIPTS = ROOT / "scripts"
if not SCRIPTS.exists():
    raise SystemExit("Run this script from the ActionAudit repository root.")

sys.path.insert(0, str(SCRIPTS))

import phase5j_realdata_recalibration_fragility as recal  # noqa: E402


OUT = ROOT / "revision_kdd2027" / "phase8a_freeze_provenance"
OUT.mkdir(parents=True, exist_ok=True)


def run_git(args: list[str]) -> str:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return p.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def first_commit_for_path(path: str) -> dict[str, str]:
    txt = run_git([
        "log", "--all", "--follow", "--reverse",
        "--format=%H|%cI|%s", "--", path
    ])
    line = txt.splitlines()[0] if txt else ""
    if not line:
        return {"commit": "", "date": "", "subject": ""}
    parts = line.split("|", 2)
    while len(parts) < 3:
        parts.append("")
    return {"commit": parts[0], "date": parts[1], "subject": parts[2]}


def first_commit_containing(search_text: str, path: str) -> dict[str, str]:
    txt = run_git([
        "log", "--all", "--reverse",
        "-S", search_text,
        "--format=%H|%cI|%s", "--", path
    ])
    line = txt.splitlines()[0] if txt else ""
    if not line:
        return {"commit": "", "date": "", "subject": ""}
    parts = line.split("|", 2)
    while len(parts) < 3:
        parts.append("")
    return {"commit": parts[0], "date": parts[1], "subject": parts[2]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_dataset(dataset: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cfg = recal.DATASET_CONFIG[dataset]
    input_path = ROOT / cfg["default_input"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing required frozen input for {dataset}: {input_path}"
        )

    raw = pd.read_csv(input_path, low_memory=False)

    fixed, edges, pop_hash = recal.prepare_dataset(
        dataset=dataset,
        raw=raw,
        n_bins=5,
        seed=2027,
    )

    split = recal.assign_analysis_split(
        fixed,
        calibration_fraction=0.5,
        seed=2027 + 11,
    )

    counts = (
        split.groupby(["analysis_split", "outcome"])
        .size()
        .rename("n")
        .reset_index()
    )

    totals = (
        split.groupby("analysis_split")
        .size()
        .rename("n")
        .reset_index()
    )
    totals["outcome"] = "ALL"

    split_counts = pd.concat(
        [
            totals[["analysis_split", "outcome", "n"]],
            counts[["analysis_split", "outcome", "n"]],
        ],
        ignore_index=True,
    )
    split_counts.insert(0, "dataset", cfg["display_name"])

    edge_rows = []
    for i, edge in enumerate(edges):
        edge_rows.append({
            "dataset": cfg["display_name"],
            "edge_index": i,
            "edge": float(edge),
            "edge_display": (
                "-inf" if np.isneginf(edge)
                else "+inf" if np.isposinf(edge)
                else format(float(edge), ".17g")
            ),
            "population_hash": pop_hash,
        })

    return split, edges, pd.DataFrame(edge_rows), split_counts


# ----------------------------------------------------------------------
# 1. Exact frozen configuration
# ----------------------------------------------------------------------

all_edges = []
all_splits = []
dataset_rows = []

for dataset in ["support2", "mimic"]:
    split, edges, edge_df, split_df = reconstruct_dataset(dataset)
    all_edges.append(edge_df)
    all_splits.append(split_df)

    cfg = recal.DATASET_CONFIG[dataset]
    dataset_rows.append({
        "dataset": cfg["display_name"],
        "n_total": int(len(split)),
        "n_calibration": int((split["analysis_split"] == "calibration").sum()),
        "n_evaluation": int((split["analysis_split"] == "test").sum()),
        "n_outcome_1_total": int(split["outcome"].sum()),
        "n_bins": 5,
        "calibration_fraction": 0.5,
        "split_seed": 2038,
        "base_seed": 2027,
        "bootstrap_replicates": 1000,
        "minimum_observed_split_n": 20,
    })

edges_df = pd.concat(all_edges, ignore_index=True)
splits_df = pd.concat(all_splits, ignore_index=True)
config_df = pd.DataFrame(dataset_rows)

edges_df.to_csv(OUT / "exact_risk_bin_edges.csv", index=False)
splits_df.to_csv(OUT / "exact_split_counts.csv", index=False)
config_df.to_csv(OUT / "exact_recalibration_configuration.csv", index=False)


# ----------------------------------------------------------------------
# 2. Exact 4-part decision rule from implementation
# ----------------------------------------------------------------------

decision_rows = [
    {
        "check_number": 1,
        "fit_policy": "aggressive_early",
        "method": "logistic",
        "metric": "restricted_ici_change",
        "bootstrap_quantity": "q975",
        "criterion": "q975 < 0",
        "meaning": "restricted ICI improves",
    },
    {
        "check_number": 2,
        "fit_policy": "aggressive_early",
        "method": "logistic",
        "metric": "reference_ici_change",
        "bootstrap_quantity": "q025",
        "criterion": "q025 > 0",
        "meaning": "linked/reference ICI worsens",
    },
    {
        "check_number": 3,
        "fit_policy": "aggressive_early",
        "method": "logistic",
        "metric": "restricted_log_loss_change",
        "bootstrap_quantity": "q975",
        "criterion": "q975 < 0",
        "meaning": "restricted log loss improves",
    },
    {
        "check_number": 4,
        "fit_policy": "aggressive_early",
        "method": "logistic",
        "metric": "reference_log_loss_change",
        "bootstrap_quantity": "q025",
        "criterion": "q025 > 0",
        "meaning": "linked/reference log loss worsens",
    },
]
decision_df = pd.DataFrame(decision_rows)

source_text = inspect.getsource(recal.build_h7_core_checks)
required_tokens = [
    "restricted_ici_change",
    "reference_ici_change",
    "restricted_log_loss_change",
    "reference_log_loss_change",
    'quantile="q975"',
    'quantile="q025"',
]
source_matches = all(token in source_text for token in required_tokens)

decision_df["verified_against_current_source"] = source_matches
decision_df.to_csv(OUT / "exact_h7_decision_rule.csv", index=False)
(OUT / "build_h7_core_checks_source.txt").write_text(
    source_text, encoding="utf-8"
)


# ----------------------------------------------------------------------
# 3. Git provenance
# ----------------------------------------------------------------------

recal_path = "scripts/phase5j_realdata_recalibration_fragility.py"
freeze_path = "scripts/phase5j_h7_mixed_evidence_freeze.py"
table_path = "results/tables/table_06_h7_recalibration.csv"

provenance_rows = []

for label, path in [
    ("recalibration_implementation_file", recal_path),
    ("cross_dataset_freeze_file", freeze_path),
    ("aggregate_recalibration_table", table_path),
]:
    rec = first_commit_for_path(path)
    provenance_rows.append({
        "evidence": label,
        "path_or_search": path,
        **rec,
    })

for label, needle, path in [
    (
        "first_commit_adding_core_decision_function",
        "def build_h7_core_checks",
        recal_path,
    ),
    (
        "first_commit_adding_restricted_ici_check",
        "restricted_ici_change",
        recal_path,
    ),
    (
        "first_commit_adding_reference_logloss_check",
        "reference_log_loss_change",
        recal_path,
    ),
    (
        "first_commit_adding_cross_dataset_rule_use",
        "core_h7_checks_passed",
        freeze_path,
    ),
]:
    rec = first_commit_containing(needle, path)
    provenance_rows.append({
        "evidence": label,
        "path_or_search": f"{path} :: -S {needle}",
        **rec,
    })

prov_df = pd.DataFrame(provenance_rows)
prov_df.to_csv(OUT / "git_provenance.csv", index=False)

head = run_git(["rev-parse", "HEAD"])
head_date = run_git(["show", "-s", "--format=%cI", "HEAD"])
head_subject = run_git(["show", "-s", "--format=%s", "HEAD"])
repo_status = run_git(["status", "--short"])
total_commits_text = run_git(["rev-list", "--all", "--count"])
try:
    total_commits = int(total_commits_text)
except Exception:
    total_commits = -1

source_path = ROOT / recal_path
source_hash = sha256_file(source_path)

# Can Git establish ordering between rule code and checked-in aggregate table?
rule_row = prov_df.loc[
    prov_df["evidence"] == "first_commit_adding_core_decision_function"
]
table_row = prov_df.loc[
    prov_df["evidence"] == "aggregate_recalibration_table"
]

rule_commit = rule_row["commit"].iloc[0] if len(rule_row) else ""
table_commit = table_row["commit"].iloc[0] if len(table_row) else ""

ordering_relation = "unknown"
if rule_commit and table_commit:
    anc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", rule_commit, table_commit],
        cwd=ROOT,
        check=False,
    )
    if anc.returncode == 0:
        ordering_relation = (
            "rule commit is an ancestor of/equal to first table-touch commit"
        )
    else:
        ordering_relation = (
            "Git does not show rule commit as ancestor of first table-touch commit"
        )

# Important limitation: commits cannot prove when a human first inspected
# linked outcomes or locally computed results.
chronology_limitation = (
    "Git history can establish when code/files were committed, but cannot by "
    "itself prove when linked outcomes or locally generated recalibration "
    "results were first inspected. Do not claim such chronology unless an "
    "independent dated artifact/log/commit establishes it."
)

metadata = {
    "phase": "8A",
    "type": "audit_only_no_new_experiment",
    "current_head": head,
    "current_head_date": head_date,
    "current_head_subject": head_subject,
    "total_git_commits_visible": total_commits,
    "working_tree_clean": repo_status == "",
    "working_tree_status": repo_status,
    "recalibration_source_sha256": source_hash,
    "decision_rule_matches_expected_current_source": source_matches,
    "git_rule_vs_table_ordering": ordering_relation,
    "chronology_limitation": chronology_limitation,
}

(OUT / "phase8a_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

validation = []

def add_check(name, value, expected, passed):
    validation.append({
        "check": name,
        "value": value,
        "expected": expected,
        "passed": bool(passed),
    })

for row in dataset_rows:
    expected_n = 1821 if row["dataset"] == "SUPPORT2" else 31012
    add_check(
        f"{row['dataset']}_expected_n",
        row["n_total"],
        expected_n,
        row["n_total"] == expected_n,
    )
    add_check(
        f"{row['dataset']}_split_complete",
        row["n_calibration"] + row["n_evaluation"],
        row["n_total"],
        row["n_calibration"] + row["n_evaluation"] == row["n_total"],
    )

add_check(
    "decision_rule_source_verified",
    source_matches,
    True,
    source_matches,
)
add_check(
    "five_bins_each_dataset",
    int(len(edges_df) / 2 - 1),
    5,
    len(edges_df) == 12,
)

validation_df = pd.DataFrame(validation)
validation_df.to_csv(OUT / "validation_checks.csv", index=False)

configuration_ok = bool(validation_df["passed"].all())

if configuration_ok:
    status = "phase8a_configuration_reconstructed"
else:
    status = "phase8a_requires_review"

(OUT / "phase8a_status.txt").write_text(status + "\n", encoding="utf-8")

# Human-readable report.
lines = []
lines.append("# Phase 8A — Freeze and Provenance Audit")
lines.append("")
lines.append(f"Status: `{status}`")
lines.append("")
lines.append("## Exact frozen configuration")
lines.append("")
lines.append(config_df.to_markdown(index=False))
lines.append("")
lines.append("## Exact risk-bin edges")
lines.append("")
lines.append(edges_df[
    ["dataset", "edge_index", "edge_display", "population_hash"]
].to_markdown(index=False))
lines.append("")
lines.append("## Exact split counts")
lines.append("")
lines.append(splits_df.to_markdown(index=False))
lines.append("")
lines.append("## Four-part recalibration rule")
lines.append("")
lines.append(decision_df[
    ["check_number", "metric", "bootstrap_quantity", "criterion", "meaning"]
].to_markdown(index=False))
lines.append("")
lines.append("## Git provenance")
lines.append("")
lines.append(prov_df.to_markdown(index=False))
lines.append("")
lines.append("## Provenance interpretation")
lines.append("")
lines.append(f"- Current HEAD: `{head}` ({head_date})")
lines.append(f"- Visible Git commits: {total_commits}")
lines.append(f"- Rule/table ordering: {ordering_relation}")
lines.append(f"- {chronology_limitation}")
lines.append("")
lines.append(
    "**Do not state that the configuration was frozen before first inspection "
    "of linked outcomes unless an independent dated record establishes that fact.**"
)

(OUT / "phase8a_report.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

print(status)
print("\nExact configuration:")
print(config_df.to_string(index=False))
print("\nExact risk-bin edges:")
print(edges_df[
    ["dataset", "edge_index", "edge_display"]
].to_string(index=False))
print("\nExact split counts:")
print(splits_df.to_string(index=False))
print("\nGit provenance:")
print(prov_df.to_string(index=False))
print("\nValidation:")
print(validation_df.to_string(index=False))
print("\nIMPORTANT:")
print(chronology_limitation)
