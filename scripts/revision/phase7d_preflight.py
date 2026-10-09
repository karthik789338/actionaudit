# scripts/revision/phase7d_preflight.py

from pathlib import Path
import pandas as pd
import importlib
import inspect


ROOT = Path(".")
ARTIFACTS = ROOT / "artifacts"


def show_csv(path):
    print("\n" + "=" * 90)
    print(path)
    print("=" * 90)

    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:
        print("READ ERROR:", repr(e))
        return

    print("columns:")
    for c in df.columns:
        print(" ", c)

    print("\nfirst rows:")
    print(df.to_string(index=False))


print("=" * 90)
print("PHASE 7D PREFLIGHT")
print("=" * 90)

# ------------------------------------------------------------------
# 1. Locate likely frozen cohorts / predictions
# ------------------------------------------------------------------

patterns = [
    "*support*.csv",
    "*support*.csv.gz",
    "*cohort*.csv",
    "*cohort*.csv.gz",
    "*prediction*.csv",
    "*prediction*.csv.gz",
]

found = []

for pattern in patterns:
    for p in ARTIFACTS.rglob(pattern):
        if p.is_file():
            found.append(p)

found = sorted(set(found))

print("\nCandidate artifact files:")
for p in found:
    print(p)

# ------------------------------------------------------------------
# 2. Inspect especially relevant files
# ------------------------------------------------------------------

priority_strings = [
    "phase5c",
    "support",
    "prediction",
    "cohort",
]

priority = []

for p in found:
    s = str(p).lower()
    if any(x in s for x in priority_strings):
        priority.append(p)

print("\n\nInspecting candidate CSV schemas...")

for p in priority[:30]:
    show_csv(p)

# ------------------------------------------------------------------
# 3. Show ActionAudit package files
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("ACTIONAUDIT SOURCE FILES")
print("=" * 90)

src = ROOT / "src" / "actionaudit"

if src.exists():
    for p in sorted(src.glob("*.py")):
        print(p)
else:
    pkg = ROOT / "actionaudit"
    if pkg.exists():
        for p in sorted(pkg.glob("*.py")):
            print(p)
    else:
        print("Could not locate actionaudit package directory.")

# ------------------------------------------------------------------
# 4. Search likely calibration functions
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("CALIBRATION FUNCTION DISCOVERY")
print("=" * 90)

candidate_modules = [
    "actionaudit.calibration",
    "actionaudit.metrics",
    "actionaudit.evaluation",
    "actionaudit.discrete_survival",
]

for module_name in candidate_modules:
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"\n{module_name}: unavailable ({e})")
        continue

    print(f"\nMODULE: {module_name}")

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            lname = name.lower()

            if any(
                key in lname
                for key in [
                    "calib",
                    "ici",
                    "bin",
                    "integrat",
                    "metric",
                ]
            ):
                try:
                    sig = inspect.signature(obj)
                except Exception:
                    sig = "(signature unavailable)"

                print(f"  {name}{sig}")

# ------------------------------------------------------------------
# 5. Explicitly inspect known MIMIC artifacts if present
# ------------------------------------------------------------------

known = [
    Path(
        "artifacts/phase5c_frozen_cohort/"
        "primary_inpatient_cohort.csv.gz"
    ),
    Path(
        "artifacts/phase5c_admission_model/"
        "test_predictions.csv.gz"
    ),
]

print("\n" + "=" * 90)
print("KNOWN MIMIC INPUTS")
print("=" * 90)

for p in known:
    if p.exists():
        show_csv(p)
    else:
        print("MISSING:", p)

print("\nPHASE 7D PREFLIGHT: COMPLETE")
