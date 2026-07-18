# ActionAudit

ActionAudit audits fixed-horizon predictive calibration when operational
actions determine whether outcomes remain observable.

## Public release contents

This repository contains:

- source code;
- experiment and validation scripts;
- tests and configuration files;
- aggregate manuscript tables;
- publication figures;
- aggregate figure-source data;
- frozen claim and validation manifests.

## Data availability

No patient-level clinical data are distributed through this repository.

### SUPPORT2

The study used the complete SUPPORT2 source dataset containing 9,105 patients.
The frozen paper evaluation split contained 1,821 patients and used a 60-day
mortality horizon.

Researchers should obtain SUPPORT2 from its authorized source and run the
provided cohort and preprocessing scripts.

### MIMIC-IV

The study constructed a frozen cohort of 155,059 admissions from MIMIC-IV
version 3.1. The test set contained 31,012 admissions and used a 30-day
mortality horizon.

Researchers must independently obtain credentialed MIMIC-IV access through
PhysioNet and accept the applicable data-use agreement.

No MIMIC-IV rows, derived patient-level cohorts, patient-level predictions,
or trained clinical model files are included here.

## Aggregate results

The `results/` directory contains only publication-safe aggregate tables,
figures, figure-source summaries, and validation manifests.

## Reproduction

1. Create the Python environment.
2. Obtain the source datasets through their authorized providers.
3. Follow the dataset-specific preprocessing instructions.
4. Run the phase scripts in numerical order.
5. Run Phase 6A and Phase 6B to regenerate the final claims, tables, and figures.

## Interpretation

The real-data observation-policy experiments are controlled emulations. They
do not estimate causal or historical effects of actual hospital policies.
