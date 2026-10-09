# Phase 8A — Freeze and Provenance Audit

Status: `phase8a_configuration_reconstructed`

## Exact frozen configuration

| dataset   |   n_total |   n_calibration |   n_evaluation |   n_outcome_1_total |   n_bins |   calibration_fraction |   split_seed |   base_seed |   bootstrap_replicates |   minimum_observed_split_n |
|:----------|----------:|----------------:|---------------:|--------------------:|---------:|-----------------------:|-------------:|------------:|-----------------------:|---------------------------:|
| SUPPORT2  |      1821 |             910 |            911 |                 632 |        5 |                    0.5 |         2038 |        2027 |                   1000 |                         20 |
| MIMIC-IV  |     31012 |           15506 |          15506 |                2816 |        5 |                    0.5 |         2038 |        2027 |                   1000 |                         20 |

## Exact risk-bin edges

| dataset   |   edge_index | edge_display          | population_hash                                                  |
|:----------|-------------:|:----------------------|:-----------------------------------------------------------------|
| SUPPORT2  |            0 | -inf                  | 218c5b223256b28fbe3a632cab3a858c8bbd4c6ee9a3ebcafc1b62e2557937ba |
| SUPPORT2  |            1 | 0.10875690370680161   | 218c5b223256b28fbe3a632cab3a858c8bbd4c6ee9a3ebcafc1b62e2557937ba |
| SUPPORT2  |            2 | 0.20714846156356659   | 218c5b223256b28fbe3a632cab3a858c8bbd4c6ee9a3ebcafc1b62e2557937ba |
| SUPPORT2  |            3 | 0.36589217418781955   | 218c5b223256b28fbe3a632cab3a858c8bbd4c6ee9a3ebcafc1b62e2557937ba |
| SUPPORT2  |            4 | 0.60878606228565002   | 218c5b223256b28fbe3a632cab3a858c8bbd4c6ee9a3ebcafc1b62e2557937ba |
| SUPPORT2  |            5 | +inf                  | 218c5b223256b28fbe3a632cab3a858c8bbd4c6ee9a3ebcafc1b62e2557937ba |
| MIMIC-IV  |            0 | -inf                  | d5934afbfff478f7cb5e9ddddaf2e240537c788f2d43cc097cae4dc8199b4a1b |
| MIMIC-IV  |            1 | 0.0061575980070088607 | d5934afbfff478f7cb5e9ddddaf2e240537c788f2d43cc097cae4dc8199b4a1b |
| MIMIC-IV  |            2 | 0.028693120246951205  | d5934afbfff478f7cb5e9ddddaf2e240537c788f2d43cc097cae4dc8199b4a1b |
| MIMIC-IV  |            3 | 0.08216318081930489   | d5934afbfff478f7cb5e9ddddaf2e240537c788f2d43cc097cae4dc8199b4a1b |
| MIMIC-IV  |            4 | 0.16349946528998208   | d5934afbfff478f7cb5e9ddddaf2e240537c788f2d43cc097cae4dc8199b4a1b |
| MIMIC-IV  |            5 | +inf                  | d5934afbfff478f7cb5e9ddddaf2e240537c788f2d43cc097cae4dc8199b4a1b |

## Exact split counts

| dataset   | analysis_split   | outcome   |     n |
|:----------|:-----------------|:----------|------:|
| SUPPORT2  | calibration      | ALL       |   910 |
| SUPPORT2  | test             | ALL       |   911 |
| SUPPORT2  | calibration      | 0         |   594 |
| SUPPORT2  | calibration      | 1         |   316 |
| SUPPORT2  | test             | 0         |   595 |
| SUPPORT2  | test             | 1         |   316 |
| MIMIC-IV  | calibration      | ALL       | 15506 |
| MIMIC-IV  | test             | ALL       | 15506 |
| MIMIC-IV  | calibration      | 0         | 14098 |
| MIMIC-IV  | calibration      | 1         |  1408 |
| MIMIC-IV  | test             | 0         | 14098 |
| MIMIC-IV  | test             | 1         |  1408 |

## Four-part recalibration rule

|   check_number | metric                     | bootstrap_quantity   | criterion   | meaning                           |
|---------------:|:---------------------------|:---------------------|:------------|:----------------------------------|
|              1 | restricted_ici_change      | q975                 | q975 < 0    | restricted ICI improves           |
|              2 | reference_ici_change       | q025                 | q025 > 0    | linked/reference ICI worsens      |
|              3 | restricted_log_loss_change | q975                 | q975 < 0    | restricted log loss improves      |
|              4 | reference_log_loss_change  | q025                 | q025 > 0    | linked/reference log loss worsens |

## Git provenance

| evidence                                    | path_or_search                                                                      | commit                                   | date                      | subject                                                  |
|:--------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------------|:--------------------------|:---------------------------------------------------------|
| recalibration_implementation_file           | scripts/phase5j_realdata_recalibration_fragility.py                                 | 3693f741e7995181fb48b2bd4356acf0e4d18b00 | 2026-07-18T07:33:33+00:00 | Release reproducible code and aggregate KDD 2027 results |
| cross_dataset_freeze_file                   | scripts/phase5j_h7_mixed_evidence_freeze.py                                         | 3693f741e7995181fb48b2bd4356acf0e4d18b00 | 2026-07-18T07:33:33+00:00 | Release reproducible code and aggregate KDD 2027 results |
| aggregate_recalibration_table               | results/tables/table_06_h7_recalibration.csv                                        | 3693f741e7995181fb48b2bd4356acf0e4d18b00 | 2026-07-18T07:33:33+00:00 | Release reproducible code and aggregate KDD 2027 results |
| first_commit_adding_core_decision_function  | scripts/phase5j_realdata_recalibration_fragility.py :: -S def build_h7_core_checks  | 3693f741e7995181fb48b2bd4356acf0e4d18b00 | 2026-07-18T07:33:33+00:00 | Release reproducible code and aggregate KDD 2027 results |
| first_commit_adding_restricted_ici_check    | scripts/phase5j_realdata_recalibration_fragility.py :: -S restricted_ici_change     | 3693f741e7995181fb48b2bd4356acf0e4d18b00 | 2026-07-18T07:33:33+00:00 | Release reproducible code and aggregate KDD 2027 results |
| first_commit_adding_reference_logloss_check | scripts/phase5j_realdata_recalibration_fragility.py :: -S reference_log_loss_change | 3693f741e7995181fb48b2bd4356acf0e4d18b00 | 2026-07-18T07:33:33+00:00 | Release reproducible code and aggregate KDD 2027 results |
| first_commit_adding_cross_dataset_rule_use  | scripts/phase5j_h7_mixed_evidence_freeze.py :: -S core_h7_checks_passed             |                                          |                           |                                                          |

## Provenance interpretation

- Current HEAD: `3693f741e7995181fb48b2bd4356acf0e4d18b00` (2026-07-18T07:33:33+00:00)
- Visible Git commits: 1
- Rule/table ordering: rule commit is an ancestor of/equal to first table-touch commit
- Git history can establish when code/files were committed, but cannot by itself prove when linked outcomes or locally generated recalibration results were first inspected. Do not claim such chronology unless an independent dated artifact/log/commit establishes it.

**Do not state that the configuration was frozen before first inspection of linked outcomes unless an independent dated record establishes that fact.**
