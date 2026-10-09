# Phase 8B — Method Specification Audit

Status: `phase8b_method_specification_requires_review`

This phase performs no new model fitting or experiment.

## Estimator equations and assumptions

| estimator                                   | estimand_or_formula                                                                                                                 | conditioning_or_assumption                                                                                                                                                                                                                  | implementation_evidence              |
|:--------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------|
| Linked/reference calibration                | m_ref(p) = E[Y(tau) | P=p]                                                                                                          | Uses linked fixed-horizon outcomes for the full reference population.                                                                                                                                                                       | src/actionaudit/metrics.py           |
| Restricted complete case                    | m_obs(p) = E[Y(tau) | P=p, R(tau)=1]                                                                                                | Uses only outcomes visible before the operational boundary.                                                                                                                                                                                 | src/actionaudit/metrics.py           |
| Binary-observability IPW (Horvitz-Thompson) | Within bin B: mean_i[ R_i Y_i / e_i ], e_i=P(R_i=1 | X_i), with lower-tail clipping.                                                | Requires conditional exchangeability of Y and R given nuisance features X, correct/adequate observation model for pure IPW, and positivity e_i>0 on target support.                                                                         | src/actionaudit/dr.py                |
| Binary-observability IPW (Hajek)            | Within bin B: sum_i R_i Y_i/e_i divided by sum_i R_i/e_i.                                                                           | Same observation-model and positivity conditions as binary IPW; normalization changes finite-sample behavior.                                                                                                                               | src/actionaudit/dr.py                |
| Binary-observability outcome regression     | Within bin B: mean_i[m_hat(X_i)], where m_hat approximates E[Y | X, R=1] and is projected to the full bin population.               | Requires the outcome regression to transport from visible to hidden cases conditional on included features.                                                                                                                                 | src/actionaudit/dr.py                |
| Binary-observability AIPW                   | Within bin B: mean_i[ m_hat(X_i) + R_i/e_i * (Y_i-m_hat(X_i)) ].                                                                    | Doubly robust under the usual missing-at-random/selective-label conditions when either the observation model or outcome model is correct, plus positivity and regularity; clipping changes the finite-sample target/bias-variance tradeoff. | src/actionaudit/dr.py                |
| Kaplan-Meier                                | 1 - S_hat_KM(tau) within each fixed risk bin.                                                                                       | Treats hospital exit as right censoring; standard KM recovery requires non-informative censoring within the analysis stratum.                                                                                                               | src/actionaudit/km.py                |
| Survival IPCW                               | Event contribution for an observed event at time t_i: Delta_i / max(G_hat(t_i- | X_i), c); bin risk is the mean event contribution. | Uses censoring survival over the risk set, not a binary horizon observability propensity. Requires conditionally independent censoring given modeled history/features and positivity of G.                                                  | src/actionaudit/discrete_survival.py |
| Survival outcome regression                 | m_hat_i(tau)=1-product_t(1-h_hat_event(t | history_i)); bin risk is mean_i[m_hat_i(tau)].                                           | Requires an adequate event-hazard model over observed follow-up and transport to the full target population.                                                                                                                                | src/actionaudit/discrete_survival.py |

## Binning / empty-bin / integration rules

| item                        | verified_rule                                                                                                                                                                                             | source                                               |
|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------|
| Bin construction            | Reference-cohort quantile edges are computed from the fixed prediction distribution; duplicate quantile edges are removed; first/last edges are set to -inf/+inf.                                         | src/actionaudit/metrics.py and src/actionaudit/km.py |
| Assignment                  | Predictions are assigned with pandas.cut(..., include_lowest=True). Estimator paths explicitly error if any prediction is unassigned.                                                                     | src/actionaudit/km.py, dr.py, discrete_survival.py   |
| Empty bins                  | groupby(..., observed=True) means bins containing no records do not produce estimator rows. Quantile-edge de-duplication can reduce the realized number of bins when predictions have ties.               | src/actionaudit/metrics.py                           |
| ActionAudit bin eligibility | A bin is eligible only when n_bin >= min_bin_size AND n_observed > 0. Ineligible bins receive NaN calibration quantities.                                                                                 | src/actionaudit/metrics.py                           |
| Integration distribution    | Each eligible bin receives reference weight n_bin/n_total. For the core ActionAudit scalar summaries, weights over eligible bins are renormalized to sum to one before ICI/calibration-error integration. | src/actionaudit/metrics.py                           |
| Sparse-bin robustness       | Phase 7E separately varied bin count and minimum-bin-size rules; this is empirical robustness evidence rather than part of the original estimator definition.                                             | revision_kdd2027/phase7e_*                           |

## Localized-sensitivity subgroup counts

Required subgroup-count artifacts were not found.

## Bootstrap subgroup reselection

| dataset   | procedure                                                                                                                                                                                                      | reselected_within_each_bootstrap   | source                                            |
|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------|:--------------------------------------------------|
| SUPPORT2  | Each patient-level bootstrap sample is passed through calculate_point_results; the closest/minimum eligible subgroup is recomputed from that resample rather than holding the original closest subgroup fixed. | True                               | scripts/phase5i_support2_partition_sensitivity.py |
| MIMIC-IV  | Bootstrap action results are grouped by bootstrap_id and the minimum distance-to-tipping is computed within each replicate; therefore the identity of the closest action may change by replicate.              | True                               | scripts/phase5i_cross_dataset_h5_freeze.py        |

## Verified settings

| component                              | setting                                                                 | value                            | source                                              |
|:---------------------------------------|:------------------------------------------------------------------------|:---------------------------------|:----------------------------------------------------|
| Recalibration audit                    | risk bins                                                               | 5                                | Phase 8A reconstruction                             |
| Recalibration audit                    | calibration/evaluation fraction                                         | 0.5 / 0.5 stratified             | Phase 8A reconstruction                             |
| Recalibration audit                    | split seed                                                              | 2038                             | Phase 8A reconstruction                             |
| Recalibration audit                    | bootstrap replicates                                                    | 1000                             | Phase 8A reconstruction                             |
| Recalibration audit                    | base seed                                                               | 2027                             | Phase 8A reconstruction                             |
| SUPPORT2 binary DR benchmark           | horizon / bins / propensity clip / bootstraps / seed                    | 60d / 5 / 0.05 / 1000 / 2027     | scripts/phase2b_support2_dr.py defaults             |
| MIMIC practitioner estimator benchmark | horizon / bins / crossfit repeats / propensity clip / bootstraps / seed | 30d / 5 / 5 / 0.01 / 1000 / 2027 | scripts/phase5g_practitioner_estimators.py defaults |
| MIMIC rich/AIPW benchmark              | reported propensity clipping grid                                       | 0.01, 0.025, 0.05                | scripts/phase5g_rich_baseline_estimators.py         |

## Validation

| check                                          | passed   |
|:-----------------------------------------------|:---------|
| binary_ipw_formula_present                     | True     |
| binary_aipw_formula_present                    | True     |
| binary_propensity_lower_clip_only              | True     |
| survival_ipcw_uses_censor_survival_before_time | True     |
| km_treats_exit_as_censoring                    | True     |
| quantile_edges_from_reference_prediction       | True     |
| duplicate_edges_removed                        | True     |
| eligibility_requires_min_n_and_observed        | True     |
| eligible_weights_renormalized                  | True     |
| support2_subgroup_counts_recovered             | False    |
| mimic_subgroup_counts_recovered                | False    |
| support2_closest_group_reselected_in_bootstrap | True     |
| mimic_closest_group_reselected_in_bootstrap    | True     |
