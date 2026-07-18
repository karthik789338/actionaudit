| Dataset | Estimator | Reference-calibration MAE | Empirical reading |
| --- | --- | --- | --- |
| SUPPORT2 | Complete case | 0.407 | Large residual error when analysis used only directly observed outcomes. |
| SUPPORT2 | Kaplan–Meier | 0.087 | Substantial improvement, but not the best recovery. |
| SUPPORT2 | Survival outcome regression | 0.056 | Best SUPPORT2 recovery under the available identification regime. |
| SUPPORT2 | IPCW | 0.116 | Improved over complete case but remained less accurate than outcome regression. |
| MIMIC-IV | Score-only outcome regression | 0.149 | Best unaugmented MIMIC estimator, but materially separated from the linked reference. |
| MIMIC-IV | Rich logistic outcome model | 0.152 | Additional covariates did not restore the linked calibration target. |
| MIMIC-IV | Kaplan–Meier | 0.157 | Limited recovery under the severe hospital-boundary observation pattern. |
| MIMIC-IV | Histogram gradient boosting | 0.168 | Greater model flexibility did not improve recovery. |
| MIMIC-IV | Random forest | 0.168 | Nonlinear nuisance estimation alone did not resolve the identification and overlap limitations. |
| MIMIC-IV | RF-AIPW, clipping 0.05 | 0.420 | Augmentation was unstable under severe positivity limitations. |
