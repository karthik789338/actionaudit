from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("artifacts")
OUT = Path(
    "revision_kdd2027/"
    "phase7a2_multimodel_overlap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = [
    {
        "name": "score_logistic",
        "path": (
            ROOT
            / "phase5g_practitioner_estimators"
            / "cross_fitted_nuisance_predictions.csv.gz"
        ),
        "probability": "observation_probability",
        "summary_path": (
            ROOT
            / "phase5g_practitioner_estimators"
            / "estimator_metric_summary.csv"
        ),
        "or_method": "pooled_survival_or",
        "ipcw_method": "score_ipcw",
        "aipw_method": "score_aipw",
    },
    {
        "name": "rich_logistic",
        "path": (
            ROOT
            / "phase5g_rich_baseline_estimators"
            / "cross_fitted_rich_nuisance_predictions.csv.gz"
        ),
        "probability": "rich_observation_probability",
        "summary_path": (
            ROOT
            / "phase5g_rich_baseline_estimators"
            / "rich_estimator_metric_summary.csv"
        ),
        "or_method": "rich_pooled_survival_or",
        "ipcw_method": "rich_ipcw_clip_0p05",
        "aipw_method": "rich_aipw_clip_0p05",
    },
    {
        "name": "hgb",
        "path": (
            ROOT
            / "phase5g_nonlinear_estimators"
            / "cross_fitted_hgb_nuisance_predictions.csv.gz"
        ),
        "probability": "hgb_observation_probability",
        "summary_path": (
            ROOT
            / "phase5g_nonlinear_estimators"
            / "nonlinear_estimator_metric_summary.csv"
        ),
        "or_method": "hgb_survival_or",
        "ipcw_method": "hgb_ipcw_clip_0p05",
        "aipw_method": "hgb_aipw_clip_0p05",
    },
    {
        "name": "random_forest",
        "path": (
            ROOT
            / "phase5g_random_forest_estimators"
            / "cross_fitted_rf_nuisance_predictions.csv.gz"
        ),
        "probability": "rf_observation_probability",
        "summary_path": (
            ROOT
            / "phase5g_random_forest_estimators"
            / "random_forest_metric_summary.csv"
        ),
        "or_method": "rf_survival_or",
        "ipcw_method": "rf_ipcw_clip_0p05",
        "aipw_method": "rf_aipw_clip_0p05",
    },
]


def ess(weights):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights[
        np.isfinite(weights)
        & (weights > 0)
    ]

    return float(
        weights.sum() ** 2
        / np.sum(
            weights ** 2
        )
    )


def get_error(
    table,
    method,
):
    row = table.loc[
        table["method"].eq(
            method
        )
    ]

    if len(row) != 1:
        return np.nan

    return float(
        row.iloc[0][
            "bin_curve_mae"
        ]
    )


rows = []
quantile_rows = []

for specification in MODELS:
    path = specification[
        "path"
    ]

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    data = pd.read_csv(
        path
    )

    summary = pd.read_csv(
        specification[
            "summary_path"
        ]
    )

    observed = (
        data["observed"]
        .astype(int)
        .to_numpy()
    )

    probability = (
        data[
            specification[
                "probability"
            ]
        ]
        .astype(float)
        .to_numpy()
    )

    if len(data) != 31012:
        raise ValueError(
            f"{specification['name']}: "
            f"expected 31012 rows, "
            f"got {len(data)}"
        )

    if np.any(
        ~np.isfinite(
            probability
        )
    ):
        raise ValueError(
            f"{specification['name']}: "
            "non-finite probabilities"
        )

    visible = probability[
        observed == 1
    ]

    hidden = probability[
        observed == 0
    ]

    row = {
        "model": (
            specification[
                "name"
            ]
        ),
        "n": len(data),
        "n_observed": int(
            observed.sum()
        ),
        "observed_fraction": float(
            observed.mean()
        ),
        "raw_min": float(
            probability.min()
        ),
        "q001": float(
            np.quantile(
                probability,
                0.001,
            )
        ),
        "q005": float(
            np.quantile(
                probability,
                0.005,
            )
        ),
        "q01": float(
            np.quantile(
                probability,
                0.01,
            )
        ),
        "q025": float(
            np.quantile(
                probability,
                0.025,
            )
        ),
        "q05": float(
            np.quantile(
                probability,
                0.05,
            )
        ),
        "median": float(
            np.median(
                probability
            )
        ),
        "observed_median": float(
            np.median(
                visible
            )
        ),
        "hidden_median": float(
            np.median(
                hidden
            )
        ),
        "fraction_all_lt_0p01": float(
            np.mean(
                probability
                < 0.01
            )
        ),
        "fraction_all_lt_0p025": float(
            np.mean(
                probability
                < 0.025
            )
        ),
        "fraction_all_lt_0p05": float(
            np.mean(
                probability
                < 0.05
            )
        ),
        "fraction_observed_lt_0p01": float(
            np.mean(
                visible
                < 0.01
            )
        ),
        "fraction_observed_lt_0p025": float(
            np.mean(
                visible
                < 0.025
            )
        ),
        "fraction_observed_lt_0p05": float(
            np.mean(
                visible
                < 0.05
            )
        ),
        "or_mae": get_error(
            summary,
            specification[
                "or_method"
            ],
        ),
        "ipcw_mae": get_error(
            summary,
            specification[
                "ipcw_method"
            ],
        ),
        "aipw_mae": get_error(
            summary,
            specification[
                "aipw_method"
            ],
        ),
    }

    for clip in [
        0.01,
        0.025,
        0.05,
        0.10,
    ]:
        clipped = np.maximum(
            probability,
            clip,
        )

        weights = (
            observed
            / clipped
        )

        current_ess = ess(
            weights
        )

        row[
            f"ess_clip_{clip}"
        ] = current_ess

        row[
            f"ess_full_fraction_clip_{clip}"
        ] = (
            current_ess
            / len(data)
        )

        row[
            f"ess_observed_fraction_clip_{clip}"
        ] = (
            current_ess
            / observed.sum()
        )

        row[
            f"fraction_observed_clipped_{clip}"
        ] = float(
            np.mean(
                visible
                < clip
            )
        )

    rows.append(
        row
    )

    for group, values in [
        (
            "all",
            probability,
        ),
        (
            "observed",
            visible,
        ),
        (
            "hidden",
            hidden,
        ),
    ]:
        for quantile in [
            0,
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.0,
        ]:
            quantile_rows.append(
                {
                    "model": (
                        specification[
                            "name"
                        ]
                    ),
                    "group": group,
                    "quantile": (
                        quantile
                    ),
                    "probability": float(
                        np.quantile(
                            values,
                            quantile,
                        )
                    ),
                }
            )


results = pd.DataFrame(
    rows
)

quantiles = pd.DataFrame(
    quantile_rows
)

results.to_csv(
    OUT
    / "multimodel_overlap_summary.csv",
    index=False,
)

quantiles.to_csv(
    OUT
    / "multimodel_probability_quantiles.csv",
    index=False,
)


print(
    "=" * 120
)

print(
    "ACTIONAUDIT PHASE 7A-2: "
    "MIMIC MULTI-MODEL OVERLAP DIAGNOSTICS"
)

print(
    "=" * 120
)

display_columns = [
    "model",
    "raw_min",
    "q01",
    "q05",
    "median",
    "fraction_all_lt_0p01",
    "fraction_all_lt_0p05",
    "fraction_observed_lt_0p05",
    "ess_clip_0.05",
    "ess_full_fraction_clip_0.05",
    "or_mae",
    "ipcw_mae",
    "aipw_mae",
]

print()

print(
    results[
        display_columns
    ].to_string(
        index=False
    )
)

print()

checks = {
    "all_models_n_31012": bool(
        (
            results[
                "n"
            ]
            == 31012
        ).all()
    ),
    "all_models_n_observed_2425": bool(
        (
            results[
                "n_observed"
            ]
            == 2425
        ).all()
    ),
    "all_probabilities_positive": bool(
        (
            results[
                "raw_min"
            ]
            > 0
        ).all()
    ),
    "all_errors_available": bool(
        results[
            [
                "or_mae",
                "ipcw_mae",
                "aipw_mae",
            ]
        ]
        .notna()
        .all()
        .all()
    ),
}

for name, value in (
    checks.items()
):
    print(
        f"{name}: "
        f"{'PASS' if value else 'FAIL'}"
    )

status = (
    "phase7a2_multimodel_overlap_passed"
    if all(
        checks.values()
    )
    else (
        "phase7a2_multimodel_overlap_failed"
    )
)

(
    OUT
    / "phase7a2_status.txt"
).write_text(
    status
    + "\n"
)

print()
print(
    "STATUS:",
    status,
)
