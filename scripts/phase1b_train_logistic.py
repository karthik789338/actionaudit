"""Phase 1B Part 1: Train and freeze the SUPPORT2 logistic model.

This script:

1. Loads SUPPORT2.
2. Creates the reference 60-day outcome.
3. Selects leakage-safe baseline features.
4. Creates fixed train, validation, and test splits.
5. Tunes logistic-regression regularization on validation data.
6. Refits on train plus validation.
7. Generates and saves frozen test predictions.

The test set is never used for model selection.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from actionaudit.modeling import (
    create_feature_manifest,
    create_patient_split,
    tune_and_refit_logistic,
)
from actionaudit.support2 import (
    build_support2_horizon_view,
    load_support2_csv,
    prediction_feature_columns,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train and freeze the SUPPORT2 logistic mortality model."
    )

    parser.add_argument(
        "--input",
        default="data/raw/support2.csv",
        help="Path to SUPPORT2 CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/phase1b_logistic",
        help="Directory for Phase 1B artifacts.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Mortality prediction horizon in days.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help="Random seed.",
    )

    return parser.parse_args()


def calculate_test_metrics(
    outcome: pd.Series,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Calculate reference test-set model metrics."""

    return {
        "test_n": int(len(outcome)),
        "test_reference_event_rate": float(outcome.mean()),
        "test_reference_log_loss": float(
            log_loss(
                outcome,
                prediction,
                labels=[0, 1],
            )
        ),
        "test_reference_brier": float(
            brier_score_loss(
                outcome,
                prediction,
            )
        ),
        "test_reference_roc_auc": float(
            roc_auc_score(
                outcome,
                prediction,
            )
        ),
        "test_reference_average_precision": float(
            average_precision_score(
                outcome,
                prediction,
            )
        ),
    }


def main() -> None:
    """Train the model and save frozen test predictions."""

    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading SUPPORT2...")

    data = load_support2_csv(args.input)

    horizon_view = build_support2_horizon_view(
        data,
        args.horizon,
    ).frame

    analysis = (
        horizon_view.loc[
            horizon_view["reference_known"].eq(1)
        ]
        .reset_index(drop=True)
    )

    if analysis.empty:
        raise RuntimeError(
            "No patients have known reference outcomes."
        )

    print(
        f"Reference-known patients: {len(analysis):,}"
    )

    candidate_features = prediction_feature_columns(data)

    feature_manifest = create_feature_manifest(
        analysis,
        candidate_features,
    )

    included_features = (
        feature_manifest.loc[
            feature_manifest["included"],
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    if not included_features:
        raise RuntimeError(
            "No usable baseline features remain."
        )

    print(
        f"Included baseline features: {len(included_features)}"
    )

    outcome = (
        analysis["outcome"]
        .astype(int)
        .reset_index(drop=True)
    )

    split = create_patient_split(
        analysis,
        outcome,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
        random_seed=args.seed,
    )

    print(
        f"Split: {len(split.train_index)} train, "
        f"{len(split.validation_index)} validation, "
        f"{len(split.test_index)} test"
    )

    print(
        f"Stratification method: "
        f"{split.stratification_method}"
    )

    selection = tune_and_refit_logistic(
        analysis,
        outcome,
        split,
        included_features,
        c_grid=(
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ),
        random_seed=args.seed,
    )

    print(
        f"Selected logistic C: {selection.best_c}"
    )

    test_data = (
        analysis
        .iloc[split.test_index]
        .copy()
        .reset_index(drop=True)
    )

    test_prediction = selection.model.predict_proba(
        test_data[included_features]
    )[:, 1]

    if (
        np.isnan(test_prediction).any()
        or np.isinf(test_prediction).any()
    ):
        raise RuntimeError(
            "Model produced non-finite test predictions."
        )

    test_outcome = (
        test_data["outcome"]
        .astype(int)
    )

    test_metrics = calculate_test_metrics(
        test_outcome,
        test_prediction,
    )

    frozen_predictions = pd.DataFrame(
        {
            "patient_id": test_data[
                "patient_id"
            ].astype(int),
            "prediction": test_prediction,
            "outcome": test_outcome,
            "observed": test_data[
                "observed"
            ].astype(int),
            "action": test_data["action"],
            "exit_time": test_data[
                "exit_time"
            ],
            "post_discharge_event_by_horizon": test_data[
                "post_discharge_event_by_horizon"
            ].astype(int),
        }
    )

    split_labels = np.full(
        len(analysis),
        "",
        dtype=object,
    )

    split_labels[
        split.train_index
    ] = "train"

    split_labels[
        split.validation_index
    ] = "validation"

    split_labels[
        split.test_index
    ] = "test"

    split_manifest = pd.DataFrame(
        {
            "patient_id": analysis[
                "patient_id"
            ].astype(int),
            "split": split_labels,
            "outcome": outcome,
        }
    )

    if "hosp" in analysis.columns:
        split_manifest["hospital"] = analysis["hosp"]

    feature_manifest.to_csv(
        output_dir / "feature_manifest.csv",
        index=False,
    )

    split_manifest.to_csv(
        output_dir / "split_manifest.csv",
        index=False,
    )

    selection.selection_table.to_csv(
        output_dir / "model_selection.csv",
        index=False,
    )

    frozen_predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    joblib.dump(
        selection.model,
        output_dir / "frozen_logistic_model.joblib",
    )

    test_metric_frame = pd.DataFrame(
        {
            "metric": list(test_metrics.keys()),
            "value": list(test_metrics.values()),
        }
    )

    test_metric_frame.to_csv(
        output_dir / "test_model_metrics.csv",
        index=False,
    )

    metadata = {
        "phase": "1B",
        "part": "training",
        "experiment": (
            "Frozen SUPPORT2 logistic mortality model"
        ),
        "horizon_days": args.horizon,
        "model": "L2 logistic regression",
        "best_c": selection.best_c,
        "train_rows": int(
            len(split.train_index)
        ),
        "validation_rows": int(
            len(split.validation_index)
        ),
        "training_rows_after_refit": int(
            len(split.train_index)
            + len(split.validation_index)
        ),
        "test_rows": int(
            len(split.test_index)
        ),
        "stratification_method": (
            split.stratification_method
        ),
        "n_features_before_encoding": len(
            included_features
        ),
        "included_features": included_features,
        "numeric_features": (
            selection.numeric_features
        ),
        "categorical_features": (
            selection.categorical_features
        ),
        "test_predictions_frozen": True,
        "test_used_for_model_selection": False,
        "random_seed": args.seed,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "test_metrics": test_metrics,
    }

    metadata_path = (
        output_dir
        / "training_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nValidation model selection")
    print(
        selection.selection_table.to_string(
            index=False
        )
    )

    print("\nFrozen test-set model metrics")

    for metric, value in test_metrics.items():
        if metric == "test_n":
            print(
                f"{metric}: {int(value)}"
            )
        else:
            print(
                f"{metric}: {value:.6f}"
            )

    print(
        "\nSaved frozen predictions to: "
        f"{output_dir / 'test_predictions.csv'}"
    )

    print(
        "Saved model to: "
        f"{output_dir / 'frozen_logistic_model.joblib'}"
    )

    print(
        "\nPhase 1B Part 1 complete."
    )


if __name__ == "__main__":
    main()
