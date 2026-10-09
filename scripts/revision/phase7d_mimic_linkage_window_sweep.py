# scripts/revision/phase7d_mimic_linkage_window_sweep.py

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt


DEFAULT_WINDOWS = [0, 1, 3, 7, 14, 21, 30]

EXPECTED_N = 31012
EXPECTED_OBSERVED = 2425
HORIZON_DAYS = 30.0

MANUSCRIPT_LINKED_ICI = 0.001
MANUSCRIPT_RESTRICTED_ICI = 0.487


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 7D: MIMIC-IV real-data linkage-window sweep."
    )

    parser.add_argument(
        "--cohort",
        default=(
            "artifacts/phase5c_frozen_cohort/"
            "primary_inpatient_cohort.csv.gz"
        ),
    )

    parser.add_argument(
        "--predictions",
        default=(
            "artifacts/phase5c_admission_model/"
            "test_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--risk-bins",
        default=(
            "artifacts/phase5g_practitioner_estimators/"
            "cross_fitted_nuisance_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--windows",
        default="0,1,3,7,14,21,30",
        help="Post-discharge linked follow-up windows in days.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default="revision_kdd2027/phase7d_mimic_linkage_window",
    )

    return parser.parse_args()


def add_check(rows, check, value, expected, passed):
    rows.append(
        {
            "check": check,
            "value": value,
            "expected": expected,
            "passed": bool(passed),
        }
    )


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def load_data(cohort_path, prediction_path, bin_path):
    print("Loading frozen MIMIC-IV inputs...")

    cohort = pd.read_csv(
        cohort_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "dod",
            "horizon_time",
            "outcome",
            "observed",
            "exit_time",
        ],
    )

    pred = pd.read_csv(prediction_path)

    bins = pd.read_csv(
        bin_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "prediction",
            "risk_bin",
        ],
    ).rename(columns={"prediction": "bin_prediction"})

    keys = ["subject_id", "hadm_id"]

    if pred.duplicated(keys).any():
        raise RuntimeError("Duplicate test prediction keys.")

    if cohort.duplicated(keys).any():
        raise RuntimeError("Duplicate frozen cohort keys.")

    if bins.duplicated(keys).any():
        raise RuntimeError("Duplicate risk-bin keys.")

    # Start from the frozen test predictions so the test population
    # remains exactly the original 31,012 admissions.
    frame = pred.merge(
        cohort[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "dod",
                "horizon_time",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    frame = frame.merge(
        bins,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if frame["risk_bin"].isna().any():
        missing = int(frame["risk_bin"].isna().sum())
        raise RuntimeError(
            f"{missing} test admissions lack frozen risk bins."
        )

    pred_difference = np.max(
        np.abs(
            frame["prediction"].to_numpy(dtype=float)
            - frame["bin_prediction"].to_numpy(dtype=float)
        )
    )

    if pred_difference > 1e-10:
        raise RuntimeError(
            "Predictions do not agree with frozen risk-bin artifact. "
            f"Maximum difference={pred_difference}"
        )

    frame = frame.drop(columns=["bin_prediction"])

    frame["admittime"] = pd.to_datetime(
        frame["admittime"],
        errors="coerce",
    )

    frame["dischtime"] = pd.to_datetime(
        frame["dischtime"],
        errors="coerce",
    )

    frame["deathtime"] = pd.to_datetime(
        frame["deathtime"],
        errors="coerce",
    )

    frame["dod"] = pd.to_datetime(
        frame["dod"],
        errors="coerce",
    )

    frame["horizon_time"] = pd.to_datetime(
        frame["horizon_time"],
        errors="coerce",
    )

    frame["outcome"] = frame["outcome"].astype(int)
    frame["observed"] = frame["observed"].astype(int)
    frame["risk_bin"] = frame["risk_bin"].astype(int)

    print("Test admissions:", len(frame))
    print("Native observed:", int(frame["observed"].sum()))
    print(
        "Native observed fraction:",
        float(frame["observed"].mean()),
    )

    return frame


def derive_post_discharge_death_lag(frame):
    """
    Estimate linked-death lag after hospital discharge.

    MIMIC-IV deathtime gives timestamp information when available.
    The subject-level DOD field can be date-resolution only.

    For deaths after discharge:
      1. use exact deathtime if it is after discharge;
      2. otherwise use calendar-day difference from DOD.

    Native window=0 always uses the original frozen observed flag,
    so date-only same-day deaths are never retroactively made
    hospital-visible.
    """

    dischtime = frame["dischtime"]
    deathtime = frame["deathtime"]
    dod = frame["dod"]

    exact_lag = (
        (deathtime - dischtime).dt.total_seconds()
        / 86400.0
    )

    exact_valid = (
        deathtime.notna()
        & dischtime.notna()
        & (exact_lag >= 0)
    )

    calendar_lag = (
        dod.dt.normalize()
        - dischtime.dt.normalize()
    ).dt.days.astype(float)

    calendar_valid = (
        dod.notna()
        & dischtime.notna()
        & (calendar_lag >= 0)
    )

    lag = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    lag.loc[calendar_valid] = calendar_lag.loc[
        calendar_valid
    ]

    lag.loc[exact_valid] = exact_lag.loc[
        exact_valid
    ]

    return lag


def build_visibility(frame, death_lag, extension_days):
    native = frame["observed"].to_numpy(dtype=bool)

    if extension_days == 0:
        return native.copy()

    outcome = frame["outcome"].to_numpy(dtype=int)
    exit_time = frame["exit_time"].to_numpy(dtype=float)
    death_lag_values = death_lag.to_numpy(dtype=float)

    # If post-discharge linkage extends all the way from discharge
    # to the 30-day horizon, the final binary outcome is determined.
    reaches_horizon = (
        exit_time + float(extension_days)
        >= HORIZON_DAYS - 1e-12
    )

    # A linked post-discharge death makes Y(30d)=1 known even when
    # the entire remaining horizon has not yet elapsed.
    captures_death = (
        (~native)
        & (outcome == 1)
        & np.isfinite(death_lag_values)
        & (
            death_lag_values
            <= float(extension_days) + 1e-12
        )
    )

    return native | reaches_horizon | captures_death


def reference_bin_table(frame):
    table = (
        frame.groupby("risk_bin", sort=True)
        .agg(
            n_reference=("outcome", "size"),
            prediction_mean=("prediction", "mean"),
            linked_event_rate=("outcome", "mean"),
        )
        .reset_index()
    )

    table["reference_weight"] = (
        table["n_reference"] / len(frame)
    )

    return table


def linked_ici(reference):
    return float(
        np.sum(
            reference["reference_weight"]
            * np.abs(
                reference["linked_event_rate"]
                - reference["prediction_mean"]
            )
        )
    )


def restricted_metrics(frame, visible, reference):
    work = frame.loc[
        visible,
        [
            "risk_bin",
            "outcome",
            "prediction",
        ],
    ].copy()

    observed_table = (
        work.groupby("risk_bin", sort=True)
        .agg(
            n_observed=("outcome", "size"),
            observed_event_rate=("outcome", "mean"),
        )
        .reset_index()
    )

    table = reference.merge(
        observed_table,
        on="risk_bin",
        how="left",
        validate="one_to_one",
    )

    if table["n_observed"].isna().any():
        empty_bins = table.loc[
            table["n_observed"].isna(),
            "risk_bin",
        ].tolist()

        raise RuntimeError(
            f"Observed calibration has empty bins: {empty_bins}"
        )

    ici = float(
        np.sum(
            table["reference_weight"]
            * np.abs(
                table["observed_event_rate"]
                - table["prediction_mean"]
            )
        )
    )

    signed = float(
        np.sum(
            table["reference_weight"]
            * (
                table["observed_event_rate"]
                - table["prediction_mean"]
            )
        )
    )

    curve_mae = float(
        np.sum(
            table["reference_weight"]
            * np.abs(
                table["observed_event_rate"]
                - table["linked_event_rate"]
            )
        )
    )

    return ici, signed, curve_mae, table


def estimate_score_propensity(frame, visible, seed):
    """
    Five-fold cross-fitted score-only logistic observation model.

    This is used only as an overlap diagnostic and a conventional
    IPCW benchmark. It is not claimed to identify the target when
    support is inadequate.
    """

    y = np.asarray(visible, dtype=int)

    if np.unique(y).size == 1:
        return np.ones(len(y), dtype=float), np.nan

    x = safe_logit(
        frame["prediction"].to_numpy(dtype=float)
    ).reshape(-1, 1)

    class_counts = np.bincount(y, minlength=2)
    min_class = int(class_counts.min())

    if min_class < 2:
        model = LogisticRegression(
            C=1e6,
            max_iter=2000,
            solver="lbfgs",
        )
        model.fit(x, y)
        g = model.predict_proba(x)[:, 1]
    else:
        n_splits = min(5, min_class)

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        g = np.zeros(len(y), dtype=float)

        for train_idx, test_idx in splitter.split(x, y):
            model = LogisticRegression(
                C=1e6,
                max_iter=2000,
                solver="lbfgs",
            )

            model.fit(
                x[train_idx],
                y[train_idx],
            )

            g[test_idx] = model.predict_proba(
                x[test_idx]
            )[:, 1]

    g = np.clip(g, 1e-6, 1.0)

    auc = roc_auc_score(y, g)

    return g, float(auc)


def ipcw_metrics(
    frame,
    visible,
    propensity,
    reference,
    clip=0.05,
):
    observed = np.asarray(visible, dtype=bool)

    g = np.clip(
        propensity,
        clip,
        1.0,
    )

    weights = 1.0 / g[observed]

    ess = float(
        weights.sum() ** 2
        / np.square(weights).sum()
    )

    work = frame.loc[
        observed,
        [
            "risk_bin",
            "outcome",
        ],
    ].copy()

    work["weight"] = weights
    work["weighted_outcome"] = (
        work["weight"] * work["outcome"]
    )

    grouped = (
        work.groupby("risk_bin", sort=True)
        .agg(
            weight_sum=("weight", "sum"),
            weighted_outcome_sum=(
                "weighted_outcome",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["ipcw_event_rate"] = (
        grouped["weighted_outcome_sum"]
        / grouped["weight_sum"]
    )

    table = reference.merge(
        grouped[
            [
                "risk_bin",
                "ipcw_event_rate",
            ]
        ],
        on="risk_bin",
        how="left",
        validate="one_to_one",
    )

    if table["ipcw_event_rate"].isna().any():
        return np.nan, np.nan, ess

    ici = float(
        np.sum(
            table["reference_weight"]
            * np.abs(
                table["ipcw_event_rate"]
                - table["prediction_mean"]
            )
        )
    )

    curve_mae = float(
        np.sum(
            table["reference_weight"]
            * np.abs(
                table["ipcw_event_rate"]
                - table["linked_event_rate"]
            )
        )
    )

    return ici, curve_mae, ess


def bootstrap_main_metrics(
    frame,
    visibility,
    windows,
    n_bootstrap,
    seed,
):
    if n_bootstrap <= 0:
        return pd.DataFrame()

    print(
        f"\nRunning {n_bootstrap} paired bootstrap replicates..."
    )

    rng = np.random.default_rng(seed)

    prediction = frame["prediction"].to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy(dtype=float)

    original_bins = frame["risk_bin"].to_numpy(dtype=int)
    unique_bins = np.sort(np.unique(original_bins))

    bin_map = {
        value: index
        for index, value in enumerate(unique_bins)
    }

    bin_code = np.array(
        [bin_map[value] for value in original_bins],
        dtype=int,
    )

    k_bins = len(unique_bins)
    n = len(frame)

    rows = []

    for b in range(n_bootstrap):
        sample = rng.integers(
            0,
            n,
            size=n,
        )

        sb = bin_code[sample]
        sy = outcome[sample]
        sp = prediction[sample]

        ref_n = np.bincount(
            sb,
            minlength=k_bins,
        ).astype(float)

        pred_sum = np.bincount(
            sb,
            weights=sp,
            minlength=k_bins,
        )

        outcome_sum = np.bincount(
            sb,
            weights=sy,
            minlength=k_bins,
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            pred_mean = pred_sum / ref_n
            ref_rate = outcome_sum / ref_n

        ref_weight = ref_n / n

        linked = np.nansum(
            ref_weight
            * np.abs(
                ref_rate - pred_mean
            )
        )

        for window in windows:
            obs = visibility[window][sample]

            obs_bins = sb[obs]
            obs_y = sy[obs]

            obs_n = np.bincount(
                obs_bins,
                minlength=k_bins,
            ).astype(float)

            obs_y_sum = np.bincount(
                obs_bins,
                weights=obs_y,
                minlength=k_bins,
            )

            if np.any(obs_n == 0):
                continue

            obs_rate = obs_y_sum / obs_n

            restricted = np.sum(
                ref_weight
                * np.abs(
                    obs_rate - pred_mean
                )
            )

            curve_mae = np.sum(
                ref_weight
                * np.abs(
                    obs_rate - ref_rate
                )
            )

            rows.append(
                {
                    "bootstrap": b,
                    "extension_days": window,
                    "linked_ici": linked,
                    "restricted_ici": restricted,
                    "curve_mae_vs_linked": curve_mae,
                    "observed_fraction": float(
                        obs.mean()
                    ),
                }
            )

        if (b + 1) % 100 == 0:
            print(
                f"  bootstrap {b + 1}/{n_bootstrap}"
            )

    boot = pd.DataFrame(rows)

    summaries = []

    for window in windows:
        part = boot.loc[
            boot["extension_days"] == window
        ]

        if part.empty:
            continue

        row = {
            "extension_days": window,
            "bootstrap_replicates": len(part),
        }

        for metric in [
            "linked_ici",
            "restricted_ici",
            "curve_mae_vs_linked",
            "observed_fraction",
        ]:
            values = part[metric].to_numpy(dtype=float)

            row[f"{metric}_mean"] = float(
                np.mean(values)
            )

            row[f"{metric}_q025"] = float(
                np.quantile(values, 0.025)
            )

            row[f"{metric}_q975"] = float(
                np.quantile(values, 0.975)
            )

        summaries.append(row)

    return pd.DataFrame(summaries)


def make_plot(results, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        results["observed_fraction"],
        results["restricted_ici"],
        marker="o",
        label="Restricted audit",
    )

    ax.plot(
        results["observed_fraction"],
        results["linked_ici"],
        linestyle="--",
        label="Linked reference",
    )

    ax.plot(
        results["observed_fraction"],
        results["score_ipcw_ici_clip0p05"],
        marker="s",
        label="Score-IPCW, clip 0.05",
    )

    for _, row in results.iterrows():
        ax.annotate(
            f"+{int(row['extension_days'])}d",
            (
                row["observed_fraction"],
                row["restricted_ici"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Fraction of 30-day outcomes observable")
    ax.set_ylabel("Integrated calibration index")
    ax.set_title(
        "MIMIC-IV calibration across linked follow-up windows"
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows = sorted(
        {
            int(x.strip())
            for x in args.windows.split(",")
            if x.strip()
        }
    )

    if windows[0] != 0:
        raise ValueError(
            "Window sweep must include native window 0."
        )

    if windows[-1] < 30:
        raise ValueError(
            "Window sweep must include at least 30 days."
        )

    print("=" * 88)
    print(
        "ACTIONAUDIT PHASE 7D: "
        "MIMIC-IV REAL-DATA LINKAGE-WINDOW SWEEP"
    )
    print("=" * 88)

    print("Windows:", windows)

    frame = load_data(
        args.cohort,
        args.predictions,
        args.risk_bins,
    )

    death_lag = derive_post_discharge_death_lag(
        frame
    )

    hidden_death = (
        (frame["observed"] == 0)
        & (frame["outcome"] == 1)
    )

    hidden_death_count = int(
        hidden_death.sum()
    )

    hidden_death_with_date = int(
        (
            hidden_death
            & death_lag.notna()
        ).sum()
    )

    print(
        "\nHidden linked deaths:",
        hidden_death_count,
    )

    print(
        "Hidden linked deaths with usable "
        "post-discharge death timing:",
        hidden_death_with_date,
    )

    reference = reference_bin_table(frame)
    ref_ici = linked_ici(reference)

    print("\nFrozen reference bins")
    print(reference.to_string(index=False))

    print(
        "\nFixed-bin linked ICI:",
        f"{ref_ici:.6f}",
    )

    visibility = {}
    result_rows = []
    bin_rows = []

    for window in windows:
        visible = build_visibility(
            frame,
            death_lag,
            window,
        )

        visibility[window] = visible

        (
            restricted_ici,
            signed_distortion,
            curve_mae,
            bin_table,
        ) = restricted_metrics(
            frame,
            visible,
            reference,
        )

        g, observation_auc = (
            estimate_score_propensity(
                frame,
                visible,
                args.seed,
            )
        )

        (
            ipcw_ici,
            ipcw_curve_mae,
            ipcw_ess,
        ) = ipcw_metrics(
            frame,
            visible,
            g,
            reference,
            clip=0.05,
        )

        if np.all(visible):
            g_q01 = 1.0
            g_q05 = 1.0
            g_median = 1.0
            fraction_g_lt_005 = 0.0
        else:
            g_q01 = float(
                np.quantile(g, 0.01)
            )
            g_q05 = float(
                np.quantile(g, 0.05)
            )
            g_median = float(
                np.quantile(g, 0.50)
            )
            fraction_g_lt_005 = float(
                np.mean(g < 0.05)
            )

        result_rows.append(
            {
                "extension_days": window,
                "n": len(frame),
                "n_observed": int(
                    visible.sum()
                ),
                "observed_fraction": float(
                    visible.mean()
                ),
                "linked_event_rate": float(
                    frame["outcome"].mean()
                ),
                "observed_event_rate": float(
                    frame.loc[
                        visible,
                        "outcome",
                    ].mean()
                ),
                "linked_ici": ref_ici,
                "restricted_ici": restricted_ici,
                "restricted_signed_distortion": (
                    signed_distortion
                ),
                "restricted_curve_mae_vs_linked": (
                    curve_mae
                ),
                "score_observation_auc": (
                    observation_auc
                ),
                "score_g_q01": g_q01,
                "score_g_q05": g_q05,
                "score_g_median": g_median,
                "fraction_score_g_lt_0p05": (
                    fraction_g_lt_005
                ),
                "score_ipcw_ess_clip0p05": (
                    ipcw_ess
                ),
                "score_ipcw_ici_clip0p05": (
                    ipcw_ici
                ),
                "score_ipcw_curve_mae_vs_linked_clip0p05": (
                    ipcw_curve_mae
                ),
            }
        )

        bin_table = bin_table.copy()
        bin_table.insert(
            0,
            "extension_days",
            window,
        )

        bin_rows.append(bin_table)

        print("\n" + "-" * 88)
        print(
            f"Linked follow-up after discharge: "
            f"{window} days"
        )
        print("-" * 88)

        print(
            "Observed fraction :",
            f"{visible.mean():.6f}",
        )

        print(
            "Restricted ICI    :",
            f"{restricted_ici:.6f}",
        )

        print(
            "Curve MAE linked  :",
            f"{curve_mae:.6f}",
        )

        print(
            "Score g q05       :",
            f"{g_q05:.6f}",
        )

        print(
            "IPCW ESS clip .05 :",
            f"{ipcw_ess:.1f}",
        )

        print(
            "IPCW ICI clip .05 :",
            f"{ipcw_ici:.6f}",
        )

    results = pd.DataFrame(result_rows)

    bins_all = pd.concat(
        bin_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Paired bootstrap for main calibration quantities
    # --------------------------------------------------------------

    bootstrap_summary = bootstrap_main_metrics(
        frame,
        visibility,
        windows,
        args.bootstrap,
        args.seed,
    )

    # --------------------------------------------------------------
    # Validation gates
    # --------------------------------------------------------------

    checks = []

    add_check(
        checks,
        "expected_test_admissions",
        len(frame),
        EXPECTED_N,
        len(frame) == EXPECTED_N,
    )

    add_check(
        checks,
        "expected_native_observed_count",
        int(
            visibility[0].sum()
        ),
        EXPECTED_OBSERVED,
        int(
            visibility[0].sum()
        )
        == EXPECTED_OBSERVED,
    )

    native_fraction_matches = np.isclose(
        visibility[0].mean(),
        frame["observed"].mean(),
        atol=1e-12,
    )

    add_check(
        checks,
        "native_visibility_exactly_reproduced",
        native_fraction_matches,
        True,
        native_fraction_matches,
    )

    observed_fractions = (
        results["observed_fraction"]
        .to_numpy(dtype=float)
    )

    monotone = bool(
        np.all(
            np.diff(observed_fractions)
            >= -1e-12
        )
    )

    add_check(
        checks,
        "observed_fraction_monotone",
        monotone,
        True,
        monotone,
    )

    final_complete = bool(
        np.isclose(
            observed_fractions[-1],
            1.0,
            atol=1e-12,
        )
    )

    add_check(
        checks,
        "thirty_day_extension_complete",
        observed_fractions[-1],
        1.0,
        final_complete,
    )

    linked_rate_invariant = (
        results["linked_event_rate"].max()
        - results["linked_event_rate"].min()
    )

    add_check(
        checks,
        "linked_event_rate_invariant",
        linked_rate_invariant,
        0.0,
        linked_rate_invariant <= 1e-12,
    )

    linked_ici_invariant = (
        results["linked_ici"].max()
        - results["linked_ici"].min()
    )

    add_check(
        checks,
        "linked_ici_invariant",
        linked_ici_invariant,
        0.0,
        linked_ici_invariant <= 1e-12,
    )

    native_restricted = float(
        results.loc[
            results["extension_days"] == 0,
            "restricted_ici",
        ].iloc[0]
    )

    linked_close = (
        abs(
            ref_ici
            - MANUSCRIPT_LINKED_ICI
        )
        <= 0.02
    )

    add_check(
        checks,
        "linked_ici_consistent_with_manuscript",
        ref_ici,
        MANUSCRIPT_LINKED_ICI,
        linked_close,
    )

    restricted_close = (
        abs(
            native_restricted
            - MANUSCRIPT_RESTRICTED_ICI
        )
        <= 0.05
    )

    add_check(
        checks,
        "native_restricted_ici_consistent_with_manuscript",
        native_restricted,
        MANUSCRIPT_RESTRICTED_ICI,
        restricted_close,
    )

    finite_columns = [
        "observed_fraction",
        "linked_event_rate",
        "observed_event_rate",
        "linked_ici",
        "restricted_ici",
        "restricted_curve_mae_vs_linked",
        "score_g_q01",
        "score_g_q05",
        "score_g_median",
        "score_ipcw_ess_clip0p05",
        "score_ipcw_ici_clip0p05",
    ]

    all_finite = bool(
        np.isfinite(
            results[
                finite_columns
            ].to_numpy(dtype=float)
        ).all()
    )

    add_check(
        checks,
        "all_required_numeric_outputs_finite",
        all_finite,
        True,
        all_finite,
    )

    checks_df = pd.DataFrame(checks)

    passed = bool(
        checks_df["passed"].all()
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    results.to_csv(
        output_dir / "boundary_sweep.csv",
        index=False,
    )

    bins_all.to_csv(
        output_dir / "bin_level_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    bootstrap_summary.to_csv(
        output_dir / "bootstrap_summary.csv",
        index=False,
    )

    checks_df.to_csv(
        output_dir / "validation_checks.csv",
        index=False,
    )

    make_plot(
        results,
        output_dir
        / "phase7d_linkage_window_sweep.png",
    )

    definition_text = f"""ActionAudit Phase 7D boundary definition

Dataset:
MIMIC-IV frozen Phase 5C test cohort.

Population:
Exactly {len(frame)} frozen test admissions.

Outcome:
Fixed 30-day linked all-cause mortality.

Prediction:
Frozen Phase 5C admission-time prediction.

Calibration bins:
Frozen risk_bin assignments from the Phase 5G practitioner
nuisance artifact.

Linkage-window interpretation:
Window 0 reproduces the original hospital-only visibility rule.
For a positive post-discharge outcome, an extension makes the
outcome observable when the linked death occurs within the added
post-discharge follow-up window.
For a negative outcome, the label becomes determinable once the
added follow-up reaches the fixed 30-day horizon.
Window 30 therefore determines every 30-day binary outcome.

Death timing:
Exact post-discharge deathtime is used where available. Otherwise
MIMIC-IV subject-level DOD is used at calendar-day resolution.
Window 0 always uses the original frozen hospital visibility flag,
so date-only linked information cannot alter the native boundary.

Interpretation:
This is a retrospective real-data-grounded linkage-window sweep.
It is not an estimate of a causal hospital policy effect and does
not claim that these linkage windows correspond to actual historical
data-sharing policies.
"""

    (
        output_dir / "boundary_definition.txt"
    ).write_text(definition_text)

    status = (
        "phase7d_mimic_linkage_window_passed"
        if passed
        else "phase7d_mimic_linkage_window_failed"
    )

    (
        output_dir / "phase7d_status.txt"
    ).write_text(status + "\n")

    print("\n" + "=" * 88)
    print("PHASE 7D SUMMARY")
    print("=" * 88)

    print("\nBoundary sweep")
    print(
        results.to_string(
            index=False,
        )
    )

    print("\nBootstrap summary")
    print(
        bootstrap_summary.to_string(
            index=False,
        )
    )

    print("\nValidation checks")
    print(
        checks_df.to_string(
            index=False,
        )
    )

    print("\nSTATUS:", status)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
