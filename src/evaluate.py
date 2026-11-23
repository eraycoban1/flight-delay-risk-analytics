"""
evaluate.py
-----------
Performs Out-of-Time (OOT) validation on unseen future months (July-August 2026)
comparing the Logistic Regression baseline and Optuna-tuned LightGBM model.
Generates comparative ROC, Precision-Recall, and Calibration curves plus risk tiers.
"""

from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def build_oot_features(oot_dir: Path, feature_sql_path: Path) -> pd.DataFrame:
    """Runs the exact DuckDB schema and T-180 min rolling feature engineering on OOT files."""
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE raw_flights AS
            SELECT
                CAST(column00 AS VARCHAR) AS marketing_airline,
                CAST(column01 AS INTEGER) AS flight_number,
                CAST(column04 AS VARCHAR) AS operating_airline,
                CAST(column06 AS VARCHAR) AS origin,
                CAST(column07 AS VARCHAR) AS dest,
                STRPTIME(CAST(column08 AS VARCHAR), '%Y%m%d')::DATE AS flight_date,
                CAST(column09 AS INTEGER) AS day_of_week,
                LPAD(CAST(column10 AS VARCHAR), 4, '0') AS crs_dep_time_str,
                LPAD(CAST(column13 AS VARCHAR), 4, '0') AS crs_arr_time_str,
                CAST(column16 AS INTEGER) AS cancelled,
                CAST(column17 AS INTEGER) AS diverted,
                CAST(column18 AS DOUBLE)  AS crs_elapsed_time,
                CAST(column20 AS DOUBLE)  AS dep_delay,
                CAST(column21 AS DOUBLE)  AS arr_delay,
                NULLIF(TRIM(CAST(column25 AS VARCHAR)), '') AS tail_number,
                COALESCE(CAST(column30 AS DOUBLE), 0.0) AS carrier_delay,
                COALESCE(CAST(column31 AS DOUBLE), 0.0) AS weather_delay,
                COALESCE(CAST(column32 AS DOUBLE), 0.0) AS nas_delay,
                COALESCE(CAST(column33 AS DOUBLE), 0.0) AS security_delay,
                COALESCE(CAST(column34 AS DOUBLE), 0.0) AS late_aircraft_delay,
                CAST(column70 AS VARCHAR) AS duplicate_flag
            FROM read_csv(
                '{(oot_dir / "*.asc").as_posix()}',
                delim = '|',
                header = false,
                auto_detect = true,
                null_padding = true
            );
        """)
        feature_sql = feature_sql_path.read_text(encoding="utf-8")
        con.execute(feature_sql)
        return con.execute("SELECT * FROM model_features").df()
    finally:
        con.close()


def evaluate_oot() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 150

    project_root = Path(__file__).resolve().parent.parent
    oot_dir = project_root / "data" / "oot"
    feature_sql_path = project_root / "sql" / "02_feature_query.sql"
    model_path = project_root / "models" / "flight_delay_models.joblib"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    oot_files = list(oot_dir.glob("*.asc"))
    if not oot_files:
        raise FileNotFoundError(
            f"No .asc files found in {oot_dir}. Place 202607 & 202608 .asc files there."
        )

    print(f"[INFO] Found {len(oot_files)} unseen future file(s) in {oot_dir}")
    df_oot = build_oot_features(oot_dir, feature_sql_path)
    print(
        f"[INFO] OOT evaluation dataset: {len(df_oot):,} rows | "
        f"Actual Delay Rate: {df_oot['is_delayed'].mean()*100:.2f}%"
    )

    artifact = joblib.load(model_path)
    logit_pipe = artifact["logit_pipeline"]
    lgbm_pipe = artifact["lgbm_pipeline"]
    frequent_dests = set(artifact["frequent_dests"])
    feature_cols = artifact["feature_cols"]
    logit_thresh = artifact["logit_threshold"]
    lgbm_thresh = artifact["lgbm_threshold"]

    df_oot["dest_grouped"] = np.where(
        df_oot["dest"].isin(frequent_dests), df_oot["dest"], "OTHER"
    )
    X_oot = df_oot[feature_cols].copy()
    y_oot = df_oot["is_delayed"].astype(int)

    logit_probs = logit_pipe.predict_proba(X_oot)[:, 1]
    lgbm_probs = lgbm_pipe.predict_proba(X_oot)[:, 1]

    print("\n--- OUT-OF-TIME (JULY-AUGUST 2026) GENERALIZATION BENCHMARK ---")
    for name, probs in [("Logistic Regression", logit_probs), ("Optuna LightGBM", lgbm_probs)]:
        roc = roc_auc_score(y_oot, probs)
        pr = average_precision_score(y_oot, probs)
        brier = brier_score_loss(y_oot, probs)
        print(f"[{name:19s}] ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Brier: {brier:.4f}")

    # Operational Risk Tier Segmentation using LightGBM calibrated probabilities
    df_oot["pred_prob"] = lgbm_probs
    df_oot["risk_tier"] = pd.cut(
        df_oot["pred_prob"],
        bins=[0.0, 0.20, 0.35, 1.0],
        labels=["Low Risk (<20%)", "Moderate Risk (20-35%)", "High Risk (>35%)"],
    )
    risk_summary = (
        df_oot.groupby("risk_tier", observed=False)
        .agg(
            Flight_Count=("is_delayed", "count"),
            Share_Pct=("is_delayed", lambda s: len(s) / len(df_oot) * 100),
            Actual_Delay_Rate_Pct=("is_delayed", lambda s: s.mean() * 100),
            Avg_Predicted_Prob_Pct=("pred_prob", lambda s: s.mean() * 100),
        )
        .round(2)
    )
    print("\n--- OPERATIONAL RISK TIER SEGMENTATION (OOT - LIGHTGBM) ---")
    print(risk_summary.to_string())

    print(f"\n--- CLASSIFICATION REPORT: LOGISTIC REGRESSION (Threshold = {logit_thresh:.3f}) ---")
    print(classification_report(y_oot, (logit_probs >= logit_thresh).astype(int), target_names=["On-Time (0)", "Delayed (1)"]))

    print(f"--- CLASSIFICATION REPORT: OPTUNA LIGHTGBM (Threshold = {lgbm_thresh:.3f}) ---")
    print(classification_report(y_oot, (lgbm_probs >= lgbm_thresh).astype(int), target_names=["On-Time (0)", "Delayed (1)"]))

    # Plot 1: Comparative ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for name, probs, color in [
        ("Logistic Regression", logit_probs, "#1f77b4"),
        ("Optuna LightGBM", lgbm_probs, "#ff7f0e"),
    ]:
        fpr, tpr, _ = roc_curve(y_oot, probs)
        prec, rec, _ = precision_recall_curve(y_oot, probs)
        axes[0].plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC = {roc_auc_score(y_oot, probs):.3f})")
        axes[1].plot(rec, prec, lw=2.2, color=color, label=f"{name} (PR-AUC = {average_precision_score(y_oot, probs):.3f})")

    axes[0].plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.2, label="Random (AUC = 0.500)")
    axes[0].set_title("Out-of-Time ROC Curve (Jul-Aug 2026)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10.5)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10.5)
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].axhline(y_oot.mean(), color="#d62728", linestyle="--", lw=1.2, label=f"Baseline ({y_oot.mean():.3f})")
    axes[1].set_title("Out-of-Time Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10.5)
    axes[1].set_ylabel("Precision", fontsize=10.5)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(figures_dir / "08_oot_roc_and_pr_curves.png")
    plt.close()

    # Plot 2: Comparative Probability Calibration Curve
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, probs, color in [
        ("Logistic Regression", logit_probs, "#1f77b4"),
        ("Optuna LightGBM", lgbm_probs, "#ff7f0e"),
    ]:
        prob_true, prob_pred = calibration_curve(y_oot, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", lw=2, color=color, label=name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", label="Perfectly Calibrated")
    ax.set_title("Out-of-Time Probability Calibration (Jul-Aug 2026)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Delay Probability", fontsize=10.5)
    ax.set_ylabel("Observed Fraction of Delayed Flights", fontsize=10.5)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "09_oot_calibration_curve.png")
    plt.close()

    print("[SUCCESS] OOT evaluation completed and figures saved to reports/figures/.")


if __name__ == "__main__":
    evaluate_oot()