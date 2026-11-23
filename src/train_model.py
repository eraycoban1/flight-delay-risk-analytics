"""
train_model.py
--------------
End-to-end modeling, hyperparameter tuning, explainability, and error analysis:
1. Statsmodels Logit: Clean statistical inference & Odds Ratios (excluding collinear sum).
2. Scikit-Learn Logistic Regression: Interpretable linear baseline.
3. Optuna-Tuned LightGBM: Non-linear gradient boosting with rolling congestion features.
4. SHAP Explainability: Dependence plot (net_turnaround_buffer) & Waterfall plot.
5. Post-Hoc Error Analysis: Root-cause breakdown of False Negatives & False Positives.
6. Probability Calibration Curve: Reliability diagram on chronological validation set.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import shap
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def find_optimal_threshold(y_true: pd.Series, probs: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train_and_analyze() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 150

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "flights_clean.parquet"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading clean analytical dataset...")
    df = pd.read_parquet(data_path)
    df = df.sort_values(by=["flight_date", "crs_dep_mins"]).reset_index(drop=True)

    # Chronological 80/20 Split
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    top_dests = train_df["dest"].value_counts(normalize=True)
    frequent_dests = set(top_dests[top_dests >= 0.01].index)

    for subset in (train_df, val_df):
        subset["dest_grouped"] = np.where(
            subset["dest"].isin(frequent_dests), subset["dest"], "OTHER"
        )

    # Full feature set for tree-based LightGBM (handles collinear derived features naturally)
    gbm_numeric_features = [
        "scheduled_dep_hour",
        "scheduled_flight_mins",
        "origin_daily_flight_volume",
        "origin_hourly_flight_volume",
        "route_daily_frequency",
        "origin_rolling_3h_dep_delay",
        "origin_rolling_3h_delay_rate",
        "carrier_origin_hist_delay_rate",
        "known_prev_aircraft_delay",
        "scheduled_turnaround_mins",
        "net_turnaround_buffer",
        "delay_to_buffer_ratio",
    ]
    # Linear models exclude 'net_turnaround_buffer' (exact linear combination) to prevent VIF inflation
    linear_numeric_features = [
        f for f in gbm_numeric_features if f != "net_turnaround_buffer"
    ]

    binary_features = ["is_weekend"]
    categorical_features = [
        "operating_airline",
        "origin",
        "dest_grouped",
        "dep_time_block",
    ]

    all_feature_cols = gbm_numeric_features + binary_features + categorical_features
    X_train, y_train = train_df[all_feature_cols], train_df["is_delayed"].astype(int)
    X_val, y_val = val_df[all_feature_cols], val_df["is_delayed"].astype(int)

    print(
        f"[INFO] Chronological Split -> Train: {len(X_train):,} ({train_df['flight_date'].min()} to {train_df['flight_date'].max()})"
    )
    print(
        f"[INFO] Chronological Split -> Val  : {len(X_val):,} ({val_df['flight_date'].min()} to {val_df['flight_date'].max()})"
    )

    # =========================================================================
    # ENGINE 1: STATSMODELS LOGIT (Clean Odds Ratios without Collinear Sum)
    # =========================================================================
    print("\n--- ENGINE 1: STATSMODELS STATISTICAL INFERENCE ---")
    scaler_sm = StandardScaler()
    X_train_sm = pd.DataFrame(
        scaler_sm.fit_transform(X_train[linear_numeric_features]),
        columns=linear_numeric_features,
        index=X_train.index,
    )
    X_train_sm["is_weekend"] = X_train["is_weekend"]

    dummies_sm = pd.get_dummies(
        X_train[["operating_airline", "origin", "dep_time_block"]],
        drop_first=True,
        dtype=float,
    )
    X_train_sm = pd.concat([X_train_sm, dummies_sm], axis=1)
    X_train_sm = sm.add_constant(X_train_sm)

    logit_model = sm.Logit(y_train, X_train_sm).fit(disp=False)

    conf = logit_model.conf_int()
    conf["Odds_Ratio"] = logit_model.params
    conf.columns = ["CI_Lower_95", "CI_Upper_95", "Odds_Ratio"]
    odds_df = np.exp(conf)[["Odds_Ratio", "CI_Lower_95", "CI_Upper_95"]]
    odds_df["p_value"] = logit_model.pvalues
    odds_df = odds_df.drop(index="const").sort_values(by="Odds_Ratio", ascending=False)
    odds_df.to_csv(reports_dir / "odds_ratios_summary.csv")

    print("[INFO] Top 8 Risk-Increasing Factors (Clean Odds Ratios):")
    print(odds_df.head(8).round(4).to_string())

    # =========================================================================
    # ENGINE 2: LOGISTIC REGRESSION BASELINE
    # =========================================================================
    print("\n--- ENGINE 2: LOGISTIC REGRESSION BASELINE ---")
    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), linear_numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )

    logit_pipeline = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
        ]
    )
    logit_pipeline.fit(X_train, y_train)

    logit_val_probs = logit_pipeline.predict_proba(X_val)[:, 1]
    logit_roc = roc_auc_score(y_val, logit_val_probs)
    logit_pr = average_precision_score(y_val, logit_val_probs)
    logit_brier = brier_score_loss(y_val, logit_val_probs)
    logit_thresh, logit_f1 = find_optimal_threshold(y_val, logit_val_probs)

    print(f"[LOGIT] ROC-AUC: {logit_roc:.4f} | PR-AUC: {logit_pr:.4f} | Brier: {logit_brier:.4f}")

    # =========================================================================
    # ENGINE 3: OPTUNA-TUNED LIGHTGBM BENCHMARK
    # =========================================================================
    print("\n--- ENGINE 3: OPTUNA HYPERPARAMETER TUNING (LIGHTGBM) ---")
    gbm_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", gbm_numeric_features + binary_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ]
    )

    X_train_enc = gbm_preprocessor.fit_transform(X_train)
    X_val_enc = gbm_preprocessor.transform(X_val)
    encoded_feature_names = gbm_numeric_features + binary_features + categorical_features

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 350,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "subsample_freq": 1,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train_enc, y_train)
        preds = model.predict_proba(X_val_enc)[:, 1]
        return average_precision_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    print(f"[OPTUNA] Best Trial PR-AUC: {study.best_value:.4f} | Params: {best_params}")

    best_lgbm = LGBMClassifier(
        n_estimators=400,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        **best_params,
    )
    best_lgbm.fit(X_train_enc, y_train)

    lgbm_pipeline = Pipeline(
        steps=[
            ("preprocessor", gbm_preprocessor),
            ("classifier", best_lgbm),
        ]
    )

    lgbm_val_probs = best_lgbm.predict_proba(X_val_enc)[:, 1]
    lgbm_roc = roc_auc_score(y_val, lgbm_val_probs)
    lgbm_pr = average_precision_score(y_val, lgbm_val_probs)
    lgbm_brier = brier_score_loss(y_val, lgbm_val_probs)
    lgbm_thresh, lgbm_f1 = find_optimal_threshold(y_val, lgbm_val_probs)

    print(f"[LGBM ] ROC-AUC: {lgbm_roc:.4f} | PR-AUC: {lgbm_pr:.4f} | Brier: {lgbm_brier:.4f}")
    print(f"[LGBM ] F1-Optimal Threshold: {lgbm_thresh:.4f} (Val F1: {lgbm_f1:.4f})")

    # =========================================================================
    # STEP 2: POST-HOC ERROR ANALYSIS (FALSE NEGATIVES & FALSE POSITIVES)
    # =========================================================================
    print("\n--- POST-HOC ERROR ANALYSIS (VALIDATION SET) ---")
    val_analysis = val_df.copy()
    val_analysis["pred_prob"] = lgbm_val_probs
    val_analysis["pred_label"] = (lgbm_val_probs >= lgbm_thresh).astype(int)

    fn_df = val_analysis[(val_analysis["is_delayed"] == 1) & (val_analysis["pred_label"] == 0)]
    fp_df = val_analysis[(val_analysis["is_delayed"] == 0) & (val_analysis["pred_label"] == 1)]
    tp_df = val_analysis[(val_analysis["is_delayed"] == 1) & (val_analysis["pred_label"] == 1)]

    print(f"[ERROR] True Positives : {len(tp_df):,} | False Negatives: {len(fn_df):,} | False Positives: {len(fp_df):,}")

    # Root-cause breakdown of missed delays (False Negatives)
    delay_causes = ["carrier_delay", "weather_delay", "nas_delay", "late_aircraft_delay", "security_delay"]
    fn_cause_shares = (fn_df[delay_causes].sum() / fn_df[delay_causes].sum().sum() * 100).round(2)
    print("\n[FN ROOT CAUSE] Share of Total Delay Minutes in Missed Delays (False Negatives):")
    for cause, pct in fn_cause_shares.items():
        print(f"  - {cause:22s}: {pct:.2f}%")

    # Turnaround recovery analysis in False Positives vs True Positives
    print("\n[FP RECOVERY] Mean Operational Metrics (False Positives vs True Positives):")
    print(f"  - Mean Net Turnaround Buffer -> FP (Recovered): {fp_df['net_turnaround_buffer'].mean():.1f} mins | TP (Delayed): {tp_df['net_turnaround_buffer'].mean():.1f} mins")
    print(f"  - Mean Rolling 3h Dep Delay  -> FP (Recovered): {fp_df['origin_rolling_3h_dep_delay'].mean():.1f} mins | TP (Delayed): {tp_df['origin_rolling_3h_dep_delay'].mean():.1f} mins")

    # =========================================================================
    # STEP 3: SHAP EXPLAINABILITY (DEPENDENCE & WATERFALL PLOTS)
    # =========================================================================
    print("\n--- SHAP EXPLAINABILITY ANALYSIS ---")
    X_val_enc_df = pd.DataFrame(X_val_enc, columns=encoded_feature_names).astype(float)
    shap_sample = X_val_enc_df.iloc[:5000].copy()

    explainer = shap.TreeExplainer(best_lgbm)
    shap_values = explainer(shap_sample)

    # Plot A: SHAP Dependence Plot for net_turnaround_buffer
    fig, ax = plt.subplots(figsize=(9, 5.5))
    shap.dependence_plot(
        "net_turnaround_buffer",
        shap_values.values,
        shap_sample,
        interaction_index="origin_rolling_3h_dep_delay",
        ax=ax,
        show=False,
    )
    ax.set_title(
        "SHAP Dependence: Non-Linear Threshold Effect of Net Turnaround Buffer",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "05_shap_dependence_turnaround.png")
    plt.close()

    # Plot B: SHAP Waterfall Plot for a High-Risk Sample Flight
    sample_probs = best_lgbm.predict_proba(shap_sample)[:, 1]
    high_risk_idx = int(np.argmax(sample_probs))
    flight_meta = val_df.iloc[high_risk_idx]
    print(
        f"[SHAP] Waterfall Flight Example: {flight_meta['operating_airline']}{flight_meta['flight_number']} "
        f"({flight_meta['origin']}->{flight_meta['dest']}) | Predicted Delay Risk: {sample_probs[high_risk_idx]*100:.1f}%"
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=10, show=False)
    plt.title(
        f"SHAP Explanation: Flight {flight_meta['operating_airline']}{flight_meta['flight_number']} "
        f"({flight_meta['origin']}->{flight_meta['dest']}) — Risk: {sample_probs[high_risk_idx]*100:.1f}%",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "06_shap_waterfall_single_flight.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # STEP 4: VALIDATION CALIBRATION CURVE (RELIABILITY DIAGRAM)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, probs, color in [
        ("Logistic Regression", logit_val_probs, "#1f77b4"),
        ("Optuna LightGBM", lgbm_val_probs, "#ff7f0e"),
    ]:
        prob_true, prob_pred = calibration_curve(y_val, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", lw=2, color=color, label=name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", label="Perfectly Calibrated")
    ax.set_title("Validation Probability Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Delay Probability", fontsize=10.5)
    ax.set_ylabel("Observed Fraction of Delayed Flights", fontsize=10.5)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "07_validation_calibration_curve.png")
    plt.close()

    # Save artifacts
    artifact = {
        "logit_pipeline": logit_pipeline,
        "lgbm_pipeline": lgbm_pipeline,
        "frequent_dests": list(frequent_dests),
        "feature_cols": all_feature_cols,
        "logit_threshold": logit_thresh,
        "lgbm_threshold": lgbm_thresh,
        "val_metrics": {
            "logit": {"roc_auc": logit_roc, "pr_auc": logit_pr, "brier": logit_brier},
            "lgbm": {"roc_auc": lgbm_roc, "pr_auc": lgbm_pr, "brier": lgbm_brier},
        },
    }
    model_path = models_dir / "flight_delay_models.joblib"
    joblib.dump(artifact, model_path)
    print(f"\n[SUCCESS] All models, SHAP plots, Calibration diagram, and Error Analysis completed!")


if __name__ == "__main__":
    train_and_analyze()