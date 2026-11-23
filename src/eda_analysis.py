"""
eda_analysis.py
---------------
Performs statistical exploratory data analysis (EDA), formal hypothesis testing
(Chi-Square & Mann-Whitney U), Variance Inflation Factor (VIF) multicollinearity
diagnostics, and generates publication-ready figures for the Flight Delay Risk project.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def run_eda() -> None:
    # Set visual style
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams["figure.dpi"] = 150

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "flights_clean.parquet"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    print("[INFO] Loading processed Parquet dataset...")
    df = pd.read_parquet(data_path)
    print(f"[INFO] Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 1. Missing Value & Integrity Check
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"[CHECK] Total missing values across all features: {total_missing}")

    # 2. Statistical Hypothesis Testing
    print("\n--- STATISTICAL HYPOTHESIS TESTS ---")

    # Test A: Chi-Square Test of Independence (Time-of-Day Block vs. Flight Delay)
    contingency_time = pd.crosstab(df["dep_time_block"], df["is_delayed"])
    chi2_stat, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_time)
    print(
        f"[TEST 1] Chi-Square (dep_time_block vs is_delayed): "
        f"chi2 = {chi2_stat:.2f}, dof = {dof}, p-value = {p_val_chi2:.4e}"
    )

    # Test B: Mann-Whitney U Test (Previous Aircraft Arrival Delay: Delayed vs On-Time)
    delayed_prev = df.loc[df["is_delayed"] == 1, "prev_aircraft_arr_delay"]
    ontime_prev = df.loc[df["is_delayed"] == 0, "prev_aircraft_arr_delay"]
    u_stat, p_val_mw = stats.mannwhitneyu(
        delayed_prev, ontime_prev, alternative="greater"
    )
    print(
        f"[TEST 2] Mann-Whitney U (prev_aircraft_arr_delay | delayed > on-time): "
        f"U = {u_stat:.2e}, p-value = {p_val_mw:.4e}"
    )

    # Test C: Mann-Whitney U Test (Hourly Airport Congestion: Delayed vs On-Time)
    delayed_vol = df.loc[df["is_delayed"] == 1, "origin_hourly_flight_volume"]
    ontime_vol = df.loc[df["is_delayed"] == 0, "origin_hourly_flight_volume"]
    u_vol, p_val_vol = stats.mannwhitneyu(
        delayed_vol, ontime_vol, alternative="greater"
    )
    print(
        f"[TEST 3] Mann-Whitney U (origin_hourly_flight_volume | delayed > on-time): "
        f"U = {u_vol:.2e}, p-value = {p_val_vol:.4e}"
    )

    # 3. Multicollinearity Diagnostics (Variance Inflation Factor - VIF)
    print("\n--- MULTICOLLINEARITY DIAGNOSTICS (VIF) ---")
    numeric_features = [
        "scheduled_dep_hour",
        "scheduled_flight_mins",
        "origin_daily_flight_volume",
        "origin_hourly_flight_volume",
        "carrier_origin_hist_delay_rate",
        "prev_aircraft_arr_delay",
        "scheduled_turnaround_mins",
        "is_weekend",
    ]
    X_vif = add_constant(df[numeric_features].dropna())
    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [
            variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])
        ],
    })
    vif_df = vif_df[vif_df["Feature"] != "const"].sort_values(
        by="VIF", ascending=False
    )
    print(vif_df.to_string(index=False))

    # 4. Generate Visualizations
    print("\n[INFO] Generating EDA visualizations in reports/figures/...")

    # Figure 1: Delay Rate by Scheduled Departure Time Block
    time_order = [
        "Early_Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Night_RedEye",
    ]
    time_summary = (
        df.groupby("dep_time_block", observed=False)["is_delayed"]
        .agg(["mean", "count"])
        .reindex(time_order)
        .reset_index()
    )
    time_summary["delay_pct"] = time_summary["mean"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = sns.barplot(
        data=time_summary,
        x="dep_time_block",
        y="delay_pct",
        hue="dep_time_block",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.axhline(
        df["is_delayed"].mean() * 100,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall Mean ({df['is_delayed'].mean()*100:.1f}%)",
    )
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(
        "Arrival Delay Rate (>=15 mins) by Departure Time Block",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Scheduled Departure Time Block", fontsize=11)
    ax.set_ylabel("Delay Rate (%)", fontsize=11)
    ax.set_ylim(0, time_summary["delay_pct"].max() * 1.18)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "01_delay_rate_by_time_block.png")
    plt.close()

    # Figure 2: Delay Rate across Top 10 US Origin Hubs
    origin_summary = (
        df.groupby("origin")["is_delayed"]
        .agg(["mean", "count"])
        .sort_values(by="mean", ascending=False)
        .reset_index()
    )
    origin_summary["delay_pct"] = origin_summary["mean"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(
        data=origin_summary,
        x="origin",
        y="delay_pct",
        hue="origin",
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.axhline(
        df["is_delayed"].mean() * 100,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Top-10 Hubs Mean ({df['is_delayed'].mean()*100:.1f}%)",
    )
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_title(
        "Arrival Delay Rate Across Top 10 Busiest US Origin Airports",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Origin Airport (IATA Code)", fontsize=11)
    ax.set_ylabel("Delay Rate (%)", fontsize=11)
    ax.set_ylim(0, origin_summary["delay_pct"].max() * 1.18)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "02_delay_rate_by_origin.png")
    plt.close()

    # Figure 3: Correlation Heatmap of Continuous Predictors
    corr_matrix = df[numeric_features].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        "Pearson Correlation Matrix of Pre-Flight Numerical Features",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "03_feature_correlation_heatmap.png")
    plt.close()

    print("[SUCCESS] EDA completed. 3 figures saved to reports/figures/.")


if __name__ == "__main__":
    run_eda()