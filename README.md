# Flight Delay Risk Analytics: End-to-End Statistical & Machine Learning Pipeline

An end-to-end, leak-free predictive analytics and statistical inference pipeline engineered to quantify **pre-flight arrival delay risk ($\ge 15$ minutes)** at the **T-180 minute (3-hour prior)** operational horizon across the 10 busiest U.S. hub airports.

Built on **3.85+ million raw U.S. Department of Transportation (BTS Form 234)** flight records (January–August 2026), this project combines high-throughput **DuckDB SQL** feature engineering, **Statsmodels** econometric inference (Odds Ratios & VIF diagnostics), **Scikit-Learn** linear baselines, **Optuna-tuned LightGBM**, **SHAP** explainability, and **Out-of-Time (OOT)** validation.

---

## Executive Summary & Key Findings

* **High-Throughput SQL ETL:** Ingested and transformed **3,857,016 raw pipe-delimited ASCII records** into a leak-free analytical dataset in **6.54 seconds** using in-memory DuckDB and SQL Window Functions.
* **Strict Temporal & Horizon Integrity:** Enforced a **T-180 minute prediction cutoff** (`RANGE BETWEEN 360 PRECEDING AND 180 PRECEDING`) so that rolling congestion metrics and aircraft propagation delays rely strictly on information physically observable 3 hours before scheduled departure.
* **Strong Out-of-Time (OOT) Generalization:** Trained chronologically on **Jan–Jun 2026** and evaluated on unseen peak-summer traffic (**Jul–Aug 2026**, $N = 400,000$):
  * **Optuna LightGBM:** Achieved **0.7154 ROC-AUC** and **0.5281 PR-AUC** (vs. `0.2933` baseline), capturing **74% of all delayed flights** at the F1-optimal threshold (`0.239`).
  * **Logistic Regression Baseline:** Achieved **0.6924 ROC-AUC** and **0.4767 PR-AUC**, providing full parametric interpretability.
* **Near-Perfect Probability Calibration:** Across 400,000 unseen summer flights, predicted delay probabilities aligned within **<0.6 percentage points** of actual observed delay rates across all operational risk tiers.

---

## Model Performance & Out-of-Time (Jul–Aug 2026) Benchmark

| Model Engine | Chronological Val ROC-AUC | Chronological Val PR-AUC | Unseen OOT ROC-AUC | Unseen OOT PR-AUC | OOT Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2 Baseline)** | 0.6985 | 0.4578 | **0.6924** | **0.4767** | 0.1874 |
| **Optuna-Tuned LightGBM** | 0.7274 | 0.5151 | **0.7154** | **0.5281** | **0.1802** |

### Operational Risk Tier Segmentation (Unseen Jul–Aug 2026 Test Set)

| Operational Risk Tier | Flight Volume | Portfolio Share | Avg. Predicted Risk | Actual Observed Delay Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Low Risk (<20%)** | 139,062 | 34.77% | 14.36% | **14.27%** |
| **Moderate Risk (20–35%)** | 141,568 | 35.39% | 26.49% | **26.43%** |
| **High Risk (>35%)** | 119,370 | 29.84% | 49.70% | **50.31%** |

---

## Statistical Inference & Key Delay Drivers

Using `statsmodels.Logit` with standardized continuous predictors and Variance Inflation Factor (VIF) multicollinearity filtering:

1. **Aircraft Delay Propagation (`known_prev_aircraft_delay`):** A 1-SD increase in known incoming aircraft delay increases the odds of arrival delay by **+33.5%** ($\text{OR} = 1.3352$, $95\% \text{ CI } [1.310, 1.361]$, $p < 0.001$).
2. **Rolling 3-Hour Hub Congestion (`origin_rolling_3h_delay_rate`):** Acts as a real-time proxy for convective weather and ATC ground stops; a 1-SD surge increases delay odds by **+28.3%** ($\text{OR} = 1.2827$, $p < 0.001$).
3. **Carrier-Hub Historical Reliability (`carrier_origin_hist_delay_rate`):** Expanding historical delay rate increases odds by **+24.3%** ($\text{OR} = 1.2428$, $p < 0.001$).
4. **Early Morning Buffer Advantage (`dep_time_block_Early_Morning`):** Flights departing between 05:00–08:59 exhibit **45.3% lower delay odds** ($\text{OR} = 0.5466$, $p < 0.001$) compared to afternoon departures due to overnight schedule resets.

---

## Post-Hoc Error Analysis (Root-Cause Breakdown)

Analyzing **False Negatives** (delays missed at the T-180 min horizon) against post-flight BTS delay cause attributions revealed:
* **Carrier Technical/Crew Issues (`carrier_delay`):** **39.41%** of missed delay minutes.
* **Late Inbound Aircraft (`late_aircraft_delay`):** **32.04%** (aircraft whose inbound leg landed inside the T-180 min window).
* **National Airspace / ATC (`nas_delay`):** **22.30%**.
* **Severe Weather (`weather_delay`):** **6.18%**.

*Actionable Insight:* Narrowing the prediction horizon from **T-180 minutes** to **T-60 minutes** would expose inbound aircraft telemetry for the 32.04% late-aircraft segment, substantially lifting recall.

---

## Project Architecture

```text
flight-delay-risk-analytics/
├── data/
│   ├── raw/                     # BTS Form 234 training files (Jan-Jun 2026)
│   ├── oot/                     # Unseen Out-of-Time test files (Jul-Aug 2026)
│   └── processed/               # Snappy-compressed Parquet analytical dataset
├── models/
│   └── flight_delay_models.joblib # Serialized Logistic Regression & LightGBM pipelines
├── reports/
│   ├── figures/                 # EDA, Odds Ratio Forest Plot, SHAP, ROC/PR & Calibration plots
│   └── odds_ratios_summary.csv  # Full Statsmodels inference table (OR, 95% CI, p-values)
├── sql/
│   ├── 01_schema_setup.sql      # Headerless ASCII ingestion & strict schema mapping
│   └── 02_feature_query.sql     # T-180 min rolling window & turnaround feature engineering
├── src/
│   ├── build_dataset.py         # DuckDB ETL & Parquet export pipeline
│   ├── eda_analysis.py          # Hypothesis tests (Chi-Square, Mann-Whitney U) & VIF checks
│   ├── train_model.py           # Dual-engine training, Optuna tuning, SHAP & Error Analysis
│   └── evaluate.py              # Out-of-Time (Jul-Aug 2026) validation & risk segmentation
├── .gitignore
├── README.md
└── requirements.txt


# 1. Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Linux/macOS: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build leak-free Parquet dataset from raw BTS .asc files
python src/build_dataset.py

# 4. Run statistical hypothesis tests, VIF diagnostics, and EDA plots
python src/eda_analysis.py

# 5. Train models, tune LightGBM via Optuna, and generate SHAP & Error Analysis
python src/train_model.py

# 6. Evaluate generalization on unseen Out-of-Time data (Jul-Aug 2026)
python src/evaluate.py
