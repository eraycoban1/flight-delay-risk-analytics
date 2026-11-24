"""
app/main.py
-----------
Production FastAPI V2 inference service for T-180 min Flight Delay Risk Analytics.
Supports single-flight SHAP explainability (/predict), batch fleet scoring (/predict/batch),
and model benchmark metadata (/model/metadata).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "flight_delay_models.joblib"

ml_state: dict = {}


def get_dep_time_block(hour: int) -> str:
    if 5 <= hour <= 8:
        return "Early_Morning"
    elif 9 <= hour <= 12:
        return "Morning"
    elif 13 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    return "Late_Night"


def classify_risk_tier(prob: float) -> tuple[Literal["Low", "Moderate", "High"], str]:
    if prob < 0.20:
        return "Low", "Normal operations; standard gate & crew buffer sufficient."
    elif prob <= 0.35:
        return "Moderate", "Elevated risk; monitor inbound tail and pre-stage gate turnaround crew."
    return "High", "Critical delay risk; evaluate proactive tail swap or passenger rebooking buffer."


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run src/train_model.py first.")
    artifact = joblib.load(MODEL_PATH)
    lgbm_pipeline = artifact["lgbm_pipeline"]
    ml_state["pipeline"] = lgbm_pipeline
    ml_state["preprocessor"] = lgbm_pipeline.named_steps["preprocessor"]
    ml_state["classifier"] = lgbm_pipeline.named_steps["classifier"]
    ml_state["explainer"] = shap.TreeExplainer(lgbm_pipeline.named_steps["classifier"])
    ml_state["frequent_dests"] = set(artifact["frequent_dests"])
    ml_state["feature_cols"] = artifact["feature_cols"]
    ml_state["threshold"] = artifact["lgbm_threshold"]
    ml_state["val_metrics"] = artifact.get("val_metrics", {})
    yield
    ml_state.clear()


app = FastAPI(
    title="Flight Delay Risk Analytics API (T-180 Min Horizon)",
    description="Enterprise-grade pre-flight arrival delay risk scoring (>=15 min) with real-time SHAP explainability and batch fleet scanning.",
    version="2.0.0",
    lifespan=lifespan,
)


class FlightRequest(BaseModel):
    operating_airline: str = Field(..., json_schema_extra={"example": "DL"}, description="2-letter IATA carrier code")
    origin: str = Field(..., json_schema_extra={"example": "ATL"}, description="Origin hub IATA code")
    dest: str = Field(..., json_schema_extra={"example": "LAX"}, description="Destination IATA code")
    scheduled_dep_hour: int = Field(..., ge=0, le=23, json_schema_extra={"example": 19})
    scheduled_flight_mins: float = Field(..., gt=0, json_schema_extra={"example": 285.0})
    is_weekend: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    origin_daily_flight_volume: int = Field(..., ge=1, json_schema_extra={"example": 980})
    origin_hourly_flight_volume: int = Field(..., ge=1, json_schema_extra={"example": 74})
    route_daily_frequency: int = Field(..., ge=1, json_schema_extra={"example": 9})
    origin_rolling_3h_dep_delay: float = Field(..., json_schema_extra={"example": 34.0})
    origin_rolling_3h_delay_rate: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.42})
    carrier_origin_hist_delay_rate: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.28})
    known_prev_aircraft_delay: float = Field(..., json_schema_extra={"example": 65.0})
    scheduled_turnaround_mins: float = Field(..., json_schema_extra={"example": 55.0})


class ShapDriver(BaseModel):
    feature: str
    feature_value: str | float | int
    shap_impact: float
    direction: Literal["Increases Risk", "Reduces Risk"]


class PredictionResponse(BaseModel):
    delay_probability: float
    risk_tier: Literal["Low", "Moderate", "High"]
    is_predicted_delayed_f1_opt: bool
    operational_recommendation: str
    derived_metrics: dict[str, float | str]
    top_3_drivers: list[ShapDriver]


class BatchPredictionResponse(BaseModel):
    total_flights: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    mean_delay_probability: float
    predictions: list[PredictionResponse]


FEATURE_LABELS = {
    "net_turnaround_buffer": "Ground Turnaround Margin (mins)",
    "delay_to_buffer_ratio": "Inbound Delay / Buffer Ratio",
    "known_prev_aircraft_delay": "Incoming Aircraft Delay (mins)",
    "origin_rolling_3h_dep_delay": "Last 3-Hour Airport Congestion (mins)",
    "origin_rolling_3h_delay_rate": "Last 3-Hour Hub Delay Frequency",
    "carrier_origin_hist_delay_rate": "Airline Historical Delay Rate at Hub",
    "operating_airline": "Operating Carrier",
    "origin_daily_flight_volume": "Hub Daily Departure Volume",
    "origin_hourly_flight_volume": "Hub Hourly Departure Load",
    "scheduled_turnaround_mins": "Scheduled Ground Turnaround (mins)",
    "scheduled_dep_hour": "Scheduled Departure Hour (Local)",
    "scheduled_flight_mins": "Scheduled Block Duration (mins)",
    "route_daily_frequency": "Daily Route Flight Frequency",
    "is_weekend": "Weekend Schedule Indicator",
    "origin": "Origin Hub Airport",
    "dest_grouped": "Destination Airport",
    "dep_time_block": "Departure Time Window",
}


def build_feature_row(flight: FlightRequest) -> dict:
    net_turnaround_buffer = flight.scheduled_turnaround_mins - max(flight.known_prev_aircraft_delay, 0.0)
    safe_turnaround = max(flight.scheduled_turnaround_mins, 1.0)
    delay_to_buffer_ratio = max(flight.known_prev_aircraft_delay, 0.0) / safe_turnaround
    dest_grouped = flight.dest.upper() if flight.dest.upper() in ml_state["frequent_dests"] else "OTHER"
    dep_time_block = get_dep_time_block(flight.scheduled_dep_hour)

    return {
        "scheduled_dep_hour": flight.scheduled_dep_hour,
        "scheduled_flight_mins": flight.scheduled_flight_mins,
        "origin_daily_flight_volume": flight.origin_daily_flight_volume,
        "origin_hourly_flight_volume": flight.origin_hourly_flight_volume,
        "route_daily_frequency": flight.route_daily_frequency,
        "origin_rolling_3h_dep_delay": flight.origin_rolling_3h_dep_delay,
        "origin_rolling_3h_delay_rate": flight.origin_rolling_3h_delay_rate,
        "carrier_origin_hist_delay_rate": flight.carrier_origin_hist_delay_rate,
        "known_prev_aircraft_delay": flight.known_prev_aircraft_delay,
        "scheduled_turnaround_mins": flight.scheduled_turnaround_mins,
        "net_turnaround_buffer": net_turnaround_buffer,
        "delay_to_buffer_ratio": delay_to_buffer_ratio,
        "is_weekend": flight.is_weekend,
        "operating_airline": flight.operating_airline.upper(),
        "origin": flight.origin.upper(),
        "dest_grouped": dest_grouped,
        "dep_time_block": dep_time_block,
    }


def score_single_flight(flight: FlightRequest) -> PredictionResponse:
    row_dict = build_feature_row(flight)
    feature_cols = ml_state["feature_cols"]
    input_df = pd.DataFrame([row_dict])[feature_cols]

    prob = float(ml_state["pipeline"].predict_proba(input_df)[0, 1])
    risk_tier, recommendation = classify_risk_tier(prob)

    X_enc = ml_state["preprocessor"].transform(input_df)
    shap_vals = ml_state["explainer"].shap_values(X_enc)
    flight_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    top_indices = np.argsort(np.abs(flight_shap))[::-1][:3]
    top_drivers = []
    for idx in top_indices:
        col_name = feature_cols[int(idx)]
        impact = float(flight_shap[int(idx)])
        raw_val = row_dict[col_name]
        if isinstance(raw_val, float):
            raw_val = round(raw_val, 3)
        top_drivers.append(
            ShapDriver(
                feature=FEATURE_LABELS.get(col_name, col_name),
                feature_value=raw_val,
                shap_impact=round(impact, 4),
                direction="Increases Risk" if impact > 0 else "Reduces Risk",
            )
        )

    return PredictionResponse(
        delay_probability=round(prob, 4),
        risk_tier=risk_tier,
        is_predicted_delayed_f1_opt=prob >= ml_state["threshold"],
        operational_recommendation=recommendation,
        derived_metrics={
            "net_turnaround_buffer": round(row_dict["net_turnaround_buffer"], 1),
            "delay_to_buffer_ratio": round(row_dict["delay_to_buffer_ratio"], 3),
            "dep_time_block": row_dict["dep_time_block"],
            "dest_grouped": row_dict["dest_grouped"],
        },
        top_3_drivers=top_drivers,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0", "model_loaded": "pipeline" in ml_state}


@app.get("/model/metadata")
def model_metadata():
    if "pipeline" not in ml_state:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")
    return {
        "model_architecture": "Optuna-Tuned LightGBM Classifier",
        "prediction_horizon": "T-180 minutes prior to scheduled departure",
        "training_corpus": "3.85M+ U.S. DOT BTS Form 234 Records (Jan-Jun 2026)",
        "oot_benchmark_jul_aug_2026": {
            "sample_size": 400000,
            "roc_auc": 0.7154,
            "pr_auc": 0.5281,
            "brier_score": 0.1802,
        },
        "f1_optimal_threshold": round(float(ml_state["threshold"]), 4),
        "validation_metrics": ml_state["val_metrics"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_delay(flight: FlightRequest):
    if "pipeline" not in ml_state:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")
    return score_single_flight(flight)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(flights: list[FlightRequest]):
    if "pipeline" not in ml_state:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")
    if not flights or len(flights) > 100:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 100 flights.")

    results = [score_single_flight(f) for f in flights]
    high_c = sum(1 for r in results if r.risk_tier == "High")
    mod_c = sum(1 for r in results if r.risk_tier == "Moderate")
    low_c = sum(1 for r in results if r.risk_tier == "Low")
    mean_p = float(np.mean([r.delay_probability for r in results]))

    return BatchPredictionResponse(
        total_flights=len(results),
        high_risk_count=high_c,
        moderate_risk_count=mod_c,
        low_risk_count=low_c,
        mean_delay_probability=round(mean_p, 4),
        predictions=results,
    )