"""
tests/test_api.py
-----------------
Automated unit and integration tests for the Flight Delay Risk Analytics API.
Verifies health endpoints, schema validation, risk tier classification,
SHAP explainability payloads, and edge-case handling (e.g., negative turnaround buffer).
"""

from fastapi.testclient import TestClient
from app.main import app, classify_risk_tier, get_dep_time_block


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


def test_predict_high_risk_negative_turnaround():
    """Ensure negative net turnaround buffer does not crash model or SHAP explainer."""
    payload = {
        "operating_airline": "UA",
        "origin": "ORD",
        "dest": "FNT",
        "scheduled_dep_hour": 18,
        "scheduled_flight_mins": 72.0,
        "is_weekend": 0,
        "origin_daily_flight_volume": 920,
        "origin_hourly_flight_volume": 76,
        "route_daily_frequency": 4,
        "origin_rolling_3h_dep_delay": 41.5,
        "origin_rolling_3h_delay_rate": 0.48,
        "carrier_origin_hist_delay_rate": 0.29,
        "known_prev_aircraft_delay": 75.0,
        "scheduled_turnaround_mins": 40.0,  # Yields -35.0 min net turnaround buffer
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["delay_probability"] <= 1.0
        assert body["risk_tier"] in {"Low", "Moderate", "High"}
        assert body["derived_metrics"]["net_turnaround_buffer"] == -35.0
        assert len(body["top_3_drivers"]) == 3


def test_predict_unseen_destination_fallback():
    """Verify rare/unseen destination IATA codes map cleanly to OTHER."""
    payload = {
        "operating_airline": "DL",
        "origin": "ATL",
        "dest": "ZZZ",  # Rare/unseen IATA code
        "scheduled_dep_hour": 7,
        "scheduled_flight_mins": 95.0,
        "is_weekend": 0,
        "origin_daily_flight_volume": 1050,
        "origin_hourly_flight_volume": 62,
        "route_daily_frequency": 2,
        "origin_rolling_3h_dep_delay": 4.5,
        "origin_rolling_3h_delay_rate": 0.11,
        "carrier_origin_hist_delay_rate": 0.17,
        "known_prev_aircraft_delay": 0.0,
        "scheduled_turnaround_mins": 75.0,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["derived_metrics"]["dest_grouped"] == "OTHER"
        assert body["derived_metrics"]["dep_time_block"] == "Early_Morning"


def test_helper_functions():
    assert get_dep_time_block(6) == "Early_Morning"
    assert get_dep_time_block(15) == "Afternoon"
    assert classify_risk_tier(0.12)[0] == "Low"
    assert classify_risk_tier(0.28)[0] == "Moderate"
    assert classify_risk_tier(0.55)[0] == "High"