import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hospital_management import (
    forecast_admissions,
    generate_hospital_dataset,
    recommend_staff_schedule,
    route_ambulance,
)


def test_generate_hospital_dataset_returns_rows():
    rows = generate_hospital_dataset(30)
    assert isinstance(rows, list)
    assert rows
    assert {"date", "department", "admissions", "icu_beds_used", "available_beds"}.issubset(rows[0].keys())


def test_forecast_returns_seven_day_predictions():
    rows = generate_hospital_dataset(60)
    forecast = forecast_admissions(rows, horizon=7)
    assert len(forecast) == 7
    assert forecast[0]["predicted_admissions"] >= 0


def test_staff_schedule_recommendation_is_reasonable():
    rows = generate_hospital_dataset(60)
    forecast = forecast_admissions(rows, horizon=7)
    recommendations = recommend_staff_schedule(forecast, rows)
    assert len(recommendations) >= 1
    assert recommendations[0]["shift"] in {"Morning", "Afternoon", "Night"}


def test_ambulance_routing_prefers_available_hospital():
    hospitals = [
        {"name": "City General", "distance_km": 4.0, "available_icu": 3, "capacity": 12},
        {"name": "North Hospital", "distance_km": 8.0, "available_icu": 6, "capacity": 14},
    ]
    result = route_ambulance(hospitals, required_beds=2)
    assert result["name"] == "North Hospital"
