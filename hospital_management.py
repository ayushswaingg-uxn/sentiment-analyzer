from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from typing import Any, Dict, List

random.seed(42)

DEPARTMENTS = [
    "emergency",
    "icu",
    "general_ward",
    "pediatrics",
    "maternity",
    "cardiology",
    "neurology",
    "oncology",
]

DEPARTMENT_MULTIPLIERS = {
    "emergency": 1.25,
    "icu": 0.95,
    "general_ward": 1.1,
    "pediatrics": 0.8,
    "maternity": 0.75,
    "cardiology": 0.9,
    "neurology": 0.85,
    "oncology": 0.88,
}

ICU_RATIO = {
    "emergency": 0.22,
    "icu": 0.78,
    "general_ward": 0.14,
    "pediatrics": 0.1,
    "maternity": 0.08,
    "cardiology": 0.12,
    "neurology": 0.13,
    "oncology": 0.15,
}

BED_CAPACITY = {
    "emergency": 28,
    "icu": 16,
    "general_ward": 38,
    "pediatrics": 18,
    "maternity": 14,
    "cardiology": 20,
    "neurology": 18,
    "oncology": 17,
}


def generate_hospital_dataset(days: int = 180) -> List[Dict[str, Any]]:
    """Create a synthetic hospital admissions dataset with bed-usage signals."""
    rows: List[Dict[str, Any]] = []
    start_date = date(2024, 1, 1)

    for offset in range(days):
        current_date = start_date + timedelta(days=offset)
        for department in DEPARTMENTS:
            weekly_signal = math.sin(2 * math.pi * offset / 7)
            trend = 0.03 * offset
            seasonal = 0.6 if department in {"emergency", "icu"} else 0.35
            admissions = int(
                round(
                    (18 + trend + weekly_signal * 6 + seasonal)
                    * DEPARTMENT_MULTIPLIERS[department]
                    + random.gauss(0, 2.2)
                )
            )
            admissions = max(0, admissions)
            icu_beds_used = int(round(admissions * ICU_RATIO[department] + random.gauss(0.0, 1.2)))
            available_beds = max(0, BED_CAPACITY[department] - icu_beds_used + int(random.gauss(0, 1.2)))
            rows.append(
                {
                    "date": current_date,
                    "department": department,
                    "admissions": admissions,
                    "icu_beds_used": icu_beds_used,
                    "available_beds": available_beds,
                }
            )

    rows.sort(key=lambda item: (item["date"], item["department"]))
    return rows


def forecast_admissions(df: List[Dict[str, Any]], horizon: int = 7) -> List[Dict[str, Any]]:
    """Forecast the next 7 days of admissions using a simple seasonal trend model."""
    daily_totals: Dict[date, int] = {}
    for row in df:
        daily_totals[row["date"]] = daily_totals.get(row["date"], 0) + int(row["admissions"])

    ordered_dates = sorted(daily_totals)
    if not ordered_dates:
        return [{"date": date.today() + timedelta(days=offset), "predicted_admissions": 0} for offset in range(horizon)]

    series = [float(daily_totals[d]) for d in ordered_dates]
    x = list(range(len(series)))
    slope = sum((xi - sum(x) / len(x)) * (yi - sum(series) / len(series)) for xi, yi in zip(x, series)) / sum((xi - sum(x) / len(x)) ** 2 for xi in x) if len(x) > 1 else 0.0
    last_value = float(series[-1])

    predictions: List[Dict[str, Any]] = []
    for offset in range(horizon):
        weekly_adjustment = 4 * math.sin(2 * math.pi * (len(series) + offset) / 7)
        predicted = max(0, int(round(last_value + slope * (offset + 1) + weekly_adjustment)))
        future_date = ordered_dates[-1] + timedelta(days=offset + 1)
        predictions.append({"date": future_date, "predicted_admissions": predicted})

    return predictions


def recommend_staff_schedule(forecast: List[Dict[str, Any]], df: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recommend staffing by shift based on predicted demand and current ICU pressure."""
    peak_demand = max(item["predicted_admissions"] for item in forecast) if forecast else 0
    avg_icu_beds = sum(item["icu_beds_used"] for item in df) / len(df) if df else 0.0
    level = "high" if peak_demand > 80 or avg_icu_beds > 12 else "moderate"

    shift_plan = [
        {"shift": "Morning", "nurses": max(6, int(round(peak_demand / 12))), "reason": "Early peak workload"},
        {"shift": "Afternoon", "nurses": max(7, int(round(peak_demand / 10))), "reason": "High arrival volume"},
        {"shift": "Night", "nurses": max(5, int(round(peak_demand / 14))), "reason": "Reduced but critical coverage"},
    ]

    if level == "high":
        for item in shift_plan:
            item["nurses"] += 2
            item["reason"] = "High demand and ICU pressure"

    return shift_plan


def route_ambulance(hospitals: List[Dict[str, Any]], required_beds: int = 2) -> Dict[str, Any]:
    """Select the best hospital based on ICU bed availability first, then distance."""
    candidates = [h for h in hospitals if h.get("available_icu", 0) >= required_beds]
    if not candidates:
        return {"name": "No suitable hospital", "distance_km": None, "available_icu": 0}

    return sorted(
        candidates,
        key=lambda h: (-h.get("available_icu", 0), h.get("distance_km", float("inf"))),
    )[0]


def build_dashboard_snapshot(days: int = 120, horizon: int = 7) -> Dict[str, Any]:
    """Build a compact dashboard snapshot with today's availability, shortages, and routing advice."""
    df = generate_hospital_dataset(days)
    forecast = forecast_admissions(df, horizon=horizon)
    latest_date = max(item["date"] for item in df)
    current_state = [item for item in df if item["date"] == latest_date]

    shortages: List[Dict[str, Any]] = []
    for department in DEPARTMENTS:
        current = next(item for item in current_state if item["department"] == department)
        projected_admissions = int(forecast[-1]["predicted_admissions"])
        expected_shortage = max(0, projected_admissions - current["available_beds"])
        if expected_shortage > 0:
            shortages.append(
                {
                    "department": department,
                    "available_beds": int(current["available_beds"]),
                    "projected_admissions": projected_admissions,
                    "expected_shortage": expected_shortage,
                }
            )

    hospitals = [
        {"name": "City General", "distance_km": 4.2, "available_icu": 3, "capacity": 14},
        {"name": "North Hospital", "distance_km": 8.1, "available_icu": 6, "capacity": 16},
        {"name": "Metro Care", "distance_km": 6.4, "available_icu": 2, "capacity": 12},
    ]
    ambulance_route = route_ambulance(hospitals, required_beds=2)

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "forecast": forecast,
        "current_availability": current_state,
        "shortages": shortages,
        "staff_schedule": recommend_staff_schedule(forecast, df),
        "ambulance_route": ambulance_route,
    }


def render_dashboard_html(snapshot: Dict[str, Any]) -> str:
    """Render a simple HTML dashboard that can be served locally."""
    current_rows = "".join(
        f"<li><strong>{escape(item['department'])}</strong>: {item['available_beds']} beds free, ICU usage {item['icu_beds_used']}</li>"
        for item in snapshot["current_availability"]
    )
    shortage_rows = "".join(
        f"<li>{escape(item['department'])}: {item['expected_shortage']} predicted shortage over available beds</li>"
        for item in snapshot["shortages"]
    )
    schedule_rows = "".join(
        f"<li>{escape(item['shift'])}: {item['nurses']} nurses — {escape(item['reason'])}</li>"
        for item in snapshot["staff_schedule"]
    )
    forecast_items = "".join(
        f"<li>{item['date'].strftime('%Y-%m-%d')}: {item['predicted_admissions']} admissions</li>"
        for item in snapshot["forecast"]
    )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta http-equiv=\"refresh\" content=\"30\">
  <title>Hospital Bed and Resource Management Dashboard</title>
  <style>
    body {{font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; background: linear-gradient(135deg, #eff6ff, #f8fafc); color: #0f172a;}}
    .card {{background: white; border-radius: 14px; padding: 18px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(15,23,42,0.08);}}
    h1, h2 {{margin-top: 0; color: #1d4ed8;}}
    .highlight {{color: #2563eb; font-weight: 700;}}
    .badge {{display:inline-block; padding:6px 10px; background:#dbeafe; color:#1d4ed8; border-radius:999px; font-size:0.9em; margin-bottom:10px;}}
    ul {{line-height: 1.6;}}
  </style>
</head>
<body>
  <h1>Hospital Bed and Resource Management Dashboard</h1>
  <div class=\"badge\">CSE Project Prototype</div>
  <p class=\"highlight\">Generated at {snapshot['generated_at']}</p>
  <div class=\"card\">
    <h2>Current Bed Availability</h2>
    <ul>{current_rows}</ul>
  </div>
  <div class=\"card\">
    <h2>Predicted Shortages</h2>
    <ul>{shortage_rows or '<li>No projected shortages</li>'}</ul>
  </div>
  <div class=\"card\">
    <h2>7-Day Admissions Forecast</h2>
    <ul>{forecast_items}</ul>
  </div>
  <div class=\"card\">
    <h2>Suggested Staff Scheduling</h2>
    <ul>{schedule_rows}</ul>
  </div>
  <div class=\"card\">
    <h2>Ambulance Routing Recommendation</h2>
    <p>{snapshot['ambulance_route']['name']} is the best option with {snapshot['ambulance_route']['available_icu']} ICU beds available and {snapshot['ambulance_route']['distance_km']} km away.</p>
  </div>
</body>
</html>"""


class HospitalDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/summary":
            payload = json.dumps(build_dashboard_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        html = render_dashboard_html(build_dashboard_snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def launch_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Serve the hospital dashboard locally."""
    server = ThreadingHTTPServer((host, port), HospitalDashboardHandler)
    print(f"Dashboard available at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    launch_dashboard()
