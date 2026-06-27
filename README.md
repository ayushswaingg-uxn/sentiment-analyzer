# Hospital Bed and Resource Management System

This project presents a smart hospital resource management system designed to support hospital administrators, government health departments, and emergency services during peak demand periods such as pandemics or disasters.

## Project Objective
The system predicts patient admissions, estimates ICU occupancy, recommends staff scheduling, and helps direct ambulances to hospitals with available ICU beds. It demonstrates how data-driven decision support can improve hospital preparedness and reduce delays in emergency care.

## Key Features
- Generates synthetic historical hospital admission and ICU occupancy data for multiple departments.
- Forecasts the next 7 days of admissions using a simple trend-based predictive model.
- Recommends staffing levels for morning, afternoon, and night shifts.
- Identifies likely shortages in bed availability and highlights them in a dashboard.
- Suggests the best hospital for ambulance transfer based on ICU capacity and travel distance.

## Technology Used
- Python
- Standard library web server for the dashboard
- Custom forecasting logic for admission prediction
- Simple decision rules for resource allocation and routing

## How to Run
```bash
python main.py
```

Then open http://127.0.0.1:8050 in your browser.

## Project Scope
This prototype is suitable for a CSE academic project because it combines:
- data generation and preprocessing,
- forecasting and decision support,
- resource optimization,
- and a user-facing dashboard.
