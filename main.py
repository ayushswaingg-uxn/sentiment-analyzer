"""Workspace entry point for the hospital management prototype."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "g data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from hospital_management import launch_dashboard


if __name__ == "__main__":
    launch_dashboard()
