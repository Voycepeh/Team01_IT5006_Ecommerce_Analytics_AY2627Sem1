"""Pytest path setup for the Streamlit dashboard modules."""

import sys
from pathlib import Path

STREAMLIT_DIR = Path(__file__).resolve().parents[1] / "deployment" / "streamlit"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))
