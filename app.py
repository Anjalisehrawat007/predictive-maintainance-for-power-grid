"""Dashboard entry point:  streamlit run app.py"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from dashboard.main import run  # noqa: E402

run()
