"""Run exploratory data analysis: python eda.py"""
import warnings

from src.analysis.eda import run_eda
from src.utils.config import load_config

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    run_eda(load_config())
