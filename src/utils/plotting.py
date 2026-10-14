"""Shared visual language for report figures and the dashboard.

Palette = validated reference categorical order (fixed slot order, never
cycled) + a reserved status palette that is never reused for series.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"NORMAL": "#0ca30c", "WARNING": "#fab219", "CRITICAL": "#d03b3b"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = "RdBu_r"
FAILURE_TYPES = ["thermal_runaway", "load_oscillation", "intermittent_arcing", "sudden"]
TYPE_COLOR = {t: SERIES[i] for i, t in enumerate(FAILURE_TYPES)}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "legend.fontsize": 8, "legend.frameon": False, "font.size": 9,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "figure.dpi": 110, "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
