"""Project-wide logging setup."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("sgpm")
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root.getChild(name)
