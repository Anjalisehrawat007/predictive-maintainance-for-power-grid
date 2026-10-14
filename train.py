"""Run the experiment pipeline.

Examples
--------
python train.py --stage all            # every stage, then the final selection
python train.py --stage architectures  # a single stage (uses experiments/best_config.json)
python train.py --stage final          # per-family best models + selection + artefacts
"""
from __future__ import annotations

import argparse
import warnings

from src.data.dataset import build_processed
from src.data.pipeline import prepare
from src.preprocessing.leakage import run_all_checks
from src.training.final import run_final
from src.training.stages import ORDER, STAGES
from src.utils.config import load_config
from src.utils.logging import get_logger

warnings.filterwarnings("ignore")
log = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "final", "ensemble", "leakage", *ORDER])
    ap.add_argument("--from-stage", default=None, choices=ORDER, help="with --stage all: resume from here")
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_processed(cfg)
    if args.stage in ("all", "leakage"):
        run_all_checks(cfg, prepare(cfg), seq_len=96)
    if args.stage == "all":
        start = ORDER.index(args.from_stage) if args.from_stage else 0
        for s in ORDER[start:]:
            log.info("########## STAGE %s ##########", s)
            STAGES[s](cfg)
        run_final(cfg)
    elif args.stage == "final":
        run_final(cfg)
    elif args.stage == "ensemble":
        from src.training.ensemble import run_ensemble

        run_ensemble(cfg)
    elif args.stage != "leakage":
        STAGES[args.stage](cfg)


if __name__ == "__main__":
    main()
