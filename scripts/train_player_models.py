"""Train PTS / REB / AST regressors for all eligible players.

Usage:
    python scripts/train_player_models.py
    python scripts/train_player_models.py --season 2024-25
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_ml.training.train_player import main

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None)
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    main(season=args.season, tune=args.tune)
    print(f"run completed: {date.today()}")