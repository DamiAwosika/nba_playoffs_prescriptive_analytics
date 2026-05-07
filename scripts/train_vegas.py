"""Train the Vegas-augmented model family.

Trains logreg + RF + XGB + soft-vote ensemble on the subset of games that
have Vegas odds in the betting_odds table. Includes vegas_home_win_prob as
a feature. Saves bundles under a `_vegas` suffix:

    models/logreg_vegas_v1.joblib
    models/rf_vegas_v1.joblib
    models/xgb_vegas_v1.joblib
    models/vegas_ensemble_v1.joblib
    models/stack_vegas_v1.joblib   (only if the OOF stack succeeds)

These models live alongside the base models. At inference time, the router
in scripts/predict_playoffs.py picks per game.

Usage:
    python scripts/train_vegas.py
    python scripts/train_vegas.py --tune --n-iter 50
"""
from __future__ import annotations
import argparse
from pathlib import Path

from sqlalchemy import select

from nba_ml.config import settings
from nba_ml.db.base import SessionLocal
from nba_ml.db.models import BettingOdds
from nba_ml.features.matchup import FEATURE_COLUMNS_VEGAS
from nba_ml.training.train import train


def _games_with_pregame_odds() -> set[str]:
    """Only count games whose odds were archived BEFORE tipoff. Post-game
    rows are excluded — training on them creates a covert outcome leak."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(BettingOdds.game_id).where(BettingOdds.is_pregame.is_(True))
        ).scalars().all()
    finally:
        db.close()
    return {gid for gid in rows if gid is not None}


def main(out_dir: Path, feature_version: str, tune: bool, n_iter: int) -> None:
    odds_game_ids = _games_with_pregame_odds()
    if not odds_game_ids:
        raise SystemExit(
            "No PRE-GAME betting_odds rows found. Schedule scripts/archive_odds.py "
            "to run daily — pre-game odds are captured before tipoff and accumulate "
            "over time. Once you have a few hundred pre-game-covered games, re-run "
            "this script. (Historical odds from old run_etl runs are intentionally "
            "excluded because they may contain post-game / settled values.)"
        )
    print(f"Pre-game-Vegas-covered games available: {len(odds_game_ids)}")
    if len(odds_game_ids) < 200:
        print(
            f"  WARNING: only {len(odds_game_ids)} pre-game-covered games — "
            "training will likely overfit. Consider waiting until you have "
            "300+ before relying on these models."
        )

    train(
        out_dir=out_dir,
        feature_version=feature_version,
        tune=tune,
        n_iter=n_iter,
        feature_columns=FEATURE_COLUMNS_VEGAS,
        row_filter=lambda df: df[df["game_id"].isin(odds_game_ids)],
        model_suffix="_vegas",
        ensemble_name="vegas_ensemble",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=settings.models_dir)
    p.add_argument("--feature-version", default=settings.feature_version)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-iter", type=int, default=20,
                   help="Tuning combos per model. Keep modest — Vegas slice is ~450 train rows.")
    args = p.parse_args()
    main(args.out_dir, args.feature_version, args.tune, args.n_iter)
