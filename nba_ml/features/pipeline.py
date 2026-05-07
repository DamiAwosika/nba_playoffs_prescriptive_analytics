"""Orchestrator: read raw Game rows, compute team features, upsert them."""
from __future__ import annotations
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from nba_ml.config import settings
from nba_ml.db.models import Game, TeamGameFeature
from nba_ml.features.builders import add_rolling_team_features, compute_post_game_elos
from nba_ml.ingest.load_games import _upsert


_FEATURE_FIELDS = [
    "off_rating_roll5", "def_rating_roll5", "net_rating_roll10",
    "pace_roll5", "win_pct_roll10", "rest_days", "is_b2b", "elo",
]


def recompute_team_features(db: Session, feature_version: str | None = None) -> int:
    feature_version = feature_version or settings.feature_version

    rows = db.execute(select(Game)).scalars().all()
    if not rows:
        return 0

    df = pd.DataFrame([{
        "game_id": r.game_id, "game_date": r.game_date, "season": r.season,
        "team_id": r.team_id, "opponent_id": r.opponent_id, "is_home": r.is_home,
        "pts": r.pts, "pts_allowed": r.pts_allowed,
        "fga": r.fga, "fta": r.fta, "tov": r.tov, "oreb": r.oreb,
        "opp_fga": r.opp_fga, "opp_fta": r.opp_fta,
        "opp_tov": r.opp_tov, "opp_oreb": r.opp_oreb,
        "won": r.won,
    } for r in rows])
    df["game_date"] = pd.to_datetime(df["game_date"])

    feats = add_rolling_team_features(df)
    elos = compute_post_game_elos(df)
    feats = feats.merge(elos, on=["game_id", "team_id"], how="left")

    out = []
    for _, r in feats.iterrows():
        out.append({
            "game_id": r["game_id"],
            "team_id": int(r["team_id"]),
            "game_date": r["game_date"].date(),
            "feature_version": feature_version,
            **{f: (None if pd.isna(r[f]) else r[f]) for f in _FEATURE_FIELDS},
        })

    _upsert(db, TeamGameFeature.__table__, out,
            ["game_id", "team_id", "feature_version"])
    db.commit()
    return len(out)
