"""Inference orchestration: resolve active rosters, build features, predict."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from sqlalchemy.orm import Session

from nba_ml.features.matchup import (
    active_roster_as_of, build_matchup_features,
)
from nba_ml.inference.loader import LoadedModel


@dataclass
class PredictionResult:
    home_win_prob: float
    feature_version: str
    model_version: str
    home_active_count: int
    away_active_count: int


def predict_matchup(
    db: Session,
    model: LoadedModel,
    home_team_id: int,
    away_team_id: int,
    game_date: date,
) -> PredictionResult | None:
    home_active = active_roster_as_of(db, home_team_id, game_date)
    away_active = active_roster_as_of(db, away_team_id, game_date)

    feats = build_matchup_features(
        db, home_team_id, away_team_id, game_date,
        feature_version=model.feature_version,
        home_active=home_active, away_active=away_active,
    )
    if feats is None:
        return None

    return PredictionResult(
        home_win_prob=model.predict_proba(feats),
        feature_version=model.feature_version,
        model_version=model.version,
        home_active_count=len(home_active),
        away_active_count=len(away_active),
    )
