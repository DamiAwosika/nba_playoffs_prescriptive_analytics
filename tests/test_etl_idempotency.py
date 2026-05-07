"""Re-running ETL with overlapping rows must update, not duplicate."""
from __future__ import annotations
from datetime import date, datetime

from sqlalchemy import func, select

from nba_ml.db.models import Game, PlayerAvailability
from nba_ml.ingest.load_games import upsert_availability, upsert_games


def _game_row(pts: int) -> dict:
    return {
        "game_id": "G1", "game_date": date(2025, 1, 1), "season": "2024-25",
        "team_id": 1, "opponent_id": 2, "is_home": True,
        "pts": pts, "pts_allowed": 90,
        "fga": 85, "fta": 20, "tov": 13, "oreb": 10,
        "opp_fga": 86, "opp_fta": 18, "opp_tov": 12, "opp_oreb": 9,
        "won": pts > 90,
        "ingested_at": datetime.utcnow(),
    }


def test_repeated_game_upsert_does_not_duplicate(db):
    upsert_games(db, [_game_row(100)])
    upsert_games(db, [_game_row(110)])
    db.commit()

    n = db.execute(select(func.count()).select_from(Game)).scalar_one()
    pts = db.execute(select(Game.pts)).scalar_one()
    assert n == 1
    assert pts == 110


def test_availability_upsert_updates_status(db):
    base = {
        "game_id": "GX", "game_date": date(2025, 2, 1),
        "player_id": 100, "team_id": 1,
        "reported_at": datetime.utcnow(),
    }
    upsert_availability(db, [{**base, "status": "questionable", "reason": "ankle"}])
    upsert_availability(db, [{**base, "status": "out", "reason": "ankle"}])
    db.commit()

    rows = db.execute(select(PlayerAvailability)).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == "out"
