"""Idempotent schema migrations. Safe to run multiple times.

What it does:
  1. Creates any tables in the SQLAlchemy models that don't exist yet
     (e.g., betting_odds) — same as Base.metadata.create_all.
  2. Adds team_game_features.elo column if it's missing — SQLite's
     create_all does NOT alter existing tables, hence this script.

Usage:
  python scripts/migrate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text

from nba_ml.db.base import engine
from nba_ml.db.models import Base


def main() -> None:
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        insp = inspect(conn)
        if "team_game_features" in insp.get_table_names():
            existing = {c["name"] for c in insp.get_columns("team_game_features")}
            if "elo" not in existing:
                conn.execute(text(
                    "ALTER TABLE team_game_features ADD COLUMN elo FLOAT"
                ))
                print("added: team_game_features.elo")
            else:
                print("ok: team_game_features.elo already present")
        if "betting_odds" in insp.get_table_names():
            existing = {c["name"] for c in insp.get_columns("betting_odds")}
            if "is_pregame" not in existing:
                conn.execute(text(
                    "ALTER TABLE betting_odds ADD COLUMN is_pregame BOOLEAN DEFAULT 0"
                ))
                print("added: betting_odds.is_pregame")
            else:
                print("ok: betting_odds.is_pregame already present")
        else:
            print("warning: betting_odds table missing (create_all should have made it)")

        # Extended player box-score columns for Hollinger Game Score / TS%.
        if "player_games" in insp.get_table_names():
            existing = {c["name"] for c in insp.get_columns("player_games")}
            for col in ("fgm", "fga", "ftm", "fta", "oreb", "dreb",
                        "stl", "blk", "tov", "pf"):
                if col not in existing:
                    conn.execute(text(
                        f"ALTER TABLE player_games ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0"
                    ))
                    print(f"added: player_games.{col}")
                else:
                    print(f"ok: player_games.{col} already present")
    print("migration done.")


if __name__ == "__main__":
    main()
