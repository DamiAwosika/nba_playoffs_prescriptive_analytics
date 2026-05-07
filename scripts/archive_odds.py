"""Daily Vegas odds archival.

Pulls current odds for games happening TODAY (and optionally further out),
stamps them is_pregame=True, and stores in betting_odds. Schedule this to
run daily — over weeks/months you'll accumulate a clean pre-game odds
corpus that's safe to train on.

Why this script exists:
    balldontlie's /v2/odds endpoint returns whatever odds are most recent
    at query time. For completed games, that's in-game or settled odds —
    which encode the outcome and produce ~99% accuracy training leaks.
    Pre-game odds, on the other hand, can only be captured BEFORE the game
    starts. This script does that.

Recommended schedule (cron / Task Scheduler):
    Run once mid-morning and once a few hours before tipoff:
        0 9,16 * * *   python C:\\dev\\nba\\scripts\\archive_odds.py

Usage:
    python scripts/archive_odds.py                  # default: today only
    python scripts/archive_odds.py --days-ahead 2   # today + next 2 days

Cleanup of contaminated rows from old run_etl runs (one-time, optional):
    python -c "from nba_ml.db.base import engine; from sqlalchemy import text; \\
        engine.connect().execute(text(\\"DELETE FROM betting_odds WHERE is_pregame=0\\"))"
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta

from nba_ml.db.base import SessionLocal, init_db
from nba_ml.ingest.balldontlie_client import BdlClient
from nba_ml.ingest.load_games import upsert_betting_odds, upsert_teams


def main(days_ahead: int) -> None:
    init_db()
    client = BdlClient()

    today = date.today()
    end = today + timedelta(days=days_ahead)

    upcoming = client.fetch_upcoming_games(today, end, postseason=None)
    if not upcoming:
        print(f"No upcoming games {today} -> {end}.")
        return

    odds_by_game = client.fetch_odds(today, end)
    upcoming_by_id = {g["game_id"]: g for g in upcoming}

    rows = []
    for gid, prob in odds_by_game.items():
        meta = upcoming_by_id.get(gid)
        if meta is None:
            continue
        rows.append({
            "game_id": gid,
            "game_date": meta["date"],
            "home_team_id": meta["home_team_id"],
            "away_team_id": meta["away_team_id"],
            "vegas_home_win_prob": prob,
            "source": "balldontlie_archive",
            "is_pregame": True,
        })

    db = SessionLocal()
    try:
        upsert_teams(db, client.fetch_teams())
        db.commit()
        upsert_betting_odds(db, rows)
        db.commit()
    finally:
        db.close()

    print(
        f"Archived {len(rows)} pre-game odds for {len(upcoming)} upcoming games "
        f"({today} -> {end})."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--days-ahead", type=int, default=5,
        help="Look forward this many days from today (default: 1 — today + tomorrow).",
    )
    args = p.parse_args()
    main(args.days_ahead)
    print(f"run completed: {date.today()}")
