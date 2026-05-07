"""End-to-end ingest -> features pipeline using balldontlie.io as the source.

Default range: yesterday and the 6 days prior. Override with --start/--end.

Examples:
    python scripts/run_etl.py
    python scripts/run_etl.py --start 2026-04-01 --end 2026-04-27
    python scripts/run_etl.py --start 2025-10-22 --end 2026-04-15   # full season
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta

from nba_ml.db.base import SessionLocal, init_db
from nba_ml.features.pipeline import recompute_team_features
from nba_ml.ingest.balldontlie_client import BdlClient
from nba_ml.ingest.load_games import (
    upsert_games, upsert_player_games, upsert_players, upsert_teams,
)


def _default_dates() -> tuple[date, date]:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=60)
    return start, end


def main(start: date, end: date) -> None:
    print(f"Ingesting NBA data from {start} to {end} via balldontlie")
    init_db()

    client = BdlClient()
    teams = client.fetch_teams()
    print(f"Teams: {len(teams)}")

    games, player_games, players = client.fetch_games_and_stats(start, end)
    print(
        f"Games: {len(games)//2} unique | "
        f"Player-game rows: {len(player_games)} | "
        f"Players: {len(players)}"
    )
    # NOTE: Vegas odds are deliberately NOT fetched here. balldontlie's /v2/odds
    # returns "most recent" odds, which for completed games means contaminated
    # post-game / settled values. Pre-game odds are captured exclusively by
    # scripts/archive_odds.py (daily, for upcoming games) and by
    # scripts/predict_playoffs.py (live, just before tipoff).

    if not games:
        print("No completed games in this date range. Try a wider/earlier window.")
        return

    db = SessionLocal()
    try:
        upsert_teams(db, teams)
        upsert_players(db, players)
        db.commit()
        upsert_games(db, games)
        upsert_player_games(db, player_games)
        db.commit()
        n = recompute_team_features(db)
        print(f"Team features upserted: {n}")
    finally:
        db.close()
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    ds, de = _default_dates()
    p.add_argument("--start", type=date.fromisoformat, default=ds,
                   help="Inclusive start date YYYY-MM-DD (default: 7 days ago)")
    p.add_argument("--end", type=date.fromisoformat, default=de,
                   help="Inclusive end date YYYY-MM-DD (default: yesterday)")
    args = p.parse_args()
    main(args.start, args.end)
    print(f"run completed: {date.today()}")
