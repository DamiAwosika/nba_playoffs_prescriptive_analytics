"""Daily player-prop archival.

Same shape as archive_odds.py, at the (player, market) grain. Run on a
schedule alongside archive_odds.py — over time you accumulate a clean
pre-game player-prop corpus that's safe to train on (the player_pts_*_vegas
variants depend on it).

Why this exists:
    balldontlie's /v2/player_props endpoint returns whatever line is current
    at query time. For completed games that's settled / in-game data which
    leaks the outcome. Pre-game props can only be captured BEFORE tipoff —
    this script does that.

Usage:
    python scripts/archive_player_props.py                  # today only
    python scripts/archive_player_props.py --days-ahead 2   # today + 2 days
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta

from sqlalchemy import select

from nba_ml.db.base import SessionLocal, init_db
from nba_ml.db.models import PlayerGame
from nba_ml.ingest.balldontlie_client import BdlClient
from nba_ml.ingest.load_games import (
    upsert_players, upsert_player_props, upsert_teams,
)


def _resolve_team_ids(db, player_ids: list[int]) -> dict[int, int]:
    """Map player_id -> their most recent team_id (so each prop row carries
    a team_id for downstream queries). Falls back silently for unknowns."""
    if not player_ids:
        return {}
    sub = db.execute(
        select(PlayerGame.player_id, PlayerGame.team_id, PlayerGame.game_date)
        .where(PlayerGame.player_id.in_(player_ids))
        .order_by(PlayerGame.game_date.desc())
    ).all()
    out: dict[int, int] = {}
    for pid, tid, _gd in sub:
        out.setdefault(pid, tid)
    return out


def main(days_ahead: int) -> None:
    init_db()
    client = BdlClient()

    today = date.today()
    end = today + timedelta(days=days_ahead)

    rows = client.fetch_player_props(today, end)
    if not rows:
        print(f"No player props returned for {today} -> {end}.")
        return

    db = SessionLocal()
    try:
        upsert_teams(db, client.fetch_teams())
        db.commit()

        # Fill in team_id from each player's most recent appearance.
        player_ids = sorted({r["player_id"] for r in rows})
        team_by_player = _resolve_team_ids(db, player_ids)
        enriched = []
        for r in rows:
            r = dict(r)
            r["team_id"] = team_by_player.get(r["player_id"])
            r["source"] = "balldontlie_archive"
            r["is_pregame"] = True
            # Skip rows where we can't attribute to a team — schema requires it.
            if r["team_id"] is None:
                continue
            enriched.append(r)

        upsert_player_props(db, enriched)
        db.commit()
    finally:
        db.close()

    n_per_market: dict[str, int] = {}
    for r in enriched:
        n_per_market[r["market"]] = n_per_market.get(r["market"], 0) + 1
    print(
        f"Archived {len(enriched)} pre-game player props "
        f"({today} -> {end}): {n_per_market}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--days-ahead", type=int, default=5,
        help="Look forward this many days from today (default: 1).",
    )
    args = p.parse_args()
    main(args.days_ahead)
    print(f"run completed: {date.today()}")
