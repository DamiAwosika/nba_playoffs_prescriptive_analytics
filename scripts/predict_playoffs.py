"""Predict win probability for upcoming NBA playoff games.

Pipeline per run:
  1. Pull scheduled postseason games from balldontlie.
  2. Refresh injury data -> PlayerAvailability snapshot.
  3. Pull LIVE Vegas odds -> BettingOdds (these are pre-game; the game
     hasn't happened yet, so the leak that contaminates HISTORICAL odds
     doesn't apply here).
  4. Run every BASE model (no vegas-augmented; see DESIGN NOTE below)
     and route between the base ensemble (always available) and the base
     individual models for the comparison columns.
  5. Show Vegas's implied probability as a side-by-side REFERENCE column
     so you can compare your model to the market.

DESIGN NOTE — Vegas-augmented models are intentionally excluded from the
routing decision because balldontlie's /v2/odds endpoint returns the most
recent odds (which for completed games means in-game / settled odds) —
training a model on that contaminates it with the outcome. To enable
Vegas-augmented training honestly, archive odds daily for *tomorrow's*
games over weeks/months, then train on that pre-game-only corpus.

Run order:
    1. python scripts/run_etl.py --start <date> --end <yesterday>
    2. python -m nba_ml.training.train [--tune --n-iter 100]
    3. python scripts/predict_playoffs.py
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from nba_ml.db.base import SessionLocal, init_db
from nba_ml.db.models import BettingOdds, PlayerGame
from nba_ml.ingest.balldontlie_client import BdlClient
from nba_ml.ingest.load_games import (
    upsert_availability, upsert_betting_odds, upsert_teams,
)
from nba_ml.inference.loader import get_loaded_model, list_available_models
from nba_ml.inference.service import predict_matchup

ROSTER_LOOKBACK_DAYS = 30
BASE_ROUTER_NAME = "ensemble"
# Names of model bundles that are EXCLUDED from the prediction columns and
# the routing decision, because they are trained on contaminated odds.
EXCLUDED_BUNDLES = {
    "logreg_vegas", "rf_vegas", "xgb_vegas",
    "vegas_ensemble", "stack_vegas",
}


def _build_availability(
    db: Session, client: BdlClient, upcoming: list[dict],
) -> list[dict]:
    team_ids = sorted(
        {g["home_team_id"] for g in upcoming}
        | {g["away_team_id"] for g in upcoming}
    )
    injuries = client.fetch_injuries(team_ids)
    cutoff = date.today() - timedelta(days=ROSTER_LOOKBACK_DAYS)

    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for g in upcoming:
        for tid in (g["home_team_id"], g["away_team_id"]):
            recent_pids = db.execute(
                select(PlayerGame.player_id).distinct().where(
                    PlayerGame.team_id == tid,
                    PlayerGame.game_date >= cutoff,
                )
            ).scalars().all()
            for pid in recent_pids:
                key = (g["game_id"], pid)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "game_id": g["game_id"],
                    "game_date": g["date"],
                    "player_id": pid,
                    "team_id": tid,
                    "status": injuries.get(pid, "active"),
                    "reason": None,
                    "reported_at": datetime.utcnow(),
                })
    return rows


def _fetch_and_upsert_live_odds(
    db: Session, client: BdlClient, upcoming: list[dict],
) -> int:
    if not upcoming:
        return 0
    start = min(g["date"] for g in upcoming)
    end = max(g["date"] for g in upcoming)
    odds_by_game = client.fetch_odds(start, end)

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
            "source": "balldontlie_live",
            "is_pregame": True,
        })
    upsert_betting_odds(db, rows)
    return len(rows)


def _vegas_prob_for(
    db: Session, home_team_id: int, away_team_id: int, game_date: date,
) -> float | None:
    """Most-recent PRE-GAME Vegas implied probability for this matchup, or None."""
    row = db.execute(
        select(BettingOdds.vegas_home_win_prob).where(
            BettingOdds.game_date == game_date,
            BettingOdds.home_team_id == home_team_id,
            BettingOdds.away_team_id == away_team_id,
            BettingOdds.is_pregame.is_(True),
        ).order_by(BettingOdds.fetched_at.desc()).limit(1)
    ).scalar_one_or_none()
    return float(row) if row is not None else None


def main(days_ahead: int) -> None:
    init_db()
    client = BdlClient()

    today = date.today()
    end = today + timedelta(days=days_ahead)

    upcoming = client.fetch_upcoming_games(today, end, postseason=True)
    if not upcoming:
        print(f"No upcoming playoff games found between {today} and {end}.")
        return

    db = SessionLocal()
    try:
        upsert_teams(db, client.fetch_teams())
        db.commit()

        avail = _build_availability(db, client, upcoming)
        upsert_availability(db, avail)
        db.commit()
        n_out = sum(1 for r in avail if r["status"] != "active")
        print(f"Availability snapshot: {len(avail)} rows, {n_out} non-active.")

        n_odds = _fetch_and_upsert_live_odds(db, client, upcoming)
        db.commit()
        print(f"Live odds upserted: {n_odds} games")

        all_names = list_available_models()
        names = [n for n in all_names if n not in EXCLUDED_BUNDLES]
        if not names:
            print("No usable trained models found. Run training first.")
            return
        models = {n: get_loaded_model(n) for n in names}

        router = (
            models.get(BASE_ROUTER_NAME) or models.get("logreg")
            or next(iter(models.values()))
        )

        header = (
            f"{'Date':<12} {'Matchup':<22}"
            + "".join(f" {n:>10}" for n in names)
            + f" {'ROUTED':>10}"
            + f" {'VEGAS_REF':>10}"
        )
        print()
        print(header)
        print("-" * len(header))
        for g in sorted(upcoming, key=lambda x: (x["date"], x["home_abbr"])):
            cells = []
            for name, m in models.items():
                r = predict_matchup(
                    db, m, g["home_team_id"], g["away_team_id"], g["date"],
                )
                cells.append(
                    f" {r.home_win_prob:>10.3f}" if r else f" {'n/a':>10}"
                )

            r = predict_matchup(
                db, router, g["home_team_id"], g["away_team_id"], g["date"],
            )
            routed = f" {r.home_win_prob:>10.3f}" if r else f" {'n/a':>10}"

            vp = _vegas_prob_for(
                db, g["home_team_id"], g["away_team_id"], g["date"],
            )
            vegas_cell = f" {vp:>10.3f}" if vp is not None else f" {'n/a':>10}"

            mu = f"{g['home_abbr']} (H) vs {g['away_abbr']}"
            print(f"{g['date']!s:<12} {mu:<22}"
                  + "".join(cells) + routed + vegas_cell)
        print(
            "\nValues are P(home team wins).\n"
            "  ROUTED    = our base ensemble's prediction (used in production).\n"
            "  VEGAS_REF = market-implied probability for this game; shown as a\n"
            "              REFERENCE for comparison only — not a model input."
        )
        print(
            "\nVegas-augmented models exist on disk but are excluded from the\n"
            "table because their training data is contaminated by post-game odds.\n"
            "See predict_playoffs.py docstring for the path forward."
        )
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days-ahead", type=int, default=7,
                   help="How many days of upcoming games to predict (default: 7)")
    args = p.parse_args()
    main(args.days_ahead)
    print(f"run completed: {date.today()}")


