"""Diagnostic: prints what get_bracket() sees so we can pinpoint why
some series show 0-0. Run from project root: python scripts/diagnose_bracket.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func
from nba_ml.db.base import SessionLocal
from nba_ml.db.models import Game
from webapp.data import (
    _all_teams, _compute_seeds, _current_playoff_season, _identify_series,
)

db = SessionLocal()
try:
    season = _current_playoff_season(db)
    print(f"=== Current playoff season: {season!r} ===\n")

    # 1. Raw count of playoff games per (team_id, opponent_id) pair this season.
    print("--- Raw playoff-game pairs in DB (is_playoff=True, current season) ---")
    rows = db.execute(
        select(
            Game.team_id, Game.opponent_id,
            func.count().label("n"),
            func.sum(Game.is_home.cast(__import__("sqlalchemy").Integer)).label("home_rows"),
        ).where(
            Game.season == season,
            Game.is_playoff.is_(True),
        ).group_by(Game.team_id, Game.opponent_id)
        .order_by(Game.team_id, Game.opponent_id)
    ).all()
    for t, o, n, hr in rows:
        print(f"  team_id={t:>3}  opp_id={o:>3}  n_games={n}  is_home_true_rows={hr}")
    print(f"  TOTAL distinct directional pairs: {len(rows)}\n")

    # 2. What _identify_series produces.
    teams_by_id = _all_teams(db)
    series = _identify_series(db, season, teams_by_id)
    print(f"--- _identify_series returned {len(series)} series ---")
    for s in series:
        t1, t2 = s["team1"], s["team2"]
        print(f"  {t1['abbreviation']:>3}({t1['team_id']}) vs {t2['abbreviation']:>3}({t2['team_id']})  "
              f"score {t1['series_wins']}-{t2['series_wins']}  "
              f"status={s['status']}  n_games={s['n_games']}")
    print()

    # 3. Seeds.
    seeds = _compute_seeds(db, season, teams_by_id)
    print("--- Seeds by conference ---")
    for conf in ("West", "East"):
        items = [(seed, tid, teams_by_id[tid]["abbreviation"])
                 for tid, seed in seeds.items()
                 if teams_by_id.get(tid, {}).get("conference") == conf]
        items.sort()
        print(f"  {conf}:")
        for seed, tid, abbr in items:
            print(f"    {seed}. {abbr} (team_id={tid})")
    print()

    # 4. Cross-check: for each canonical R1 matchup, can we find the actual series?
    from webapp.data import CANONICAL_R1_MATCHUPS, _find_series_for_teams
    print("--- Canonical R1 lookup test ---")
    for conf in ("West", "East"):
        conf_seeds = {seed: tid for tid, seed in seeds.items()
                      if teams_by_id.get(tid, {}).get("conference") == conf}
        for high, low in CANONICAL_R1_MATCHUPS:
            t_high = conf_seeds.get(high)
            t_low = conf_seeds.get(low)
            found = _find_series_for_teams(series, t_high, t_low) if t_high and t_low else None
            ah = teams_by_id.get(t_high, {}).get("abbreviation", "?")
            al = teams_by_id.get(t_low, {}).get("abbreviation", "?")
            mark = "FOUND" if found else "MISSING"
            print(f"  {conf} {high}v{low}: {ah}({t_high}) vs {al}({t_low}) -> {mark}")
finally:
    db.close()
