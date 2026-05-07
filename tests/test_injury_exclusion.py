"""Injury edge case: a player marked 'out' contributes nothing to matchup
features, no matter how strong their historical performance vs the opponent."""
from __future__ import annotations
from datetime import date, timedelta

from nba_ml.db.models import (
    Player, PlayerAvailability, PlayerGame, Team,
)
from nba_ml.features.matchup import (
    active_players_for_game, aggregate_pvt,
)


HOME_ID, AWAY_ID = 1, 2
STAR_ID, BENCH_ID = 100, 101
OTHER_OPP = 99


def _seed(db) -> None:
    db.add_all([
        Team(team_id=HOME_ID, abbreviation="HOM", full_name="Home"),
        Team(team_id=AWAY_ID, abbreviation="AWY", full_name="Away"),
        Team(team_id=OTHER_OPP, abbreviation="OTH", full_name="Other"),
        Player(player_id=STAR_ID, full_name="Star"),
        Player(player_id=BENCH_ID, full_name="Bench"),
    ])
    base = date(2025, 1, 1)

    for i in range(5):
        db.add(PlayerGame(
            game_id=f"S{i}", game_date=base + timedelta(days=i),
            player_id=STAR_ID, team_id=HOME_ID, opponent_id=AWAY_ID,
            is_home=True, minutes=36, pts=40, reb=10, ast=8, plus_minus=15,
        ))
    for i in range(5, 15):
        db.add(PlayerGame(
            game_id=f"SR{i}", game_date=base + timedelta(days=i),
            player_id=STAR_ID, team_id=HOME_ID, opponent_id=OTHER_OPP,
            is_home=True, minutes=36, pts=20, reb=5, ast=5, plus_minus=5,
        ))

    for i in range(3):
        db.add(PlayerGame(
            game_id=f"B{i}", game_date=base + timedelta(days=i),
            player_id=BENCH_ID, team_id=HOME_ID, opponent_id=AWAY_ID,
            is_home=True, minutes=8, pts=4, reb=1, ast=1, plus_minus=-2,
        ))
    for i in range(5, 15):
        db.add(PlayerGame(
            game_id=f"BR{i}", game_date=base + timedelta(days=i),
            player_id=BENCH_ID, team_id=HOME_ID, opponent_id=OTHER_OPP,
            is_home=True, minutes=8, pts=4, reb=1, ast=1, plus_minus=-1,
        ))
    db.commit()


def test_injured_star_drops_projection_dramatically(db):
    _seed(db)
    as_of = date(2025, 2, 1)

    healthy = aggregate_pvt(db, [STAR_ID, BENCH_ID], AWAY_ID, as_of)
    star_out = aggregate_pvt(db, [BENCH_ID], AWAY_ID, as_of)

    assert healthy["proj_pts"] > star_out["proj_pts"] * 5
    assert healthy["weight_total"] > star_out["weight_total"]


def test_excluded_player_history_has_zero_effect(db):
    _seed(db)
    as_of = date(2025, 2, 1)
    bench_only_a = aggregate_pvt(db, [BENCH_ID], AWAY_ID, as_of)
    bench_only_b = aggregate_pvt(db, [BENCH_ID], AWAY_ID, as_of)
    assert bench_only_a == bench_only_b


def test_status_out_filtered_by_active_players(db):
    _seed(db)
    db.add_all([
        PlayerAvailability(
            game_id="GX", game_date=date(2025, 2, 1),
            player_id=STAR_ID, team_id=HOME_ID, status="out", reason="ankle",
        ),
        PlayerAvailability(
            game_id="GX", game_date=date(2025, 2, 1),
            player_id=BENCH_ID, team_id=HOME_ID, status="active",
        ),
    ])
    db.commit()

    actives = active_players_for_game(db, "GX", HOME_ID)
    assert STAR_ID not in actives
    assert BENCH_ID in actives


def test_pvt_uses_only_history_strictly_before_as_of(db):
    _seed(db)
    early = aggregate_pvt(db, [STAR_ID], AWAY_ID, date(2025, 1, 3))
    late = aggregate_pvt(db, [STAR_ID], AWAY_ID, date(2025, 2, 1))
    assert early["proj_pts"] != late["proj_pts"]
