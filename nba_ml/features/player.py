"""Player-level feature builder for next-game stat prediction.

Targets: pts / reb / ast for a single player against a single opponent
on a given date.

Temporal integrity (CLAUDE.md guardrail): every aggregation is filtered with
strict `game_date < as_of` — no row from the target game (or later) ever
contributes to its own features.

DNP rule: rows with minutes == 0 are excluded from all rolling/season averages
so a stretch of bench DNPs doesn't drag a player's average toward zero.
"""
from __future__ import annotations
from datetime import date
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from nba_ml.db.models import Game, PlayerGame, PlayerProp, TeamGameFeature

ROLL_SHORT = 5
ROLL_LONG = 10

PROP_MARKETS = ("pts", "reb", "ast", "stl", "blk", "tov")


def player_prop_line(
    db: Session, player_id: int, game_date: date, market: str,
) -> float | None:
    """Latest pre-tipoff prop line for this player+date+market.
    Returns None when no pre-game prop is on record."""
    return db.execute(
        select(PlayerProp.line).where(
            PlayerProp.player_id == player_id,
            PlayerProp.game_date == game_date,
            PlayerProp.market == market,
            PlayerProp.is_pregame.is_(True),
        ).order_by(PlayerProp.fetched_at.desc()).limit(1)
    ).scalar_one_or_none()


def has_player_props(
    db: Session, player_id: int, game_date: date,
    markets: tuple[str, ...] = PROP_MARKETS,
) -> bool:
    """True if a pre-game prop exists for this player+date for ALL markets.
    Used at training time to filter rows for the vegas-augmented variant."""
    for m in markets:
        if player_prop_line(db, player_id, game_date, m) is None:
            return False
    return True


PLAYER_FEATURE_COLUMNS: list[str] = [
    # Recent form (5g)
    "pts_roll5", "reb_roll5", "ast_roll5", "min_roll5",
    "fga_roll5", "fta_roll5", "tov_roll5", "stl_roll5", "blk_roll5",
    # Recent form (10g)
    "pts_roll10", "reb_roll10", "ast_roll10", "min_roll10",
    "stl_roll10", "blk_roll10", "tov_roll10",
    # Super-recent (3g) — latest rotation snapshot
    "min_roll3", "pts_roll3",
    # Per-minute rate stats — decouples production from playing time
    "pts_per_min_roll10", "reb_per_min_roll10", "ast_per_min_roll10",
    # Usage proxy: true shot attempts (FGA + 0.44*FTA) per minute
    "tsa_per_min_roll10",
    # Variance — streaky players & inconsistent rotations are harder to predict
    "pts_std_roll10", "min_std_roll10",
    # Hot/cold streak: recent vs season baseline
    "pts_streak", "min_streak",
    # Home/road career splits
    "pts_home_avg", "pts_road_avg",
    # Season-to-date
    "pts_season_avg", "reb_season_avg", "ast_season_avg", "min_season_avg",
    "stl_season_avg", "blk_season_avg", "tov_season_avg",
    "games_played_season",
    # vs this opponent (career)
    "pts_vs_opp", "reb_vs_opp", "ast_vs_opp",
    "stl_vs_opp", "blk_vs_opp", "tov_vs_opp",
    "n_games_vs_opp",
    # Opponent context
    "opp_pts_allowed_roll10", "opp_def_rating_roll5", "opp_pace_roll5",
    # Game context
    "is_home", "days_rest",
]

# Per-target Vegas variants. Each target uses ONLY its own prop line so a
# missing market doesn't disqualify the row for the other markets.
PLAYER_FEATURE_COLUMNS_PTS_VEGAS: list[str] = PLAYER_FEATURE_COLUMNS + ["vegas_pts_line"]
PLAYER_FEATURE_COLUMNS_REB_VEGAS: list[str] = PLAYER_FEATURE_COLUMNS + ["vegas_reb_line"]
PLAYER_FEATURE_COLUMNS_AST_VEGAS: list[str] = PLAYER_FEATURE_COLUMNS + ["vegas_ast_line"]
PLAYER_FEATURE_COLUMNS_STL_VEGAS: list[str] = PLAYER_FEATURE_COLUMNS + ["vegas_stl_line"]
PLAYER_FEATURE_COLUMNS_BLK_VEGAS: list[str] = PLAYER_FEATURE_COLUMNS + ["vegas_blk_line"]
PLAYER_FEATURE_COLUMNS_TOV_VEGAS: list[str] = PLAYER_FEATURE_COLUMNS + ["vegas_tov_line"]

VEGAS_FEATURE_COLUMNS_BY_TARGET: dict[str, list[str]] = {
    "pts": PLAYER_FEATURE_COLUMNS_PTS_VEGAS,
    "reb": PLAYER_FEATURE_COLUMNS_REB_VEGAS,
    "ast": PLAYER_FEATURE_COLUMNS_AST_VEGAS,
    "stl": PLAYER_FEATURE_COLUMNS_STL_VEGAS,
    "blk": PLAYER_FEATURE_COLUMNS_BLK_VEGAS,
    "tov": PLAYER_FEATURE_COLUMNS_TOV_VEGAS,
}


def _last_n_rows(db: Session, player_id: int, as_of: date, n: int) -> list:
    """Raw per-game stat rows for the player's last `n` games (DNPs excluded).
    Returned as list of (pts, reb, ast, minutes, fga, fta, tov) tuples."""
    return db.execute(
        select(
            PlayerGame.pts, PlayerGame.reb, PlayerGame.ast, PlayerGame.minutes,
            PlayerGame.fga, PlayerGame.fta, PlayerGame.tov,
            PlayerGame.stl, PlayerGame.blk,
        )
        .where(
            PlayerGame.player_id == player_id,
            PlayerGame.game_date < as_of,
            PlayerGame.minutes > 0,
        )
        .order_by(PlayerGame.game_date.desc())
        .limit(n)
    ).all()


def _avgs_from_rows(rows: list) -> dict:
    """Aggregate the row list returned by _last_n_rows into mean/std/n stats."""
    if not rows:
        return {
            "pts": 0.0, "reb": 0.0, "ast": 0.0, "min": 0.0,
            "fga": 0.0, "fta": 0.0, "tov": 0.0, "stl": 0.0, "blk": 0.0,
            "pts_std": 0.0, "min_std": 0.0, "n": 0,
        }
    n = len(rows)
    pts = [r[0] for r in rows]
    reb = [r[1] for r in rows]
    ast = [r[2] for r in rows]
    mins = [r[3] for r in rows]
    fga = [r[4] for r in rows]
    fta = [r[5] for r in rows]
    tov = [r[6] for r in rows]
    stl = [r[7] for r in rows]
    blk = [r[8] for r in rows]

    def _avg(xs): return sum(xs) / n
    def _std(xs):
        m = _avg(xs)
        return (sum((x - m) ** 2 for x in xs) / n) ** 0.5

    return {
        "pts": _avg(pts), "reb": _avg(reb), "ast": _avg(ast), "min": _avg(mins),
        "fga": _avg(fga), "fta": _avg(fta), "tov": _avg(tov),
        "stl": _avg(stl), "blk": _avg(blk),
        "pts_std": _std(pts), "min_std": _std(mins),
        "n": n,
    }


def _last_n_avgs(db: Session, player_id: int, as_of: date, n: int) -> dict:
    """Per-game averages + std over the player's last `n` games (DNPs excluded)."""
    return _avgs_from_rows(_last_n_rows(db, player_id, as_of, n))


def _season_avgs(db: Session, player_id: int, as_of: date, season: str) -> dict:
    """Per-game averages this season, strictly prior to `as_of`. Joins to
    Game on (game_id, team_id) so we can filter by season."""
    row = db.execute(
        select(
            func.avg(PlayerGame.pts), func.avg(PlayerGame.reb),
            func.avg(PlayerGame.ast), func.avg(PlayerGame.minutes),
            func.avg(PlayerGame.stl), func.avg(PlayerGame.blk),
            func.avg(PlayerGame.tov),
            func.count().label("games"),
        )
        .join(Game, and_(Game.game_id == PlayerGame.game_id,
                         Game.team_id == PlayerGame.team_id))
        .where(
            PlayerGame.player_id == player_id,
            PlayerGame.game_date < as_of,
            PlayerGame.minutes > 0,
            Game.season == season,
        )
    ).one()
    pts, reb, ast, mins, stl, blk, tov, games = row
    return {
        "pts": float(pts or 0.0), "reb": float(reb or 0.0),
        "ast": float(ast or 0.0), "min": float(mins or 0.0),
        "stl": float(stl or 0.0), "blk": float(blk or 0.0),
        "tov": float(tov or 0.0),
        "games": int(games or 0),
    }


def _home_road_pts(db: Session, player_id: int, as_of: date) -> dict:
    """Career PTS averages split by venue, prior to as_of."""
    rows = db.execute(
        select(PlayerGame.is_home, func.avg(PlayerGame.pts), func.count())
        .where(
            PlayerGame.player_id == player_id,
            PlayerGame.game_date < as_of,
            PlayerGame.minutes > 0,
        )
        .group_by(PlayerGame.is_home)
    ).all()
    out = {"home": 0.0, "road": 0.0}
    for is_home, avg_pts, _n in rows:
        out["home" if is_home else "road"] = float(avg_pts or 0.0)
    return out


def _vs_opp_avgs(db: Session, player_id: int, opponent_id: int,
                 as_of: date) -> dict:
    """Career averages vs this opponent, strictly prior to `as_of`."""
    row = db.execute(
        select(
            func.avg(PlayerGame.pts), func.avg(PlayerGame.reb),
            func.avg(PlayerGame.ast), func.avg(PlayerGame.stl),
            func.avg(PlayerGame.blk), func.avg(PlayerGame.tov),
            func.count(),
        ).where(
            PlayerGame.player_id == player_id,
            PlayerGame.opponent_id == opponent_id,
            PlayerGame.game_date < as_of,
            PlayerGame.minutes > 0,
        )
    ).one()
    pts, reb, ast, stl, blk, tov, n = row
    return {
        "pts": float(pts or 0.0), "reb": float(reb or 0.0),
        "ast": float(ast or 0.0), "stl": float(stl or 0.0),
        "blk": float(blk or 0.0), "tov": float(tov or 0.0),
        "n": int(n or 0),
    }


def _opp_context(db: Session, opponent_id: int, as_of: date) -> dict:
    """Opponent's recent defensive context: latest TeamGameFeature snapshot
    (def_rating, pace) plus pts_allowed averaged over their last 10 games."""
    tf = db.execute(
        select(TeamGameFeature)
        .where(
            TeamGameFeature.team_id == opponent_id,
            TeamGameFeature.game_date < as_of,
        )
        .order_by(TeamGameFeature.game_date.desc())
        .limit(1)
    ).scalar_one_or_none()

    def_rating = float(tf.def_rating_roll5) if tf and tf.def_rating_roll5 is not None else 110.0
    pace = float(tf.pace_roll5) if tf and tf.pace_roll5 is not None else 100.0

    pa_rows = db.execute(
        select(Game.pts_allowed)
        .where(
            Game.team_id == opponent_id,
            Game.game_date < as_of,
        )
        .order_by(Game.game_date.desc())
        .limit(ROLL_LONG)
    ).scalars().all()
    pts_allowed = float(sum(pa_rows) / len(pa_rows)) if pa_rows else 110.0

    return {
        "def_rating": def_rating,
        "pace": pace,
        "pts_allowed": pts_allowed,
    }


def _days_rest(db: Session, player_id: int, as_of: date) -> int:
    last = db.execute(
        select(func.max(PlayerGame.game_date))
        .where(
            PlayerGame.player_id == player_id,
            PlayerGame.game_date < as_of,
            PlayerGame.minutes > 0,
        )
    ).scalar_one_or_none()
    if last is None:
        return 7  # neutral default for first-game-ever case
    return (as_of - last).days


def _current_season(db: Session, as_of: date) -> str | None:
    """Most recent season string with a game on or before `as_of`."""
    return db.execute(
        select(Game.season)
        .where(Game.game_date <= as_of)
        .order_by(Game.game_date.desc())
        .limit(1)
    ).scalar_one_or_none()


MIN_PRIOR_GAMES = 5


def build_player_features(
    db: Session, player_id: int, opponent_id: int, as_of: date, is_home: bool,
) -> dict[str, float] | None:
    """Build the feature dict for one (player, opponent, date) prediction.

    Returns None when the player has fewer than MIN_PRIOR_GAMES strictly-prior
    appearances (with minutes > 0) — too little signal to predict.
    """
    short = _last_n_avgs(db, player_id, as_of, ROLL_SHORT)
    if short["n"] < MIN_PRIOR_GAMES:
        return None
    long_ = _last_n_avgs(db, player_id, as_of, ROLL_LONG)
    short3 = _last_n_avgs(db, player_id, as_of, 3)

    season = _current_season(db, as_of)
    season_avgs = _season_avgs(db, player_id, as_of, season) if season else {
        "pts": 0.0, "reb": 0.0, "ast": 0.0, "min": 0.0, "games": 0,
    }
    vs_opp = _vs_opp_avgs(db, player_id, opponent_id, as_of)
    opp = _opp_context(db, opponent_id, as_of)
    venue = _home_road_pts(db, player_id, as_of)

    # Per-minute rate stats over 10g (avoid div by zero with max(., 1)).
    min10 = max(long_["min"], 1.0)
    pts_per_min = long_["pts"] / min10
    reb_per_min = long_["reb"] / min10
    ast_per_min = long_["ast"] / min10
    # True shot attempts: FGA + 0.44*FTA — standard usage proxy.
    tsa10 = long_["fga"] + 0.44 * long_["fta"]
    tsa_per_min = tsa10 / min10

    return {
        "pts_roll5":  short["pts"],
        "reb_roll5":  short["reb"],
        "ast_roll5":  short["ast"],
        "min_roll5":  short["min"],
        "fga_roll5":  short["fga"],
        "fta_roll5":  short["fta"],
        "tov_roll5":  short["tov"],
        "stl_roll5":  short["stl"],
        "blk_roll5":  short["blk"],
        "pts_roll10": long_["pts"],
        "reb_roll10": long_["reb"],
        "ast_roll10": long_["ast"],
        "min_roll10": long_["min"],
        "stl_roll10": long_["stl"],
        "blk_roll10": long_["blk"],
        "tov_roll10": long_["tov"],
        "min_roll3":  short3["min"],
        "pts_roll3":  short3["pts"],
        "pts_per_min_roll10": pts_per_min,
        "reb_per_min_roll10": reb_per_min,
        "ast_per_min_roll10": ast_per_min,
        "tsa_per_min_roll10": tsa_per_min,
        "pts_std_roll10": long_["pts_std"],
        "min_std_roll10": long_["min_std"],
        "pts_streak": short["pts"] - season_avgs["pts"],
        "min_streak": short["min"] - season_avgs["min"],
        "pts_home_avg": venue["home"],
        "pts_road_avg": venue["road"],
        "pts_season_avg": season_avgs["pts"],
        "reb_season_avg": season_avgs["reb"],
        "ast_season_avg": season_avgs["ast"],
        "min_season_avg": season_avgs["min"],
        "stl_season_avg": season_avgs["stl"],
        "blk_season_avg": season_avgs["blk"],
        "tov_season_avg": season_avgs["tov"],
        "games_played_season": float(season_avgs["games"]),
        "pts_vs_opp": vs_opp["pts"],
        "reb_vs_opp": vs_opp["reb"],
        "ast_vs_opp": vs_opp["ast"],
        "stl_vs_opp": vs_opp["stl"],
        "blk_vs_opp": vs_opp["blk"],
        "tov_vs_opp": vs_opp["tov"],
        "n_games_vs_opp": float(vs_opp["n"]),
        "opp_pts_allowed_roll10": opp["pts_allowed"],
        "opp_def_rating_roll5":   opp["def_rating"],
        "opp_pace_roll5":         opp["pace"],
        "is_home": float(bool(is_home)),
        "days_rest": float(_days_rest(db, player_id, as_of)),
        # Always present in dict; base models ignore via their feature_columns.
        # Vegas-augmented variants pick them up. 0.0 sentinel when no prop;
        # the trainer filters out rows missing the relevant line so the
        # sentinel never enters the vegas-variant training set.
        "vegas_pts_line": float(player_prop_line(db, player_id, as_of, "pts") or 0.0),
        "vegas_reb_line": float(player_prop_line(db, player_id, as_of, "reb") or 0.0),
        "vegas_ast_line": float(player_prop_line(db, player_id, as_of, "ast") or 0.0),
        "vegas_stl_line": float(player_prop_line(db, player_id, as_of, "stl") or 0.0),
        "vegas_blk_line": float(player_prop_line(db, player_id, as_of, "blk") or 0.0),
        "vegas_tov_line": float(player_prop_line(db, player_id, as_of, "tov") or 0.0),
    }
