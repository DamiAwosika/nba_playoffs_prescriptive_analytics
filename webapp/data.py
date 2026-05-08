"""Data assembly for the Flask dashboard.

Three public functions:
  get_bracket()                       — full bracket payload for the index page
  get_series_predictions(t1, t2)      — predictions + Vegas reference for one matchup
  get_team_detail(t1, t2)             — side-by-side stat comparison for the modal
"""
from __future__ import annotations
from datetime import date, datetime, timedelta, timezone

_ET = timezone(timedelta(hours=-4))


def _today_et() -> date:
    """date.today() in US/Eastern so Railway (UTC) matches local behavior."""
    return datetime.now(_ET).date()

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from nba_ml.db.base import SessionLocal
from nba_ml.db.models import (
    BettingOdds, Game, Player, PlayerGame, PlayerProp, Team, TeamGameFeature,
)
from nba_ml.features.matchup import (
    FEATURE_COLUMNS, active_roster_as_of, build_matchup_features,
    has_vegas_odds, latest_team_feature, vegas_home_win_prob,
)
from nba_ml.inference.loader import get_loaded_model, list_available_models
from nba_ml.inference.player_predict import predict_player_stats

# Conference splits — used to lay the bracket out East / West.
EAST_ABBR = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
             "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST_ABBR = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
             "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}

# Names of trained models we want the dashboard to show.
DASHBOARD_MODEL_NAMES = ("logreg", "rf", "xgb", "ensemble", "stack")

# Standard NBA bracket sizes per round. The dashboard always renders this
# many slots per round; missing ones are filled with TBD placeholders.
ROUND_SIZES = {
    "first_round_west": 4,
    "conf_semis_west": 2,
    "conf_finals_west": 1,
    "finals": 1,
    "conf_finals_east": 1,
    "conf_semis_east": 2,
    "first_round_east": 4,
}

# ESPN's CDN serves clean team logos by abbreviation, but their slugs
# differ from balldontlie's for a handful of teams. Mapping here.
_ESPN_ABBR_OVERRIDE = {
    "GSW": "gs", "NOP": "no", "NYK": "ny",
    "SAS": "sa", "UTA": "utah", "WAS": "wsh",
}


def _logo_url(abbr: str) -> str:
    slug = _ESPN_ABBR_OVERRIDE.get(abbr, abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png"


# NBA bracket positions per conference, top-to-bottom in canonical order.
# The pairs feed into semis as: (slot 0 winner) vs (slot 1 winner),
# and (slot 2 winner) vs (slot 3 winner).
CANONICAL_R1_MATCHUPS: list[tuple[int, int]] = [(1, 8), (4, 5), (3, 6), (2, 7)]


# ===== Public API =====

def get_bracket() -> dict:
    """Build the bracket using canonical NBA seeding.

    Steps:
      1. Compute each playoff team's seed (1-8 per conference) from regular-
         season win count.
      2. Build first round in canonical order — 1v8 / 4v5 / 3v6 / 2v7 — for
         each conference. Missing series become TBD.
      3. Walk advancement: when a first-round series completes, fill the
         corresponding semis slot with the predetermined matchup
         (winners face each other, "Series tied 0-0"). Same for finals.
    """
    db = SessionLocal()
    try:
        season = _current_playoff_season(db)
        if not season:
            return {"season": None, "rounds": _full_tbd_rounds()}

        teams_by_id = _all_teams(db)
        seeds = _compute_seeds(db, season, teams_by_id)
        for tid, t in teams_by_id.items():
            t["seed"] = seeds.get(tid)

        # Pre-fetch regular-season records for predetermined slots.
        records = {tid: _regular_season_record(db, tid, season) for tid in seeds}

        actual_series = _identify_series(db, season, teams_by_id)
        for s in actual_series:
            s["team1"]["seed"] = seeds.get(s["team1"]["team_id"])
            s["team2"]["seed"] = seeds.get(s["team2"]["team_id"])

        west = _build_conference(
            "West", seeds, actual_series, teams_by_id, records,
        )
        east = _build_conference(
            "East", seeds, actual_series, teams_by_id, records,
        )

        # Finals: West conf final winner vs East conf final winner.
        west_champ = west["conf_final"].get("winner_id")
        east_champ = east["conf_final"].get("winner_id")
        if west_champ and east_champ:
            actual = _find_series_for_teams(actual_series, west_champ, east_champ)
            finals = actual or _predetermined_series(
                west_champ, east_champ, teams_by_id, seeds, records,
            )
        else:
            finals = _tbd_series()

        rounds = {
            "first_round_west": west["r1"],
            "conf_semis_west":  west["semis"],
            "conf_finals_west": [west["conf_final"]],
            "finals":           [finals],
            "conf_finals_east": [east["conf_final"]],
            "conf_semis_east":  east["semis"],
            "first_round_east": east["r1"],
        }
        return {"season": season, "rounds": rounds}
    finally:
        db.close()


def _compute_seeds(db: Session, season: str, teams_by_id: dict) -> dict[int, int]:
    """Compute seed (1-8) for each playoff team within its conference.

    RS-wins ranking alone is unreliable: (a) seeds 7/8 come from the play-in
    tournament, not RS standings; (b) ties (e.g. 46-36 ATL vs 46-36 TOR) are
    broken by NBA tiebreakers we don't replicate. So we use the actual R1
    pairings in the DB as ground truth and only use RS wins to disambiguate
    within-pair ordering.

    Algorithm per conference:
      1. seed 1 = top RS-wins; seed 2 = second.
      2. seed 1's R1 opponent → seed 8; seed 2's R1 opponent → seed 7.
      3. The remaining 2 R1 series each have a higher- and lower-RS team.
         The series whose higher team has more RS wins is 3v6, the other 4v5.
    """
    r1_pairs_by_conf = _find_r1_pairs(db, season, teams_by_id)

    # Per-conference RS wins for every playoff team.
    playoff_team_ids = db.execute(
        select(Game.team_id).distinct().where(
            Game.season == season,
            Game.is_playoff.is_(True),
        )
    ).scalars().all()

    by_conf: dict[str, list[tuple[int, int]]] = {"West": [], "East": []}
    for tid in playoff_team_ids:
        team = teams_by_id.get(tid)
        if not team or team.get("conference") not in by_conf:
            continue
        wins, _losses = _regular_season_record(db, tid, season)
        by_conf[team["conference"]].append((tid, wins))

    seeds: dict[int, int] = {}
    for conf, lst in by_conf.items():
        lst.sort(key=lambda x: -x[1])
        if len(lst) < 2:
            continue
        wins_of = dict(lst)
        seed1, seed2 = lst[0][0], lst[1][0]
        seeds[seed1] = 1
        seeds[seed2] = 2

        r1_pairs = r1_pairs_by_conf.get(conf, [])
        seed8 = _other_in_pair(r1_pairs, seed1)
        seed7 = _other_in_pair(r1_pairs, seed2)
        if seed8:
            seeds[seed8] = 8
        if seed7:
            seeds[seed7] = 7

        # Remaining 2 pairs → 3v6 and 4v5.
        used = {seed1, seed2, seed8, seed7}
        other_pairs = [(a, b) for a, b in r1_pairs
                       if a not in used and b not in used]
        normalized = []
        for a, b in other_pairs:
            wa, wb = wins_of.get(a, 0), wins_of.get(b, 0)
            high, low = (a, b) if wa >= wb else (b, a)
            normalized.append((high, low))
        normalized.sort(key=lambda p: -wins_of.get(p[0], 0))
        if normalized:
            high, low = normalized[0]
            seeds[high], seeds[low] = 3, 6
        if len(normalized) > 1:
            high, low = normalized[1]
            seeds[high], seeds[low] = 4, 5
    return seeds


def _find_r1_pairs(db: Session, season: str,
                   teams_by_id: dict) -> dict[str, list[tuple[int, int]]]:
    """Return the (up to 4) earliest-starting playoff series per conference.

    Uses start date as the round discriminator: NBA R1 always begins ~10 days
    before semis, so the first 4 series per conference (by start date) are R1.
    """
    rows = db.execute(
        select(
            Game.team_id, Game.opponent_id,
            func.min(Game.game_date).label("start_date"),
        ).where(
            Game.season == season,
            Game.is_playoff.is_(True),
        ).group_by(Game.team_id, Game.opponent_id)
    ).all()

    seen: set[tuple[int, int]] = set()
    series: list[tuple] = []  # (start_date, conf, t1, t2)
    for t1, t2, start in rows:
        key = tuple(sorted([t1, t2]))
        if key in seen:
            continue
        seen.add(key)
        conf = teams_by_id.get(t1, {}).get("conference")
        if conf in ("West", "East"):
            series.append((start, conf, t1, t2))

    series.sort(key=lambda s: s[0] or date.max)
    by_conf: dict[str, list[tuple[int, int]]] = {"West": [], "East": []}
    for _start, conf, t1, t2 in series:
        if len(by_conf[conf]) < 4:
            by_conf[conf].append((t1, t2))
    return by_conf


def _other_in_pair(pairs: list[tuple[int, int]], tid: int) -> int | None:
    for a, b in pairs:
        if a == tid:
            return b
        if b == tid:
            return a
    return None


def _build_conference(conf: str, seeds: dict[int, int], actual_series: list[dict],
                      teams_by_id: dict, records: dict) -> dict:
    """Returns {r1: [4 series], semis: [2 series], conf_final: series}."""
    # Map seed -> team_id within this conference
    conf_seeds = {
        seed: tid for tid, seed in seeds.items()
        if teams_by_id.get(tid, {}).get("conference") == conf
    }

    # First round in canonical bracket order
    r1: list[dict] = []
    for high, low in CANONICAL_R1_MATCHUPS:
        t_high = conf_seeds.get(high)
        t_low = conf_seeds.get(low)
        r1.append(_resolve_slot(t_high, t_low, actual_series, teams_by_id, seeds, records))

    # Semis: pair (r1[0], r1[1]) and (r1[2], r1[3])
    semis: list[dict] = []
    for top_idx, bot_idx in [(0, 1), (2, 3)]:
        winner_top = r1[top_idx].get("winner_id")
        winner_bot = r1[bot_idx].get("winner_id")
        semis.append(_resolve_slot(winner_top, winner_bot, actual_series, teams_by_id, seeds, records))

    # Conf finals
    cf_top = semis[0].get("winner_id")
    cf_bot = semis[1].get("winner_id")
    conf_final = _resolve_slot(cf_top, cf_bot, actual_series, teams_by_id, seeds, records)

    return {"r1": r1, "semis": semis, "conf_final": conf_final}


def _resolve_slot(t1_id, t2_id, actual_series, teams_by_id, seeds, records) -> dict:
    """For a bracket slot expecting two teams, return the most accurate series:
    actual (if found in DB) → predetermined (if both teams known) → TBD."""
    if t1_id and t2_id:
        actual = _find_series_for_teams(actual_series, t1_id, t2_id)
        if actual:
            return actual
        return _predetermined_series(t1_id, t2_id, teams_by_id, seeds, records)
    return _tbd_series()


def _find_series_for_teams(actual_series: list[dict], t1: int, t2: int) -> dict | None:
    target = {t1, t2}
    for s in actual_series:
        if {s["team1"]["team_id"], s["team2"]["team_id"]} == target:
            return s
    return None


def _predetermined_series(t1_id: int, t2_id: int, teams_by_id: dict,
                          seeds: dict, records: dict) -> dict:
    """A matchup whose teams are known via bracket advancement, but no games
    have been played yet. Renders as 'Series tied 0-0', clickable for
    predictions about the FIRST game of the upcoming series."""
    t1 = teams_by_id.get(t1_id, _unknown_team_meta(t1_id))
    t2 = teams_by_id.get(t2_id, _unknown_team_meta(t2_id))
    t1_w, t1_l = records.get(t1_id, (None, None))
    t2_w, t2_l = records.get(t2_id, (None, None))

    return {
        "team1": {**t1, "regular_season_wins": t1_w, "regular_season_losses": t1_l,
                  "series_wins": 0, "seed": seeds.get(t1_id)},
        "team2": {**t2, "regular_season_wins": t2_w, "regular_season_losses": t2_l,
                  "series_wins": 0, "seed": seeds.get(t2_id)},
        "start_date": None, "last_date": None, "n_games": 0,
        "status": "in_progress", "winner_id": None,
        "conference": t1.get("conference"),
        "is_placeholder": False, "is_predetermined": True,
    }


def _unknown_team_meta(tid: int) -> dict:
    return {"team_id": tid, "abbreviation": "???", "full_name": "Unknown",
            "conference": "Unknown", "logo_url": None, "seed": None}


def _tbd_team() -> dict:
    return {
        "team_id": 0,
        "abbreviation": "TBD",
        "full_name": "TBD",
        "conference": "Unknown",
        "regular_season_wins": None,
        "regular_season_losses": None,
        "series_wins": 0,
        "logo_url": None,
    }


def _tbd_series() -> dict:
    return {
        "team1": _tbd_team(),
        "team2": _tbd_team(),
        "start_date": None,
        "n_games": 0,
        "status": "tbd",
        "winner_id": None,
        "conference": "Unknown",
        "is_placeholder": True,
    }


def _full_tbd_rounds() -> dict:
    return {key: [_tbd_series() for _ in range(size)]
            for key, size in ROUND_SIZES.items()}


def get_series_predictions(team1_id: int, team2_id: int,
                            as_of: date | None = None) -> dict:
    """Quick prediction summary for a series. Used by the click handler.
    `as_of` defaults to tomorrow; pass an earlier date for retrospective
    'what did the model predict before this past game?' views."""
    db = SessionLocal()
    try:
        as_of = as_of or (_today_et() + timedelta(days=1))
        models = _available_models()

        preds_t1_home = {n: _safe_predict(db, m, team1_id, team2_id, as_of)
                         for n, m in models.items()}
        preds_t2_home = {n: _safe_predict(db, m, team2_id, team1_id, as_of)
                         for n, m in models.items()}

        vegas_t1_home = vegas_home_win_prob(db, team1_id, team2_id, as_of)
        vegas_t1_home = None if vegas_t1_home == 0.5 else vegas_t1_home
        vegas_t2_home = vegas_home_win_prob(db, team2_id, team1_id, as_of)
        vegas_t2_home = None if vegas_t2_home == 0.5 else vegas_t2_home

        return {
            "team1_at_home": {"predictions": preds_t1_home,
                              "vegas_home_win_prob": vegas_t1_home},
            "team2_at_home": {"predictions": preds_t2_home,
                              "vegas_home_win_prob": vegas_t2_home},
        }
    finally:
        db.close()


def get_team_roster(team_id: int) -> dict:
    """Per-player season totals + per-game averages for a team's roster.

    Aggregates ALL games (regular season + playoffs) for the current season,
    joining PlayerGame -> Game by game_id to filter on Game.season.

    Computes Hollinger Game Score per game:
       GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB
              + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

    Game Score is the per-game version of Hollinger PER — true PER additionally
    normalizes against league pace and minutes (centered at 15), which we don't
    compute. Game Score is the standard practical roster-level metric.
    """
    db = SessionLocal()
    try:
        season = _current_playoff_season(db) or _latest_season(db)
        team = _team_info(db, team_id)

        rows = db.execute(
            select(
                PlayerGame.player_id,
                func.count().label("games"),
                func.sum(PlayerGame.minutes).label("min_total"),
                func.sum(PlayerGame.pts).label("pts_total"),
                func.sum(PlayerGame.reb).label("reb_total"),
                func.sum(PlayerGame.ast).label("ast_total"),
                func.sum(PlayerGame.plus_minus).label("pm_total"),
                func.sum(PlayerGame.fgm).label("fgm_total"),
                func.sum(PlayerGame.fga).label("fga_total"),
                func.sum(PlayerGame.ftm).label("ftm_total"),
                func.sum(PlayerGame.fta).label("fta_total"),
                func.sum(PlayerGame.oreb).label("oreb_total"),
                func.sum(PlayerGame.dreb).label("dreb_total"),
                func.sum(PlayerGame.stl).label("stl_total"),
                func.sum(PlayerGame.blk).label("blk_total"),
                func.sum(PlayerGame.tov).label("tov_total"),
                func.sum(PlayerGame.pf).label("pf_total"),
            )
            .join(Game, and_(Game.game_id == PlayerGame.game_id,
                             Game.team_id == PlayerGame.team_id))
            .where(
                PlayerGame.team_id == team_id,
                Game.season == season,
                # Exclude DNPs: balldontlie returns a row for every player on
                # the active roster, including ones who didn't play. Counting
                # those would inflate G and deflate per-game averages.
                PlayerGame.minutes > 0,
            )
            .group_by(PlayerGame.player_id)
        ).all()

        if not rows:
            return {"team": team, "season": season, "players": []}

        from nba_ml.db.models import Player
        names = dict(db.execute(
            select(Player.player_id, Player.full_name).where(
                Player.player_id.in_([r[0] for r in rows])
            )
        ).all())

        players = []
        for r in rows:
            g = r.games or 1
            ppg  = (r.pts_total or 0) / g
            rpg  = (r.reb_total or 0) / g
            apg  = (r.ast_total or 0) / g
            mpg  = (r.min_total or 0) / g
            pmpg = (r.pm_total  or 0) / g
            fgm, fga = (r.fgm_total or 0), (r.fga_total or 0)
            ftm, fta = (r.ftm_total or 0), (r.fta_total or 0)
            oreb, dreb = (r.oreb_total or 0), (r.dreb_total or 0)
            stl, blk, tov, pf = (r.stl_total or 0), (r.blk_total or 0), (r.tov_total or 0), (r.pf_total or 0)

            # Hollinger Game Score (per-game).
            gmsc_total = (
                (r.pts_total or 0)
                + 0.4 * fgm - 0.7 * fga
                - 0.4 * (fta - ftm)
                + 0.7 * oreb + 0.3 * dreb
                + stl + 0.7 * apg * g  # AST scaled back to total via *g not needed; rewrite below
            )
            # Simpler: compute on totals then divide.
            gmsc_total = (
                (r.pts_total or 0)
                + 0.4 * fgm - 0.7 * fga
                - 0.4 * (fta - ftm)
                + 0.7 * oreb + 0.3 * dreb
                + stl
                + 0.7 * (r.ast_total or 0)
                + 0.7 * blk
                - 0.4 * pf
                - tov
            )
            gmsc = gmsc_total / g

            # True shooting %: PTS / (2 * (FGA + 0.44*FTA))
            ts_denom = 2 * (fga + 0.44 * fta)
            ts_pct = (r.pts_total or 0) / ts_denom if ts_denom > 0 else None

            players.append({
                "player_id": r.player_id,
                "name": names.get(r.player_id, f"Player {r.player_id}"),
                "games": g,
                "mpg": round(mpg, 1),
                "ppg": round(ppg, 1),
                "rpg": round(rpg, 1),
                "apg": round(apg, 1),
                "spg": round(stl / g, 1),
                "bpg": round(blk / g, 1),
                "topg": round(tov / g, 1),
                "plus_minus": round(pmpg, 1),
                "ts_pct": round(ts_pct, 3) if ts_pct is not None else None,
                "game_score": round(gmsc, 1),
            })
        players.sort(key=lambda p: -p["mpg"])
        return {"team": team, "season": season, "players": players}
    finally:
        db.close()


def get_head_to_head(team1_id: int, team2_id: int) -> dict:
    """Regular-season head-to-head between two teams for the current season.

    Returns:
      - team record (wins for each side)
      - team-level per-game averages in those H2H games (pts, reb, ast, +/-)
      - per-player averages in those H2H games for each team
    Excludes playoff games — postseason matchups already render in the
    bracket / Playoff averages section.
    """
    db = SessionLocal()
    try:
        season = _current_playoff_season(db) or _latest_season(db)
        team1 = _team_info(db, team1_id)
        team2 = _team_info(db, team2_id)

        # H2H game IDs this regular season (one row per game from team1's POV).
        h2h_games = db.execute(
            select(Game.game_id, Game.won, Game.pts, Game.pts_allowed)
            .where(
                Game.season == season,
                Game.is_playoff.is_(False),
                Game.team_id == team1_id,
                Game.opponent_id == team2_id,
            )
        ).all()

        n = len(h2h_games)
        team1_wins = sum(1 for g in h2h_games if g.won)
        team2_wins = n - team1_wins
        game_ids = [g.game_id for g in h2h_games]

        return {
            "team1": team1,
            "team2": team2,
            "season": season,
            "n_games": n,
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "team1_avgs": _h2h_team_avgs(db, team1_id, team2_id, season),
            "team2_avgs": _h2h_team_avgs(db, team2_id, team1_id, season),
            "team1_players": _h2h_player_avgs(db, team1_id, team2_id, game_ids),
            "team2_players": _h2h_player_avgs(db, team2_id, team1_id, game_ids),
        }
    finally:
        db.close()


def _h2h_team_avgs(db: Session, team_id: int, opp_id: int, season: str) -> dict:
    """Team-level per-game averages from this season's RS H2H games."""
    games = db.execute(
        select(Game).where(
            Game.season == season,
            Game.is_playoff.is_(False),
            Game.team_id == team_id,
            Game.opponent_id == opp_id,
        )
    ).scalars().all()
    n = len(games)
    if n == 0:
        return {"games": 0, "pts": None, "pts_allowed": None,
                "reb": None, "ast": None}

    pts = sum(g.pts for g in games) / n
    pa = sum(g.pts_allowed for g in games) / n

    pg_rows = db.execute(
        select(PlayerGame.game_id,
               func.sum(PlayerGame.reb).label("reb"),
               func.sum(PlayerGame.ast).label("ast"))
        .where(
            PlayerGame.team_id == team_id,
            PlayerGame.game_id.in_([g.game_id for g in games]),
        ).group_by(PlayerGame.game_id)
    ).all()
    reb = (sum(r[1] or 0 for r in pg_rows) / len(pg_rows)) if pg_rows else 0
    ast = (sum(r[2] or 0 for r in pg_rows) / len(pg_rows)) if pg_rows else 0
    return {"games": n, "pts": round(pts, 1), "pts_allowed": round(pa, 1),
            "reb": round(reb, 1), "ast": round(ast, 1)}


def _h2h_player_avgs(db: Session, team_id: int, opp_id: int,
                     game_ids: list) -> list:
    """Per-player averages in H2H games. Only includes players who played
    at least one of these games (minutes > 0)."""
    if not game_ids:
        return []
    rows = db.execute(
        select(
            PlayerGame.player_id,
            func.count().label("games"),
            func.sum(PlayerGame.minutes).label("min_total"),
            func.sum(PlayerGame.pts).label("pts_total"),
            func.sum(PlayerGame.reb).label("reb_total"),
            func.sum(PlayerGame.ast).label("ast_total"),
            func.sum(PlayerGame.plus_minus).label("pm_total"),
            func.sum(PlayerGame.stl).label("stl_total"),
            func.sum(PlayerGame.blk).label("blk_total"),
            func.sum(PlayerGame.tov).label("tov_total"),
            func.sum(PlayerGame.fgm).label("fgm_total"),
            func.sum(PlayerGame.fga).label("fga_total"),
            func.sum(PlayerGame.ftm).label("ftm_total"),
            func.sum(PlayerGame.fta).label("fta_total"),
            func.sum(PlayerGame.oreb).label("oreb_total"),
            func.sum(PlayerGame.dreb).label("dreb_total"),
            func.sum(PlayerGame.pf).label("pf_total"),
        ).where(
            PlayerGame.team_id == team_id,
            PlayerGame.opponent_id == opp_id,
            PlayerGame.game_id.in_(game_ids),
            PlayerGame.minutes > 0,
        ).group_by(PlayerGame.player_id)
    ).all()

    if not rows:
        return []

    from nba_ml.db.models import Player
    names = dict(db.execute(
        select(Player.player_id, Player.full_name).where(
            Player.player_id.in_([r.player_id for r in rows])
        )
    ).all())

    players = []
    for r in rows:
        g = r.games or 1
        fgm, fga = (r.fgm_total or 0), (r.fga_total or 0)
        ftm, fta = (r.ftm_total or 0), (r.fta_total or 0)
        oreb, dreb = (r.oreb_total or 0), (r.dreb_total or 0)
        stl, blk, tov, pf = (r.stl_total or 0), (r.blk_total or 0), (r.tov_total or 0), (r.pf_total or 0)

        ts_denom = 2 * (fga + 0.44 * fta)
        ts_pct = (r.pts_total or 0) / ts_denom if ts_denom > 0 else None

        gmsc_total = (
            (r.pts_total or 0)
            + 0.4 * fgm - 0.7 * fga
            - 0.4 * (fta - ftm)
            + 0.7 * oreb + 0.3 * dreb
            + stl
            + 0.7 * (r.ast_total or 0)
            + 0.7 * blk
            - 0.4 * pf
            - tov
        )

        players.append({
            "player_id": r.player_id,
            "name": names.get(r.player_id, f"Player {r.player_id}"),
            "games": g,
            "mpg": round((r.min_total or 0) / g, 1),
            "ppg": round((r.pts_total or 0) / g, 1),
            "rpg": round((r.reb_total or 0) / g, 1),
            "apg": round((r.ast_total or 0) / g, 1),
            "spg": round(stl / g, 1),
            "bpg": round(blk / g, 1),
            "topg": round(tov / g, 1),
            "plus_minus": round((r.pm_total or 0) / g, 1),
            "ts_pct": round(ts_pct, 3) if ts_pct is not None else None,
            "game_score": round(gmsc_total / g, 1),
        })
    players.sort(key=lambda p: -p["mpg"])
    return players


def get_player_predictions(team1_id: int, team2_id: int) -> dict:
    """Per-player PTS/REB/AST predictions for a matchup.

    Series state determines prediction date:
      - ongoing / predetermined: predict for the next game (use the nearest
        future game date from the schedule so prop lookups match)
      - completed: predict as-of the last game of the series
    """
    db = SessionLocal()
    try:
        season = _current_playoff_season(db) or _latest_season(db)
        teams_by_id = _all_teams(db)

        actual_series = _identify_series(db, season, teams_by_id) if season else []
        series = _find_series_for_teams(actual_series, team1_id, team2_id)

        if series and series.get("status") == "completed" and series.get("last_date"):
            as_of = date.fromisoformat(series["last_date"])
            context = "last_game"
        else:
            as_of = _next_game_date(db, team1_id, team2_id) or (_today_et() + timedelta(days=1))
            context = "next_game"

        # Determine home team for prediction context.
        # For ongoing: alternate from the last game played.
        # For completed: use the last game's actual home team.
        t1_is_home = _infer_home_team(db, team1_id, team2_id, season, as_of)

        team1 = _team_info(db, team1_id)
        team2 = _team_info(db, team2_id)

        t1_roster = _roster_near_date(db, team1_id, as_of)
        t2_roster = _roster_near_date(db, team2_id, as_of)

        t1_active = set(active_roster_as_of(db, team1_id, as_of))
        t2_active = set(active_roster_as_of(db, team2_id, as_of))

        names = dict(db.execute(
            select(Player.player_id, Player.full_name).where(
                Player.player_id.in_(t1_roster + t2_roster)
            )
        ).all())

        t1_preds = _predict_roster(db, t1_roster, team2_id, as_of, t1_is_home, names, t1_active)
        t2_preds = _predict_roster(db, t2_roster, team1_id, as_of, not t1_is_home, names, t2_active)

        return {
            "prediction_context": context,
            "prediction_date": as_of.isoformat(),
            "team1": team1,
            "team2": team2,
            "team1_players": t1_preds,
            "team2_players": t2_preds,
        }
    finally:
        db.close()


def _predict_roster(
    db: Session, player_ids: list[int], opp_id: int,
    as_of: date, is_home: bool, names: dict,
    active_ids: set[int] | None = None,
) -> list[dict]:
    has_availability = bool(active_ids)
    results = []
    for pid in player_ids:
        is_active = (not has_availability) or (pid in active_ids)
        entry = {
            "player_id": pid,
            "name": names.get(pid, f"Player {pid}"),
            "active": is_active,
        }
        if not is_active:
            for target in ("pts", "reb", "ast", "stl", "blk", "tov"):
                entry[target] = None
                entry[f"{target}_detail"] = {}
            entry["variants"] = {}
            entry["prop_lines"] = {}
            results.append(entry)
            continue

        pred = predict_player_stats(db, pid, opp_id, as_of, is_home)
        if pred is None:
            continue
        for target in ("pts", "reb", "ast", "stl", "blk", "tov"):
            t_preds = pred.get(target, {})
            entry[target] = t_preds.get("ensemble") or t_preds.get("rf") or t_preds.get("linear")
            entry[f"{target}_detail"] = t_preds
        entry["variants"] = pred.get("variants", {})
        entry["prop_lines"] = pred.get("prop_lines", {})
        results.append(entry)
    active = [p for p in results if p["active"]]
    inactive = [p for p in results if not p["active"]]
    active.sort(key=lambda p: -(p.get("pts") or 0))
    inactive.sort(key=lambda p: p["name"])
    return active + inactive


def _next_game_date(db: Session, t1: int, t2: int) -> date | None:
    """Nearest prop-covered or scheduled game date for this matchup."""
    today = _today_et()
    # Props are keyed by team_id — check either team.
    prop_date = db.execute(
        select(PlayerProp.game_date).where(
            PlayerProp.is_pregame.is_(True),
            PlayerProp.game_date >= today,
            PlayerProp.team_id.in_([t1, t2]),
        ).order_by(PlayerProp.game_date.asc()).limit(1)
    ).scalar_one_or_none()
    if prop_date:
        return prop_date
    return None


def _infer_home_team(db: Session, t1: int, t2: int, season: str | None,
                     as_of: date) -> bool:
    """Determine whether team1 is home for the next game using the NBA
    playoff 2-2-1-1-1 format.

    Game 1 home team has home-court advantage (higher seed). The pattern:
      Game 1,2: HCA team home | Game 3,4: other team home
      Game 5: HCA team | Game 6: other team | Game 7: HCA team
    """
    if not season:
        return True
    # All playoff games between these teams, from team1's perspective.
    games = db.execute(
        select(Game.is_home).where(
            Game.season == season,
            Game.is_playoff.is_(True),
            Game.team_id == t1,
            Game.opponent_id == t2,
        ).order_by(Game.game_date.asc())
    ).scalars().all()

    if not games:
        return True

    # Game 1 tells us who has home-court advantage.
    t1_has_hca = games[0]  # True if t1 was home in game 1
    next_game_num = len(games) + 1

    # 2-2-1-1-1: HCA team is home in games 1, 2, 5, 7
    hca_home_games = {1, 2, 5, 7}
    hca_is_home = next_game_num in hca_home_games

    if t1_has_hca:
        return hca_is_home
    return not hca_is_home


def _latest_season(db: Session) -> str | None:
    return db.execute(
        select(Game.season).order_by(Game.season.desc()).limit(1)
    ).scalar_one_or_none()


def get_team_detail(team1_id: int, team2_id: int) -> dict:
    """Side-by-side comparison: bracket modal calls this when a series is opened.

    Picks the prediction date based on series state:
      - completed: predict the deciding (last) game using stats from before it
      - ongoing / predetermined: predict tomorrow's game using current stats
    """
    db = SessionLocal()
    try:
        season = _current_playoff_season(db)
        teams_by_id = _all_teams(db)
        seeds = _compute_seeds(db, season, teams_by_id) if season else {}
        for tid, t in teams_by_id.items():
            t["seed"] = seeds.get(tid)

        team1 = teams_by_id.get(team1_id, _unknown_team_meta(team1_id))
        team2 = teams_by_id.get(team2_id, _unknown_team_meta(team2_id))

        team1_playoff = _team_playoff_averages(db, team1_id, season)
        team2_playoff = _team_playoff_averages(db, team2_id, season)
        team1_features = _team_feature_snapshot(db, team1_id)
        team2_features = _team_feature_snapshot(db, team2_id)

        # Decide prediction date: deciding game for completed series, else tomorrow.
        actual_series = _identify_series(db, season, teams_by_id) if season else []
        series = _find_series_for_teams(actual_series, team1_id, team2_id)

        if series and series.get("status") == "completed" and series.get("last_date"):
            as_of = date.fromisoformat(series["last_date"]) + timedelta(days=1)
            context = "deciding_game"
        else:
            as_of = _next_game_date(db, team1_id, team2_id) or (_today_et() + timedelta(days=1))
            context = "next_game"

        preds_summary = get_series_predictions(team1_id, team2_id, as_of=as_of)

        return {
            "team1": team1,
            "team2": team2,
            "team1_playoff_stats": team1_playoff,
            "team2_playoff_stats": team2_playoff,
            "team1_features": team1_features,
            "team2_features": team2_features,
            "preds": preds_summary,
            "prediction_context": context,
            "prediction_date": as_of.isoformat(),
            "series_status": series.get("status") if series else "predetermined",
        }
    finally:
        db.close()


# ===== Helpers (private) =====

def _current_playoff_season(db: Session) -> str | None:
    return db.execute(
        select(Game.season).where(Game.is_playoff.is_(True))
        .order_by(Game.season.desc()).limit(1)
    ).scalar_one_or_none()


def _all_teams(db: Session) -> dict[int, dict]:
    teams = db.execute(select(Team)).scalars().all()
    return {
        t.team_id: {
            "team_id": t.team_id,
            "abbreviation": t.abbreviation,
            "full_name": t.full_name,
            "conference": _conference_of(t.abbreviation),
            "logo_url": _logo_url(t.abbreviation),
        }
        for t in teams
    }


def _conference_of(abbr: str) -> str:
    if abbr in EAST_ABBR:
        return "East"
    if abbr in WEST_ABBR:
        return "West"
    return "Unknown"


def _team_info(db: Session, team_id: int) -> dict:
    t = db.execute(select(Team).where(Team.team_id == team_id)).scalar_one_or_none()
    if not t:
        return {"team_id": team_id, "abbreviation": "???", "full_name": "Unknown",
                "conference": "Unknown", "logo_url": None}
    return {"team_id": team_id, "abbreviation": t.abbreviation,
            "full_name": t.full_name, "conference": _conference_of(t.abbreviation),
            "logo_url": _logo_url(t.abbreviation)}


def _regular_season_record(db: Session, team_id: int, season: str) -> tuple[int, int]:
    rows = db.execute(
        select(Game.won).where(
            Game.team_id == team_id,
            Game.season == season,
            Game.is_playoff.is_(False),
        )
    ).scalars().all()
    wins = sum(rows)
    return wins, len(rows) - wins


def _identify_series(db: Session, season: str, teams_by_id: dict) -> list[dict]:
    """Find every unique playoff matchup this season + compute series state.

    Identifies pairs from any playoff row (each game has one row per team
    perspective). Deliberately does NOT filter by is_home — that flag has
    been observed to be unset/inconsistent for some ingested games, which
    would silently drop entire series from the bracket.
    """
    pairs = db.execute(
        select(
            Game.team_id, Game.opponent_id,
            func.min(Game.game_date).label("start_date"),
            func.max(Game.game_date).label("last_date"),
            func.count().label("n_games"),
        ).where(
            Game.season == season,
            Game.is_playoff.is_(True),
        ).group_by(Game.team_id, Game.opponent_id)
    ).all()

    seen: set[tuple] = set()
    series_list = []
    for t1, t2, start, _last, n in pairs:
        key = tuple(sorted([t1, t2]))
        if key in seen:
            continue
        seen.add(key)

        t1_wins, t2_wins = _series_score(db, t1, t2, season)
        t1_rs_w, t1_rs_l = _regular_season_record(db, t1, season)
        t2_rs_w, t2_rs_l = _regular_season_record(db, t2, season)

        winner = t1 if t1_wins == 4 else t2 if t2_wins == 4 else None
        status = "completed" if winner else "in_progress"

        t1_meta = teams_by_id.get(t1, {"team_id": t1, "abbreviation": "???",
                                       "full_name": "Unknown", "conference": "Unknown"})
        t2_meta = teams_by_id.get(t2, {"team_id": t2, "abbreviation": "???",
                                       "full_name": "Unknown", "conference": "Unknown"})

        series_list.append({
            "team1": {**t1_meta, "regular_season_wins": t1_rs_w,
                      "regular_season_losses": t1_rs_l, "series_wins": t1_wins},
            "team2": {**t2_meta, "regular_season_wins": t2_rs_w,
                      "regular_season_losses": t2_rs_l, "series_wins": t2_wins},
            "start_date": start.isoformat() if start else None,
            "last_date": _last.isoformat() if _last else None,
            "n_games": n,
            "status": status,
            "winner_id": winner,
            "conference": (t1_meta.get("conference") if
                           t1_meta.get("conference") == t2_meta.get("conference")
                           else "Mixed"),
            "is_predetermined": False,
        })

    return sorted(series_list, key=lambda s: s["start_date"] or "")


def _series_score(db: Session, t1: int, t2: int, season: str) -> tuple[int, int]:
    """Wins for t1 and t2 in this playoff matchup so far.

    Reads from t1's perspective (one row per game) — does not depend on
    is_home, which is unreliable in some ingested rows.
    """
    rows = db.execute(
        select(Game.won).where(
            Game.season == season,
            Game.is_playoff.is_(True),
            Game.team_id == t1,
            Game.opponent_id == t2,
        )
    ).scalars().all()
    t1_wins = sum(1 for won in rows if won)
    t2_wins = len(rows) - t1_wins
    return t1_wins, t2_wins


def _empty_rounds() -> dict:
    return {
        "first_round_west": [], "first_round_east": [],
        "conf_semis_west": [], "conf_semis_east": [],
        "conf_finals_west": [], "conf_finals_east": [],
        "finals": [],
    }


def _group_into_rounds(series: list[dict]) -> dict[str, list[dict]]:
    """Cluster series into rounds by start-date gap.

    NBA playoffs: 8 first-round series start within ~3 days of each other,
    then ~10-day gap, then 4 conf-semis series, etc. We pick gaps in the
    sorted start dates to identify round boundaries — robust across seasons.
    """
    if not series:
        return _empty_rounds()

    # Sort by start date (already sorted, but defensive).
    s_sorted = sorted(series, key=lambda x: x["start_date"] or "")
    dates = [x["start_date"] or "" for x in s_sorted]

    # Compute gaps in days between consecutive series starts.
    bucket_idx = [0]
    for i in range(1, len(s_sorted)):
        prev = date.fromisoformat(dates[i - 1])
        cur = date.fromisoformat(dates[i])
        if (cur - prev).days >= 5:        # 5+ day gap = new round
            bucket_idx.append(i)
    bucket_idx.append(len(s_sorted))

    buckets = []
    for i in range(len(bucket_idx) - 1):
        buckets.append(s_sorted[bucket_idx[i]:bucket_idx[i + 1]])

    # Map up to 4 buckets into the 4 NBA round names.
    round_names = ["first_round", "conf_semis", "conf_finals", "finals"]
    rounds = _empty_rounds()
    for i, bucket in enumerate(buckets[:4]):
        rname = round_names[i]
        if rname == "finals":
            rounds["finals"].extend(bucket)
            continue
        for s in bucket:
            conf = s.get("conference", "Unknown").lower()
            key = f"{rname}_{conf}" if conf in ("west", "east") else f"{rname}_west"
            rounds.setdefault(key, []).append(s)
    return rounds


def _team_playoff_averages(db: Session, team_id: int, season: str) -> dict:
    """Per-game averages over this team's playoff games this season."""
    if not season:
        return {"games": 0, "pts": 0, "reb": 0, "ast": 0, "pts_allowed": 0}

    games = db.execute(
        select(Game).where(
            Game.team_id == team_id,
            Game.season == season,
            Game.is_playoff.is_(True),
        )
    ).scalars().all()
    n = len(games)
    if n == 0:
        return {"games": 0, "pts": 0, "reb": 0, "ast": 0, "pts_allowed": 0}

    pts_avg = sum(g.pts for g in games) / n
    pts_allowed_avg = sum(g.pts_allowed for g in games) / n

    # REB/AST live only on PlayerGame rows. Sum per game then average.
    pg_rows = db.execute(
        select(PlayerGame.game_id,
               func.sum(PlayerGame.reb).label("reb"),
               func.sum(PlayerGame.ast).label("ast"))
        .where(
            PlayerGame.team_id == team_id,
            PlayerGame.game_id.in_([g.game_id for g in games]),
        )
        .group_by(PlayerGame.game_id)
    ).all()
    reb_avg = (sum(r[1] or 0 for r in pg_rows) / len(pg_rows)) if pg_rows else 0
    ast_avg = (sum(r[2] or 0 for r in pg_rows) / len(pg_rows)) if pg_rows else 0

    return {
        "games": n,
        "pts": round(pts_avg, 1),
        "reb": round(reb_avg, 1),
        "ast": round(ast_avg, 1),
        "pts_allowed": round(pts_allowed_avg, 1),
    }


def _team_feature_snapshot(db: Session, team_id: int) -> dict:
    """Latest TeamGameFeature row for this team — exposes the model's view of them."""
    tf = db.execute(
        select(TeamGameFeature).where(TeamGameFeature.team_id == team_id)
        .order_by(TeamGameFeature.game_date.desc()).limit(1)
    ).scalar_one_or_none()
    if tf is None:
        return {}
    return {
        "off_rating_roll5": _round(tf.off_rating_roll5),
        "def_rating_roll5": _round(tf.def_rating_roll5),
        "net_rating_roll10": _round(tf.net_rating_roll10),
        "win_pct_roll10": _round(tf.win_pct_roll10, 3),
        "pace_roll5": _round(tf.pace_roll5),
        "elo": _round(tf.elo),
    }


def _round(x, n: int = 2):
    return round(x, n) if isinstance(x, (int, float)) else None


def _available_models() -> dict:
    """Load only the models we actually want to display."""
    available = set(list_available_models())
    return {n: get_loaded_model(n) for n in DASHBOARD_MODEL_NAMES if n in available}


_ROSTER_LOOKBACK_DAYS = 30


def _roster_near_date(db: Session, team_id: int, as_of: date,
                      lookback_days: int = _ROSTER_LOOKBACK_DAYS) -> list[int]:
    """Players who played for this team in the lookback window BEFORE as_of.
    Works for both future (today's roster) and historical (roster at the
    time of a past deciding game) predictions."""
    cutoff = as_of - timedelta(days=lookback_days)
    return list(db.execute(
        select(PlayerGame.player_id).distinct().where(
            PlayerGame.team_id == team_id,
            PlayerGame.game_date >= cutoff,
            PlayerGame.game_date < as_of,
        )
    ).scalars().all())


def _safe_predict(db, model, home_id, away_id, game_date) -> float | None:
    """Like predict_matchup but falls back to lookback-window roster when no
    PlayerAvailability snapshot exists for game_date."""
    home_active = active_roster_as_of(db, home_id, game_date) or _roster_near_date(db, home_id, game_date)
    away_active = active_roster_as_of(db, away_id, game_date) or _roster_near_date(db, away_id, game_date)
    feats = build_matchup_features(
        db, home_id, away_id, game_date,
        feature_version=model.feature_version,
        home_active=home_active, away_active=away_active,
    )
    if feats is None:
        return None
    return float(model.predict_proba(feats))
