"""Roster-aware matchup feature builder.

Injury rule: a player marked 'out' (or any non-active status) for the target
game is excluded from `player_ids` before aggregation. Their entire history
vs the opponent therefore contributes zero — the prediction sees the team
without that player.

Recent-minutes weighting ensures stars dominate the projection: we weight each
available player's avg-vs-opponent stats by their share of a 48-minute game
over the last `RECENT_MINUTES_GAMES` games. Removing a 36-min/g star drops the
projection by ~75%; removing an 8-min/g bench player barely moves it.
"""
from __future__ import annotations
from datetime import date
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from nba_ml.db.models import (
    BettingOdds, Game, PlayerAvailability, PlayerGame, TeamGameFeature,
)

ACTIVE_STATUSES: tuple[str, ...] = ("active",)
RECENT_MINUTES_GAMES = 10
HOME_ROAD_LOOKBACK = 20
H2H_LOOKBACK = 10

FEATURE_COLUMNS: list[str] = [
    # Team rolling form
    "home_off_roll5", "home_def_roll5", "home_net_roll10",
    "home_win_pct_roll10", "home_is_b2b",
    "away_off_roll5", "away_def_roll5", "away_net_roll10",
    "away_win_pct_roll10", "away_is_b2b",
    # Differentials — dominant signal pattern in this dataset
    "net_rating_diff", "off_rating_diff", "def_rating_diff",
    "win_pct_diff", "pvt_proj_pts_diff", "rest_advantage",
    # Elo (post-game snapshot of each team's strength + closed-form prediction)
    "elo_diff", "elo_win_prob",
    # Home-court advantage signal (per-team, not a constant)
    "home_home_win_pct_l20", "away_road_win_pct_l20",
    # Head-to-head between these two teams
    "h2h_home_win_pct",
    # Player-vs-opponent
    "home_pvt_proj_pts", "away_pvt_proj_pts",
]

# Vegas-augmented variant — same as above plus the Vegas signal.
# Used by the Vegas-augmented model trained only on games that have odds.
FEATURE_COLUMNS_VEGAS: list[str] = FEATURE_COLUMNS + ["vegas_home_win_prob"]

# Mirrored from features/builders.py — used to compute the closed-form
# Elo win probability from team rating diff at predict time.
_ELO_HCA = 100.0
_ELO_BASE = 1500.0


# ---- Roster + availability ----

def active_players_for_game(db: Session, game_id: str, team_id: int) -> list[int]:
    """Players reported active for `team_id` in `game_id`.

    Falls back to "who actually played" (PlayerGame rows with minutes > 0)
    when no availability snapshot exists — common during training on history.
    """
    rows = db.execute(
        select(PlayerAvailability.player_id).where(
            PlayerAvailability.game_id == game_id,
            PlayerAvailability.team_id == team_id,
            PlayerAvailability.status.in_(ACTIVE_STATUSES),
        )
    ).scalars().all()
    if rows:
        return list(rows)
    rows = db.execute(
        select(PlayerGame.player_id).where(
            PlayerGame.game_id == game_id,
            PlayerGame.team_id == team_id,
            PlayerGame.minutes > 0,
        )
    ).scalars().all()
    return list(rows)


def active_roster_as_of(db: Session, team_id: int, game_date: date) -> list[int]:
    """Active roster for the upcoming game on `game_date` for `team_id`.
    Used at inference when the caller has only (team_id, date), not game_id."""
    return list(db.execute(
        select(PlayerAvailability.player_id).where(
            PlayerAvailability.team_id == team_id,
            PlayerAvailability.game_date == game_date,
            PlayerAvailability.status.in_(ACTIVE_STATUSES),
        )
    ).scalars().all())


# ---- Player-vs-team aggregation ----

def _recent_minutes(
    db: Session, player_ids: list[int], as_of: date, n: int = RECENT_MINUTES_GAMES
) -> dict[int, float]:
    if not player_ids:
        return {}
    sub = (
        select(
            PlayerGame.player_id,
            PlayerGame.minutes,
            func.row_number().over(
                partition_by=PlayerGame.player_id,
                order_by=PlayerGame.game_date.desc(),
            ).label("rn"),
        )
        .where(
            PlayerGame.player_id.in_(player_ids),
            PlayerGame.game_date < as_of,
        )
        .subquery()
    )
    rows = db.execute(
        select(sub.c.player_id, func.avg(sub.c.minutes))
        .where(sub.c.rn <= n)
        .group_by(sub.c.player_id)
    ).all()
    return {pid: float(m or 0.0) for pid, m in rows}


def aggregate_pvt(
    db: Session, player_ids: list[int], opponent_id: int, as_of: date,
) -> dict[str, float]:
    """Roster-weighted projected contribution vs `opponent_id`."""
    if not player_ids:
        return {"proj_pts": 0.0, "proj_pm": 0.0, "weight_total": 0.0}

    rows = db.execute(
        select(
            PlayerGame.player_id,
            func.avg(PlayerGame.pts).label("avg_pts"),
            func.avg(PlayerGame.plus_minus).label("avg_pm"),
        )
        .where(
            PlayerGame.player_id.in_(player_ids),
            PlayerGame.opponent_id == opponent_id,
            PlayerGame.game_date < as_of,
        )
        .group_by(PlayerGame.player_id)
    ).all()

    minutes = _recent_minutes(db, player_ids, as_of)
    proj_pts = proj_pm = total_w = 0.0
    for pid, avg_pts, avg_pm in rows:
        w = minutes.get(pid, 0.0) / 48.0
        proj_pts += w * float(avg_pts or 0.0)
        proj_pm += w * float(avg_pm or 0.0)
        total_w += w
    return {"proj_pts": proj_pts, "proj_pm": proj_pm, "weight_total": total_w}


# ---- Team feature lookup ----

def vegas_home_win_prob(
    db: Session, home_team_id: int, away_team_id: int, game_date: date,
) -> float:
    """Latest PRE-GAME fair-prob from the betting_odds table for this matchup.
    Returns 0.5 (neutral) when no pre-game odds exist — post-game/in-game odds
    (is_pregame=False) are excluded because they encode the outcome."""
    row = db.execute(
        select(BettingOdds.vegas_home_win_prob).where(
            BettingOdds.game_date == game_date,
            BettingOdds.home_team_id == home_team_id,
            BettingOdds.away_team_id == away_team_id,
            BettingOdds.is_pregame.is_(True),
        ).order_by(BettingOdds.fetched_at.desc()).limit(1)
    ).scalar_one_or_none()
    return float(row) if row is not None else 0.5


def latest_team_feature(
    db: Session, team_id: int, as_of: date, feature_version: str
) -> TeamGameFeature | None:
    return db.execute(
        select(TeamGameFeature)
        .where(
            TeamGameFeature.team_id == team_id,
            TeamGameFeature.game_date < as_of,
            TeamGameFeature.feature_version == feature_version,
        )
        .order_by(TeamGameFeature.game_date.desc())
        .limit(1)
    ).scalar_one_or_none()


# ---- Home/road win % (home-court signal) ----

def home_or_road_win_pct(
    db: Session, team_id: int, as_of: date, *, at_home: bool,
    n: int = HOME_ROAD_LOOKBACK,
) -> float:
    """Win % over `team_id`'s last `n` games at home (at_home=True) or on the road.
    Defaults to 0.5 when the team has no recent games."""
    rows = db.execute(
        select(Game.won).where(
            Game.team_id == team_id,
            Game.game_date < as_of,
            Game.is_home.is_(at_home),
        ).order_by(Game.game_date.desc()).limit(n)
    ).scalars().all()
    if not rows:
        return 0.5
    return sum(rows) / len(rows)


# ---- Head-to-head ----

def head_to_head(
    db: Session, home_id: int, away_id: int, as_of: date, n: int = H2H_LOOKBACK,
) -> tuple[float, int]:
    """Recent meetings between these two teams.

    We query rows where is_home is True (each game is stored twice — once per
    team's perspective; this dedupes). Returns (home_team_win_pct, n_meetings)."""
    rows = db.execute(
        select(Game.team_id, Game.won).where(
            Game.game_date < as_of,
            Game.is_home.is_(True),
            (
                ((Game.team_id == home_id) & (Game.opponent_id == away_id))
                | ((Game.team_id == away_id) & (Game.opponent_id == home_id))
            ),
        ).order_by(Game.game_date.desc()).limit(n)
    ).all()
    if not rows:
        return 0.5, 0
    home_wins = 0
    for tid, won in rows:
        # tid is the team that was home in that historical row.
        # A "home_id win" in the future-matchup sense means home_id won
        # that meeting, regardless of which side they were on.
        if (tid == home_id and won) or (tid == away_id and not won):
            home_wins += 1
    return home_wins / len(rows), len(rows)


# ---- Top-level feature builder ----

def build_matchup_features(
    db: Session,
    home_team_id: int,
    away_team_id: int,
    as_of: date,
    feature_version: str,
    home_active: list[int],
    away_active: list[int],
) -> dict[str, float] | None:
    """Build the feature dict for one matchup. Returns None if either team
    lacks an active roster or prior team features."""
    if not home_active or not away_active:
        return None

    home_tf = latest_team_feature(db, home_team_id, as_of, feature_version)
    away_tf = latest_team_feature(db, away_team_id, as_of, feature_version)
    if home_tf is None or away_tf is None:
        return None

    home_pvt = aggregate_pvt(db, home_active, away_team_id, as_of)
    away_pvt = aggregate_pvt(db, away_active, home_team_id, as_of)

    h2h_pct, _ = head_to_head(db, home_team_id, away_team_id, as_of)

    home_off = home_tf.off_rating_roll5 or 0.0
    home_def = home_tf.def_rating_roll5 or 0.0
    home_net = home_tf.net_rating_roll10 or 0.0
    home_wp = home_tf.win_pct_roll10 or 0.5
    away_off = away_tf.off_rating_roll5 or 0.0
    away_def = away_tf.def_rating_roll5 or 0.0
    away_net = away_tf.net_rating_roll10 or 0.0
    away_wp = away_tf.win_pct_roll10 or 0.5
    home_is_b2b = float(bool(home_tf.is_b2b))
    away_is_b2b = float(bool(away_tf.is_b2b))
    home_pvt_pts = home_pvt["proj_pts"]
    away_pvt_pts = away_pvt["proj_pts"]
    home_elo = home_tf.elo if home_tf.elo is not None else _ELO_BASE
    away_elo = away_tf.elo if away_tf.elo is not None else _ELO_BASE
    elo_diff = home_elo - away_elo
    elo_p = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - _ELO_HCA) / 400.0))

    return {
        "home_off_roll5": home_off,
        "home_def_roll5": home_def,
        "home_net_roll10": home_net,
        "home_win_pct_roll10": home_wp,
        "home_is_b2b": home_is_b2b,
        "away_off_roll5": away_off,
        "away_def_roll5": away_def,
        "away_net_roll10": away_net,
        "away_win_pct_roll10": away_wp,
        "away_is_b2b": away_is_b2b,
        "net_rating_diff": home_net - away_net,
        "off_rating_diff": home_off - away_off,
        # def_rating: lower is better, so a positive diff means home defends WORSE.
        "def_rating_diff": home_def - away_def,
        "win_pct_diff": home_wp - away_wp,
        "pvt_proj_pts_diff": home_pvt_pts - away_pvt_pts,
        # +1 home is fresher, -1 home is on a b2b while away rested, 0 same.
        "rest_advantage": away_is_b2b - home_is_b2b,
        "elo_diff": elo_diff,
        "elo_win_prob": elo_p,
        "home_home_win_pct_l20": home_or_road_win_pct(
            db, home_team_id, as_of, at_home=True,
        ),
        "away_road_win_pct_l20": home_or_road_win_pct(
            db, away_team_id, as_of, at_home=False,
        ),
        "h2h_home_win_pct": h2h_pct,
        "home_pvt_proj_pts": home_pvt_pts,
        "away_pvt_proj_pts": away_pvt_pts,
        # Always present in the dict; base models ignore it via their own
        # feature_columns. Vegas-augmented models pick it up.
        "vegas_home_win_prob": vegas_home_win_prob(
            db, home_team_id, away_team_id, as_of,
        ),
    }


def has_vegas_odds(
    db: Session, home_team_id: int, away_team_id: int, game_date: date,
) -> bool:
    """True when a PRE-GAME BettingOdds row exists for this matchup."""
    row = db.execute(
        select(BettingOdds.id).where(
            BettingOdds.game_date == game_date,
            BettingOdds.home_team_id == home_team_id,
            BettingOdds.away_team_id == away_team_id,
            BettingOdds.is_pregame.is_(True),
        ).limit(1)
    ).scalar_one_or_none()
    return row is not None
