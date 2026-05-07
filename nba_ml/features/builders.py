"""Team-level rolling features.

Leakage rule: every rolling metric is computed by sorting per-team chronologically,
shifting by 1, then rolling. The shift moves each team's history forward by one
game so a row's window covers games strictly prior to that row's date.
"""
from __future__ import annotations
import pandas as pd

POSS_COEF = 0.44

# Elo parameters. Standard 538-style values for NBA.
ELO_K = 20.0           # how aggressively a single result moves the rating
ELO_HCA = 100.0        # home-court advantage in Elo points
ELO_BASE = 1500.0      # rating new teams start at
ELO_SEASON_REGRESS = 0.25  # at season boundary, regress 25% toward 1505
ELO_SEASON_MEAN = 1505.0


def compute_post_game_elos(games: pd.DataFrame) -> pd.DataFrame:
    """For each (game_id, team_id) emit the team's Elo AFTER that game.

    Use the post-game Elo as the team's "current strength" snapshot — for
    predicting their NEXT game, this is exactly what you want (no staleness).

    Each season boundary regresses each team 25% toward 1505 (538 convention).
    """
    home_only = games[games["is_home"]].sort_values(["game_date", "game_id"]).copy()

    elos: dict[int, float] = {}
    last_season: dict[int, str] = {}
    out: list[dict] = []

    for _, r in home_only.iterrows():
        season = r["season"]
        home_id = int(r["team_id"])
        away_id = int(r["opponent_id"])

        for tid in (home_id, away_id):
            if tid in last_season and last_season[tid] != season:
                elos[tid] = (elos[tid] * (1 - ELO_SEASON_REGRESS)
                             + ELO_SEASON_MEAN * ELO_SEASON_REGRESS)
            last_season[tid] = season

        home_elo = elos.get(home_id, ELO_BASE)
        away_elo = elos.get(away_id, ELO_BASE)

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - ELO_HCA) / 400.0))
        actual_home = 1.0 if bool(r["won"]) else 0.0
        delta = ELO_K * (actual_home - expected_home)

        elos[home_id] = home_elo + delta
        elos[away_id] = away_elo - delta

        out.append({"game_id": r["game_id"], "team_id": home_id, "elo": elos[home_id]})
        out.append({"game_id": r["game_id"], "team_id": away_id, "elo": elos[away_id]})

    return pd.DataFrame(out)


def _possessions(df: pd.DataFrame, side: str = "team") -> pd.Series:
    if side == "team":
        return df["fga"] + POSS_COEF * df["fta"] - df["oreb"] + df["tov"]
    return df["opp_fga"] + POSS_COEF * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"]


def add_rolling_team_features(games: pd.DataFrame) -> pd.DataFrame:
    """Inputs `games` long-format (one row per team-perspective per game).

    Required columns: game_id, game_date, team_id, won, plus pts/pts_allowed and
    the four-factor inputs (fga, fta, tov, oreb, opp_*).
    """
    g = games.sort_values(["team_id", "game_date", "game_id"]).copy()

    poss = _possessions(g, "team")
    opp_poss = _possessions(g, "opp")
    avg_poss = (poss + opp_poss) / 2

    g["_off"] = 100 * g["pts"] / avg_poss
    g["_def"] = 100 * g["pts_allowed"] / avg_poss
    g["_pace"] = avg_poss
    g["_win"] = g["won"].astype(float)

    grp = g.groupby("team_id", sort=False, group_keys=False)

    def _roll(col: str, w: int) -> pd.Series:
        return grp[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    g["off_rating_roll5"] = _roll("_off", 5)
    g["def_rating_roll5"] = _roll("_def", 5)
    g["pace_roll5"] = _roll("_pace", 5)
    g["net_rating_roll10"] = _roll("_off", 10) - _roll("_def", 10)
    g["win_pct_roll10"] = _roll("_win", 10)

    prev_date = grp["game_date"].shift(1)
    g["rest_days"] = (g["game_date"] - prev_date).dt.days
    g["is_b2b"] = g["rest_days"] == 1

    return g.drop(columns=[c for c in g.columns if c.startswith("_")])
