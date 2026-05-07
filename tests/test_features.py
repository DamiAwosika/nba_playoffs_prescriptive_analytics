"""Leakage guard: a row's rolling feature must not change when that same
row's outcome is mutated. If it does, future data is leaking into the past."""
from __future__ import annotations
import pandas as pd

from nba_ml.features.builders import add_rolling_team_features


def _synthetic_games(n_games_per_team: int = 20) -> pd.DataFrame:
    rows = []
    for team_id in (1, 2):
        for i in range(n_games_per_team):
            rows.append({
                "game_id": f"G{team_id}-{i}",
                "game_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "team_id": team_id,
                "opponent_id": 3 - team_id,
                "pts": 100 + i,
                "pts_allowed": 95 + i,
                "fga": 85, "fta": 20, "tov": 13, "oreb": 10,
                "opp_fga": 86, "opp_fta": 18, "opp_tov": 12, "opp_oreb": 9,
                "won": (100 + i) > (95 + i),
            })
    return pd.DataFrame(rows)


def test_rolling_does_not_see_current_row():
    df = _synthetic_games()
    base = add_rolling_team_features(df.copy()).set_index(["team_id", "game_id"])

    mutated = df.copy()
    target = mutated.index[10]
    mutated.loc[target, ["pts", "pts_allowed", "won"]] = [999, 0, True]
    mutated_feats = add_rolling_team_features(mutated).set_index(["team_id", "game_id"])

    cols = ["off_rating_roll5", "def_rating_roll5",
            "net_rating_roll10", "win_pct_roll10"]
    target_key = (df.loc[target, "team_id"], df.loc[target, "game_id"])
    pd.testing.assert_series_equal(
        base.loc[target_key, cols], mutated_feats.loc[target_key, cols],
    )


def test_first_game_features_are_nan():
    df = _synthetic_games()
    feats = add_rolling_team_features(df).sort_values(["team_id", "game_date"])
    first = feats.groupby("team_id").head(1)
    assert first["off_rating_roll5"].isna().all()
    assert first["rest_days"].isna().all()
