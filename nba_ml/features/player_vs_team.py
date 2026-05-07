"""Player-vs-team historical features.

For every (player, opponent) pair, we compute rolling means of the player's
performance against that opponent. Same leakage guard as team features:
sort by date, then shift(1) before the rolling op so the window covers prior
meetings only.

These features are computed offline for analysis; at inference, we don't need
the per-row history — we aggregate via `features.matchup.aggregate_pvt`,
which queries the raw `PlayerGame` rows and applies the same `< as_of` filter.
"""
from __future__ import annotations
import pandas as pd

PVT_WINDOW = 10


def add_player_vs_team_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """Input columns required: player_id, opponent_id, game_date, game_id,
    pts, plus_minus, minutes."""
    g = player_games.sort_values(
        ["player_id", "opponent_id", "game_date", "game_id"]
    ).copy()
    grp = g.groupby(["player_id", "opponent_id"], sort=False, group_keys=False)

    def _roll(col: str, w: int = PVT_WINDOW) -> pd.Series:
        return grp[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    g["pts_vs_opp_avg"] = _roll("pts")
    g["pm_vs_opp_avg"] = _roll("plus_minus")
    g["min_vs_opp_avg"] = _roll("minutes")
    g["meetings_vs_opp"] = grp.cumcount()
    return g
