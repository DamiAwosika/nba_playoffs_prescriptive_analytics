"""nba_api -> our schema.

One source endpoint, called twice (T then P), and we shape the rows to match
the dicts upstream of `upsert_*`. Date range is pushed to the API so we don't
download a whole season just to ingest a week.

Injury data: nba_api has no clean injury endpoint. `fetch_availability`
returns [] and we rely on the training-time fallback in `features.matchup`,
which infers actives from "who actually played".
"""
from __future__ import annotations
import time
from datetime import date

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.static import teams as static_teams

API_SLEEP_SECONDS = 0.7
NBA_TIMEOUT = 60
NBA_MAX_RETRIES = 3

# stats.nba.com filters anything that doesn't look like a real browser hitting
# nba.com. Default nba_api headers occasionally get blocked, returning an empty
# body that json.loads can't parse. These mirror what nba.com itself sends.
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


def season_for_date(d: date) -> str:
    """NBA season string for a given date.

    Season N starts in October of year N. Jan-Jun dates belong to season N-1.
    Jul-Sep is the off-season; we still bucket those into the prior season.
    """
    start_year = d.year if d.month >= 10 else d.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _is_home(matchup: str) -> bool:
    return " vs. " in matchup


def _league_log(season: str, start: date, end: date, side: str) -> pd.DataFrame:
    """One call to LeagueGameLog with browser-like headers + retry/backoff.
    side='T' for team-level rows, 'P' for player-level rows."""
    last_err: Exception | None = None
    for attempt in range(NBA_MAX_RETRIES):
        try:
            return LeagueGameLog(
                season=season,
                player_or_team_abbreviation=side,
                date_from_nullable=start.strftime("%m/%d/%Y"),
                date_to_nullable=end.strftime("%m/%d/%Y"),
                headers=NBA_HEADERS,
                timeout=NBA_TIMEOUT,
            ).get_data_frames()[0]
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"stats.nba.com call failed after {NBA_MAX_RETRIES} retries "
        f"(season={season}, side={side}, {start}..{end}). "
        "Common causes: rate limit (wait a minute and retry), your IP/network "
        "is blocked by stats.nba.com (try a VPN), or the upstream is down. "
        f"Last error: {last_err!r}"
    ) from last_err


def _team_log(season: str, start: date, end: date) -> pd.DataFrame:
    return _league_log(season, start, end, "T")


def _player_log(season: str, start: date, end: date) -> pd.DataFrame:
    return _league_log(season, start, end, "P")


def fetch_teams() -> list[dict]:
    """Static team metadata. No network call."""
    return [
        {"team_id": int(t["id"]),
         "abbreviation": t["abbreviation"],
         "full_name": t["full_name"]}
        for t in static_teams.get_teams()
    ]


def fetch_games(start: date, end: date) -> list[dict]:
    season = season_for_date(end)
    df = _team_log(season, start, end)
    rows: list[dict] = []
    for _, pair in df.groupby("GAME_ID"):
        if len(pair) != 2:
            continue
        a, b = pair.iloc[0], pair.iloc[1]
        for me, opp in ((a, b), (b, a)):
            rows.append({
                "game_id": str(me["GAME_ID"]),
                "game_date": pd.to_datetime(me["GAME_DATE"]).date(),
                "season": season,
                "team_id": int(me["TEAM_ID"]),
                "opponent_id": int(opp["TEAM_ID"]),
                "is_home": _is_home(me["MATCHUP"]),
                "pts": int(me["PTS"]),    "pts_allowed": int(opp["PTS"]),
                "fga": int(me["FGA"]),    "opp_fga": int(opp["FGA"]),
                "fta": int(me["FTA"]),    "opp_fta": int(opp["FTA"]),
                "tov": int(me["TOV"]),    "opp_tov": int(opp["TOV"]),
                "oreb": int(me["OREB"]),  "opp_oreb": int(opp["OREB"]),
                "won": me["WL"] == "W",
            })
    return rows


def fetch_player_games_and_players(
    start: date, end: date
) -> tuple[list[dict], list[dict]]:
    """Returns (player_game_rows, player_rows) in one pass.

    Building the player rows here avoids a second pull of the same data — the
    player log already contains every PLAYER_ID + PLAYER_NAME we need.
    """
    season = season_for_date(end)
    team_df = _team_log(season, start, end)

    opp_map: dict[tuple, int] = {}
    home_map: dict[tuple, bool] = {}
    for _, pair in team_df.groupby("GAME_ID"):
        if len(pair) != 2:
            continue
        a, b = pair.iloc[0], pair.iloc[1]
        opp_map[(a["GAME_ID"], a["TEAM_ID"])] = int(b["TEAM_ID"])
        opp_map[(b["GAME_ID"], b["TEAM_ID"])] = int(a["TEAM_ID"])
        home_map[(a["GAME_ID"], a["TEAM_ID"])] = _is_home(a["MATCHUP"])
        home_map[(b["GAME_ID"], b["TEAM_ID"])] = _is_home(b["MATCHUP"])

    time.sleep(API_SLEEP_SECONDS)
    p_df = _player_log(season, start, end)

    pg_rows: list[dict] = []
    for _, p in p_df.iterrows():
        key = (p["GAME_ID"], p["TEAM_ID"])
        if key not in opp_map:
            continue
        pg_rows.append({
            "game_id": str(p["GAME_ID"]),
            "game_date": pd.to_datetime(p["GAME_DATE"]).date(),
            "player_id": int(p["PLAYER_ID"]),
            "team_id": int(p["TEAM_ID"]),
            "opponent_id": opp_map[key],
            "is_home": home_map[key],
            "minutes": float(p["MIN"] or 0),
            "pts": int(p["PTS"] or 0),
            "reb": int(p["REB"] or 0),
            "ast": int(p["AST"] or 0),
            "plus_minus": float(p["PLUS_MINUS"] or 0),
        })

    seen = p_df.drop_duplicates("PLAYER_ID")
    player_rows = [
        {"player_id": int(r["PLAYER_ID"]), "full_name": str(r["PLAYER_NAME"])}
        for _, r in seen.iterrows()
    ]
    return pg_rows, player_rows


def fetch_availability(start: date, end: date) -> list[dict]:
    """No clean nba_api endpoint. The training fallback in matchup.py infers
    actives from "who actually played", which is sufficient for backtests.
    Wire a real source (NBA injury report PDF, third-party feed) here for live
    inference."""
    return []
