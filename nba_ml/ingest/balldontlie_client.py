"""balldontlie.io -> our schema.

Requires NBA_BALLDONTLIE_API_KEY in env or .env file (free tier works).

We use (all on GOAT tier):
  /v1/teams                — team metadata
  /v1/box_scores/{date}    — completed games + per-player box scores (one call/date)
  /v1/games                — schedule (start_date/end_date), used for upcoming
  /v1/player_injuries      — current injury statuses (powers PlayerAvailability)

Pagination is cursor-based: each response carries meta.next_cursor.
"""
from __future__ import annotations
import time
from datetime import date, timedelta
from typing import Any, Iterator

import requests

from nba_ml.config import settings

BASE_URL = "https://api.balldontlie.io/v1"
BASE_URL_V2 = "https://api.balldontlie.io/v2"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.4


class BdlError(RuntimeError):
    pass


class BdlClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.balldontlie_api_key
        if not self.api_key:
            raise BdlError(
                "balldontlie API key not configured. Set NBA_BALLDONTLIE_API_KEY "
                "in your environment or in a .env file at the project root."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

    def _get(self, path: str, params: dict | None = None, *, base: str = BASE_URL) -> dict:
        url = f"{base}{path}"
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if r.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise BdlError(
            f"GET {url} failed after {MAX_RETRIES} retries: {last_err!r}"
        ) from last_err

    def _paginate(self, path: str, params: dict | None = None) -> Iterator[dict]:
        params = dict(params or {})
        params.setdefault("per_page", 100)
        while True:
            payload = self._get(path, params)
            for row in payload.get("data", []):
                yield row
            cursor = payload.get("meta", {}).get("next_cursor")
            if not cursor:
                return
            params["cursor"] = cursor
            time.sleep(SLEEP_BETWEEN)

    # ---- Teams ----

    def fetch_teams(self) -> list[dict]:
        # balldontlie /teams includes ~15 defunct franchises. Real current
        # teams have conference set to "East" or "West".
        return [
            {"team_id": int(t["id"]),
             "abbreviation": t["abbreviation"],
             "full_name": t["full_name"]}
            for t in self._paginate("/teams")
            if t.get("conference") in ("East", "West")
        ]

    # ---- Completed games (one date at a time, via box_scores) ----

    def _box_scores_for_date(self, d: date) -> list[dict]:
        return self._get("/box_scores", {"date": d.isoformat()}).get("data", [])

    def fetch_games_and_stats(
        self, start: date, end: date,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Walk every date in range, pull box scores, derive
        (game_rows, player_game_rows, unique_player_rows)."""
        game_rows: list[dict] = []
        pg_rows: list[dict] = []
        players_seen: dict[int, dict] = {}

        d = start
        while d <= end:
            for box in self._box_scores_for_date(d):
                if (box.get("status") or "").strip() != "Final":
                    continue
                game_rows.extend(_box_to_game_rows(box))
                pg, pl = _box_to_player_rows(box)
                pg_rows.extend(pg)
                for p in pl:
                    players_seen[p["player_id"]] = p
            time.sleep(SLEEP_BETWEEN)
            d += timedelta(days=1)
        return game_rows, pg_rows, list(players_seen.values())

    # ---- Upcoming games ----

    def fetch_upcoming_games(
        self, start: date, end: date, postseason: bool | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        if postseason is not None:
            params["postseason"] = "true" if postseason else "false"

        upcoming = []
        for g in self._paginate("/games", params):
            status = (g.get("status") or "").strip()
            if status == "Final":
                continue
            upcoming.append({
                "game_id": str(g["id"]),
                "date": _parse_date(g["date"]),
                "home_team_id": int(g["home_team"]["id"]),
                "away_team_id": int(g["visitor_team"]["id"]),
                "home_abbr": g["home_team"]["abbreviation"],
                "away_abbr": g["visitor_team"]["abbreviation"],
                "postseason": bool(g.get("postseason")),
                "status": status,
            })
        return upcoming

    # ---- Betting odds (GOAT tier) ----

    def fetch_odds(self, start: date, end: date) -> dict[str, float]:
        """Pre-game moneyline odds aggregated per game.

        balldontlie's /v2/odds returns one row per (game, vendor). We compute
        the no-vig fair home-win probability from each vendor's moneyline,
        then take the median across vendors for stability. Returns
        {game_id_str: vegas_home_win_prob}.

        Caller is responsible for joining game_id back to (game_date, home, away).
        """
        per_game: dict[str, list[float]] = {}
        d = start
        while d <= end:
            try:
                payload = self._get(
                    "/odds", {"dates[]": [d.isoformat()]}, base=BASE_URL_V2,
                )
            except BdlError as e:
                print(f"  odds {d}: {e!r}")
                payload = {}
            for item in payload.get("data", []):
                gid = item.get("game_id")
                home_ml = item.get("moneyline_home_odds")
                away_ml = item.get("moneyline_away_odds")
                if gid is None or home_ml is None or away_ml is None:
                    continue
                hi = _ml_to_implied(home_ml)
                ai = _ml_to_implied(away_ml)
                total = hi + ai
                if total <= 0:
                    continue
                per_game.setdefault(str(gid), []).append(hi / total)
            time.sleep(SLEEP_BETWEEN)
            d += timedelta(days=1)
        return {gid: _median(probs) for gid, probs in per_game.items()}

    # ---- Player props (GOAT tier) ----
    #
    # Endpoint: GET /v2/odds/player_props?game_id=<int>
    # Required param is game_id, so we have to query per game (not per date).
    # Response items shape:
    #   {id, game_id, player_id, vendor, prop_type, line_value: str,
    #    market: { type: "over_under", over_odds, under_odds, ... }
    #             or { type: "milestone", ... }, updated_at}

    # balldontlie prop_type -> our canonical short names. We only train on
    # the over/under markets for total points/rebounds/assists; everything
    # else (1q splits, milestones, combos like points_rebounds_assists) is
    # ignored at the feature layer for now.
    _PROP_MARKET_MAP = {
        "points": "pts", "rebounds": "reb", "assists": "ast",
        "steals": "stl", "blocks": "blk", "turnovers": "tov",
    }

    def fetch_player_props(
        self, start: date, end: date, game_ids: list[str] | None = None,
    ) -> list[dict]:
        """Pull PTS / REB / AST player props for games in [start, end].

        If `game_ids` is provided we use it directly; otherwise we resolve
        the upcoming games for the date range via /v1/games. Returns rows
        shaped for upsert_player_props:
            {game_id, game_date, player_id, market, line, over_odds, under_odds}
        Vendors are medianed per (game, player, market).

        team_id is filled in by the caller (the prop row itself doesn't
        carry team information).
        """
        from collections import defaultdict

        if game_ids is None:
            upcoming = self.fetch_upcoming_games(start, end, postseason=None)
            games_meta = {g["game_id"]: g["date"] for g in upcoming}
        else:
            # Caller passed explicit ids — assume game_date == start for archive use.
            games_meta = {gid: start for gid in game_ids}

        if not games_meta:
            return []

        # {(game_id, player_id, market): [(line, over_odds, under_odds, gdate), ...]}
        bucket: dict[tuple, list] = defaultdict(list)

        for gid, gdate in games_meta.items():
            try:
                payload = self._get(
                    "/odds/player_props", {"game_id": int(gid)}, base=BASE_URL_V2,
                )
            except BdlError as e:
                print(f"  player_props game {gid}: {e!r}")
                continue

            for item in payload.get("data", []):
                pid = item.get("player_id")
                prop_type = item.get("prop_type")
                market_name = self._PROP_MARKET_MAP.get(str(prop_type))
                line_value = item.get("line_value")
                if pid is None or market_name is None or line_value is None:
                    continue
                try:
                    line = float(line_value)
                except (TypeError, ValueError):
                    continue

                # market is an object — only over_under variants carry
                # over/under odds we care about. Milestones we skip.
                m = item.get("market") or {}
                if str(m.get("type")) != "over_under":
                    continue
                over_odds = _coerce_int(m.get("over_odds"))
                under_odds = _coerce_int(m.get("under_odds"))

                bucket[(str(gid), int(pid), market_name)].append(
                    (line, over_odds, under_odds, gdate),
                )
            time.sleep(SLEEP_BETWEEN)

        rows: list[dict] = []
        for (gid, pid, market_name), entries in bucket.items():
            lines = [e[0] for e in entries]
            over_odds = [e[1] for e in entries if e[1] is not None]
            under_odds = [e[2] for e in entries if e[2] is not None]
            gdate = entries[0][3]
            rows.append({
                "game_id": gid,
                "game_date": gdate,
                "player_id": pid,
                "market": market_name,
                "line": _median(lines),
                "over_odds": int(_median(over_odds)) if over_odds else None,
                "under_odds": int(_median(under_odds)) if under_odds else None,
            })
        return rows

    # ---- Injuries -> status map ----

    def fetch_injuries(self, team_ids: list[int] | None = None) -> dict[int, str]:
        """Returns {player_id: normalized_status}.

        Players absent from the dict are assumed active by the caller.
        Statuses normalized to: 'out' | 'doubtful' | 'questionable'.
        """
        params: dict[str, Any] = {}
        if team_ids:
            params["team_ids[]"] = team_ids
        out: dict[int, str] = {}
        for r in self._paginate("/player_injuries", params):
            pid = int(r["player"]["id"])
            raw = (r.get("status") or "").lower()
            if "out" in raw:
                out[pid] = "out"
            elif "doubt" in raw:
                out[pid] = "doubtful"
            elif "questionable" in raw or "day to day" in raw or "game time" in raw:
                out[pid] = "questionable"
            else:
                out[pid] = "out"  # be conservative on unrecognized statuses
        return out


# ---- Pure helpers (testable without network) ----

def _parse_date(s: str) -> date:
    return date.fromisoformat(s[:10])


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _coerce_int(x) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _ml_to_implied(ml) -> float:
    """American moneyline -> implied probability."""
    try:
        x = float(ml)
    except (TypeError, ValueError):
        return 0.0
    if x > 0:
        return 100.0 / (x + 100.0)
    if x < 0:
        return abs(x) / (abs(x) + 100.0)
    return 0.0


def _parse_minutes(s: Any) -> float:
    if not s:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if ":" in s:
        m, sec = s.split(":")
        return float(m or 0) + float(sec or 0) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


def _sum(players: list[dict], field: str) -> int:
    return sum(int(p.get(field) or 0) for p in players)


def _box_to_game_rows(box: dict) -> list[dict]:
    home = box["home_team"]
    away = box["visitor_team"]
    home_p = home.get("players", []) or []
    away_p = away.get("players", []) or []

    home_pts = int(box.get("home_team_score") or 0)
    away_pts = int(box.get("visitor_team_score") or 0)
    season_yr = int(box.get("season") or 0)
    season = f"{season_yr}-{(season_yr + 1) % 100:02d}" if season_yr else ""
    game_id = str(box["id"])
    game_date = _parse_date(box["date"])
    is_playoff = bool(box.get("postseason"))

    home_fga, home_fta = _sum(home_p, "fga"), _sum(home_p, "fta")
    home_tov, home_oreb = _sum(home_p, "turnover"), _sum(home_p, "oreb")
    away_fga, away_fta = _sum(away_p, "fga"), _sum(away_p, "fta")
    away_tov, away_oreb = _sum(away_p, "turnover"), _sum(away_p, "oreb")

    base = dict(game_id=game_id, game_date=game_date, season=season,
                is_playoff=is_playoff)
    return [
        {
            **base, "team_id": int(home["id"]), "opponent_id": int(away["id"]),
            "is_home": True,
            "pts": home_pts, "pts_allowed": away_pts,
            "fga": home_fga, "fta": home_fta, "tov": home_tov, "oreb": home_oreb,
            "opp_fga": away_fga, "opp_fta": away_fta,
            "opp_tov": away_tov, "opp_oreb": away_oreb,
            "won": home_pts > away_pts,
        },
        {
            **base, "team_id": int(away["id"]), "opponent_id": int(home["id"]),
            "is_home": False,
            "pts": away_pts, "pts_allowed": home_pts,
            "fga": away_fga, "fta": away_fta, "tov": away_tov, "oreb": away_oreb,
            "opp_fga": home_fga, "opp_fta": home_fta,
            "opp_tov": home_tov, "opp_oreb": home_oreb,
            "won": away_pts > home_pts,
        },
    ]


def _box_to_player_rows(box: dict) -> tuple[list[dict], list[dict]]:
    home, away = box["home_team"], box["visitor_team"]
    game_id = str(box["id"])
    game_date = _parse_date(box["date"])

    pg: list[dict] = []
    players: list[dict] = []
    for team, opp, is_home in [(home, away, True), (away, home, False)]:
        for ps in team.get("players", []) or []:
            p = ps["player"]
            pid = int(p["id"])
            full_name = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            players.append({"player_id": pid, "full_name": full_name or "Unknown"})
            pg.append({
                "game_id": game_id,
                "game_date": game_date,
                "player_id": pid,
                "team_id": int(team["id"]),
                "opponent_id": int(opp["id"]),
                "is_home": is_home,
                "minutes": _parse_minutes(ps.get("min")),
                "pts": int(ps.get("pts") or 0),
                "reb": int(ps.get("reb") or 0),
                "ast": int(ps.get("ast") or 0),
                "plus_minus": float(ps.get("plus_minus") or 0),
                "fgm": int(ps.get("fgm") or 0),
                "fga": int(ps.get("fga") or 0),
                "ftm": int(ps.get("ftm") or 0),
                "fta": int(ps.get("fta") or 0),
                "oreb": int(ps.get("oreb") or 0),
                "dreb": int(ps.get("dreb") or 0),
                "stl": int(ps.get("stl") or 0),
                "blk": int(ps.get("blk") or 0),
                "tov": int(ps.get("turnover") or 0),
                "pf": int(ps.get("pf") or 0),
            })
    return pg, players
