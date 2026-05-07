"""Idempotent upserts for raw box-score and availability data.

Idempotency comes from the UniqueConstraint on each table. Re-running these
on overlapping date ranges updates corrected rows in place.
"""
from __future__ import annotations
from typing import Iterable, Mapping
from sqlalchemy import Table
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from nba_ml.db.models import (
    BettingOdds, Game, Player, PlayerAvailability, PlayerGame, PlayerProp, Team,
)


# SQLite caps a single statement at 999 params on builds < 3.32 and 32766 on
# newer ones. Postgres goes up to ~65535. Stay well under the SQLite floor so
# the same code works on any SQLite version without a runtime check.
MAX_PARAMS_PER_STATEMENT = 800


def _upsert(db: Session, table: Table, rows: list[Mapping], conflict_cols: list[str]) -> None:
    if not rows:
        return
    rows = list(rows)
    n_per_row = len(rows[0])
    chunk_size = max(1, MAX_PARAMS_PER_STATEMENT // max(n_per_row, 1))
    for start in range(0, len(rows), chunk_size):
        _upsert_chunk(db, table, rows[start:start + chunk_size], conflict_cols)


def _upsert_chunk(
    db: Session, table: Table, rows: list[Mapping], conflict_cols: list[str],
) -> None:
    dialect = db.bind.dialect.name
    if dialect == "postgresql":
        stmt = pg_insert(table).values(rows)
        update_cols = {
            c.name: stmt.excluded[c.name] for c in table.columns
            if c.name not in {"id", "ingested_at", "computed_at", "reported_at"}
            and c.name not in conflict_cols
        }
        stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
    elif dialect == "sqlite":
        stmt = sqlite_insert(table).values(rows)
        update_cols = {
            c.name: getattr(stmt.excluded, c.name) for c in table.columns
            if c.name not in {"id", "ingested_at", "computed_at", "reported_at"}
            and c.name not in conflict_cols
        }
        stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
    else:
        raise NotImplementedError(f"Upsert not implemented for dialect {dialect!r}")
    db.execute(stmt)


def upsert_teams(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, Team.__table__, list(rows), ["team_id"])


def upsert_players(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, Player.__table__, list(rows), ["player_id"])


def upsert_games(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, Game.__table__, list(rows), ["game_id", "team_id"])


def upsert_player_games(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, PlayerGame.__table__, list(rows), ["game_id", "player_id"])


def upsert_availability(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, PlayerAvailability.__table__, list(rows), ["game_id", "player_id"])


def upsert_betting_odds(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, BettingOdds.__table__, list(rows),
            ["game_date", "home_team_id", "away_team_id", "source"])


def upsert_player_props(db: Session, rows: Iterable[Mapping]) -> None:
    _upsert(db, PlayerProp.__table__, list(rows),
            ["game_id", "player_id", "market", "source"])
