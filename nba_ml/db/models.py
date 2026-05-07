from datetime import date, datetime
from sqlalchemy import (
    Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Index,
    Integer, String, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    team_id: Mapped[int] = mapped_column(primary_key=True)
    abbreviation: Mapped[str] = mapped_column(String(3), unique=True)
    full_name: Mapped[str] = mapped_column(String(64))


class Player(Base):
    __tablename__ = "players"
    player_id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(128))


class Game(Base):
    """Raw box score, one row per (game, team) perspective."""
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(16), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(7), index=True)

    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    opponent_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    is_home: Mapped[bool] = mapped_column(Boolean)

    pts: Mapped[int]
    pts_allowed: Mapped[int]
    fga: Mapped[int]
    fta: Mapped[int]
    tov: Mapped[int]
    oreb: Mapped[int]
    opp_fga: Mapped[int]
    opp_fta: Mapped[int]
    opp_tov: Mapped[int]
    opp_oreb: Mapped[int]

    won: Mapped[bool] = mapped_column(Boolean)
    is_playoff: Mapped[bool] = mapped_column(Boolean, default=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", name="uq_game_team"),
        Index("ix_games_team_date", "team_id", "game_date"),
        Index("ix_games_season_playoff", "season", "is_playoff"),
        CheckConstraint("team_id != opponent_id", name="ck_team_neq_opp"),
    )


class PlayerGame(Base):
    """Raw per-player per-game box score. Idempotent on (game_id, player_id)."""
    __tablename__ = "player_games"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(16), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    opponent_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean)

    minutes: Mapped[float]
    pts: Mapped[int]
    reb: Mapped[int]
    ast: Mapped[int]
    plus_minus: Mapped[float]

    # Box-score components needed for Hollinger Game Score / true shooting / etc.
    # Default 0 so the column can be added to existing tables non-destructively;
    # rows ingested before backfill will have zeros and skew Game Score low until
    # ETL is re-run for those games.
    fgm: Mapped[int] = mapped_column(Integer, default=0)
    fga: Mapped[int] = mapped_column(Integer, default=0)
    ftm: Mapped[int] = mapped_column(Integer, default=0)
    fta: Mapped[int] = mapped_column(Integer, default=0)
    oreb: Mapped[int] = mapped_column(Integer, default=0)
    dreb: Mapped[int] = mapped_column(Integer, default=0)
    stl: Mapped[int] = mapped_column(Integer, default=0)
    blk: Mapped[int] = mapped_column(Integer, default=0)
    tov: Mapped[int] = mapped_column(Integer, default=0)
    pf: Mapped[int] = mapped_column(Integer, default=0)

    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_game"),
        Index("ix_player_opp_date", "player_id", "opponent_id", "game_date"),
    )


class PlayerAvailability(Base):
    """Pre-tipoff availability snapshot. status in {'active','out','questionable','doubtful','gtd'}.

    Used at inference to mask out injured players: their historical performance
    is excluded from matchup feature aggregation entirely.
    """
    __tablename__ = "player_availability"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(16), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reported_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_avail_game_player"),
        Index("ix_avail_team_date", "team_id", "game_date"),
    )


class TeamGameFeature(Base):
    """Engineered team features. Computed only from games strictly prior."""
    __tablename__ = "team_game_features"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(16), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)

    off_rating_roll5: Mapped[float | None] = mapped_column(Float, nullable=True)
    def_rating_roll5: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_rating_roll10: Mapped[float | None] = mapped_column(Float, nullable=True)
    pace_roll5: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_pct_roll10: Mapped[float | None] = mapped_column(Float, nullable=True)
    rest_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_b2b: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Post-game Elo rating. Use as the team's "current strength" for the next game.
    elo: Mapped[float | None] = mapped_column(Float, nullable=True)

    feature_version: Mapped[str] = mapped_column(String(16))
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", "feature_version",
                         name="uq_tfeat_game_team_ver"),
    )


class PlayerProp(Base):
    """Pre-tipoff player prop line (PTS / REB / AST over/under).

    Mirrors the BettingOdds idea at the player+market grain. is_pregame=True
    means the row was captured before tipoff — only those rows are safe to
    train on or use for inference. Post-tipoff rows leak the outcome the
    same way settled team odds do.
    """
    __tablename__ = "player_props"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(16), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    # Canonical names: 'pts', 'reb', 'ast'.
    market: Mapped[str] = mapped_column(String(8), index=True)
    line: Mapped[float] = mapped_column(Float)
    over_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    under_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="balldontlie")
    is_pregame: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "game_id", "player_id", "market", "source",
            name="uq_player_prop",
        ),
        Index("ix_props_player_date", "player_id", "game_date"),
    )


class BettingOdds(Base):
    """Pre-tipoff Vegas odds. Keyed by (game_date, home_team_id, away_team_id, source)
    so multiple sportsbooks can coexist; we read the first one matching at predict time.
    """
    __tablename__ = "betting_odds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    vegas_home_win_prob: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="balldontlie")
    # True only when the row was captured BEFORE tipoff. Daily archival sets
    # this; historical/post-game pulls leave it False so they get filtered
    # out of training and the inference lookup.
    is_pregame: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_date", "home_team_id", "away_team_id", "source",
                         name="uq_odds_date_teams_source"),
    )
