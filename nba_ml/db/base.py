from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_ml.config import settings
from nba_ml.db.models import Base

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
