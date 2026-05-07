from typing import Generator
from sqlalchemy.orm import Session

from nba_ml.db.base import SessionLocal
from nba_ml.inference.loader import LoadedModel, get_loaded_model


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_model() -> LoadedModel:
    return get_loaded_model()
