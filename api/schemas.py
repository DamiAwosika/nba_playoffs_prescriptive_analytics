from datetime import date
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date = Field(..., description="Tipoff date of the game to predict")


class PredictResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    home_win_prob: float
    model_version: str
    feature_version: str
    home_active_count: int
    away_active_count: int
