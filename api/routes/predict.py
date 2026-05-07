from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_model
from api.schemas import PredictRequest, PredictResponse
from nba_ml.inference.loader import LoadedModel
from nba_ml.inference.service import predict_matchup

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    model: LoadedModel = Depends(get_model),
) -> PredictResponse:
    if req.home_team_id == req.away_team_id:
        raise HTTPException(400, "home and away teams must differ")

    result = predict_matchup(
        db, model, req.home_team_id, req.away_team_id, req.game_date,
    )
    if result is None:
        raise HTTPException(
            404,
            "insufficient data: missing prior team features or empty active roster",
        )

    return PredictResponse(
        home_team_id=req.home_team_id,
        away_team_id=req.away_team_id,
        home_win_prob=result.home_win_prob,
        model_version=result.model_version,
        feature_version=result.feature_version,
        home_active_count=result.home_active_count,
        away_active_count=result.away_active_count,
    )
