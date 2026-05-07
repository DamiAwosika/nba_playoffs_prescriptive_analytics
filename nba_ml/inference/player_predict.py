"""Load trained player models and produce next-game stat predictions.

Usage:
    from nba_ml.inference.player_predict import predict_player_stats
    out = predict_player_stats(db, player_id=237, opponent_team_id=14,
                               as_of=date(2026,5,4), is_home=True)
    # -> {"pts": {"linear": 27.4, "rf": 26.1, "xgb": 27.9, "ensemble": 27.1},
    #     "reb": {...}, "ast": {...}, "stl": {...}, "blk": {...}, "tov": {...},
    #     "feature_count_vs_opp": 11, "games_played_season": 60}

Returns None for a target/model combo if no bundle is present.
Returns None for the whole call if the player has too little prior history
(see MIN_PRIOR_GAMES in features.player).
"""
from __future__ import annotations
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from nba_ml.config import settings
from nba_ml.features.player import build_player_features, player_prop_line

TARGETS = ("pts", "reb", "ast", "stl", "blk", "tov")
MODEL_TYPES = ("linear", "rf", "xgb")


@lru_cache(maxsize=64)
def _load_bundle(target: str, model_type: str, variant: str = "base") -> dict | None:
    """variant: 'base' or 'vegas'."""
    suffix = "_vegas" if variant == "vegas" else ""
    path: Path = settings.models_dir / f"player_{target}_{model_type}{suffix}_v1.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _predict_one(bundle: dict, features: dict[str, float]) -> float:
    cols = bundle["feature_columns"]
    X = pd.DataFrame([features], columns=cols)
    pred = float(bundle["model"].predict(X)[0])
    return max(0.0, pred)  # stat counts can't be negative


def predict_player_stats(
    db: Session, player_id: int, opponent_team_id: int,
    as_of: date, is_home: bool,
) -> dict | None:
    """Predict next-game PTS / REB / AST / STL / BLK / TOV for a player.

    Per target, if a pre-game prop line is available for that target on
    `as_of`, the vegas-augmented model is preferred over the base model.
    Falls back to base when no prop / no vegas bundle is on disk.
    """
    feats = build_player_features(db, player_id, opponent_team_id, as_of, is_home)
    if feats is None:
        return None

    out: dict = {"variants": {}, "prop_lines": {}}
    for target in TARGETS:
        prop = player_prop_line(db, player_id, as_of, target)
        # Use vegas variant only when the prop is actually present (non-zero
        # sentinel from feature builder + bundle exists on disk).
        use_vegas = prop is not None
        out["prop_lines"][target] = float(prop) if prop is not None else None

        target_preds: dict[str, float] = {}
        used_variant = "base"
        for mtype in MODEL_TYPES:
            bundle = _load_bundle(target, mtype, "vegas") if use_vegas else None
            if bundle is None:
                bundle = _load_bundle(target, mtype, "base")
                used_variant = "base"
            else:
                used_variant = "vegas"
            if bundle is None:
                continue
            target_preds[mtype] = round(_predict_one(bundle, feats), 1)
        if target_preds:
            target_preds["ensemble"] = round(
                float(np.mean(list(target_preds.values()))), 1
            )
        out[target] = target_preds
        out["variants"][target] = used_variant

    out["games_played_season"] = int(feats.get("games_played_season", 0))
    out["n_games_vs_opp"] = int(feats.get("n_games_vs_opp", 0))
    return out


def list_player_models() -> dict[str, dict[str, list[str]]]:
    """Returns {target: {variant: [model_types_available]}} for inspection."""
    available: dict[str, dict[str, list[str]]] = {
        t: {"base": [], "vegas": []} for t in TARGETS
    }
    for target in TARGETS:
        for variant in ("base", "vegas"):
            for mtype in MODEL_TYPES:
                if _load_bundle(target, mtype, variant) is not None:
                    available[target][variant].append(mtype)
    return available
