"""Train per-target regression models that predict player PTS / REB / AST /
STL / BLK / TOV for the next game.

For each target T in {pts, reb, ast, stl, blk, tov} and each model in {linear, rf, xgb}:
  - Build (features, T) over all PlayerGame rows that have >= MIN_PRIOR_GAMES
    of strictly-prior history.
  - Holdout = most recent 20% by date (no shuffling — random splits leak).
  - Save to models_dir/player_{T}_{model}_v1.joblib.

Bundle schema:
  {
    "model": fitted estimator,
    "feature_columns": list[str],
    "target": "pts" | "reb" | "ast" | "stl" | "blk" | "tov",
    "model_type": "linear" | "rf" | "xgb",
    "feature_version": "v1",
    "version": "v1",
    "metrics": {"mae": ..., "rmse": ..., "r2": ...},
  }
"""
from __future__ import annotations
import argparse
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from nba_ml.config import settings
from nba_ml.db.base import SessionLocal
from nba_ml.db.models import Game, PlayerGame
from nba_ml.features.player import (
    PLAYER_FEATURE_COLUMNS, VEGAS_FEATURE_COLUMNS_BY_TARGET,
    build_player_features, player_prop_line,
)

MODEL_VERSION = "v1"
TARGETS = ("pts", "reb", "ast", "stl", "blk", "tov")
HOLDOUT_FRAC = 0.20


# ---- Dataset assembly ----

def _iter_training_rows(db: Session, season: str | None = None):
    """Yield (player_id, opponent_id, game_date, is_home, pts, reb, ast,
    stl, blk, tov) for every actually-played PlayerGame row.

    Joining Game on (game_id, team_id) lets us filter by season and gives us
    is_home from the Game row (PlayerGame also has is_home but we take it
    from Game for consistency with other training pipelines).
    """
    q = (
        select(
            PlayerGame.player_id, PlayerGame.opponent_id,
            PlayerGame.game_date, PlayerGame.is_home,
            PlayerGame.pts, PlayerGame.reb, PlayerGame.ast,
            PlayerGame.stl, PlayerGame.blk, PlayerGame.tov,
        )
        .join(Game, and_(Game.game_id == PlayerGame.game_id,
                         Game.team_id == PlayerGame.team_id))
        .where(PlayerGame.minutes > 0)
        .order_by(PlayerGame.game_date.asc())
    )
    if season:
        q = q.where(Game.season == season)
    return db.execute(q).all()


def build_dataset(db: Session, season: str | None = None) -> pd.DataFrame:
    """Build the full (features + targets + date) DataFrame for training.

    Computes features per row on the fly. Skips rows where the player has
    too few prior games (build_player_features returns None).
    """
    rows = _iter_training_rows(db, season)
    print(f"  scanning {len(rows)} player-game rows for feature build...")

    records: list[dict] = []
    skipped = 0
    for i, (pid, opp, gdate, is_home, pts, reb, ast, stl, blk, tov) in enumerate(rows):
        if i and i % 5000 == 0:
            print(f"    {i}/{len(rows)} processed, {len(records)} kept, {skipped} skipped")
        feats = build_player_features(db, pid, opp, gdate, is_home)
        if feats is None:
            skipped += 1
            continue
        feats["pts"] = pts
        feats["reb"] = reb
        feats["ast"] = ast
        feats["stl"] = stl
        feats["blk"] = blk
        feats["tov"] = tov
        feats["_game_date"] = gdate
        records.append(feats)

    print(f"  built {len(records)} usable rows, skipped {skipped} (insufficient history)")
    return pd.DataFrame.from_records(records)


# ---- Models ----

def _make_linear() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def _make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )


def _make_xgb() -> "XGBRegressor":
    return XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        objective="reg:squarederror", tree_method="hist",
        random_state=42, n_jobs=-1,
    )


def _model_factory(model_type: str):
    if model_type == "linear": return _make_linear()
    if model_type == "rf":     return _make_rf()
    if model_type == "xgb":    return _make_xgb()
    raise ValueError(f"unknown model_type {model_type!r}")


# RandomizedSearchCV spaces — sampled n_iter times each.
PARAM_DIST = {
    "linear": {"reg__alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
    "rf": {
        "n_estimators":   [200, 400, 600, 800],
        "max_depth":      [6, 8, 10, 12, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features":   ["sqrt", 0.5, 0.7],
    },
    "xgb": {
        "n_estimators":     [400, 600, 800, 1200],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.02, 0.03, 0.05],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 10],
    },
}
TUNE_N_ITER = {"linear": 6, "rf": 15, "xgb": 25}


def _tune(model_type: str, X_tr: pd.DataFrame, y_tr: pd.Series):
    """RandomizedSearchCV over PARAM_DIST[model_type] using TimeSeriesSplit
    so CV folds respect chronological order (no future info leaks back)."""
    base = _model_factory(model_type)
    cv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        base, param_distributions=PARAM_DIST[model_type],
        n_iter=TUNE_N_ITER[model_type], cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_


# ---- Training driver ----

def _temporal_split(df: pd.DataFrame, holdout_frac: float):
    df = df.sort_values("_game_date").reset_index(drop=True)
    cutoff = int(len(df) * (1 - holdout_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def _eval(model, X_test, y_test) -> dict:
    pred = np.clip(model.predict(X_test), 0, None)  # stat counts can't be negative
    return {
        "mae":  float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2":   float(r2_score(y_test, pred)),
    }


def train_one(target: str, model_type: str, train: pd.DataFrame,
              test: pd.DataFrame, tune: bool = False,
              feature_columns: list[str] | None = None) -> dict[str, Any]:
    cols = feature_columns or PLAYER_FEATURE_COLUMNS
    X_tr = train[cols]
    y_tr = train[target]
    X_te = test[cols]
    y_te = test[target]

    best_params: dict | None = None
    if tune:
        model, best_params = _tune(model_type, X_tr, y_tr)
    else:
        model = _model_factory(model_type)
        model.fit(X_tr, y_tr)
    metrics = _eval(model, X_te, y_te)
    return {"model": model, "metrics": metrics, "best_params": best_params}


def main(season: str | None = None, tune: bool = False) -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    db: Session = SessionLocal()
    try:
        print("=== building dataset ===")
        df = build_dataset(db, season=season)
    finally:
        db.close()

    if df.empty:
        print("no training rows — exiting")
        return

    train, test = _temporal_split(df, HOLDOUT_FRAC)
    print(f"train: {len(train)} rows, holdout: {len(test)} rows "
          f"(date range: {df['_game_date'].min()} .. {df['_game_date'].max()})")

    model_types = ["linear", "rf"] + (["xgb"] if HAS_XGB else [])

    print("\n=== training ===")
    print(f"{'target':<8} {'model':<8} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    print("-" * 40)

    for target in TARGETS:
        for mtype in model_types:
            result = train_one(target, mtype, train, test, tune=tune)
            m = result["metrics"]
            print(f"{target:<8} {mtype:<8} {m['mae']:>7.2f} {m['rmse']:>7.2f} {m['r2']:>7.3f}")
            if result["best_params"]:
                print(f"         best params: {result['best_params']}")

            bundle = {
                "model": result["model"],
                "feature_columns": list(PLAYER_FEATURE_COLUMNS),
                "target": target,
                "model_type": mtype,
                "feature_version": settings.feature_version,
                "version": MODEL_VERSION,
                "metrics": m,
                "best_params": result["best_params"],
            }
            path = settings.models_dir / f"player_{target}_{mtype}_{MODEL_VERSION}.joblib"
            joblib.dump(bundle, path)
            print(f"         saved {path.name}")

    # ---- Vegas-augmented variants ----
    # Per target, filter to rows where that target's prop line was archived
    # pre-game. This subset will be empty until archive_player_props.py has
    # run for a while; we skip cleanly when too few rows exist.
    print("\n=== vegas-augmented variants (per-target) ===")
    print(f"{'target':<8} {'model':<8} {'MAE':>7} {'RMSE':>7} {'R2':>7}  rows")
    print("-" * 52)
    MIN_VEGAS_ROWS = 200
    for target in TARGETS:
        line_col = f"vegas_{target}_line"
        if line_col not in train.columns:
            print(f"{target:<8} skipped — no vegas prop column ({line_col}).")
            continue
        # Sentinel 0.0 means no pre-game prop was found for that row.
        train_v = train[train[line_col] > 0]
        test_v = test[test[line_col] > 0]
        if len(train_v) < MIN_VEGAS_ROWS or len(test_v) < 20:
            print(f"{target:<8} skipped — only {len(train_v)} train / {len(test_v)} test "
                  f"rows have pre-game prop lines (need ≥ {MIN_VEGAS_ROWS} train).")
            continue

        cols = VEGAS_FEATURE_COLUMNS_BY_TARGET[target]
        for mtype in model_types:
            result = train_one(target, mtype, train_v, test_v, tune=tune,
                               feature_columns=cols)
            m = result["metrics"]
            print(f"{target:<8} {mtype:<8} {m['mae']:>7.2f} {m['rmse']:>7.2f} "
                  f"{m['r2']:>7.3f}  {len(train_v)}")
            if result["best_params"]:
                print(f"         best params: {result['best_params']}")

            bundle = {
                "model": result["model"],
                "feature_columns": list(cols),
                "target": target,
                "model_type": mtype,
                "variant": "vegas",
                "feature_version": settings.feature_version,
                "version": MODEL_VERSION,
                "metrics": m,
                "best_params": result["best_params"],
            }
            path = settings.models_dir / f"player_{target}_{mtype}_vegas_{MODEL_VERSION}.joblib"
            joblib.dump(bundle, path)
            print(f"         saved {path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None,
                    help="Restrict training to this season (e.g. 2024-25). "
                         "Default uses all seasons in the DB.")
    ap.add_argument("--tune", action="store_true",
                    help="Run RandomizedSearchCV (TimeSeriesSplit, n_iter per "
                         "model type) before final fit. Slower (~5-10x).")
    args = ap.parse_args()
    main(season=args.season, tune=args.tune)
