"""Train calibrated logistic regression / random forest / xgboost on home-win
outcomes. Saves one bundle per model to {models_dir}/{name}_v1.joblib.

Evaluation is a temporal holdout: the most recent 20% of games (by date) are
held out — no random shuffling, since random splits would leak future info."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from datetime import date
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from nba_ml.config import settings
from nba_ml.db.base import SessionLocal
from nba_ml.db.models import Game
from nba_ml.features.matchup import (
    FEATURE_COLUMNS, FEATURE_COLUMNS_VEGAS,
    active_players_for_game, build_matchup_features,
)

MODEL_VERSION = "v1"

# Monotonic priors for XGBoost. +1 = "feature increase pushes P(home win) up",
# -1 = "feature increase pushes P(home win) down", 0 = no constraint.
# Encodes basketball-true direction so XGB can't fit non-monotonic noise.
_MONOTONE_BY_FEATURE: dict[str, int] = {
    "home_off_roll5":         +1,
    "home_def_roll5":         -1,  # higher def_rating = giving up more = worse
    "home_net_roll10":        +1,
    "home_win_pct_roll10":    +1,
    "home_is_b2b":            -1,  # tired team
    "away_off_roll5":         -1,
    "away_def_roll5":         +1,
    "away_net_roll10":        -1,
    "away_win_pct_roll10":    -1,
    "away_is_b2b":            +1,  # tired opponent
    "net_rating_diff":        +1,
    "off_rating_diff":        +1,
    "def_rating_diff":        -1,  # positive = home defends worse
    "win_pct_diff":           +1,
    "pvt_proj_pts_diff":      +1,
    "rest_advantage":         +1,
    "home_home_win_pct_l20":  +1,
    "away_road_win_pct_l20":  -1,
    "h2h_home_win_pct":       +1,
    "home_pvt_proj_pts":      +1,
    "away_pvt_proj_pts":      -1,
    "elo_diff":               +1,
    "elo_win_prob":           +1,
    "vegas_home_win_prob":    +1,
}


def _xgb_monotone_tuple(feature_columns: list[str] | None = None) -> tuple:
    """Constraint tuple in feature_columns order."""
    cols = feature_columns or FEATURE_COLUMNS
    return tuple(_MONOTONE_BY_FEATURE.get(c, 0) for c in cols)


# Search spaces for --tune. RandomizedSearchCV samples n_iter combos.
# XGB space is widened (slower learning_rate × more rounds) since tree models
# now have ~3 seasons of training data and can use the extra capacity.
PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "logreg": {
        # clf__C addresses the LogisticRegression nested inside the Pipeline.
        "clf__C": [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
    "rf": {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "xgb": {
        # n_estimators NOT here — chosen by early stopping in the final fit.
        # CV uses XGB's constructor default (1500 below) for fair param search.
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
}


def _tune(name: str, base, X_train, y_train, n_iter: int, feature_columns: list[str]):
    """RandomizedSearchCV over PARAM_DISTRIBUTIONS[name], scored by neg log_loss
    using TimeSeriesSplit (no temporal leakage during CV)."""
    if name not in PARAM_DISTRIBUTIONS:
        return base
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base, PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter, cv=tscv, scoring="neg_log_loss",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  {name}: best CV log_loss={-search.best_score_:.4f}  "
          f"params={search.best_params_}")
    if name == "xgb" and HAS_XGB:
        return _xgb_refit_with_early_stopping(
            search.best_params_, X_train, y_train, feature_columns,
        )
    return search.best_estimator_


def _xgb_refit_with_early_stopping(
    best_params: dict, X_train, y_train, feature_columns: list[str],
):
    """Two-stage XGB refit:
      1) Carve last 10% of train as time-ordered validation.
      2) Fit with n_estimators=3000 + early_stopping_rounds=50 -> discover best n.
      3) Refit on FULL train with that best n (no early stopping needed).
    """
    n = len(X_train)
    split = max(1, int(n * 0.9))
    X_tr, y_tr = X_train.iloc[:split], y_train[:split]
    X_val, y_val = X_train.iloc[split:], y_train[split:]

    params = {k: v for k, v in best_params.items() if k != "n_estimators"}
    monotone = _xgb_monotone_tuple(feature_columns)

    es = XGBClassifier(
        **params, n_estimators=3000, early_stopping_rounds=50,
        eval_metric="logloss", random_state=42, tree_method="hist",
        monotone_constraints=monotone,
    )
    es.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_n = int(es.best_iteration) + 1
    print(f"  xgb early-stopped at {best_n} rounds (of 3000) — refitting on full train")

    final = XGBClassifier(
        **params, n_estimators=best_n,
        eval_metric="logloss", random_state=42, tree_method="hist",
        monotone_constraints=monotone,
    )
    final.fit(X_train, y_train)
    return final


def build_training_frame(db: Session, feature_version: str) -> pd.DataFrame:
    home_games = db.execute(
        select(Game).where(Game.is_home.is_(True)).order_by(Game.game_date)
    ).scalars().all()

    rows: list[dict] = []
    for g in home_games:
        ha = active_players_for_game(db, g.game_id, g.team_id)
        aa = active_players_for_game(db, g.game_id, g.opponent_id)
        feats = build_matchup_features(
            db, g.team_id, g.opponent_id, g.game_date, feature_version,
            home_active=ha, away_active=aa,
        )
        if feats is None:
            continue
        feats["game_id"] = g.game_id
        feats["home_won"] = int(g.won)
        feats["game_date"] = g.game_date
        rows.append(feats)
    return pd.DataFrame(rows)


def _make_base_models(feature_columns: list[str] | None = None) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        ),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=1500, max_depth=4, learning_rate=0.02,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, tree_method="hist",
            monotone_constraints=_xgb_monotone_tuple(feature_columns),
        )
    else:
        print("xgboost not installed — skipping XGB. `pip install xgboost` to enable.")
    return models


def train(
    out_dir: Path,
    feature_version: str,
    tune: bool = False,
    n_iter: int = 20,
    *,
    feature_columns: list[str] | None = None,
    row_filter=None,
    model_suffix: str = "",
    ensemble_name: str = "ensemble",
) -> None:
    """Train logreg + RF + XGB + soft-vote ensemble.

    Parameters
    ----------
    feature_columns : which columns of the training frame to feed the model.
        Defaults to FEATURE_COLUMNS (the base, no-Vegas set).
    row_filter : optional callable(df) -> df. Used by the Vegas-augmented
        trainer to restrict training to games with odds.
    model_suffix : appended to filenames, e.g. "_vegas" -> logreg_vegas_v1.joblib.
    ensemble_name : filename stem for the saved ensemble bundle.
    """
    cols = feature_columns or FEATURE_COLUMNS
    db = SessionLocal()
    try:
        df = build_training_frame(db, feature_version)
    finally:
        db.close()

    if df.empty:
        raise SystemExit(
            "No training rows produced. Run scripts/run_etl.py first to populate "
            "raw box scores and team features."
        )

    if row_filter is not None:
        n_before = len(df)
        df = row_filter(df).reset_index(drop=True)
        print(f"Row filter: {n_before} -> {len(df)} rows")
        if df.empty:
            raise SystemExit("No rows left after filter; nothing to train.")

    df = df.sort_values("game_date").reset_index(drop=True)
    split = max(1, int(len(df) * 0.8))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train_df[cols], train_df["home_won"]
    X_test, y_test = test_df[cols], test_df["home_won"]

    print(
        f"Training on {len(train_df)} games "
        f"({train_df['game_date'].min()} -> {train_df['game_date'].max()}), "
        f"holding out {len(test_df)} most-recent games for evaluation."
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict] = {}

    if tune:
        print(f"\nTuning each model with RandomizedSearchCV (n_iter={n_iter}, "
              "TimeSeriesSplit cv=3, scoring=neg_log_loss)...")

    fitted_bases: dict[str, Any] = {}
    fitted_calibrated: dict[str, Any] = {}
    for name, base in _make_base_models(cols).items():
        if tune:
            base = _tune(name, base, X_train, y_train, n_iter=n_iter, feature_columns=cols)
        calib_method = "sigmoid" if name == "logreg" else "isotonic"
        model = CalibratedClassifierCV(base, method=calib_method, cv=3)
        model.fit(X_train, y_train)
        fitted_bases[name] = base
        fitted_calibrated[name] = model

        train_proba = model.predict_proba(X_train)[:, 1]
        metrics[name] = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train": _eval(y_train, train_proba),
        }
        if len(test_df):
            test_proba = model.predict_proba(X_test)[:, 1]
            metrics[name]["test"] = _eval(y_test, test_proba)

        joblib.dump({
            "model": model,
            "feature_columns": cols,
            "feature_version": feature_version,
            "version": MODEL_VERSION,
            "model_type": name,
        }, out_dir / f"{name}{model_suffix}_{MODEL_VERSION}.joblib")

    # ---- Soft-vote ensemble ----
    # Mean of the calibrated base predictions. Cheap, robust on small samples,
    # and avoids the meta-overfit risk that stacking has when n_train is small.
    ens_metrics = _save_ensemble(
        fitted_calibrated, X_train, y_train, X_test, y_test,
        feature_columns=cols, model_suffix=model_suffix,
        feature_version=feature_version, out_dir=out_dir, name=ensemble_name,
    )
    if ens_metrics is not None:
        metrics[ensemble_name] = ens_metrics

    # ---- Stacking meta-model (kept; still useful when n_train is large) ----
    stack_metrics = _train_stack(
        fitted_bases, fitted_calibrated, X_train, y_train, X_test, y_test,
        feature_version, out_dir, model_suffix=model_suffix,
    )
    if stack_metrics is not None:
        metrics[f"stack{model_suffix}"] = stack_metrics

    _print_metrics_table(metrics)
    (out_dir / f"metrics{model_suffix}.json").write_text(json.dumps(metrics, indent=2))


def _save_ensemble(
    fitted_calibrated: dict[str, Any], X_train, y_train, X_test, y_test,
    *, feature_columns: list[str], model_suffix: str,
    feature_version: str, out_dir: Path, name: str,
) -> dict | None:
    """Soft-vote ensemble (mean of member predict_probas). Saves a bundle that
    just lists which member models to load — the actual mean happens at
    inference time inside VotingEnsemble."""
    members = list(fitted_calibrated.keys())
    if len(members) < 2:
        return None

    train_proba = np.mean(
        [fitted_calibrated[m].predict_proba(X_train)[:, 1] for m in members], axis=0,
    )
    out: dict[str, Any] = {
        "n_train": len(X_train), "n_test": len(X_test),
        "train": _eval(y_train, train_proba),
    }
    if len(X_test):
        test_proba = np.mean(
            [fitted_calibrated[m].predict_proba(X_test)[:, 1] for m in members], axis=0,
        )
        out["test"] = _eval(y_test, test_proba)

    member_names = [f"{m}{model_suffix}" for m in members]
    joblib.dump({
        "model_type": "ensemble",
        "member_names": member_names,
        "feature_columns": feature_columns,
        "feature_version": feature_version,
        "version": MODEL_VERSION,
    }, out_dir / f"{name}_{MODEL_VERSION}.joblib")
    print(f"  ensemble '{name}' = soft-vote over {member_names}")
    return out


def _train_stack(
    fitted_bases: dict[str, Any], fitted_calibrated: dict[str, Any],
    X_train, y_train, X_test, y_test,
    feature_version: str, out_dir: Path, *, model_suffix: str = "",
) -> dict | None:
    """Stack base-model OOF probabilities with a logreg meta. Saves stack_v1.joblib."""
    if len(fitted_bases) < 2:
        return None
    print("\nStacking: generating OOF base predictions (manual TimeSeriesSplit cv=5)...")

    # Manual TimeSeriesSplit loop — the leading rows are never in any test fold,
    # so they have no OOF prediction. We accumulate predictions where we have
    # them and train the meta only on those rows.
    tscv = TimeSeriesSplit(n_splits=5)
    n = len(X_train)
    oof: dict[str, np.ndarray] = {}
    for name, base in fitted_bases.items():
        preds = np.full(n, np.nan)
        try:
            for tr_idx, val_idx in tscv.split(X_train):
                m = clone(base)
                m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                preds[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        except Exception as e:
            print(f"  OOF for {name} failed ({e!r}); skipping stack.")
            return None
        oof[name] = preds

    base_names = list(oof.keys())
    valid = ~np.isnan(oof[base_names[0]])
    n_valid = int(valid.sum())
    if n_valid < 50:
        print(f"  only {n_valid} rows with OOF predictions; skipping stack.")
        return None

    # Use the SAVED (suffixed) names as the meta's column labels so the column
    # names at fit time match the column names StackedModel uses at inference.
    saved_base_names = [f"{n}{model_suffix}" for n in base_names]

    X_meta_train = pd.DataFrame({
        saved: oof[orig][valid]
        for orig, saved in zip(base_names, saved_base_names)
    })
    y_meta_train = y_train.iloc[valid] if hasattr(y_train, "iloc") else y_train[valid]

    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta_train, y_meta_train)

    # For test: feed the CALIBRATED base predictions to the meta with the same
    # (suffixed) column names the meta expects.
    test_base = pd.DataFrame({
        saved: fitted_calibrated[orig].predict_proba(X_test)[:, 1]
        for orig, saved in zip(base_names, saved_base_names)
    })
    test_proba = meta.predict_proba(test_base)[:, 1]
    train_proba = meta.predict_proba(X_meta_train)[:, 1]
    joblib.dump({
        "meta_model": meta,
        "base_model_names": saved_base_names,
        "feature_columns": saved_base_names,
        "feature_version": feature_version,
        "version": MODEL_VERSION,
        "model_type": "stack",
    }, out_dir / f"stack{model_suffix}_{MODEL_VERSION}.joblib")

    weights = dict(zip(base_names, meta.coef_[0]))
    print(f"  stack meta weights: {weights}, intercept={meta.intercept_[0]:+.3f}")
    print(f"  stack trained on {n_valid} OOF rows (of {n} train rows)")

    out: dict[str, Any] = {
        "n_train": n_valid, "n_test": len(X_test),
        "train": _eval(y_meta_train, train_proba),
    }
    if len(X_test):
        out["test"] = _eval(y_test, test_proba)
    return out


def _eval(y_true, y_proba) -> dict[str, float]:
    """Compute every metric in one place. Notes on interpretation for binary
    classification:
      log_loss : primary metric. Lower is better. Penalizes confident wrongness.
      accuracy : threshold@0.5. Coarse — ignores how (un)sure the model was.
      brier    : MSE on probabilities. Lower is better. Equivalent to RMSE^2.
      rmse     : sqrt(brier). Same info as brier, just on the original scale.
      r2       : sklearn's r2_score on (y, p). Defined but weak meaning for
                 binary targets — included for comparison only.
      roc_auc  : ranking quality. 0.5 = coin flip, 1.0 = perfect."""
    y_pred = (y_proba >= 0.5).astype(int)
    has_both_classes = len(set(y_true)) > 1
    return {
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier":    float(brier_score_loss(y_true, y_proba)),
        "rmse":     float(np.sqrt(mean_squared_error(y_true, y_proba))),
        "r2":       float(r2_score(y_true, y_proba)),
        "roc_auc":  float(roc_auc_score(y_true, y_proba)) if has_both_classes else float("nan"),
    }


def _print_metrics_table(metrics: dict[str, dict]) -> None:
    cols = ["log_loss", "accuracy", "brier", "rmse", "r2", "roc_auc"]
    header = f"{'model':<8} {'set':<6}" + "".join(f" {c:>9}" for c in cols)
    print()
    print("Train vs holdout (lower log_loss/brier/rmse, higher accuracy/r2/roc_auc = better):")
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        for split in ("train", "test"):
            if split not in m:
                continue
            row = m[split]
            cells = "".join(f" {row[c]:>9.4f}" for c in cols)
            print(f"{name:<8} {split:<6}{cells}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=settings.models_dir)
    p.add_argument("--feature-version", default=settings.feature_version)
    p.add_argument("--tune", action="store_true",
                   help="Run RandomizedSearchCV over each model's PARAM_DISTRIBUTIONS")
    p.add_argument("--n-iter", type=int, default=20,
                   help="Sampled hyperparameter combos per model when --tune")
    args = p.parse_args()
    train(args.out_dir, args.feature_version, tune=args.tune, n_iter=args.n_iter)
    print(f"run completed: {date.today()}")