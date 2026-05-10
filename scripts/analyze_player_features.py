"""Feature-importance & error analysis for player stat regression models.

Run AFTER `python nba_ml/training/train_player.py`. Produces PNGs in analysis/:
  player_feature_importance.png    permutation importance per target+model
  player_correlation.png           feature correlation heatmap
  player_distributions.png         per-feature histograms
  player_residuals.png             predicted vs actual scatter + residual distribution
  player_error_by_target.png       MAE/RMSE comparison across targets and models

Usage:
  python scripts/analyze_player_features.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nba_ml.config import settings
from nba_ml.db.base import SessionLocal
from nba_ml.features.player import PLAYER_FEATURE_COLUMNS
from nba_ml.training.train_player import TARGETS, build_dataset, _temporal_split, HOLDOUT_FRAC


def _load_player_models(models_dir: Path) -> dict[str, dict]:
    """Load all player_*_v1.joblib bundles. Returns {name: bundle}."""
    bundles = {}
    for path in sorted(models_dir.glob("player_*_v1.joblib")):
        b = joblib.load(path)
        if b.get("variant") == "vegas":
            continue
        key = f"{b['target']}_{b['model_type']}"
        bundles[key] = b
    return bundles


def plot_feature_importance(bundles, X_test, y_test_dict, out_dir):
    """Permutation importance for each target's best model (by MAE)."""
    best_per_target = {}
    for key, b in bundles.items():
        t = b["target"]
        if t not in best_per_target:
            best_per_target[t] = (key, b)
        else:
            cur_mae = best_per_target[t][1]["metrics"]["mae"]
            if b["metrics"]["mae"] < cur_mae:
                best_per_target[t] = (key, b)

    targets = [t for t in TARGETS if t in best_per_target]
    n = len(targets)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, target in enumerate(targets):
        key, b = best_per_target[target]
        model = b["model"]
        feat_cols = b["feature_columns"]
        y_t = y_test_dict[target]

        pi = permutation_importance(
            model, X_test[feat_cols], y_t,
            scoring="neg_mean_absolute_error", n_repeats=8,
            random_state=42, n_jobs=-1,
        )
        top_k = min(15, len(feat_cols))
        order = np.argsort(pi.importances_mean)[-top_k:]
        ax = axes[i]
        ax.barh(
            [feat_cols[j] for j in order],
            pi.importances_mean[order],
            xerr=pi.importances_std[order],
            color="steelblue",
        )
        ax.set_title(f"{target.upper()} ({b['model_type']})\nTop {top_k} permutation importance")
        ax.tick_params(axis="y", labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Player Feature Importance (permutation, higher = more important)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "player_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  saved player_feature_importance.png")


def plot_correlation(X, out_dir):
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Player Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "player_correlation.png", dpi=120)
    plt.close()

    pairs = []
    cols_list = list(corr.columns)
    for i, c1 in enumerate(cols_list):
        for j in range(i + 1, len(cols_list)):
            v = corr.iloc[i, j]
            if abs(v) > 0.7:
                pairs.append((c1, cols_list[j], v))
    if pairs:
        print("\n  Highly correlated player feature pairs (|r| > 0.7):")
        for c1, c2, v in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"    {v:+.2f}  {c1} <-> {c2}")
    print("  saved player_correlation.png")


def plot_distributions(X, out_dir):
    n = len(X.columns)
    cols_per_row = 5
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3 * rows))
    axes = axes.flatten()
    for i, col in enumerate(X.columns):
        axes[i].hist(X[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Player Feature Distributions", y=1.001, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "player_distributions.png", dpi=120)
    plt.close()
    print("  saved player_distributions.png")


def plot_residuals(bundles, X_test, y_test_dict, out_dir):
    """Predicted vs actual scatter and residual histogram per target."""
    targets = [t for t in TARGETS if any(
        b["target"] == t for b in bundles.values()
    )]
    n = len(targets)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, target in enumerate(targets):
        matching = [(k, b) for k, b in bundles.items() if b["target"] == target]
        best_key, best_b = min(matching, key=lambda x: x[1]["metrics"]["mae"])
        model = best_b["model"]
        feat_cols = best_b["feature_columns"]
        y_true = y_test_dict[target].values
        y_pred = np.clip(model.predict(X_test[feat_cols]), 0, None)
        residuals = y_true - y_pred

        ax = axes[0, i]
        ax.scatter(y_true, y_pred, alpha=0.15, s=8, color="steelblue")
        lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "r--", alpha=0.7, label="perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target.upper()} ({best_b['model_type']})\n"
                     f"MAE={best_b['metrics']['mae']:.2f}  R²={best_b['metrics']['r2']:.3f}")
        ax.legend(fontsize=8)

        ax = axes[1, i]
        ax.hist(residuals, bins=40, color="darkred", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", lw=1, linestyle="--")
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Count")
        mean_r = np.mean(residuals)
        ax.set_title(f"Residuals (mean={mean_r:+.2f}, std={np.std(residuals):.2f})")

    plt.suptitle("Predicted vs Actual & Residual Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "player_residuals.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  saved player_residuals.png")


def plot_error_comparison(bundles, out_dir):
    """Bar chart comparing MAE and RMSE across all targets and model types."""
    data = []
    for key, b in bundles.items():
        data.append({
            "target": b["target"].upper(),
            "model": b["model_type"],
            "MAE": b["metrics"]["mae"],
            "RMSE": b["metrics"]["rmse"],
            "R2": b["metrics"]["r2"],
        })
    df = pd.DataFrame(data)
    if df.empty:
        return

    targets = [t for t in ["PTS", "REB", "AST", "STL", "BLK", "TOV"] if t in df["target"].values]
    models = sorted(df["model"].unique())
    x = np.arange(len(targets))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"linear": "steelblue", "rf": "forestgreen", "xgb": "darkorange"}

    for metric, ax, title in [("MAE", ax1, "Mean Absolute Error (lower = better)"),
                               ("R2", ax2, "R² Score (higher = better)")]:
        for j, m in enumerate(models):
            vals = []
            for t in targets:
                row = df[(df["target"] == t) & (df["model"] == m)]
                vals.append(row[metric].values[0] if len(row) else 0)
            ax.bar(x + j * width, vals, width, label=m, color=colors.get(m, "gray"))
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(targets)
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Model Performance Comparison by Target Stat", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "player_error_by_target.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  saved player_error_by_target.png")


def print_summary(bundles):
    print("\n" + "=" * 65)
    print("PLAYER MODEL ANALYSIS SUMMARY")
    print("=" * 65)

    print(f"\n{'target':<8} {'model':<8} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("-" * 40)
    for target in TARGETS:
        for key in sorted(bundles):
            b = bundles[key]
            if b["target"] != target:
                continue
            m = b["metrics"]
            print(f"{target:<8} {b['model_type']:<8} {m['mae']:>7.2f} {m['rmse']:>7.2f} {m['r2']:>7.3f}")

    print("""
Key observations for player stat regression:

1. FEATURE GROUPS
   - Rolling averages (3/5/10g) capture recent form and trends
   - Per-minute rates decouple production from playing time
   - Variance features (std, streak) capture consistency
   - Opponent context (def_rating, pace, pts_allowed) adjusts for matchup
   - Head-to-head history provides player-vs-team matchup signal

2. EXPECTED PATTERNS
   - PTS has the highest MAE (widest range) but also the most signal
   - STL/BLK have low MAE but low R² (rare events are hard to predict)
   - TOV prediction benefits from usage-rate proxies (TSA per min)

3. IMPROVING PREDICTIONS
   - More historical data increases rolling average stability
   - Vegas prop lines (when available) add strong signal for PTS/REB/AST
   - Opponent defensive matchup features are high-leverage for all targets
""")


def main(models_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = _load_player_models(models_dir)
    if not bundles:
        raise SystemExit(f"No player models in {models_dir}. Run train_player.py first.")
    print(f"Loaded {len(bundles)} player model bundles")

    db = SessionLocal()
    try:
        print("Building dataset (this may take a few minutes)...")
        df = build_dataset(db)
    finally:
        db.close()
    if df.empty:
        raise SystemExit("No training rows.")

    train, test = _temporal_split(df, HOLDOUT_FRAC)
    X_test = test[PLAYER_FEATURE_COLUMNS]
    X_train = train[PLAYER_FEATURE_COLUMNS]
    y_test_dict = {t: test[t] for t in TARGETS}
    print(f"Train: {len(train)}  Test: {len(test)}")

    print("\nGenerating plots...")
    plot_feature_importance(bundles, X_test, y_test_dict, out_dir)
    plot_correlation(X_train, out_dir)
    plot_distributions(X_train, out_dir)
    plot_residuals(bundles, X_test, y_test_dict, out_dir)
    plot_error_comparison(bundles, out_dir)
    print_summary(bundles)
    print(f"\nAll player analysis plots saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=settings.models_dir)
    p.add_argument("--out-dir", type=Path, default=Path("analysis"))
    args = p.parse_args()
    main(args.models_dir, args.out_dir)
