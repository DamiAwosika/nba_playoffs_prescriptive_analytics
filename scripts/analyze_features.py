"""Feature-importance & calibration analysis for the trained models.

Run AFTER `python -m nba_ml.training.train`. Produces six PNGs in analysis/:
  feature_importance.png       native importance + permutation importance per model
  correlation.png              feature correlation heatmap (spot redundancy)
  distributions.png            per-feature histograms (spot outliers / scale issues)
  feature_vs_target.png        decile plots — is each feature monotonic in P(home win)?
  calibration_and_confusion.png  reliability diagrams + confusion matrices

Also prints ranked importance tables and concrete next-step suggestions.

Usage:
  python scripts/analyze_features.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from nba_ml.config import settings
from nba_ml.db.base import SessionLocal
from nba_ml.features.matchup import FEATURE_COLUMNS
from nba_ml.training.train import build_training_frame


# ---- model loading ----

def load_models(models_dir: Path) -> dict[str, dict]:
    bundles: dict[str, dict] = {}
    for path in sorted(models_dir.glob("*_v1.joblib")):
        bundle = joblib.load(path)
        # Stack model operates on base-model probabilities, not raw game features
        # — its feature importance is a different analysis (the meta weights),
        # already printed during training. Skip here.
        if bundle.get("model_type") == "stack":
            continue
        bundles[bundle["model_type"]] = bundle
    return bundles


def _inner_estimators(calibrated_model) -> list:
    """CalibratedClassifierCV holds N inner base estimators (one per CV fold).
    Return them so we can average their per-fold importances."""
    return [cc.estimator for cc in calibrated_model.calibrated_classifiers_]


def native_importance(name: str, calibrated_model) -> np.ndarray:
    """Per-model importance, averaged across the calibration folds.
    logreg -> |standardized coefficient|; rf/xgb -> built-in feature_importances_."""
    inners = _inner_estimators(calibrated_model)
    rows = []
    for est in inners:
        if name == "logreg":
            rows.append(np.abs(est.named_steps["clf"].coef_[0]))
        else:
            rows.append(est.feature_importances_)
    return np.mean(rows, axis=0)


# ---- plots ----

def plot_feature_importance(bundles, X_test, y_test, out_dir):
    n = len(bundles)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 12))
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, b) in enumerate(bundles.items()):
        # Native
        imp = native_importance(name, b["model"])
        order = np.argsort(imp)
        ax = axes[0, i]
        ax.barh([FEATURE_COLUMNS[j] for j in order], imp[order], color="steelblue")
        ax.set_title(f"{name} — native importance\n"
                     f"({'|coef|' if name == 'logreg' else 'gain/gini'})")
        ax.tick_params(axis="y", labelsize=7)

        # Permutation (model-agnostic, on holdout — most reliable)
        pi = permutation_importance(
            b["model"], X_test, y_test,
            scoring="neg_log_loss", n_repeats=8, random_state=42, n_jobs=-1,
        )
        order2 = np.argsort(pi.importances_mean)
        ax = axes[1, i]
        ax.barh(
            [FEATURE_COLUMNS[j] for j in order2],
            pi.importances_mean[order2],
            xerr=pi.importances_std[order2],
            color="darkred",
        )
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{name} — permutation importance\n(higher = bigger log_loss hit when shuffled)")
        ax.tick_params(axis="y", labelsize=7)

        # Print ranking
        top = np.argsort(pi.importances_mean)[::-1]
        print(f"\n{name} permutation importance (top 10):")
        for rank, j in enumerate(top[:10], 1):
            print(f"  {rank:2}. {FEATURE_COLUMNS[j]:<28} {pi.importances_mean[j]:+.4f} "
                  f"± {pi.importances_std[j]:.4f}")
        zero_or_neg = [FEATURE_COLUMNS[j] for j in range(len(FEATURE_COLUMNS))
                       if pi.importances_mean[j] <= 0]
        if zero_or_neg:
            print(f"  features with ≤0 permutation importance: {zero_or_neg}")

    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close()


def plot_correlation(X, out_dir):
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Feature correlation matrix\n(|r|>0.8 pairs are largely redundant)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation.png", dpi=120)
    plt.close()

    pairs = []
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > 0.7:
                pairs.append((c1, cols[j], v))
    if pairs:
        print("\nHighly correlated feature pairs (|r| > 0.7):")
        for c1, c2, v in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {v:+.2f}  {c1} <-> {c2}")
    else:
        print("\nNo feature pairs with |r| > 0.7 — no obvious redundancy.")


def plot_distributions(X, out_dir):
    n = len(X.columns)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3 * rows))
    axes = axes.flatten()
    for i, col in enumerate(X.columns):
        axes[i].hist(X[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(len(X.columns), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Per-feature distributions", y=1.001, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "distributions.png", dpi=120)
    plt.close()


def plot_feature_vs_target(X, y, out_dir):
    """Bin each feature into deciles, plot mean(y) per bin. A clean monotonic
    curve = strong signal. A flat line = no signal. A zig-zag = noise that
    tree models will overfit."""
    n = len(X.columns)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3 * rows))
    axes = axes.flatten()
    base_rate = float(np.mean(y))
    monotonicity_scores: list[tuple[str, float]] = []

    for i, col in enumerate(X.columns):
        try:
            bins = pd.qcut(X[col], 10, duplicates="drop")
            means = pd.Series(y, index=X.index).groupby(bins, observed=True).mean()
            xs = np.arange(len(means))
            axes[i].plot(xs, means.values, "o-", color="darkred")
            axes[i].axhline(base_rate, color="gray", linestyle="--", alpha=0.5,
                            label=f"base {base_rate:.2f}")
            axes[i].set_title(col, fontsize=9)
            axes[i].set_ylabel("P(home won)", fontsize=7)
            axes[i].tick_params(labelsize=7)
            # Spearman-like monotonicity score: rank-correlation with bin index
            mono = float(np.corrcoef(xs, means.values)[0, 1])
            monotonicity_scores.append((col, mono))
        except Exception:
            axes[i].set_title(f"{col} (skipped)", fontsize=9)
    for j in range(len(X.columns), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Feature value vs P(home won) by decile", y=1.001, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_vs_target.png", dpi=120)
    plt.close()

    monotonicity_scores.sort(key=lambda x: -abs(x[1]))
    print("\nFeature monotonicity (corr between decile rank and win rate):")
    for col, m in monotonicity_scores:
        flag = "  weak" if abs(m) < 0.3 else "      "
        print(f"  {flag} {m:+.3f}  {col}")


def plot_calibration_and_confusion(bundles, X_test, y_test, out_dir):
    n = len(bundles)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, b) in enumerate(bundles.items()):
        proba = b["model"].predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        frac, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax = axes[0, i]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax.plot(mean_pred, frac, "o-", label=name, color="darkblue")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual win rate")
        ax.set_title(f"{name} calibration\n(closer to dashed = better-calibrated)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend()

        cm = confusion_matrix(y_test, preds)
        ax = axes[1, i]
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=14,
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["away win", "home win"])
        ax.set_yticklabels(["away win", "home win"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name} confusion matrix")

    plt.tight_layout()
    plt.savefig(out_dir / "calibration_and_confusion.png", dpi=120)
    plt.close()


def print_recommendations():
    print("""
Concrete next steps to lift RF/XGB above logreg:

1. MORE DATA (largest single lever).
   python scripts/run_etl.py --start 2022-10-18 --end 2026-04-27
   Tree models eat data. ~5000 games tends to be where they decisively beat
   linear models on this kind of problem.

2. WIDER TUNING.
   python -m nba_ml.training.train --tune --n-iter 100
   Your XGB search space has ~5000 combos; sampling only 20-25 leaves a lot
   of good regions unexplored.

3. PRUNE REDUNDANT FEATURES.
   See correlation.png. If you find pairs with |r| > 0.9 (e.g. off_rating
   ↔ net_rating), drop one. Trees handle collinearity but it bloats the
   number of splits considered.

4. PRUNE NOISY FEATURES.
   See feature_importance.png — bottom panel (permutation). Any feature
   with permutation importance ≤ 0 is actively hurting log_loss. Remove
   those from FEATURE_COLUMNS in matchup.py and retrain.

5. CHECK MONOTONICITY.
   See feature_vs_target.png. If a feature is non-monotonic (zig-zag),
   tree models can fit the noise. Consider transforming or dropping it.

6. XGB-SPECIFIC TWEAKS.
   - Lower learning_rate (0.005-0.01) with much higher n_estimators (1500+).
   - Add early_stopping_rounds with a held-out validation slice.
   - Try monotonic_constraints={"home_net_roll10": 1, "away_net_roll10": -1}
     to inject prior knowledge (more home net rating -> higher P(home wins)).

7. FEATURE INTERACTIONS.
   Add explicit differential features: (home_net_roll10 - away_net_roll10),
   (home_pace_roll5 - away_pace_roll5). Trees can derive these themselves
   given enough depth + data, but explicit features short-circuit the search.

8. CALIBRATION CHECK.
   See calibration_and_confusion.png. If a model's curve hugs the diagonal,
   probabilities are trustworthy. If it sags, predictions are overconfident
   (or underconfident — opposite direction).
""")


# ---- main ----

def main(models_dir: Path, out_dir: Path, feature_version: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_models(models_dir)
    if not bundles:
        raise SystemExit(f"No models found in {models_dir}. Train first.")
    print(f"Loaded models: {list(bundles)}")

    db = SessionLocal()
    try:
        df = build_training_frame(db, feature_version)
    finally:
        db.close()
    if df.empty:
        raise SystemExit("No training rows. Run scripts/run_etl.py first.")

    df = df.sort_values("game_date").reset_index(drop=True)
    split = max(1, int(len(df) * 0.8))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["home_won"].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["home_won"].values
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    plot_feature_importance(bundles, X_test, y_test, out_dir)
    plot_correlation(X_train, out_dir)
    plot_distributions(X_train, out_dir)
    plot_feature_vs_target(X_train, y_train, out_dir)
    plot_calibration_and_confusion(bundles, X_test, y_test, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
    print_recommendations()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=settings.models_dir)
    p.add_argument("--out-dir", type=Path, default=Path("analysis"))
    p.add_argument("--feature-version", default=settings.feature_version)
    args = p.parse_args()
    main(args.models_dir, args.out_dir, args.feature_version)
