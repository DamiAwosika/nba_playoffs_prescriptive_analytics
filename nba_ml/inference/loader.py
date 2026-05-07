from __future__ import annotations
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from nba_ml.config import settings

DEFAULT_MODEL_NAME = "logreg"


class LoadedModel:
    def __init__(self, bundle: dict, name: str):
        self.name = name
        self.model = bundle["model"]
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.feature_version: str = bundle["feature_version"]
        self.version: str = bundle["version"]

    def predict_proba(self, features: dict[str, float]) -> float:
        # DataFrame slicing by columns means features dict can carry extra keys
        # (e.g., vegas_home_win_prob) that this model doesn't use — they're
        # silently dropped here. Keeps the build_matchup_features dict universal.
        X = pd.DataFrame([features], columns=self.feature_columns)
        return float(self.model.predict_proba(X)[0, 1])


class StackedModel:
    """Loads stack_v1.joblib + every base model it references, then exposes
    the same predict_proba(features_dict) interface as LoadedModel."""

    def __init__(self, bundle: dict, base_models: dict[str, LoadedModel]):
        self.name = bundle.get("model_type", "stack")
        self.meta = bundle["meta_model"]
        self.base_model_names: list[str] = bundle["base_model_names"]
        self.base_models = base_models
        self.feature_version: str = bundle["feature_version"]
        self.version: str = bundle["version"]
        self.feature_columns: list[str] = bundle["feature_columns"]

    def predict_proba(self, features: dict[str, float]) -> float:
        base_probas = {
            n: self.base_models[n].predict_proba(features)
            for n in self.base_model_names
        }
        x = pd.DataFrame([{c: base_probas[c] for c in self.feature_columns}])
        return float(self.meta.predict_proba(x)[0, 1])


class VotingEnsemble:
    """Soft-vote ensemble: mean of member predict_proba outputs.

    Bundle stores only member names — the actual member models are loaded via
    get_loaded_model and held by reference. No retraining at load time.
    """

    def __init__(self, bundle: dict, members: list):
        self.name = bundle.get("name", "ensemble")
        self.members = members
        self.member_names: list[str] = bundle["member_names"]
        self.feature_version: str = bundle["feature_version"]
        self.version: str = bundle["version"]
        # All members share these — pick the first as authoritative.
        self.feature_columns: list[str] = bundle.get(
            "feature_columns", members[0].feature_columns,
        )

    def predict_proba(self, features: dict[str, float]) -> float:
        return float(np.mean([m.predict_proba(features) for m in self.members]))


@lru_cache(maxsize=16)
def get_loaded_model(name: str = DEFAULT_MODEL_NAME):
    path = settings.models_dir / f"{name}_v1.joblib"
    bundle = joblib.load(path)
    mtype = bundle.get("model_type")
    if mtype == "stack":
        bases = {n: get_loaded_model(n) for n in bundle["base_model_names"]}
        return StackedModel(bundle, bases)
    if mtype == "ensemble":
        members = [get_loaded_model(n) for n in bundle["member_names"]]
        return VotingEnsemble(bundle, members)
    return LoadedModel(bundle, name)


def list_available_models() -> list[str]:
    """Names of all trained *matchup* model bundles in models_dir.

    Excludes player_* regression models — those are loaded separately via
    nba_ml.inference.player_predict and don't expose predict_proba.
    """
    d = settings.models_dir
    if not d.exists():
        return []
    return sorted(
        {p.stem.rsplit("_", 1)[0] for p in d.glob("*_v1.joblib")
         if not p.name.startswith("player_")}
    )
