from __future__ import annotations

import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig

FEATURE_COLUMNS = [
    "home_rest_days",
    "away_rest_days",
    "home_travel_km",
    "away_travel_km",
    "home_lineup_score",
    "away_lineup_score",
    "home_pace",
    "away_pace",
    "home_ft_adj",
    "away_ft_adj",
    "home_to_adj",
    "away_to_adj",
    "pace_expectation",
    "expected_possession_margin",
]


def _make_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    gbr = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", gbr),
        ]
    )


def fit_models(config: PipelineConfig, train_df: pd.DataFrame) -> Tuple[Pipeline, Pipeline, dict]:
    train_df = train_df.copy()
    train_df.fillna(0, inplace=True)

    spread_model = _make_model()
    total_model = _make_model()

    spread_model.fit(train_df[FEATURE_COLUMNS], train_df["target_spread"])
    total_model.fit(train_df[FEATURE_COLUMNS], train_df["target_total"])

    metrics = {
        "spread_mae": mean_absolute_error(
            train_df["target_spread"],
            spread_model.predict(train_df[FEATURE_COLUMNS]),
        ),
        "total_mae": mean_absolute_error(
            train_df["target_total"],
            total_model.predict(train_df[FEATURE_COLUMNS]),
        ),
    }

    save_model(spread_model, config.models_dir / "spread_model.pkl")
    save_model(total_model, config.models_dir / "total_model.pkl")
    return spread_model, total_model, metrics


def save_model(model: Pipeline, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: pathlib.Path) -> Pipeline:
    return joblib.load(path)


def predict_slate(
    spread_model: Pipeline,
    total_model: Pipeline,
    slate_df: pd.DataFrame,
) -> pd.DataFrame:
    slate_df = slate_df.copy()
    slate_df.fillna(0, inplace=True)
    slate_df["predicted_spread"] = spread_model.predict(slate_df[FEATURE_COLUMNS])
    slate_df["predicted_total"] = total_model.predict(slate_df[FEATURE_COLUMNS])
    return slate_df
