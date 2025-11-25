from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class SimulationSettings:
    n_sims: int = 20000
    spread_adjustment: float = 0.0
    total_adjustment: float = 0.0


def run_simulations(
    slate: pd.DataFrame,
    settings: SimulationSettings,
) -> pd.DataFrame:
    sims = []
    for _, row in slate.iterrows():
        sims.append(_simulate_row(row, settings))
    return pd.DataFrame(sims)


def _simulate_row(row: pd.Series, settings: SimulationSettings) -> Dict[str, float]:
    total_mean = row["predicted_total"] + settings.total_adjustment
    spread_mean = row["predicted_spread"] + settings.spread_adjustment
    home_mean = (total_mean + spread_mean) / 2
    away_mean = total_mean - home_mean

    possessions = max(row.get("pace_expectation", 70), 60)
    sigma = max(np.sqrt(possessions) * 1.15, 6)

    home_draws = np.random.normal(home_mean, sigma, settings.n_sims)
    away_draws = np.random.normal(away_mean, sigma, settings.n_sims)
    margin = home_draws - away_draws
    total = home_draws + away_draws

    ats_prob = (margin > 0).mean()
    ou_prob = (total > total_mean).mean()

    return {
        "gamecode": row["gamecode"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "predicted_spread": spread_mean,
        "predicted_total": total_mean,
        "expected_possessions": possessions,
        "ats_prob_home": ats_prob,
        "ats_prob_away": 1 - ats_prob,
        "over_prob": ou_prob,
        "under_prob": 1 - ou_prob,
        "margin_p10": np.percentile(margin, 10),
        "margin_p50": np.percentile(margin, 50),
        "margin_p90": np.percentile(margin, 90),
        "total_p10": np.percentile(total, 10),
        "total_p50": np.percentile(total, 50),
        "total_p90": np.percentile(total, 90),
    }
