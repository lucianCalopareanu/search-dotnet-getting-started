from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from datetime import date

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

EUROLEAGUE_BASE = "https://live.euroleague.net/api"
HACKASTAT_BASE = "https://api.hackastat.eu/v1/euroleague"
DATA4BASKET_BASE = "https://data4basket.com/api/euroleague"
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"

DEFAULT_SEASON = 2024
DEFAULT_SIMULATIONS = 20000


@dataclass
class PipelineConfig:
    season: int = DEFAULT_SEASON
    slate_date: date | None = None
    slate_input: pathlib.Path | None = None
    cache_raw: pathlib.Path = RAW_DIR
    cache_processed: pathlib.Path = PROCESSED_DIR
    models_dir: pathlib.Path = MODELS_DIR
    n_simulations: int = DEFAULT_SIMULATIONS
    force_refresh: bool = False

    def __post_init__(self) -> None:
        self.cache_raw.mkdir(parents=True, exist_ok=True)
        self.cache_processed.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SlateInputs:
    slate_date: date
    teams: dict[str, dict]
    pace_override: float | None = None
    simulation: dict[str, float] = field(default_factory=dict)
