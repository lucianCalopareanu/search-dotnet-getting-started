from __future__ import annotations

import hashlib
import pathlib
import time
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from .config import (
    DATA4BASKET_BASE,
    EUROLEAGUE_BASE,
    HACKASTAT_BASE,
    NOMINATIM_ENDPOINT,
    PipelineConfig,
)
from . import utils

USER_AGENT = "EuroLeagueModel/1.0 (contact: analytics@example.com)"
SESSION = requests.Session()


def _cache_file(
    base_dir: pathlib.Path, name: str, extension: str = "json"
) -> pathlib.Path:
    safe = name.replace("/", "_")
    return base_dir / f"{safe}.{extension}"


def _download_json(
    url: str,
    cache: pathlib.Path,
    params: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> Any:
    if cache.exists() and not force_refresh:
        return utils.load_json(cache)

    response = SESSION.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    utils.save_json(cache, data)
    time.sleep(0.35)  # stay polite to public APIs
    return data


def fetch_euroleague_calendar(config: PipelineConfig) -> pd.DataFrame:
    season_code = f"E{config.season}"
    cache = _cache_file(config.cache_raw, f"calendar_{season_code}")
    data = _download_json(
        f"{EUROLEAGUE_BASE}/Calendar", cache, {"SeasonCode": season_code}, config.force_refresh
    )
    return pd.DataFrame(data.get("Games", []))


def fetch_euroleague_boxscore(gamecode: int, config: PipelineConfig) -> dict[str, Any]:
    season_code = f"E{config.season}"
    cache = _cache_file(config.cache_raw, f"boxscore_{season_code}_{gamecode}")
    return _download_json(
        f"{EUROLEAGUE_BASE}/Boxscore",
        cache,
        {"gamecode": gamecode, "seasoncode": season_code},
        config.force_refresh,
    )


def fetch_euroleague_pbp(gamecode: int, config: PipelineConfig) -> dict[str, Any]:
    season_code = f"E{config.season}"
    cache = _cache_file(config.cache_raw, f"pbp_{season_code}_{gamecode}")
    return _download_json(
        f"{EUROLEAGUE_BASE}/PlayByPlay",
        cache,
        {"gamecode": gamecode, "seasoncode": season_code},
        config.force_refresh,
    )


def fetch_team_metadata(config: PipelineConfig) -> pd.DataFrame:
    cache = _cache_file(config.cache_raw, "team_metadata")
    data = _download_json(f"{EUROLEAGUE_BASE}/Teams", cache, None, config.force_refresh)
    return pd.DataFrame(data.get("Teams", []))


def fetch_hackastat_team_ratings(config: PipelineConfig) -> pd.DataFrame:
    cache = _cache_file(config.cache_raw, f"hackastat_team_{config.season}")
    data = _download_json(
        f"{HACKASTAT_BASE}/seasons/{config.season}/teams/advanced",
        cache,
        None,
        config.force_refresh,
    )
    return pd.DataFrame(data)


def fetch_hackastat_lineups(config: PipelineConfig) -> pd.DataFrame:
    cache = _cache_file(config.cache_raw, f"hackastat_lineups_{config.season}")
    data = _download_json(
        f"{HACKASTAT_BASE}/seasons/{config.season}/lineups",
        cache,
        None,
        config.force_refresh,
    )
    return pd.DataFrame(data)


def fetch_data4basket_possessions(config: PipelineConfig) -> pd.DataFrame:
    cache = _cache_file(config.cache_raw, f"data4basket_possessions_{config.season}")
    data = _download_json(
        f"{DATA4BASKET_BASE}/seasons/{config.season}/possessions",
        cache,
        None,
        config.force_refresh,
    )
    return pd.json_normalize(data, max_level=1)


def fetch_data4basket_ft_to_splits(config: PipelineConfig) -> pd.DataFrame:
    cache = _cache_file(config.cache_raw, f"data4basket_ft_to_{config.season}")
    data = _download_json(
        f"{DATA4BASKET_BASE}/seasons/{config.season}/splits",
        cache,
        None,
        config.force_refresh,
    )
    return pd.DataFrame(data)


def fetch_arena_coordinates(arena_name: str, city: str | None = None) -> tuple[float, float] | None:
    params = {"q": f"{arena_name} {city or ''}".strip(), "format": "json", "limit": 1}
    # hashed cache per query
    hashed = hashlib.sha256(params["q"].encode()).hexdigest()[:10]
    cache_dir = pathlib.Path(__file__).resolve().parents[2] / "data/raw"
    cache = _cache_file(cache_dir, f"nominatim_{hashed}")
    try:
        data = _download_json(NOMINATIM_ENDPOINT, cache, params, False)
    except requests.HTTPError:
        return None
    if not data:
        return None
    lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
    return lat, lon


def download_historical_games(config: PipelineConfig) -> pd.DataFrame:
    """Download all available EuroLeague games for the season."""
    schedule = fetch_euroleague_calendar(config)
    records: list[dict[str, Any]] = []
    for _, row in tqdm(schedule.iterrows(), total=len(schedule), desc="Boxscores"):
        gamecode = int(row.get("GameCode"))
        box = fetch_euroleague_boxscore(gamecode, config)
        if not box:
            continue
        records.append(
            {
                "gamecode": gamecode,
                "date": row.get("GameDate"),
                "home_team": row.get("HomeTeam"),
                "away_team": row.get("AwayTeam"),
                "home_score": row.get("HomeScore"),
                "away_score": row.get("AwayScore"),
                "boxscore": box,
            }
        )
    return pd.DataFrame(records)
