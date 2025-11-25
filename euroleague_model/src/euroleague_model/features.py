from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .config import PipelineConfig, SlateInputs
from .data_sources import (
    download_historical_games,
    fetch_data4basket_ft_to_splits,
    fetch_data4basket_possessions,
    fetch_euroleague_calendar,
    fetch_hackastat_lineups,
    fetch_hackastat_team_ratings,
    fetch_team_metadata,
)
from .utils import haversine_km, write_parquet


def _normalize_date(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val).replace("Z", ""))


def _rest_features(schedule: pd.DataFrame) -> pd.DataFrame:
    records = []
    for team_col in ["HomeTeam", "AwayTeam"]:
        team_games = (
            schedule[["GameCode", "GameDate", team_col]]
            .rename(columns={team_col: "Team"})
            .copy()
        )
        team_games["GameDate"] = team_games["GameDate"].apply(_normalize_date)
        team_games.sort_values("GameDate", inplace=True)
        team_games["prev_date"] = team_games.groupby("Team")["GameDate"].shift(1)
        team_games["rest_days"] = (
            (team_games["GameDate"] - team_games["prev_date"]).dt.days.fillna(7)
        )
        team_games["is_home"] = 1 if team_col == "HomeTeam" else 0
        team_games.rename(
            columns={
                "GameCode": "gamecode",
                "Team": f"{team_col.lower()}_team",
                "rest_days": f"{team_col.lower()}_rest_days",
            },
            inplace=True,
        )
        records.append(team_games[["gamecode", f"{team_col.lower()}_rest_days"]])
    rest = records[0].merge(records[1], on="gamecode", how="outer")
    return rest


def _travel_features(schedule: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    meta = team_meta.rename(columns={"ClubName": "team"}).copy()
    coord_map: dict[str, tuple[float, float]] = {}
    for _, row in meta.iterrows():
        team = row["team"]
        lat = row.get("Latitude")
        lon = row.get("Longitude")
        if pd.notna(lat) and pd.notna(lon):
            coord_map[team] = (float(lat), float(lon))
            continue
        arena = row.get("Arena") or team
        city = row.get("City")
        from .data_sources import fetch_arena_coordinates

        lookup = fetch_arena_coordinates(arena, city)
        if lookup:
            coord_map[team] = lookup

    schedule = schedule.copy()
    schedule["GameDate"] = schedule["GameDate"].apply(_normalize_date)
    schedule.sort_values("GameDate", inplace=True)

    last_coord: dict[str, tuple[float, float]] = {}
    records: list[dict[str, Any]] = []

    for _, row in schedule.iterrows():
        gamecode = row["GameCode"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        arena_coord = coord_map.get(home, coord_map.get(away))

        for team, is_home in ((home, True), (away, False)):
            prev = last_coord.get(team)
            current_coord = coord_map.get(team if is_home else home)
            if not current_coord and arena_coord:
                current_coord = arena_coord
            distance = 0.0
            if prev and current_coord:
                distance = haversine_km(*prev, *current_coord)
            last_coord[team] = current_coord or prev
            records.append(
                {
                    "gamecode": gamecode,
                    f"{'home' if is_home else 'away'}_travel_km": distance,
                }
            )

    travel = pd.DataFrame(records).fillna(0)
    home = travel.groupby("gamecode")["home_travel_km"].max()
    away = travel.groupby("gamecode")["away_travel_km"].max()
    return (
        pd.DataFrame({"gamecode": home.index, "home_travel_km": home.values})
        .merge(
            pd.DataFrame({"gamecode": away.index, "away_travel_km": away.values}),
            on="gamecode",
            how="outer",
        )
        .fillna(0)
    )


def _lineup_usage_scores(lineups: pd.DataFrame) -> pd.Series:
    if lineups.empty:
        return pd.Series(dtype=float)
    df = lineups.copy()
    team_col = next((c for c in ["teamName", "team", "team_code"] if c in df.columns), None)
    poss_col = next((c for c in ["possessions", "poss"] if c in df.columns), None)
    net_col = next((c for c in ["netRating", "net_rating"] if c in df.columns), None)
    if not team_col or not poss_col or not net_col:
        return pd.Series(dtype=float)
    grouped = df.groupby(team_col)
    scores = grouped.apply(
        lambda x: (x[net_col] * (x[poss_col] / x[poss_col].sum())).sum()
    )
    scores.name = "lineup_usage_score"
    return scores


def _pace_blend(team_ratings: pd.DataFrame, possessions: pd.DataFrame) -> pd.Series:
    team_col = next((c for c in ["team", "teamName", "club"] if c in team_ratings.columns), "team")
    pace_col = next((c for c in ["pace", "tempo"] if c in team_ratings.columns), "pace")
    pace_a = team_ratings.set_index(team_col)[pace_col]
    poss_team_col = next((c for c in ["team", "teamName", "club"] if c in possessions.columns), None)
    poss_pace_col = next((c for c in ["pace", "poss_per_game"] if c in possessions.columns), None)
    if poss_team_col and poss_pace_col:
        pace_b = possessions.groupby(poss_team_col)[poss_pace_col].mean()
    else:
        pace_b = pd.Series(dtype=float)
    return (0.6 * pace_a).add(0.4 * pace_b, fill_value=0).rename("pace_blend")


def _ft_to_adjustments(splits: pd.DataFrame) -> pd.DataFrame:
    if splits.empty:
        return pd.DataFrame(columns=["team", "ft_home_adj", "to_home_adj"])
    df = splits.copy()
    team_col = next((c for c in ["team", "teamName", "club"] if c in df.columns), "team")
    home_ft = next((c for c in ["home_ft_pct", "home_ft"] if c in df.columns), None)
    away_ft = next((c for c in ["away_ft_pct", "away_ft"] if c in df.columns), None)
    home_to = next((c for c in ["home_to_pct", "home_to"] if c in df.columns), None)
    away_to = next((c for c in ["away_to_pct", "away_to"] if c in df.columns), None)
    if not home_ft or not away_ft or not home_to or not away_to:
        return pd.DataFrame(columns=["team", "ft_home_adj", "to_home_adj"])
    df["ft_home_adj"] = df[home_ft] - df[away_ft]
    df["to_home_adj"] = df[home_to] - df[away_to]
    return df.set_index(team_col)[["ft_home_adj", "to_home_adj"]]


def engineer_features(config: PipelineConfig, slate_inputs: SlateInputs | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = fetch_euroleague_calendar(config)
    history = download_historical_games(config)
    team_meta = fetch_team_metadata(config)
    team_ratings = fetch_hackastat_team_ratings(config)
    lineups = fetch_hackastat_lineups(config)
    possessions = fetch_data4basket_possessions(config)
    splits = fetch_data4basket_ft_to_splits(config)

    schedule["GameCode"] = schedule["GameCode"].astype(int)
    schedule["GameDate"] = pd.to_datetime(schedule["GameDate"])
    history["gamecode"] = history["gamecode"].astype(int)

    rest = _rest_features(schedule)
    travel = _travel_features(schedule, team_meta)
    lineup_scores = _lineup_usage_scores(lineups)
    pace = _pace_blend(team_ratings, possessions)
    ft_to = _ft_to_adjustments(splits)

    enriched = (
        history.merge(rest, on="gamecode", how="left")
        .merge(travel, on="gamecode", how="left")
    )

    enriched["home_lineup_score"] = enriched["home_team"].map(lineup_scores)
    enriched["away_lineup_score"] = enriched["away_team"].map(lineup_scores)
    enriched["home_pace"] = enriched["home_team"].map(pace)
    enriched["away_pace"] = enriched["away_team"].map(pace)
    enriched["home_ft_adj"] = enriched["home_team"].map(ft_to["ft_home_adj"])
    enriched["away_ft_adj"] = enriched["away_team"].map(ft_to["ft_home_adj"])
    enriched["home_to_adj"] = enriched["home_team"].map(ft_to["to_home_adj"])
    enriched["away_to_adj"] = enriched["away_team"].map(ft_to["to_home_adj"])

    enriched["pace_expectation"] = (
        enriched[["home_pace", "away_pace"]].mean(axis=1)
    )
    enriched["expected_possession_margin"] = (
        enriched["home_lineup_score"].fillna(0)
        - enriched["away_lineup_score"].fillna(0)
    )

    feature_cols = [
        "gamecode",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
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
    processed = enriched[feature_cols].copy()
    processed.dropna(subset=["home_score", "away_score"], inplace=True)
    processed["spread"] = processed["home_score"] - processed["away_score"]
    processed["total"] = processed["home_score"] + processed["away_score"]
    processed["target_spread"] = processed["spread"]
    processed["target_total"] = processed["total"]
    processed_path = config.cache_processed / f"features_{config.season}.parquet"
    write_parquet(processed, processed_path)

    if not slate_inputs:
        return processed, pd.DataFrame()

    slate_games = schedule[schedule["GameDate"].dt.date == slate_inputs.slate_date].copy()
    slate_games.rename(columns={"GameCode": "gamecode"}, inplace=True)
    slate_games = slate_games.merge(rest, on="gamecode", how="left").merge(
        travel, on="gamecode", how="left"
    )

    slate_games["home_lineup_score"] = slate_games["HomeTeam"].map(lineup_scores)
    slate_games["away_lineup_score"] = slate_games["AwayTeam"].map(lineup_scores)
    slate_games["home_pace"] = slate_games["HomeTeam"].map(pace)
    slate_games["away_pace"] = slate_games["AwayTeam"].map(pace)
    slate_games["home_ft_adj"] = slate_games["HomeTeam"].map(ft_to["ft_home_adj"])
    slate_games["away_ft_adj"] = slate_games["AwayTeam"].map(ft_to["ft_home_adj"])
    slate_games["home_to_adj"] = slate_games["HomeTeam"].map(ft_to["to_home_adj"])
    slate_games["away_to_adj"] = slate_games["AwayTeam"].map(ft_to["to_home_adj"])

    slate_games["pace_expectation"] = (
        slate_games[["home_pace", "away_pace"]].mean(axis=1)
    )
    slate_games["expected_possession_margin"] = (
        slate_games["home_lineup_score"].fillna(0)
        - slate_games["away_lineup_score"].fillna(0)
    )

    slate_features = slate_games.rename(
        columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "GameDate": "date",
        }
    )

    slate_features = _apply_manual_inputs(slate_features, slate_inputs)

    return processed, slate_features[
        [
            "gamecode",
            "date",
            "home_team",
            "away_team",
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
            "home_injury_usage_penalty",
            "away_injury_usage_penalty",
        ]
    ]


def _apply_manual_inputs(df: pd.DataFrame, slate_inputs: SlateInputs) -> pd.DataFrame:
    df = df.copy()
    injury_penalty = {}
    rest_overrides = {}
    travel_overrides = {}

    for team, payload in slate_inputs.teams.items():
        injuries = payload.get("injuries", [])
        penalty = sum(item.get("usage_penalty", 0.0) for item in injuries if item.get("status") != "active")
        injury_penalty[team] = penalty
        if payload.get("rest_override_days") is not None:
            rest_overrides[team] = payload["rest_override_days"]
        if payload.get("travel_override_km") is not None:
            travel_overrides[team] = payload["travel_override_km"]

    df["home_injury_usage_penalty"] = df["home_team"].map(injury_penalty).fillna(0.0)
    df["away_injury_usage_penalty"] = df["away_team"].map(injury_penalty).fillna(0.0)

    df.loc[df["home_team"].isin(rest_overrides), "home_rest_days"] = df["home_team"].map(rest_overrides)
    df.loc[df["away_team"].isin(rest_overrides), "away_rest_days"] = df["away_team"].map(rest_overrides)

    df.loc[df["home_team"].isin(travel_overrides), "home_travel_km"] = df["home_team"].map(travel_overrides)
    df.loc[df["away_team"].isin(travel_overrides), "away_travel_km"] = df["away_team"].map(travel_overrides)

    df["home_lineup_score"] = df["home_lineup_score"] - df["home_injury_usage_penalty"]
    df["away_lineup_score"] = df["away_lineup_score"] - df["away_injury_usage_penalty"]

    if slate_inputs.pace_override is not None:
        df["pace_expectation"] = slate_inputs.pace_override

    return df
