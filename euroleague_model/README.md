## EuroLeague Advanced Betting Model

This package builds and runs a possession-based predictive model for EuroLeague men's games with spread and total projections. The pipeline gathers freely available data, engineers contextual adjustments (rosters, rest, travel, pace, lineup usage, FT%/TO% splits), fits ensemble models, and runs a Monte Carlo simulation to turn expected margins into actionable distributions.

### Contents

- `requirements.txt` – Python dependencies.
- `src/euroleague_model/` – Pipeline source code (config, data acquisition, feature engineering, modeling, simulation).
- `data/` – Cached raw/processed data.
- `models/` – Serialized model artifacts.
- `injury_rest_travel_template.yaml` – Editable slate-level inputs (injuries, travel overrides, rest notes).

### Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# populate manual slate inputs
cp injury_rest_travel_template.yaml slate_inputs.yaml
# edit slate_inputs.yaml with current injuries, rest, travel flags

python -m euroleague_model.pipeline \
  --season 2024 \
  --slate-date 2025-01-15 \
  --slate-input slate_inputs.yaml \
  --n-sims 20000
```

Outputs:

- `data/raw/*.json|csv` – cached API responses.
- `data/processed/features_<season>.parquet` – modeling table.
- `models/spread_model.pkl`, `models/total_model.pkl`.
- `models/simulations_<slate-date>.csv` – Monte Carlo outcomes (spread/total distributions).
- Console summary with probabilities for ATS, OU, alt-lines.

### Data Sources (Free)

1. **EuroLeague Live API** (`https://live.euroleague.net/api/`): schedule, box scores, play-by-play, rosters, venues.
2. **Hack-a-Stat** (`https://api.hackastat.eu/v1/euroleague/...`): pace, offensive/defensive ratings, lineup possessions.
3. **Data4Basket** (`https://data4basket.com/api/euroleague/...`): advanced lineup usage, possession logs, FT/TO splits.
4. **OpenStreetMap Nominatim** (no API key): arena coordinates for travel distance estimation.
5. **Manual Slate Inputs** (YAML): injury statuses, travel quirks, national-league rest adjustments. Only file that needs frequent editing.

All APIs are rate-limited but free; caching avoids repeated calls.

### Model Highlights

- **Roster/Injury Adjustments** – current availability & usage from slate YAML + latest box scores.
- **Rest Days (EL + domestic)** – automatically computed from EuroLeague calendar; manual overrides allow adding domestic fixtures.
- **Travel Impact** – haversine distance between last venue and next venue with cumulative fatigue scoring.
- **Lineup Usage Weighting** – lineup possession shares from Hack-a-Stat/Data4Basket drive effective team ratings.
- **Pace Adjustments** – harmonizes Hack-a-Stat pace with possession logs to estimate slate pace.
- **FT% / TO% Home-Away Splits** – rolling splits transformed into matchup adjustments.
- **Expected Possession Margin** – predicted net rating * expected possessions forms the base spread; totals derived from offensive ratings & tempo.
- **Monte Carlo Simulation** – possession-level simulation generates ATS/OU probabilities, alternative lines, and distribution percentiles.

### Workflow

1. **Collect** – `data_sources.py` pulls latest data + caches locally.
2. **Enrich** – merge injuries, rest, travel, pace, lineup usage in `features.py`.
3. **Train** – `modeling.py` fits Gradient Boosting regressors for spread & total using historical games.
4. **Simulate** – `simulate.py` converts predictions into distribution via Monte Carlo.
5. **Report** – `pipeline.py` orchestrates everything and prints/saves outputs.

Re-run the pipeline whenever you need new projections. You only need to edit `slate_inputs.yaml` (from the provided template) for the latest injuries/rest/travel context before executing.

### Updating Slate Inputs (Only Manual Step)

1. Copy `injury_rest_travel_template.yaml` to a dated file (e.g., `slate_inputs_2025-01-15.yaml`).
2. For each team on the slate:
   - List current injuries with `status` (`out`, `doubtful`, `questionable`, etc.) and an estimated `usage_penalty` (share of team possessions).
   - Add `rest_override_days` when the team played a domestic league game not in the EuroLeague calendar.
   - Add `travel_override_km` if the team had an unusual travel leg (charter, trans-Atlantic, neutral court).
   - Use `domestic_fixture` to document the competition/date for traceability (optional, purely informational).
3. Optionally set `pace_override` or tweak `simulation.spread_adjustment` / `simulation.total_adjustment` if you want subjective nudges.
4. Run the pipeline with the YAML path; all other data stays automated.

### Monte Carlo Output Fields

- `predicted_spread`, `predicted_total` – model point estimates after manual adjustments.
- `ats_prob_home`, `ats_prob_away` – probability each side covers a zero line (use to compare to markets).
- `over_prob`, `under_prob` – probability game lands over/under the projected total.
- `margin_p10/p50/p90`, `total_p10/p50/p90` – percentile bands from the simulation distribution.
- `expected_possessions` – tempo context (possession-based margin is `predicted_spread / expected_possessions`).

### Notes on Free Data Sources

- All HTTP calls include a User-Agent and a gentle delay. The pipeline caches to `data/raw` so you only re-download when `--force-refresh` is supplied.
- Hack-a-Stat plus Data4Basket cover lineup usage, pace, possessions, FT%/TO% splits, and play-by-play-derived stats. Both are free with attribution.
- Arena coordinates default to EuroLeague metadata; missing values are auto-looked up via OpenStreetMap's Nominatim endpoint (no key required).

### Troubleshooting

- **No games returned** – verify `--slate-date` matches EuroLeague schedule format (YYYY-MM-DD) and that the round exists.
- **API throttling** – increase the `time.sleep` interval inside `data_sources._download_json` or run with cached files (default).
- **Missing columns** – the feature builders map several possible column names from Hack-a-Stat/Data4Basket. If the upstream schemas change, adjust the selectors in `features.py`.
- **Model reuse** – pass `--skip-train` to reuse stored `models/*.pkl` and only update slate inputs.
