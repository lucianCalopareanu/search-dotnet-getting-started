from __future__ import annotations

import argparse
from datetime import date
import pathlib

from .config import PipelineConfig, SlateInputs
from .features import engineer_features
from .modeling import fit_models, load_model, predict_slate
from .simulate import SimulationSettings, run_simulations
from .utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EuroLeague advanced betting model")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--slate-date", type=str, required=True)
    parser.add_argument("--slate-input", type=pathlib.Path, required=True)
    parser.add_argument("--n-sims", type=int, default=20000)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing models if available")
    return parser.parse_args()


def load_slate_inputs(path: pathlib.Path) -> SlateInputs:
    payload = load_yaml(path)
    return SlateInputs(
        slate_date=date.fromisoformat(payload["slate_date"]),
        teams=payload.get("teams", {}),
        pace_override=payload.get("pace_override"),
        simulation=payload.get("simulation", {}),
    )


def main() -> None:
    args = parse_args()
    slate_inputs = load_slate_inputs(args.slate_input)
    config = PipelineConfig(
        season=args.season,
        slate_date=slate_inputs.slate_date,
        slate_input=args.slate_input,
        n_simulations=args.n_sims,
        force_refresh=args.force_refresh,
    )

    train_df, slate_df = engineer_features(config, slate_inputs)

    if slate_df.empty:
        raise ValueError("No games found for slate date. Check the schedule/API.")

    spread_model_path = config.models_dir / "spread_model.pkl"
    total_model_path = config.models_dir / "total_model.pkl"

    if args.skip_train and spread_model_path.exists() and total_model_path.exists():
        spread_model = load_model(spread_model_path)
        total_model = load_model(total_model_path)
        metrics = {"info": "Loaded existing models."}
    else:
        spread_model, total_model, metrics = fit_models(config, train_df)

    slate_predictions = predict_slate(spread_model, total_model, slate_df)

    sim_settings = SimulationSettings(
        n_sims=args.n_sims,
        spread_adjustment=slate_inputs.simulation.get("spread_adjustment", 0.0),
        total_adjustment=slate_inputs.simulation.get("total_adjustment", 0.0),
    )
    simulations = run_simulations(slate_predictions, sim_settings)

    output_path = config.models_dir / f"simulations_{slate_inputs.slate_date}.csv"
    simulations.to_csv(output_path, index=False)

    print("Training metrics:", metrics)
    print(simulations.to_string(index=False))
    print(f"Simulation results saved to {output_path}")


if __name__ == "__main__":
    main()
