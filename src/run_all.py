"""Run the label build, exploration, tests, models, and experiment design."""

import argparse

from src.build import build_dataset
from src.config import OUTPUTS
from src.data import load_analysis
from src.eda import run_eda
from src.experiment import run_experiment
from src.hypotheses import run_hypotheses
from src.models import run_models


def main(force_build: bool = False, force: bool = False) -> None:
    build_dataset(force=force_build)
    df, _ = load_analysis()
    if force or not (OUTPUTS / "eda_summary.json").exists():
        run_eda(df)
    if force or not (OUTPUTS / "hypotheses.json").exists():
        run_hypotheses(df)
    if force or not (OUTPUTS / "model_metrics.json").exists():
        run_models(df)
    if force or not (OUTPUTS / "experiment.json").exists():
        run_experiment(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the unanswered-question analysis.")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute every stage.")
    args = parser.parse_args()
    main(force_build=args.force_build, force=args.force)
