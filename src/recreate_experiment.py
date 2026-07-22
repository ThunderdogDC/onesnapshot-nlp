import argparse
import os

from src.locations import EXPERIMENTS_HOME
from src.main import run_experiment


def run_experiment_from_registry(experiment_key):
    """
    Recreates an experiment from the registry file provided. Writes the experiment's outputs to a user's disk. Raises a ValueError

    Args:
        experiment_key: str
            The experiment to recreate. Note: This uses the experiment's key which is the experiment's full name

    Returns:
        None. This function simply helps to recreate the outputs from an experiment on registry (e.g. predictions.csv, eval_df.csv)
        on a user's laptop disk.
    """
    experiment_inputs_path = os.path.join(EXPERIMENTS_HOME, experiment_key, "inputs/")

    if not os.path.exists(experiment_inputs_path):
        raise ValueError(f"Experiment '{experiment_key}' not found.")

    # Log the paths to the input documents for that experiment
    experiment_config_path = os.path.join(
        experiment_inputs_path, "experiment_config.yaml"
    )
    experiment_mapping_path = os.path.join(experiment_inputs_path, "mapping.yaml")
    experiment_workstreams_path = os.path.join(
        experiment_inputs_path, "workstreams.yaml"
    )

    # Recreate the experiment
    run_experiment(
        experiment_config_path, experiment_mapping_path, experiment_workstreams_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreating experiment already done by others."
    )

    parser.add_argument(
        "-e",
        "--experiment_key",
        required=True,
        help="Name of the experiment to recreate",
    )

    args = parser.parse_args()

    run_experiment_from_registry(experiment_key=args.experiment_key)
