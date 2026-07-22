import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_HOME = os.path.join(PROJECT_DIR, "data")

NOTEBOOKS_HOME = os.path.join(PROJECT_DIR, "notebooks")

MODELS_HOME = os.path.join(PROJECT_DIR, "models")

EXPERIMENTS_HOME = os.path.join(PROJECT_DIR, "experiment_runs")
