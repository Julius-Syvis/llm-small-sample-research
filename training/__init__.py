from pathlib import Path

TRAINING_PATH = Path(__file__).resolve().parent

RESULTS_PATH = TRAINING_PATH / ".." / ".results"
CHECKPOINTS_PATH = RESULTS_PATH / "checkpoints"
LOGS_PATH = RESULTS_PATH / "logs"
OUTPUTS_PATH = RESULTS_PATH / "outputs"
