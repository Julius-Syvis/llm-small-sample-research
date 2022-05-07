from pathlib import Path

TRAINING_PATH = Path(__file__).resolve().parent
RESULTS_PATH = TRAINING_PATH / ".." / ".results"


def get_checkpoints_path(path: Path) -> Path:
    return path / "checkpoints"


def get_logs_path(path: Path) -> Path:
    return path / "logs"


def get_outputs_path(path: Path) -> Path:
    return path / "outputs"
