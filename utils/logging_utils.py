import logging
import os
from datetime import datetime

from training import LOGS_PATH


def get_fn(experiment_name: str) -> str:
    experiment_name = '_'.join(experiment_name.split(' '))
    now = datetime.now().strftime("%m-%d.%H-%M-%S")
    return f"{experiment_name}_{now}"


def setup_logging(experiment_name: str):
    fn = get_fn(experiment_name)

    logs_path = LOGS_PATH / 'logging'
    os.makedirs(logs_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_path / f"{fn}.log"),
            logging.StreamHandler()
        ]
    )

