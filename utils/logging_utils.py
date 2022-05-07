import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.core import ModelFactory
from tasks.core import Task
from training import RESULTS_PATH
from training.core import TrainConfig


def get_base_path(train_config: TrainConfig, task: Optional[Task] = None, model_factory: ModelFactory = None,
                  run_id: Optional[int] = None, metrics_path: bool = False, logging_path: bool = False) -> Path:
    # Save to .results/{ID}/{task}/{model}/{run}/{0..5}/{logs/checkpoints/outputs}
    experiment_name = train_config.experiment_name

    if train_config.do_test_overfit:
        experiment_name = f"{experiment_name}_overfit"
    elif train_config.do_test_loop:
        experiment_name = f"{experiment_name}_loop"
    elif train_config.do_few_sample:
        experiment_name = f"{experiment_name}_few_sample"

    path = RESULTS_PATH / experiment_name
    os.makedirs(path, exist_ok=True)

    if logging_path:
        return path

    path /= task.hub_dataset_name.split("/")[-1]
    os.makedirs(path, exist_ok=True)

    if metrics_path:
        return path

    path /= model_factory.model_hub_name.split("/")[-1]
    path /= str(run_id)
    os.makedirs(path, exist_ok=True)

    return path


def setup_logging(train_config: TrainConfig):
    fn = str(datetime.now().strftime("%m-%d.%H-%M-%S"))

    logs_path = get_base_path(train_config, logging_path=True)
    os.makedirs(logs_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_path / f"{fn}.log"),
            logging.StreamHandler()
        ],
        force=True,
    )
