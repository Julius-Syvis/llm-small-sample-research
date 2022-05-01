from pathlib import Path

import datasets

TASKS_PATH = Path(__file__).resolve().parent
datasets.enable_caching()
