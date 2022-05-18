from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainConfig:
    do_train: bool = True
    do_test_overfit: bool = False
    do_test_loop: bool = False
    do_few_sample: bool = True

    custom_train_sample_count: Optional[int] = None
    custom_eval_step: Optional[int] = None
    custom_max_step: Optional[int] = None

    num_runs: int = 1
    batch_size_multiplier: int = 1
    do_save: bool = True
    delete_after_save: bool = True
    early_stopping_patience: Optional[int] = None
    validation_set_size_limit: Optional[int] = None
    test_set_size_limit: Optional[int] = None

    experiment_name: str = '0'
    track_metric: Optional[str] = None
