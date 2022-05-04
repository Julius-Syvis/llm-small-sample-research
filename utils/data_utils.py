from typing import List, Union

from datasets import Dataset, DatasetDict

from utils.seed_utils import SEED


def shuffle_ds(dsd: Union[Dataset, DatasetDict]) -> DatasetDict:
    dsd = dsd.shuffle(seed=SEED)
    return dsd


def prepare_test_dsd(dsd: DatasetDict, use_test: bool) -> DatasetDict:
    dsd = prepare_dsd(dsd, False, use_test)
    return DatasetDict({
        "train": dsd["train"].train_test_split(100, seed=SEED)["test"],
        "validation": dsd["validation"].train_test_split(10, seed=SEED)["test"],
        "test": dsd["test"].train_test_split(10, seed=SEED)["test"]
    })


def prepare_dsd(dsd: DatasetDict, low_sample: bool, use_test: bool) -> DatasetDict:
    train_ds: Dataset = dsd['train']
    val_ds: Dataset = dsd['validation']

    if not use_test:
        split_subset = val_ds.train_test_split(0.5, 0.5, seed=SEED)
        val_ds = split_subset["train"]
        test_ds = split_subset["test"]
    else:
        test_ds = dsd["test"]

    if low_sample:
        train_ds = train_ds.train_test_split(1000, seed=SEED)["test"]

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
