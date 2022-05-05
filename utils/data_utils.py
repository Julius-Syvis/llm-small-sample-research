from typing import List, Union, Optional

from datasets import Dataset, DatasetDict

from utils.seed_utils import SEED


def shuffle_ds(dsd: Union[Dataset, DatasetDict]) -> DatasetDict:
    dsd = dsd.shuffle(seed=SEED)
    return dsd


def prepare_test_dsd(dsd: DatasetDict, validation_col: Optional[str], test_col: Optional[str]) -> DatasetDict:
    dsd = prepare_dsd(dsd, False, None, validation_col, test_col)
    return DatasetDict({
        "train": dsd["train"].train_test_split(100, seed=SEED)["test"],
        "validation": dsd["validation"].train_test_split(10, seed=SEED)["test"],
        "test": dsd["test"].train_test_split(10, seed=SEED)["test"]
    })


def prepare_dsd(dsd: DatasetDict, few_sample: bool, custom_train_sample_count: Optional[int],
                validation_col: Optional[str], test_col: Optional[str]) -> DatasetDict:
    assert not(validation_col is None and test_col is None)
    missing_column = validation_col is None or test_col is None

    train_ds: Dataset = dsd['train']

    if validation_col is not None:
        val_ds: Dataset = dsd[validation_col]
    if test_col is not None:
        test_ds: Dataset = dsd[test_col]

        if validation_col is None:
            val_ds = test_ds
            del test_ds

    if missing_column:
        split_subset = val_ds.train_test_split(0.5, 0.5, seed=SEED)
        val_ds = split_subset["train"]
        test_ds = split_subset["test"]

    if few_sample:
        train_sample_count = custom_train_sample_count if custom_train_sample_count else 1000
        train_ds = train_ds.train_test_split(train_sample_count, seed=SEED)["test"]

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
