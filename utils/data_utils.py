from typing import Union, Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from utils.seed_utils import SEED


def shuffle_ds(dsd: Union[Dataset, DatasetDict]) -> DatasetDict:
    dsd = dsd.shuffle(seed=SEED)
    return dsd


def prepare_cross_validation(dsd: DatasetDict, validation_col: Optional[str],
                             test_col: Optional[str], split_by_col: Optional[str]) -> DatasetDict:
    assert not (validation_col is None and test_col is None)
    missing_column = validation_col is None or test_col is None

    if validation_col is not None:
        val_ds: Dataset = dsd[validation_col]
    if test_col is not None:
        test_ds: Dataset = dsd[test_col]

        if validation_col is None:
            val_ds = test_ds
            del test_ds

    if missing_column:
        if split_by_col:
            val_sum, val_cols = 0, []
            test_sum, test_cols = 0, []

            for name, value_count in pd.value_counts(val_ds[split_by_col]).iteritems():
                if val_sum < test_sum:
                    val_sum += value_count
                    val_cols.append(name)
                else:
                    test_sum += value_count
                    test_cols.append(name)

            # Order matters here:
            test_ds = val_ds.filter(lambda x: x[split_by_col] in test_cols)
            val_ds = val_ds.filter(lambda x: x[split_by_col] in val_cols)
        else:
            split_subset = val_ds.train_test_split(0.5, 0.5, seed=SEED)
            val_ds = split_subset["train"]
            test_ds = split_subset["test"]

    return DatasetDict({
        "train": dsd['train'].train_test_split(1000)["test"],
        "validation": val_ds.train_test_split(1000)["test"],
        "test": test_ds.train_test_split(1000)["test"]
    })


def prepare_test_dsd(dsd: DatasetDict) -> DatasetDict:
    dsd = prepare_dsd(dsd, False, None)

    return DatasetDict({
        "train": dsd["train"].train_test_split(100, seed=SEED)["test"],
        "validation": dsd["validation"].train_test_split(10, seed=SEED)["test"],
        "test": dsd["test"].train_test_split(10, seed=SEED)["test"]
    })


def prepare_dsd(dsd: DatasetDict, few_sample: bool, custom_train_sample_count: Optional[int]) -> DatasetDict:
    train_ds: Dataset = dsd['train']

    if few_sample:
        train_sample_count = custom_train_sample_count if custom_train_sample_count else 1000
        train_ds = train_ds.train_test_split(train_sample_count, seed=SEED)["test"]

    return DatasetDict({
        "train": train_ds,
        "validation": dsd['validation'],
        "test": dsd['test']
    })
