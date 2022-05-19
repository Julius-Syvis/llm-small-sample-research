from functools import partial
from typing import Union, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from datasets.arrow_dataset import Batch

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

            def pick_cols(cols, batch: Batch):
                needed_idxs = [value in cols for value in batch.data[split_by_col]]
                data = {k: [v_ for (i_, v_) in enumerate(v) if needed_idxs[i_]] for k, v in batch.items()}
                return Batch(data)

            # Order matters here:
            test_ds = val_ds.map(partial(pick_cols, test_cols), batched=True)
            val_ds = val_ds.map(partial(pick_cols, val_cols), batched=True)
        else:
            split_subset = val_ds.train_test_split(0.5, 0.5, seed=SEED)
            val_ds = split_subset["train"]
            test_ds = split_subset["test"]

    return DatasetDict({
        "train": dsd['train'],
        "validation": val_ds,
        "test": test_ds
    })


def prepare_test_dsd(dsd: DatasetDict) -> DatasetDict:
    return prepare_dsd(dsd, True, 100, 10, 10)


def prepare_dsd(dsd: DatasetDict, few_sample: bool, custom_train_sample_count: Optional[int],
                validation_set_size_limit: Optional[int] = None,
                test_set_size_limit: Optional[int] = None) -> DatasetDict:
    train_ds: Dataset = dsd['train']

    if few_sample:
        train_sample_count = custom_train_sample_count or 1000

        if custom_train_sample_count > len(train_ds):
            train_ds = train_ds.train_test_split(train_sample_count, seed=SEED)["test"]

    val_ds = dsd['validation']
    if validation_set_size_limit and validation_set_size_limit > len(val_ds):
        val_ds = val_ds.train_test_split(validation_set_size_limit, seed=SEED)["test"]

    test_ds = dsd['test']
    if test_set_size_limit and test_set_size_limit > len(test_ds):
        test_ds = test_ds.train_test_split(test_set_size_limit, seed=SEED)["test"]

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
