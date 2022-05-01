from typing import List, Union

from datasets import Dataset, DatasetDict

from utils.seed_utils import SEED


def shuffle_ds(dsd: Union[Dataset, DatasetDict]) -> DatasetDict:
    dsd = dsd.shuffle(seed=SEED)
    return dsd


def prepare_test_dsd(dsd: DatasetDict) -> DatasetDict:
    dsd = prepare_dsd(dsd, False)
    return DatasetDict({
        "train": dsd["train"].train_test_split(100, seed=SEED)["test"],
        "validation": dsd["validation"].train_test_split(10, seed=SEED)["test"],
        "test": dsd["test"].train_test_split(10, seed=SEED)["test"]
    })


def prepare_dsd(dsd: DatasetDict, low_sample: bool) -> DatasetDict:
    datasets: List[Dataset] = [subset for (subset_name, subset) in dsd.items()]

    train_ds = datasets[0]
    if len(datasets) < 3:
        split_subset = datasets[1].train_test_split(0.5, 0.5, seed=SEED)
        val_ds = split_subset["train"]
        test_ds = split_subset["test"]
    else:
        val_ds = datasets[1]
        test_ds = datasets[2]

    if low_sample:
        train_ds = train_ds.train_test_split(1000, seed=SEED)["test"]

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
