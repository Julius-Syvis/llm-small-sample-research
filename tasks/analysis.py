from datasets import load_dataset, DatasetDict, Dataset, Features

from models import CACHE_DIR
from tasks.core import get_conll_2003, get_swag, get_ag_news, get_squad_v2


def analyse_task(task):
    dataset: DatasetDict = load_dataset(task.hub_dataset_name, cache_dir=CACHE_DIR)

    print(f"Analysing {task.hub_dataset_name}")
    print(f"Keys: {dataset.keys()}")

    for key in dataset.keys():
        sub_dataset: Dataset = dataset[key]

        features: Features = sub_dataset.features
        print(f"Features: {features}")
        print(f"{key} has {sub_dataset.shape} entries")


if __name__ == "__main__":
    # NER (coNLL)
    # 14042, 3251, 3454
    # [id, tokens, chunk_tags, pos_tags, ner_tags]
    analyse_task(get_conll_2003())

    # Multiple Choice (SWAG)
    # 73456, 20006, 20005
    # [video-id, fold-ind, startphrase, sent1, sent2, gold-source, ending0, ending1, ending2, ending3, label]
    analyse_task(get_swag())

    # Sentence Classification (AGNews)
    # 120000, 7600
    # [text, label, names, id]
    analyse_task(get_ag_news())

    # TODO: make this run
    # Extractive Question Answering ()
    #
    # []
    analyse_task(get_squad_v2())
