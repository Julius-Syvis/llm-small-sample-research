from datasets import DatasetDict, Dataset, Features

from tasks.core import get_conll_2003, get_swag, get_ag_news, get_squad_v2, get_wikiann_en, get_wikiann_lt


def analyse_task(task):
    dataset: DatasetDict = task.loaded_dataset

    print(f"Analysing {task.get_dataset_name()}")
    print(f"Keys: {dataset.keys()}")

    for key in dataset.keys():
        sub_dataset: Dataset = dataset[key]

        features: Features = sub_dataset.features
        print(f"Features: {features}")
        print(f"{key} has {sub_dataset.shape} entries")


if __name__ == "__main__":
    pass

    # NER (coNLL)
    # 14042, 3251, 3454
    # [id, tokens, chunk_tags, pos_tags, ner_tags]
    analyse_task(get_conll_2003())

    # NER (wikiann)
    # 10000, 10000, 10000
    # [tokens, ner_tags, langs, spans]
    analyse_task(get_wikiann_lt())

    # NER (wikiann)
    # 10000, 10000, 20000
    # [tokens, ner_tags, langs, spans]
    analyse_task(get_wikiann_en())

    # Multiple Choice (SWAG)
    # 73456, 20006, 20005
    # [video-id, fold-ind, startphrase, sent1, sent2, gold-source, ending0, ending1, ending2, ending3, label]
    analyse_task(get_swag())

    # Sentence Classification (AGNews)
    # 120000, T: 7600
    # [text, label, names, id]
    analyse_task(get_ag_news())

    # Extractive Question Answering (SQUAD 2)
    # 130319, V: 11873
    # [id, title, context, question, answers(text, answer_start)]
    analyse_task(get_squad_v2())
