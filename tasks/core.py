import abc
from functools import partial
from typing import List, Optional

from datasets import load_dataset, DatasetDict
from datasets.arrow_dataset import Batch
from transformers import PreTrainedTokenizerBase, BatchEncoding, DataCollatorForTokenClassification

from tasks.metrics import MetricHolder, SeqEvalComputer, conll_converter


class Task(abc.ABC):
    def __init__(self, hub_dataset_name: str):
        self.hub_dataset_name: str = hub_dataset_name
        self.loaded_dataset: DatasetDict = load_dataset(hub_dataset_name)

    @abc.abstractmethod
    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        raise NotImplementedError

    @abc.abstractmethod
    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        raise NotImplementedError

    @abc.abstractmethod
    def load_metrics_holder(self, **kwargs):
        raise NotImplementedError

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
        return self.loaded_dataset.map(
            partial(self._tokenize_and_align_labels, tokenizer),
            batched=True,
            remove_columns=self.loaded_dataset["train"].column_names
        )


class NERTask(Task):
    def __init__(self, hub_dataset_name: str):
        super().__init__(hub_dataset_name)

    # https://huggingface.co/course/chapter7/2
    def _align_labels_with_tokens(self, labels: List[int], word_ids: List[Optional[int]]):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:  # word_ids return the same index for longer words
                # start of a new word
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)  # -100 is ignored by CrossEntropy by default
            else:
                # same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX.
                # In conll B-Type denotes that a new phrase is starting,
                # given that two phrases of the same type follow each other
                # But this is not needed because we tokenize words by spaces either way
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            # TODO: handle unicode encoding for NER
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return DataCollatorForTokenClassification(tokenizer)  # Pads both inputs and labels

    def load_metrics_holder(self, label_names):
        metrics = [
            SeqEvalComputer()
        ]

        converters = [
            partial(conll_converter, label_names)
        ]

        return MetricHolder(metrics, converters)


class MultipleChoiceTask(Task):
    def __init__(self, hub_dataset_name: str):
        super().__init__(hub_dataset_name)
        # TODO: handle multiple choice encoding & heads


class ClassificationTask(Task):
    def __init__(self, hub_dataset_name: str):
        super().__init__(hub_dataset_name)
        # TODO: handle classification encoding & heads


class ExtractiveQuestionAnsweringTask(Task):
    def __init__(self, hub_dataset_name: str):
        super().__init__(hub_dataset_name)
        # TODO: handle classification encoding & heads


def get_conll_2003() -> NERTask:
    return NERTask("conll2003")


def get_swag() -> MultipleChoiceTask:
    return MultipleChoiceTask("swag")


def get_ag_news() -> ClassificationTask:
    return ClassificationTask("ag_news")


def get_squad_v2() -> ExtractiveQuestionAnsweringTask:
    return ExtractiveQuestionAnsweringTask("squad_v2")
