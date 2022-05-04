import abc
from functools import partial
from itertools import chain
from typing import List, Optional, Callable

from datasets import load_dataset, DatasetDict
from datasets.arrow_dataset import Batch
from transformers import PreTrainedTokenizerBase, BatchEncoding, DataCollatorForTokenClassification, PreTrainedModel, \
    DataCollator

from models import CACHE_DIR
from models.core import ModelFactory
from tasks.collators import DataCollatorForMultipleChoice
from tasks.metrics import MetricHolder, SeqEvalComputer, conll_converter, MultipleChoiceComputer


class Task(abc.ABC):
    def __init__(self, hub_dataset_name: str, use_test: bool = True):
        self.hub_dataset_name: str = hub_dataset_name
        self.use_test = use_test
        self.loaded_dataset: DatasetDict = load_dataset(hub_dataset_name, cache_dir=CACHE_DIR)

    @abc.abstractmethod
    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        raise NotImplementedError

    @abc.abstractmethod
    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        raise NotImplementedError

    @abc.abstractmethod
    def load_metrics_holder(self) -> MetricHolder:
        raise NotImplementedError

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
        return self.loaded_dataset.map(
            partial(self._tokenize_and_align_labels, tokenizer),
            batched=True,
            remove_columns=set(self.loaded_dataset["train"].column_names) - {'label'},
        )

    def get_model_loader(self, model_factory: ModelFactory) -> Callable[[], PreTrainedModel]:
        raise NotImplementedError


# https://huggingface.co/course/chapter7/2?fw=pt
class NERTask(Task):
    def __init__(self, hub_dataset_name: str, use_test: bool = True):
        super().__init__(hub_dataset_name, use_test)

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
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return DataCollatorForTokenClassification(tokenizer)  # Pads both inputs and labels

    def load_metrics_holder(self) -> MetricHolder:
        metrics = [
            SeqEvalComputer()
        ]

        converters = [
            partial(conll_converter, self._get_label_names())
        ]

        return MetricHolder(metrics, converters)

    def _get_label_names(self) -> List[str]:
        return self.loaded_dataset["train"].features["ner_tags"].feature.names

    def get_model_loader(self, model_factory) -> Callable[[], PreTrainedModel]:
        return partial(model_factory.load_token_classification_model, self._get_label_names())


# https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice
class MultipleChoiceTask(Task):
    def __init__(self, hub_dataset_name: str, use_test: bool = True):
        super().__init__(hub_dataset_name, use_test)

    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        ending_names = [f"ending{i}" for i in range(4)]
        context_name = "sent1"
        question_header_name = "sent2"

        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=512,  # No matter what, the max length is always 512
            padding=False,
        )

        # Un-flatten
        tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

        return tokenized_examples

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        return DataCollatorForMultipleChoice(tokenizer)

    def load_metrics_holder(self) -> MetricHolder:
        metrics = [
            MultipleChoiceComputer()
        ]

        return MetricHolder(metrics)

    def get_model_loader(self, model_factory: ModelFactory) -> Callable[[], PreTrainedModel]:
        return model_factory.load_multiple_choice_model


class ClassificationTask(Task):
    def __init__(self, hub_dataset_name: str, use_test: bool = True):
        super().__init__(hub_dataset_name, use_test)
        # TODO: handle classification encoding & heads
        # https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification


class ExtractiveQuestionAnsweringTask(Task):
    def __init__(self, hub_dataset_name: str, use_test: bool = True):
        super().__init__(hub_dataset_name, use_test)
        # TODO: handle classification encoding & heads
        # https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering


def get_conll_2003() -> NERTask:
    return NERTask("conll2003")


def get_swag() -> MultipleChoiceTask:
    return MultipleChoiceTask("swag", use_test=False)


def get_ag_news() -> ClassificationTask:
    return ClassificationTask("ag_news")


def get_squad_v2() -> ExtractiveQuestionAnsweringTask:
    return ExtractiveQuestionAnsweringTask("squad_v2")
