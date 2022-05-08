import abc
from functools import partial
from itertools import chain
from typing import List, Optional, Callable, Set

from datasets import load_dataset, DatasetDict
from datasets.arrow_dataset import Batch
from transformers import PreTrainedTokenizerBase, BatchEncoding, DataCollatorForTokenClassification, PreTrainedModel, \
    DataCollator, DataCollatorWithPadding

from models import CACHE_DIR
from models.core import ModelFactory
from tasks.collators import DataCollatorForMultipleChoice
from tasks.converters import CoNLLConverter, ClassificationConverter, SquadV2Converter
from tasks.metrics import MetricHolder, ClassificationComputer, AccuracyComputer, NERComputer, SquadV2Computer
from utils.data_utils import prepare_cross_validation
from utils.seed_utils import SEED


class Task(abc.ABC):
    def __init__(self, hub_dataset_name: str,
                 validation_col: Optional[str] = "validation",
                 test_col: Optional[str] = "test",
                 track_metric: Optional[str] = None,
                 greater_is_better: bool = True,
                 split_by_col: Optional[str] = None,
                 kept_cols: Optional[Set[str]] = None):
        self.hub_dataset_name: str = hub_dataset_name
        self.validation_col = validation_col
        self.test_col = test_col
        self.track_metric = track_metric
        self.greater_is_better = greater_is_better
        self.split_by_col = split_by_col
        self.kept_cols = kept_cols if kept_cols else {}
        self.loaded_dataset: DatasetDict = load_dataset(hub_dataset_name, cache_dir=CACHE_DIR, keep_in_memory=False)

    @abc.abstractmethod
    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        raise NotImplementedError

    @abc.abstractmethod
    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        raise NotImplementedError

    @abc.abstractmethod
    def load_metric_holder(self) -> MetricHolder:
        raise NotImplementedError

    def map_batch(self, kept_col: str, batch: Batch):
        len_ = len(next(iter(batch.values())))

        vals = []
        for i in range(len_):
            i_vals = [batch[key][i] for key in batch.data.keys()]
            i_lens = [len(i_val) if isinstance(i_val, list) else 1 for i_val in i_vals]
            max_len = max(i_lens)

            vals += [batch.data[kept_col][i]] * max_len

        return {'kept_col': vals}

    def add_kept_col_from_untokenized_dataset(self, prepared_dataset: DatasetDict, tokenized_dataset: DatasetDict):
        for key in prepared_dataset:
            ds = tokenized_dataset[key]
            for kept_col in self.kept_cols:
                # col = prepared_dataset[key].flatten().map(partial(map_batch, kept_col),
                #                        batched=True, remove_columns=prepared_dataset[key].flatten().column_names)
                ds = ds.add_column(kept_col, prepared_dataset[key][kept_col])
            tokenized_dataset[key] = ds

    def tokenize(self, tokenizer: PreTrainedTokenizerBase, validation_col: Optional[str] = "validation",
                 test_col: Optional[str] = "test", split_by_col: Optional[str] = None) -> DatasetDict:
        prepared_dataset = prepare_cross_validation(self.loaded_dataset, validation_col, test_col, split_by_col)
        prepared_dataset = prepared_dataset.shuffle(seed=SEED)

        tokenized_dataset = prepared_dataset.map(
            partial(self._tokenize_and_align_labels, tokenizer),
            batched=True,
            remove_columns=set(self.loaded_dataset["train"].column_names) - {'label'},
            desc="Tokenizing dataset"
        )

        return tokenized_dataset

    def get_model_loader(self, model_factory: ModelFactory) -> Callable[[], PreTrainedModel]:
        raise NotImplementedError


# https://huggingface.co/course/chapter7/2?fw=pt
class NERTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def _align_labels_with_chars(self, labels: List[int], word_ids: List[Optional[int]]):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            else:
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1

    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        if tokenizer.is_character_level:
            concatenated_tokens = [' '.join(e) for e in examples.data['tokens']]
            tokenized_inputs = tokenizer(concatenated_tokens, truncation=True, is_split_into_words=False,
                                         max_length=tokenizer.max_sequence_length)
        else:
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,
                                         max_length=tokenizer.max_sequence_length)

        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            if tokenizer.is_character_level:
                word_ids = [([j] * len(s)) for (j, s) in enumerate(examples.data['tokens'][i])]
                word_ids = list(chain(*[[None, *w] for w in word_ids]))[1:]
                word_ids = [None, *word_ids, None]

                raise NotImplementedError
            else:
                word_ids = tokenized_inputs.word_ids(i)

            aligned_labels = self._align_labels_with_tokens(labels, word_ids)
            new_labels.append(aligned_labels)

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return DataCollatorForTokenClassification(tokenizer)  # Pads both inputs and labels

    def load_metric_holder(self) -> MetricHolder:
        metrics = [
            NERComputer()
        ]

        converters = [
            CoNLLConverter(self._get_label_names())
        ]

        return MetricHolder(metrics, converters)

    def _get_label_names(self) -> List[str]:
        return self.loaded_dataset["train"].features["ner_tags"].feature.names

    def get_model_loader(self, model_factory) -> Callable[[], PreTrainedModel]:
        return partial(model_factory.load_token_classification_model, self._get_label_names())


# https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice
class MultipleChoiceTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            max_length=tokenizer.max_sequence_length,
            padding=False,
        )

        # Un-flatten
        tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

        return tokenized_examples

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        return DataCollatorForMultipleChoice(tokenizer)

    def load_metric_holder(self) -> MetricHolder:
        metrics = [
            AccuracyComputer()
        ]

        return MetricHolder(metrics)

    def get_model_loader(self, model_factory: ModelFactory) -> Callable[[], PreTrainedModel]:
        return model_factory.load_multiple_choice_model


# https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
class SequenceClassificationTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=tokenizer.max_sequence_length,
            padding=False,
        )

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def load_metric_holder(self) -> MetricHolder:
        metrics = [
            ClassificationComputer()
        ]

        converters = [
            ClassificationConverter()
        ]

        return MetricHolder(metrics, converters)

    def _get_label_names(self) -> List[str]:
        return self.loaded_dataset["train"].features['label'].names

    def get_model_loader(self, model_factory: ModelFactory) -> Callable[[], PreTrainedModel]:
        label_names = self._get_label_names()
        return partial(model_factory.load_classification_model, label_names)


# https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
class ExtractiveQuestionAnsweringTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kept_cols={"question"})

    def _tokenize_and_align_labels(self, tokenizer: PreTrainedTokenizerBase, examples: Batch) -> BatchEncoding:
        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second",
            return_offsets_mapping=True,
            max_length=tokenizer.max_sequence_length,
            padding=False,
        )

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["context"] = examples["context"]
        tokenized_examples["answers"] = examples["answers"]

        # https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):  # Pairs of (i, ith_token_offset)
            input_ids = tokenized_examples["input_ids"][i]
            cls_token_index = input_ids.index(tokenizer.cls_token_id)  # Just 0

            seq_ids = tokenized_examples.sequence_ids(i)  # 0s for question, None for [sep] and [cls], 1s for context
            answers = examples["answers"][i]

            if len(answers["answer_start"]) == 0:
                # If no answers are given, the correct label is first [cls] token
                tokenized_examples["start_positions"].append(cls_token_index)
                tokenized_examples["end_positions"].append(cls_token_index)
            else:
                # We have a single answer
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                start_of_context_token_idx = seq_ids.index(1)
                end_of_context_token_idx = len(seq_ids) - list(reversed(seq_ids)).index(1) - 1

                if not (offsets[start_of_context_token_idx][0] <= start_char
                        and offsets[end_of_context_token_idx][1] >= end_char):
                    # The required span is not in overflow
                    tokenized_examples["start_positions"].append(cls_token_index)
                    tokenized_examples["end_positions"].append(cls_token_index)
                else:
                    # The required span is in the overflow

                    # Pick first token that contains the required span
                    fitting_tokens = [i for (i, (from_, to_)) in enumerate(offsets) if (i >= start_of_context_token_idx
                                                                                        and from_ >= start_char)]
                    start_token_index = min(len(offsets), fitting_tokens[0])
                    tokenized_examples["start_positions"].append(start_token_index)

                    # Pick last token that contains the required span
                    end_token_index = [i for (i, (from_, to_)) in enumerate(offsets) if (i < end_of_context_token_idx
                                                                                         and to_ <= end_char)][-1]
                    tokenized_examples["end_positions"].append(end_token_index)

        return tokenized_examples

    def load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def load_metric_holder(self) -> MetricHolder:
        metrics = [
            SquadV2Computer()
        ]

        converters = [
            SquadV2Converter()
        ]

        return MetricHolder(metrics, converters)

    def get_model_loader(self, model_factory: ModelFactory) -> Callable[[], PreTrainedModel]:
        return model_factory.load_question_answering_model


def get_conll_2003() -> NERTask:
    return NERTask("conll2003", track_metric="f1")


def get_swag() -> MultipleChoiceTask:
    # Ignore test col because it contains -1 labels
    return MultipleChoiceTask("swag", test_col=None, track_metric="accuracy", split_by_col="video-id")


def get_ag_news() -> SequenceClassificationTask:
    return SequenceClassificationTask("ag_news", validation_col=None, track_metric="error", greater_is_better=False)


def get_squad_v2() -> ExtractiveQuestionAnsweringTask:
    return ExtractiveQuestionAnsweringTask("squad_v2", test_col=None, split_by_col="context", track_metric="f1")


def get_supported_tasks() -> List[Task]:
    return [get_conll_2003(), get_swag(), get_ag_news()]
