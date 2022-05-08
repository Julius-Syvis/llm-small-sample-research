from typing import List, Generator, Callable

from transformers import AutoModel, PreTrainedTokenizerBase, PreTrainedModel, AutoModelForTokenClassification, \
    AutoTokenizer, AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

from models import CACHE_DIR

class ModelFactory:
    def __init__(self, model_hub_name, is_character_level=False):
        self.model_hub_name = model_hub_name
        self.is_character_level = is_character_level

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_hub_name,
            add_prefix_space=True,  # needed for RoBERTa to use first token
            cache_dir=CACHE_DIR
        )

        tokenizer.is_character_level = self.is_character_level
        tokenizer.max_sequence_length = 2048 if self.is_character_level else 512

        return tokenizer

    def load_model(self) -> PreTrainedModel:
        model: PreTrainedModel = AutoModel.from_pretrained(
            self.model_hub_name,
            cache_dir=CACHE_DIR
        )

        model = model.cuda()
        return model

    def load_token_classification_model(self, label_names: List[str]) -> PreTrainedModel:
        id2label = {str(i): label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}

        model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            self.model_hub_name,
            id2label=id2label,
            label2id=label2id,
            cache_dir=CACHE_DIR
        )

        model = model.cuda()
        return model

    def load_multiple_choice_model(self) -> PreTrainedModel:
        model = AutoModelForMultipleChoice.from_pretrained(
            self.model_hub_name,
            cache_dir=CACHE_DIR
        )

        model = model.cuda()
        return model

    def load_classification_model(self, label_names: List[str]) -> PreTrainedModel:
        id2label = {str(i): label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_hub_name,
            id2label=id2label,
            label2id=label2id,
            cache_dir=CACHE_DIR
        )

        model = model.cuda()
        return model

    def load_question_answering_model(self) -> PreTrainedModel:
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_hub_name,
            cache_dir=CACHE_DIR
        )

        model = model.cuda()
        return model


def get_bert_base() -> ModelFactory:
    return ModelFactory("bert-base-cased")


def get_bert_base_uncased() -> ModelFactory:
    return ModelFactory("bert-base-uncased")


def get_roberta_base() -> ModelFactory:
    return ModelFactory("roberta-base")


def get_canine_s() -> ModelFactory:
    return ModelFactory("google/canine-s", is_character_level=True)


def get_canine_c() -> ModelFactory:
    return ModelFactory("google/canine-c", is_character_level=True)


def get_electra_base() -> ModelFactory:
    return ModelFactory("google/electra-base-discriminator")


def get_big_bird() -> ModelFactory:
    return ModelFactory("google/bigbird-roberta-base")


# TODO: fix gradient checkpointing
def get_xlnet_base() -> ModelFactory:
    return ModelFactory("xlnet-base-cased")


def get_transformer_xl() -> ModelFactory:
    return ModelFactory("transfo-xl-wt103", is_character_level=True)


def get_xlm() -> ModelFactory:
    return ModelFactory("xlm-roberta-base")


def get_supported_model_factories() -> List[ModelFactory]:
    return [get_bert_base(), get_bert_base_uncased(), get_roberta_base(),
            get_electra_base(), get_big_bird(), get_xlm()]
