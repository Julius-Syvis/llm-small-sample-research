from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List

import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel, Trainer, TrainingArguments, EarlyStoppingCallback, \
    DataCollator

from models.core import ModelFactory
from tasks.core import Task
from tasks.metrics import conll_converter, SeqEvalComputer, MetricHolder
from training import CHECKPOINTS_PATH, LOGS_PATH, OUTPUTS_PATH
from utils.data_utils import prepare_test_dsd, shuffle_ds, prepare_dsd
from utils.gpu_utils import cleanup
from utils.seed_utils import SEED


@dataclass
class TrainConfig:
    do_test_run: bool
    do_train: bool
    # run_name: str  # TODO use this as identifier / add model_name additionally ?
    # TODO: add num_runs param


class TrainSequencer:
    @staticmethod
    def train(model_factory: ModelFactory, task: Task, train_config: TrainConfig):
        tokenizer = model_factory.load_tokenizer()

        dataset_dict = task.tokenize(tokenizer)
        if train_config.do_test_run:
            dataset_dict = prepare_test_dsd(dataset_dict)
        else:
            dataset_dict = prepare_dsd(dataset_dict, True)
        dataset_dict = shuffle_ds(dataset_dict)

        label_names = task.loaded_dataset["train"].features["ner_tags"].feature.names
        model_loader = partial(model_factory.load_token_classification_model, label_names)
        metric_holder = task.load_metrics_holder(label_names)

        TrainSequencer.train_loop(model_loader, task, tokenizer, dataset_dict, train_config, metric_holder)

    @staticmethod
    @cleanup
    def train_loop(model_loader: Callable[[], PreTrainedModel],
                   task: Task, tokenizer: PreTrainedTokenizerBase,
                   dataset_dict: DatasetDict, train_config: TrainConfig, metric_holder: MetricHolder):
        data_collator = task.load_data_collator(tokenizer)
        model = model_loader()

        trainer = TrainSequencer.get_trainer(model, tokenizer, data_collator, dataset_dict, train_config)

        if train_config.do_train:
            trainer.train()
            model.save_pretrained(CHECKPOINTS_PATH / "BEST")

        if "test" in dataset_dict:
            evaluations = TrainSequencer.evaluate_dataset(trainer, dataset_dict["test"])
            metrics = TrainSequencer.compute_metrics(metric_holder, evaluations)

    @staticmethod
    def get_trainer(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, data_collator: DataCollator,
                    dataset_dict: DatasetDict, train_config: TrainConfig) -> Trainer:
        max_steps, eval_steps = TrainSequencer.get_step_counts(train_config)
        eval_strategy = "steps" if "validation" in dataset_dict else "no"

        training_args = TrainingArguments(
            output_dir=CHECKPOINTS_PATH,  # TODO: update path
            overwrite_output_dir=True,
            max_steps=max_steps,

            evaluation_strategy=eval_strategy,
            eval_steps=eval_steps,

            save_strategy=eval_strategy,
            save_steps=eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            # metric_for_best_model='ChrF', # TODO: customize

            logging_steps=eval_steps,
            logging_dir=LOGS_PATH,  # TODO: update path
            log_level='error',
            report_to="tensorboard",

            optim="adamw_torch",
            # TODO: select "reasonable" params
            weight_decay=0.01,
            max_grad_norm=1.00,
            warmup_ratio=0.25,

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            eval_accumulation_steps=1,
            fp16=True,
            fp16_full_eval=True,

            dataloader_num_workers=0,
            remove_unused_columns=True,
            dataloader_pin_memory=False,

            # TODO: evaluate whether results are deterministic with seeding (+ add seed() before call)
            seed=SEED,
            data_seed=SEED,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"] if "validation" in dataset_dict else None,
            # compute_metrics=compute_metrics, # TODO: compute metrics & allow customization for each task type
            callbacks=TrainSequencer.get_callbacks(train_config),
        )

        return trainer

    @staticmethod
    def evaluate_dataset(trainer: Trainer, dataset: Dataset) -> Dict[str, List]:
        dataset = DataLoader(
            dataset,
            sampler=trainer._get_eval_sampler(dataset),
            batch_size=1,
            collate_fn=trainer.data_collator,
            drop_last=trainer.args.dataloader_drop_last,
            pin_memory=trainer.args.dataloader_pin_memory
        )

        dataset = trainer._prepare_inputs(dataset)

        def get_entry_loss(entry):
            return trainer.prediction_step(trainer.model, entry, prediction_loss_only=False)

        # map() Calls iter() on dataset automatically
        losses, logits, labels = list(zip(*list(map(get_entry_loss, dataset))))

        def get_decoded_sentences(entry):
            return trainer.tokenizer.decode(entry['input_ids'][0])

        sentences = list(map(get_decoded_sentences, dataset))

        return {
            "loss": torch.stack(losses).cpu().numpy(),
            "logits": list(map(lambda x: x[0, :, :].cpu().numpy(), logits)),
            "labels": list(map(lambda x: x.flatten().cpu().numpy(), labels)),
            "sentences": sentences,
        }

    @staticmethod
    def compute_metrics(metric_holder: MetricHolder, evaluations: Dict[str, list]):
        metrics = metric_holder.compute_metrics(evaluations["logits"], evaluations["labels"])

        df = pd.DataFrame(evaluations)

        print(metrics)

        # os.makedirs(OUTPUTS_PATH / "outputs.tsv", exist_ok=True)
        # df.to_csv(OUTPUTS_PATH / "outputs.tsv", sep="\t")  # TODO: save under task/model_datasize

        # TODO: append to task.tsv with [identifier, model_name, dataset_size, metric_1, metric_2] (separate file for each task)
        # keep identifier for future comparisons (e.g. distilled)

    @staticmethod
    def get_step_counts(train_config: TrainConfig):
        if train_config.do_test_run:
            return 250, 50
        else:
            return 2_000, 50

    @staticmethod
    def get_callbacks(train_config: TrainConfig):
        callbacks = []

        if not train_config.do_test_run:
            callbacks.append(EarlyStoppingCallback(3, 0.0))

        return callbacks
