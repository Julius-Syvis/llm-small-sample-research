import logging
import os
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Generator

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel, Trainer, TrainingArguments, EarlyStoppingCallback, \
    DataCollator
from transformers.trainer_utils import TrainOutput

from models.core import ModelFactory
from tasks.core import Task
from tasks.metrics import MetricHolder
from training import get_outputs_path, get_logs_path, get_checkpoints_path
from training.config import TrainConfig
from utils.data_utils import prepare_test_dsd, shuffle_ds, prepare_dsd
from utils.gpu_utils import cleanup
from utils.logging_utils import setup_logging, get_base_path
from utils.seed_utils import SEED


@dataclass(frozen=True)
class MultipleTrainSequencer:
    model_factories: List[Callable[[], ModelFactory]]
    tasks: List[Callable[[], Task]]
    train_config: TrainConfig

    def __post_init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    @cleanup
    def train(self):
        setup_logging(self.train_config)

        for mf in self.model_factories:
            model_factory = mf()

            for t in self.tasks:
                task = t()

                sequencer = TrainSequencer(model_factory, task, self.train_config)
                sequencer.train()


@dataclass(frozen=True)
class TrainSequencer:
    model_factory: ModelFactory
    task: Task
    train_config: TrainConfig

    @cleanup
    def train(self):
        logging.info(f"Running experiment {self.train_config.experiment_name} "
                     f"on {self.model_factory.model_hub_name} "
                     f"with {self.task.hub_dataset_name}..")

        logging.info("Loading tokenizer..")
        tokenizer = self.model_factory.load_tokenizer()

        logging.info("Tokenizing dataset..")
        dataset_dict = self.task.tokenize(tokenizer)

        if self.train_config.do_test_overfit or self.train_config.do_test_loop:
            logging.info("Preparing test dataset..")
            dataset_dict = prepare_test_dsd(dataset_dict, self.task.validation_col, self.task.test_col)
        else:
            logging.info("Preparing full dataset..")
            dataset_dict = prepare_dsd(dataset_dict, self.train_config.do_few_sample,
                                       self.train_config.custom_train_sample_count,
                                       self.task.validation_col, self.task.test_col)

        logging.info("Shuffling..")
        dataset_dict = shuffle_ds(dataset_dict)

        logging.info("Getting model loader..")
        model_loader = self.task.get_model_loader(self.model_factory)

        logging.info("Loading metric holder..")
        metric_holder = self.task.load_metric_holder()

        for i in range(self.train_config.num_runs):
            logging.info(f"Starting training for step {i}..")
            self._train_loop(model_loader, tokenizer, dataset_dict, metric_holder, i)

    @cleanup
    def _train_loop(self, model_loader: Callable[[], PreTrainedModel], tokenizer: PreTrainedTokenizerBase,
                    dataset_dict: DatasetDict, metric_holder: MetricHolder, run_id: int):
        logging.info(f"Loading data collator..")
        data_collator = self.task.load_data_collator(tokenizer)

        logging.info(f"Loading model..")
        model = model_loader()

        logging.info(f"Getting trainer..")
        trainer = self._get_trainer(model, tokenizer, data_collator, dataset_dict, metric_holder, run_id)

        train_output = None
        if self.train_config.do_train:
            logging.info(f"Training..")
            train_output = trainer.train()
            if self.train_config.do_save:
                logging.info(f"Saving best model..")
                model.save_pretrained(self._get_checkpoints_path(run_id) / "BEST")

                if self.train_config.delete_after_save:
                    shutil.rmtree(self._get_checkpoints_path(run_id))

        if "test" in dataset_dict:
            logging.info(f"Running evaluation..")
            evaluations = self._evaluate_dataset(trainer, dataset_dict["test"])

            logging.info(f"Computing metrics..")
            metrics = self._compute_metrics(metric_holder, evaluations, train_output)

            logging.info(f"Saving results..")
            self._save_results(metrics, evaluations, run_id)

    def _get_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, data_collator: DataCollator,
                     dataset_dict: DatasetDict, metric_holder: MetricHolder, run_id: int) -> Trainer:
        max_steps, eval_steps = self._get_step_counts()
        eval_strategy = "steps" if "validation" in dataset_dict else "no"

        training_args = TrainingArguments(
            output_dir=self._get_checkpoints_path(run_id),
            # save to .results/checkpoints/{ID}/{task}/{model}/{ID}
            overwrite_output_dir=True,
            max_steps=max_steps,

            evaluation_strategy=eval_strategy,
            eval_steps=eval_steps,

            save_strategy=eval_strategy if self.train_config.do_save else "no",
            save_steps=eval_steps,
            save_total_limit=1,
            load_best_model_at_end=self.train_config.do_save,
            metric_for_best_model=self.task.track_metric,
            greater_is_better=self.task.greater_is_better,

            logging_steps=eval_steps,
            logging_dir=self._get_logs_path(run_id),  # save to .results/logs/{ID}/{task}/{model}/{ID}/
            log_level='error',
            report_to="tensorboard",

            optim="adamw_torch",
            weight_decay=0.01,
            max_grad_norm=1.00,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",  # Default
            learning_rate=2e-4,

            per_device_train_batch_size=1 * self.train_config.batch_size_multiplier,
            per_device_eval_batch_size=1 * self.train_config.batch_size_multiplier,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            eval_accumulation_steps=1 * self.train_config.batch_size_multiplier,
            fp16=True,
            fp16_full_eval=True,

            dataloader_num_workers=0,
            remove_unused_columns=True,
            dataloader_pin_memory=False,

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
            compute_metrics=lambda metrics: metric_holder.compute_metrics(metrics.predictions, metrics.label_ids),
            callbacks=self._get_callbacks(),
        )

        return trainer

    def _evaluate_dataset(self, trainer: Trainer, dataset: Dataset) -> Dict[str, List]:
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
            input_ids = entry['input_ids'][0]
            if len(input_ids.shape) == 2:
                return [trainer.tokenizer.decode(input_row) for input_row in input_ids]
            else:
                return trainer.tokenizer.decode(input_ids)

        sentences = list(map(get_decoded_sentences, dataset))

        return {
            "loss": torch.stack(losses).cpu().numpy(),
            "logits": list(map(lambda x: x[0].cpu().numpy(), logits)),
            "labels": list(map(lambda x: x.flatten().cpu().numpy(), labels)),
            "sentences": sentences,
        }

    def _compute_metrics(self, metric_holder: MetricHolder, evaluations: Dict[str, List],
                         train_output: Optional[TrainOutput]) -> Dict[str, float]:
        metrics = metric_holder.compute_metrics(evaluations["logits"], evaluations["labels"])
        metrics["test_loss"] = np.mean(evaluations["loss"])

        if train_output is not None:
            global_step, _, train_metrics = train_output
            metrics["global_step"] = global_step
            metrics = {**metrics, **train_metrics}

        return metrics

    def _save_results(self, metrics: Dict[str, float], evaluations: Dict[str, List], run_id: int):
        df_metrics = pd.DataFrame({k: [v] for (k, v) in metrics.items()})
        model_name = self.model_factory.model_hub_name.split("/")[-1]
        df_metrics["model"] = [model_name]
        df_metrics["run_id"] = [run_id]

        # Save evals to:
        # .results/outputs/{ID}/{task}/{model}/evals_XXX.tsv
        df_evals = pd.DataFrame(evaluations)
        df_path = self._get_outputs_path(run_id)
        os.makedirs(df_path, exist_ok=True)
        df_evals.to_csv(df_path / f"evals.tsv", sep="\t")

        # Save metrics to:
        # .results/outputs/{ID}/{task}/metrics.tsv
        metrics_path = self._get_metrics_path(run_id) / f"metrics.tsv"
        if os.path.exists(metrics_path):
            df_metrics_loaded = pd.read_csv(metrics_path, sep="\t")

            updatable_idx = df_metrics_loaded.index[
                (df_metrics_loaded.model == model_name) & (df_metrics_loaded.run_id == run_id)
                ]

            if len(updatable_idx) > 0:
                df_metrics_loaded.iloc[updatable_idx] = df_metrics
                df_metrics = df_metrics_loaded
            else:
                df_metrics = pd.concat([df_metrics_loaded, df_metrics])

        df_metrics.to_csv(metrics_path, sep="\t", index=False)

    def _get_metrics_path(self, run_id: int):
        return self._get_base_path(run_id, metrics_path=True)

    def _get_outputs_path(self, run_id: int):
        return get_outputs_path(self._get_base_path(run_id))

    def _get_logs_path(self, run_id: int):
        return get_logs_path(self._get_base_path(run_id))

    def _get_checkpoints_path(self, run_id: int):
        return get_checkpoints_path(self._get_base_path(run_id))

    def _get_base_path(self, run_id: int, metrics_path: bool = False):
        return get_base_path(self.train_config, self.task, self.model_factory, run_id, metrics_path)

    def _get_step_counts(self):
        if self.train_config.do_test_overfit:
            return 250, 50
        elif self.train_config.do_test_loop:
            return 1, 1
        else:
            return 2_000, 50

    def _get_callbacks(self):
        callbacks = []

        if not self.train_config.do_test_overfit and self.train_config.do_save:
            callbacks.append(EarlyStoppingCallback(5, 0.0))

        return callbacks
