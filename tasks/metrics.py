import abc
from typing import List, Dict, Optional, Tuple

import numpy as np
from datasets import load_metric, Dataset

from models import CACHE_DIR
from tasks.converters import Converter


class MetricComputer(abc.ABC):
    @abc.abstractmethod
    def compute_metrics(self, dataset: Dataset, predictions: List, references: List) -> Dict[str, float]:
        raise NotImplementedError


class NERComputer(MetricComputer):
    def __init__(self):
        self.metric = load_metric("seqeval", cache_dir=CACHE_DIR)

    def compute_metrics(self, dataset: Dataset, predictions: List[List[str]],
                        references: List[List[str]]) -> Dict[str, float]:
        computed_metrics = self.metric.compute(predictions=predictions, references=references)

        return {
            "precision": computed_metrics["overall_precision"],
            "recall": computed_metrics["overall_recall"],
            "f1": computed_metrics["overall_f1"],
            "accuracy": computed_metrics["overall_accuracy"],
        }


class ClassificationComputer(MetricComputer):
    def __init__(self):
        self.accuracy_metric = load_metric("accuracy", cache_dir=CACHE_DIR)

    def compute_metrics(self, dataset: Dataset, predictions: List[int], references: List[int]) -> Dict[str, float]:
        accuracy = self.accuracy_metric.compute(predictions=predictions, references=references)['accuracy']

        return {
            "accuracy": accuracy,
            "error": 1 - accuracy,
        }


class AccuracyComputer(MetricComputer):
    def compute_metrics(self, dataset: Dataset, predictions: List[int], references: List[int]) -> Dict[str, float]:
        accuracy = (predictions == references).astype(np.float32).mean().item()

        return {
            "accuracy": accuracy
        }


class SquadV2Computer(MetricComputer):
    def __init__(self):
        self.squad_v2_metric = load_metric("squad_v2")

    def compute_metrics(self, dataset: Dataset, predictions: List[str], references: Tuple[List, List]) -> Dict[str, float]:
        # References: two lists of ints
        predictions = [{
            "id": i,
            "prediction_text": predictions[i],
            "no_answer_probability": 0.0
        } for i in range(len(predictions))]
        references = [{"id": i, "answers": dataset["answers"][i]} for i in range(len(predictions))]

        squad_metrics = self.squad_v2_metric.compute(predictions=predictions, references=references)

        return {
            "exact": squad_metrics["exact"],
            "f1": squad_metrics["f1"]
        }


class MetricHolder(MetricComputer):
    def __init__(self, metrics: List[MetricComputer],
                 converters: Optional[List[Converter]] = None):
        self.metrics = metrics
        self.converters = converters

    def compute_metrics(self, dataset: Dataset, predictions: List, references: List) -> Dict[str, float]:
        if self.converters is not None:
            for converter in self.converters:
                predictions, references = converter.convert(dataset, predictions, references)

        computed_metrics = {}
        for metric in self.metrics:
            computed_metrics = {**computed_metrics, **metric.compute_metrics(dataset, predictions, references)}

        return computed_metrics
