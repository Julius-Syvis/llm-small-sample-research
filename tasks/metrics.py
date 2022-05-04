import abc
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
from datasets import load_metric

from models import CACHE_DIR


class MetricComputer(abc.ABC):
    @abc.abstractmethod
    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        raise NotImplementedError


class NERComputer(MetricComputer):
    def __init__(self):
        self.metric = load_metric("seqeval", cache_dir=CACHE_DIR)

    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        computed_metrics = self.metric.compute(predictions=predictions, references=references)

        return {
            "precision": computed_metrics["overall_precision"],
            "recall": computed_metrics["overall_recall"],
            "f1": computed_metrics["overall_f1"],
            "accuracy": computed_metrics["overall_accuracy"],
        }


class ClassificationComputer(MetricComputer):
    def __init__(self):
        # self.precision_metric = load_metric("precision", cache_dir=CACHE_DIR)
        # self.recall_metric = load_metric("recall", cache_dir=CACHE_DIR)
        # self.f1_metric = load_metric("f1", cache_dir=CACHE_DIR)
        self.accuracy_metric = load_metric("accuracy", cache_dir=CACHE_DIR)

    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        accuracy = self.accuracy_metric.compute(predictions=predictions, references=references)['accuracy']
        return {
            # "precision": self.precision_metric.compute(predictions=predictions, references=references, average='macro'),
            # "recall": self.recall_metric.compute(predictions=predictions, references=references, average='macro'),
            # "f1": self.f1_metric.compute(predictions=predictions, references=references, average='macro'),
            "accuracy": accuracy,
            "error": 1 - accuracy,
        }


class AccuracyComputer(MetricComputer):
    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        preds = np.argmax(predictions, axis=1)
        return {
            "accuracy": (preds == references).astype(np.float32).mean().item()
        }


class MetricHolder(MetricComputer):
    def __init__(self, metrics: List[MetricComputer],
                 converters: Optional[List[Callable[[List, List], Tuple[List, List]]]] = None):
        self.metrics = metrics
        self.converters = converters

    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        if self.converters is not None:
            for converter in self.converters:
                predictions, references = converter(predictions, references)

        computed_metrics = {}
        for metric in self.metrics:
            computed_metrics = {**computed_metrics, **metric.compute_metrics(predictions, references)}

        return computed_metrics


def conll_converter(label_names: List[str], logits: List[List[float]],
                    references: List[List[int]]) -> Tuple[List, List]:
    predictions = [np.argmax(ls, 1) for ls in logits]

    def convert_label(label):
        return [label_names[l] for l in label if l != -100]

    def convert_prediction(prediction_label):
        prediction, label = prediction_label
        return [label_names[p] for (p, l) in zip(prediction, label) if l != -100]

    true_labels = list(map(convert_label, references))
    true_predictions = list(map(convert_prediction, zip(predictions, references)))

    return true_predictions, true_labels


def classification_converter(label_names: List[str], logits: List[float], references: List[int]) -> Tuple[List, List]:
    predictions = np.argmax(logits, 1)

    # true_labels = list(map(lambda label: label_names[label[0]], references))
    # true_predictions = list(map(lambda label: label_names[label], predictions))

    return predictions, references
