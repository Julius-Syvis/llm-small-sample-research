import abc
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
from datasets import load_metric


class MetricComputer(abc.ABC):
    @abc.abstractmethod
    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        raise NotImplementedError


class SeqEvalComputer(MetricComputer):
    def __init__(self):
        self.metric = load_metric("seqeval")

    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        computed_metrics = self.metric.compute(predictions=predictions, references=references)

        return {
            "precision": computed_metrics["overall_precision"],
            "recall": computed_metrics["overall_recall"],
            "f1": computed_metrics["overall_f1"],
            "accuracy": computed_metrics["overall_accuracy"],
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


def conll_converter(label_names: List[str], logits: List[List[float]], references: List[List[int]]) -> Tuple[List, List]:
    predictions = [np.argmax(ls, 1) for ls in logits]

    def convert_label(label):
        return [label_names[l] for l in label if l != -100]

    def convert_prediction(prediction_label):
        prediction, label = prediction_label
        return [label_names[p] for (p, l) in zip(prediction, label) if l != -100]

    true_labels = list(map(convert_label, references))
    true_predictions = list(map(convert_prediction, zip(predictions, references)))

    return true_predictions, true_labels
