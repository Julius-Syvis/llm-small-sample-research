import abc
from typing import List, Tuple

import numpy as np
from datasets import Dataset


class Converter(abc.ABC):
    def convert(self, dataset: Dataset, logits: List, references: List) -> Tuple[List, List]:
        raise NotImplementedError


class CoNLLConverter(Converter):
    def __init__(self, label_names: List[str]):
        self.label_names = label_names

    def convert(self, dataset: Dataset, logits: List, references: List) -> Tuple[List, List]:
        predictions = [np.argmax(ls, 1) for ls in logits]

        def convert_label(label):
            return [self.label_names[l] for l in label if l != -100]

        def convert_prediction(prediction_label):
            prediction, label = prediction_label
            return [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]

        true_labels = list(map(convert_label, references))
        true_predictions = list(map(convert_prediction, zip(predictions, references)))

        return true_predictions, true_labels


class ClassificationConverter(Converter):
    def convert(self, dataset: Dataset, logits: List, references: List) -> Tuple[List, List]:
        predictions = np.argmax(logits, 1)

        return predictions, references


# https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/examples/pytorch/question-answering/utils_qa.py#L31
class SquadV2Converter(Converter):
    def convert(self, dataset: Dataset, logits: List, references: List) -> Tuple[List, List]:
        all_start_logits, all_end_logits = logits

        # If data arrives as [1, X] instead of [X]
        all_start_logits = [l_.flatten() for l_ in all_start_logits]
        all_end_logits = [l_.flatten() for l_ in all_end_logits]

        offset_mapping = dataset['offset_mapping']

        final_preds = []

        for i, (start_logits, end_logits) in enumerate(zip(all_start_logits, all_end_logits)):
            start_idxs = np.argsort(start_logits)[-20:]
            end_idxs = np.argsort(end_logits)[-20:]

            # Append score for 0-0
            preds = []

            # Evaluate best n x n options
            for start_idx in start_idxs:
                for end_idx in end_idxs:
                    max_length = 30  # Arbitrary for tokens. Can be longer for characters
                    if end_idx < start_idx or end_idx - start_idx + 1 > max_length:
                        continue

                    preds.append({
                        "offsets": (offset_mapping[i][start_idx][0], offset_mapping[i][end_idx][1]),
                        "score": all_start_logits[i][start_idx] + all_end_logits[i][end_idx],
                        "start_logit": all_start_logits[i][start_idx],
                        "end_logit": all_end_logits[i][end_idx]
                    })

            # Pick best n options, add null option
            preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:20]

            # Add 0 entry
            null_score = all_start_logits[i][0] + all_end_logits[i][0]
            preds.append({
                "offsets": (0, 0),
                "score": null_score,
                "start_logit": all_start_logits[i][0],
                "end_logit": all_end_logits[i][0],
            })

            scores = np.array([pred["score"] for pred in preds])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            for j, prob in enumerate(probs):
                preds[j]["prob"] = prob
                preds[j]["text"] = dataset["context"][i][preds[j]["offsets"][0]:preds[j]["offsets"][1]]

            # Pick best prediction
            i = 0
            while preds[i]["text"] == "":
                i += 1
            best_non_null_pred = preds[i]

            if null_score > best_non_null_pred["score"]:
                final_preds.append("")
            else:
                final_preds.append(best_non_null_pred["text"])

        return final_preds, references
