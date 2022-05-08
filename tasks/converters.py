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

    def convert(self, dataset: Dataset, logits: np.ndarray, references: np.ndarray) -> Tuple[List, List]:
        # Logits: [T, Seq, C]
        # References: [T, Seq]

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
    def convert(self, dataset: Dataset, logits: List[List[float]], references: List[int]) -> Tuple[List, List]:
        predictions = np.argmax(logits, -1)

        return predictions, references


# https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/examples/pytorch/question-answering/utils_qa.py#L31
class SquadV2Converter(Converter):
    def convert(self, dataset: Dataset, logits: Tuple[List, List], references: Tuple[List, List]) -> Tuple[List, List]:
        all_start_logits, all_end_logits = logits
        # Logits: [T, Seq]

        # If data arrives as [1, X] instead of [X]
        all_start_logits = [l_.flatten() for l_ in all_start_logits]
        all_end_logits = [l_.flatten() for l_ in all_end_logits]

        offset_mapping = dataset['offset_mapping']

        final_preds = []

        for i, (start_logits, end_logits) in enumerate(zip(all_start_logits, all_end_logits)):
            num_to_consider = 20  # Tune this
            start_idxs = np.argsort(start_logits)[-num_to_consider:]
            end_idxs = np.argsort(end_logits)[-num_to_consider:]

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

            # Add 0 entry
            null_score = all_start_logits[i][0] + all_end_logits[i][0]
            preds.append({
                "offsets": (0, 0),
                "score": null_score,
                "start_logit": all_start_logits[i][0],
                "end_logit": all_end_logits[i][0],
            })

            # Pick best n options
            preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:20]

            # Return best option early without threshold
            final_preds.append(dataset["context"][i][preds[0]["offsets"][0]:preds[0]["offsets"][1]])

            # scores = np.array([pred["score"] for pred in preds])
            # exp_scores = np.exp(scores - np.max(scores))
            # probs = exp_scores / exp_scores.sum()
            #
            # context = dataset["context"][i]
            # for j, prob in enumerate(probs):
            #     preds[j]["prob"] = prob
            #     preds[j]["text"] = context[preds[j]["offsets"][0]:preds[j]["offsets"][1]]
            #
            # # Pick best prediction
            # i = 0
            # while preds[i]["text"] == "":
            #     i += 1
            # best_non_null_pred = preds[i]
            #
            # if null_score > best_non_null_pred["score"]:
            #     final_preds.append("")
            # else:
            #     final_preds.append(best_non_null_pred["text"])

        return final_preds, references


class PredictionFlattener(Converter):
    def convert(self, dataset: Dataset, logits: List[int], references: List[int]) -> Tuple[List, List]:
        # Test evaluation returns references as a List[np.ndarray]] with one int each
        if not isinstance(references[0], int):
            references = np.array(references).flatten()

        return logits, references


class TupleTransposer(Converter):
    def transpose(self, values) -> List:
        num_entries = len(values[0])
        lists = [[] for _ in range(num_entries)]
        for j in range(num_entries):
            for i in range(len(values)):
                lists[j].append(values[i][j])

        return lists

    def convert(self, dataset: Dataset, logits: List, references: List) -> Tuple[List, List]:
        if isinstance(logits[0], list):
            logits = self.transpose(logits)

        if isinstance(references[0], list):
            references = self.transpose(references)

        # Logits: N arrays of [T, Seq]
        # References: N arrays of [T, L] where L is whatever suits (e.g. 1 for SQuAD)
        return logits, references
