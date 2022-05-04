from dataclasses import dataclass
from itertools import chain
from typing import List, Dict

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, List[List[int]]]]):
        label_name = "label"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding='longest',
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
