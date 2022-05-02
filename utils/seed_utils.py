import random

import numpy as np
import torch

SEED = 1234


# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/trainer_utils.py#L49
def seed_rngs():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
