import random

import numpy as np
import torch

SEED = 1234


def seed_rngs():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
