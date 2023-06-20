import random
from typing import Any

import numpy as np
import torch

EPS: float = 1e-8


class DotDict(dict):
    def __getattr__(self, name: str) -> Any:
        return self[name]


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
