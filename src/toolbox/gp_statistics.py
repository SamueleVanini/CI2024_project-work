from typing import Iterable

import numpy as np
from src.toolbox.gp_objects import Individual


class Statistics:

    def __init__(self, key_lambd=lambda x: x) -> None:
        self.key = key_lambd

    def compute_stats(self, pop: Iterable[Individual]) -> dict[str, float]:
        results = {}
        values = [self.key(ind) for ind in pop]
        results["avg"] = np.mean(values)
        results["std"] = np.std(values)
        results["min"] = np.min(values)
        results["max"] = np.max(values)
        return results
