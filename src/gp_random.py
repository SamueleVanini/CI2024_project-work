import random
import numpy as np

from typing import Any
from src.config import SEED


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class NpRndGenerator(metaclass=SingletonMeta):

    def __init__(self) -> None:
        self._gen = np.random.default_rng(SEED)

    @property
    def gen(self) -> np.random.Generator:
        return self._gen


class PyRndGenerator(metaclass=SingletonMeta):

    def __init__(self) -> None:
        self._gen = random.Random(SEED)

    @property
    def gen(self) -> random.Random:
        return self._gen


if __name__ == "__main__":
    rnd_gen = NpRndGenerator()
    a = rnd_gen.gen.random()
    print(a)
    py_gen = PyRndGenerator().gen
    print(py_gen.choice([1, 2, 3]))
