import numpy as np
from random import Random
from functools import partial

from src.gp_random import PyRndGenerator

gen: Random = PyRndGenerator().gen

# RUNS = [
#     {
#         "primitives": [np.sin, np.cos, np.pow, np.add, np.negative, np.multiply, np.subtract],
#         "terminals": [
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#         ],
#         "min_height": 2,
#         "max_height": 3,
#         "is_adf": False,
#         "mprob": 0.05,
#         "cprob": 0.95,
#         "ngen": 100,
#         "lambd": 300,
#         "mu": 100,
#         "problems": [2],
#     },
#     {
#         "primitives": [
#             np.sin,
#             np.cos,
#             np.pow,
#             np.add,
#             np.negative,
#             np.multiply,
#             np.subtract,
#             np.reciprocal,
#             np.divide,
#             np.exp,
#         ],
#         "terminals": [
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#         ],
#         "is_adf": False,
#         "min_height": 2,
#         "max_height": 4,
#         "mprob": 0.05,
#         "cprob": 0.95,
#         "ngen": 100,
#         "lambd": 300,
#         "mu": 100,
#         "problems": [2],
#     },
#     {
#         "primitives": [
#             np.sin,
#             np.cos,
#             np.pow,
#             np.add,
#             np.negative,
#             np.multiply,
#             np.subtract,
#             np.reciprocal,
#             np.divide,
#             np.exp,
#         ],
#         "terminals": [
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#         ],
#         "is_adf": False,
#         "min_height": 2,
#         "max_height": 5,
#         "mprob": 0.05,
#         "cprob": 0.95,
#         "ngen": 100,
#         "lambd": 300,
#         "mu": 100,
#         "problems": [2],
#     },
# ]

RUNS = [
    {
        "primitives": [
            np.sin,
            np.cos,
            np.pow,
            np.add,
            np.negative,
            np.multiply,
            np.subtract,
            np.reciprocal,
            np.divide,
            np.exp,
        ],
        "terminals": [
            (partial(gen.randint, -10, 10), "rand101"),
            (partial(gen.randint, -10, 10), "rand101"),
            (partial(gen.randint, -10, 10), "rand101"),
            (partial(gen.randint, -10, 10), "rand101"),
            (partial(gen.randint, -10, 10), "rand101"),
            (partial(gen.randint, -10, 10), "rand101"),
        ],
        "is_adf": False,
        "min_height": 2,
        "max_height": 7,
        "mprob": 0.05,
        "cprob": 0.95,
        "ngen": 30,
        "lambd": 600,
        "mu": 200,
        # "problems": [0, 1, 3, 4, 5, 6, 7, 8],
        "problems": [2],
    },
]
