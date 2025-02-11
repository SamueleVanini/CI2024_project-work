import numpy as np
from random import Random
from functools import partial

from src.gp_random import PyRndGenerator
from src.toolbox.gp_algorithms import gen_offsprings, tournament_selection

gen: Random = PyRndGenerator().gen

# RUNS = [
#     {
#         "primitives": [
#             np.sin,
#             np.cos,
#             np.tan,
#             np.arcsin,
#             np.asin,
#             np.arccos,
#             np.acos,
#             np.arctan,
#             np.atan,
#             np.hypot,
#             np.arctan2,
#             np.atan2,
#             np.sinh,
#             np.cosh,
#             np.tanh,
#             np.arcsinh,
#             np.asinh,
#             np.arccosh,
#             np.acosh,
#             np.arctanh,
#             np.atanh,
#             np.exp,
#             np.expm1,
#             np.exp2,
#             np.log1p,
#             np.logaddexp,
#             np.logaddexp2,
#             np.add,
#             np.reciprocal,
#             np.negative,
#             np.multiply,
#             np.divide,
#             np.subtract,
#             np.floor_divide,
#             np.fmod,
#             np.mod,
#         ],
#         "terminals": [
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#             (partial(gen.randint, -10, 10), "rand101"),
#         ],
#         "is_adf": False,
#         "min_height": 3,
#         "max_height": 6,
#         "mprob": 0.3,
#         "cprob": 0.7,
#         "ngen": 50,
#         "lambd": 400,
#         "mu": 700,
#         "gen_off_func": gen_offsprings,
#         # "problems": [1, 2, 3, 4, 5, 6, 7, 8],
#         # "problems": [3, 4, 6, 7],
#         "problems": [3],
#     }
# ]

RUNS = [
    {
        "primitives": [
            np.add,
            np.subtract,
            np.multiply,
            np.sin,
            np.cos,
            np.exp,
            np.abs,
            np.arctan,
            np.cosh,
            np.sinh,
            np.tanh,
        ],
        "terminals": [
            (partial(gen.randint, 1, 10), "rand101"),
            (partial(gen.randint, 1, 10), "rand101"),
            (partial(gen.randint, 1, 10), "rand101"),
        ],
        "gen_off_func": gen_offsprings,
        "selection_func": tournament_selection,
        "is_adf": False,
        "min_height": 4,
        "max_height": 9,
        "mprob": 0.2,
        "cprob": 0.8,
        "ngen": 200,
        "lambd": 500,
        "mu": 500,
        # "problems": [0, 1, 3, 4, 5, 6, 7, 8],
        "problems": [8],
    },
]
