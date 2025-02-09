# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray: ...


def f3(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.add(x[1], np.subtract(-7, np.subtract(np.multiply(x[1], x[1]), np.subtract(x[1], -5)))), x[1])


def f4(x: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(6, np.cos(x[1])), np.exp(np.cos(np.subtract(x[1], x[1]))))


def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(-2, np.multiply(x[1], np.reciprocal(-7)))


def f6(x: np.ndarray) -> np.ndarray:
    return np.add(x[1], np.subtract(np.divide(np.add(np.add(np.add(x[0], x[1]), x[1]), x[1]), 4), x[0]))


def f7(x: np.ndarray) -> np.ndarray:
    return np.exp(np.multiply(x[1], x[0]))


def f8(x: np.ndarray) -> np.ndarray:
    return np.multiply(
        np.multiply(
            10,
            np.add(
                np.multiply(np.multiply(9, x[5]), x[5]),
                np.add(
                    np.multiply(np.exp(x[5]), np.cos(x[5])),
                    np.add(x[5], np.add(np.multiply(x[5], np.cos(9)), np.add(9, np.multiply(x[5], np.cos(9))))),
                ),
            ),
        ),
        x[5],
    )
