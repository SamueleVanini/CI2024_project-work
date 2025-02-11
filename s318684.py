# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return np.exp(np.hypot(np.add(8, x[0]), np.logaddexp2(8, 8)))


def f3(x: np.ndarray) -> np.ndarray:
    # this is better, change it
    return np.add(
        np.add(
            np.multiply(np.multiply(np.sinh(np.sin(np.sinh(np.sinh(np.sinh(np.sin(x[1])))))), x[1]), x[1]),
            np.sinh(np.subtract(np.tanh(np.tanh(np.cos(5))), x[1])),
        ),
        np.subtract(
            np.subtract(
                np.subtract(
                    np.subtract(
                        np.subtract(
                            np.subtract(np.multiply(np.cosh(np.absolute(x[0])), np.cosh(np.cos(x[0]))), x[2]), x[2]
                        ),
                        np.absolute(np.subtract(np.sinh(np.sinh(np.sin(x[1]))), x[1])),
                    ),
                    np.add(x[1], np.add(x[1], x[2])),
                ),
                np.add(np.add(x[1], np.cos(x[0])), np.add(np.add(x[1], np.cos(10)), x[2])),
            ),
            np.add(np.add(x[1], np.cos(10)), np.cos(x[0])),
        ),
    )


def f4(x: np.ndarray) -> np.ndarray:
    return np.add(
        np.subtract(
            np.add(np.cos(np.add(np.cos(x[1]), np.cos(x[1]))), np.cos(x[1])),
            np.multiply(
                np.multiply(np.absolute(3), np.sinh(np.cos(x[1]))),
                np.tanh(np.tanh(np.subtract(np.multiply(x[0], np.sinh(np.cos(x[1]))), 7))),
            ),
        ),
        np.exp(
            np.add(
                np.cos(
                    np.add(
                        np.subtract(np.multiply(x[0], 3), np.multiply(x[0], 3)),
                        np.multiply(np.cos(3), np.subtract(np.cos(x[1]), np.tanh(np.tanh(np.tanh(3))))),
                    )
                ),
                np.cos(x[1]),
            )
        ),
    )


def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(-2, np.multiply(x[1], np.reciprocal(-7)))


def f6(x: np.ndarray) -> np.ndarray:
    return np.add(x[1], np.subtract(np.divide(np.add(np.add(np.add(x[0], x[1]), x[1]), x[1]), 4), x[0]))


def f7(x: np.ndarray) -> np.ndarray:
    return np.exp(
        np.add(
            np.arctan(
                np.multiply(
                    np.add(
                        np.cosh(np.tanh(np.sin(np.add(np.cosh(4), np.multiply(x[1], 6))))),
                        np.sin(np.multiply(x[0], x[0])),
                    ),
                    np.multiply(
                        np.add(
                            np.cosh(np.sin(np.add(np.cosh(6), np.multiply(x[1], x[1])))),
                            np.sin(np.add(np.cosh(6), np.multiply(x[0], x[0]))),
                        ),
                        np.exp(
                            np.multiply(
                                np.multiply(np.arctan(x[1]), np.arctan(x[0])),
                                np.multiply(np.multiply(np.arctan(x[1]), np.arctan(x[0])), np.multiply(x[1], x[1])),
                            )
                        ),
                    ),
                )
            ),
            np.multiply(x[0], x[1]),
        )
    )


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
