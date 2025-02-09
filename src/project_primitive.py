import numpy as np
from src.toolbox.gp_objects import PrimitiveSet


# TODO: we are not including np.sqrt since we are not impose checking on the arguments of a function
def complete_pset(nvar: int) -> PrimitiveSet:

    pset = PrimitiveSet("COMPLETE", nvar)

    # Trigonometric functions
    pset.addPrimitive(np.sin)
    pset.addPrimitive(np.cos)
    pset.addPrimitive(np.tan)
    pset.addPrimitive(np.arcsin)
    pset.addPrimitive(np.asin)
    pset.addPrimitive(np.arccos)
    pset.addPrimitive(np.acos)
    pset.addPrimitive(np.arctan)
    pset.addPrimitive(np.atan)
    pset.addPrimitive(np.hypot)
    pset.addPrimitive(np.arctan2)
    pset.addPrimitive(np.atan2)

    # Hyperbolic
    pset.addPrimitive(np.sinh)
    pset.addPrimitive(np.cosh)
    pset.addPrimitive(np.tanh)
    pset.addPrimitive(np.arcsinh)
    pset.addPrimitive(np.asinh)
    pset.addPrimitive(np.arccosh)
    pset.addPrimitive(np.acosh)
    pset.addPrimitive(np.arctanh)
    pset.addPrimitive(np.atanh)

    # Exponents and logarithms
    pset.addPrimitive(np.exp)
    pset.addPrimitive(np.expm1)
    pset.addPrimitive(np.exp2)
    # pset.addPrimitive(np.log)
    # pset.addPrimitive(np.log10)
    # pset.addPrimitive(np.log2)
    pset.addPrimitive(np.log1p)
    pset.addPrimitive(np.logaddexp)
    pset.addPrimitive(np.logaddexp2)

    # Floating point routines
    pset.addPrimitive(np.ldexp)

    # Aritmetic operations
    pset.addPrimitive(np.add)
    pset.addPrimitive(np.reciprocal)
    pset.addPrimitive(np.negative)
    pset.addPrimitive(np.multiply)
    pset.addPrimitive(np.divide)
    # pset.addPrimitive(np.power)
    pset.addPrimitive(np.subtract)
    pset.addPrimitive(np.floor_divide)
    pset.addPrimitive(np.fmod)
    pset.addPrimitive(np.mod)

    # pset.addPrimitive(np.sqrt)

    return pset


def simple_pset(nvar: int) -> PrimitiveSet:

    pset = PrimitiveSet("SIMPLE", nvar)

    pset.addPrimitive(np.sin)
    pset.addPrimitive(np.cos)
    pset.addPrimitive(np.tan)

    pset.addPrimitive(np.exp)
    pset.addPrimitive(np.exp2)
    # pset.addPrimitive(np.log)
    # pset.addPrimitive(np.log10)
    # pset.addPrimitive(np.log2)

    pset.addPrimitive(np.add)
    pset.addPrimitive(np.reciprocal)
    pset.addPrimitive(np.negative)
    pset.addPrimitive(np.multiply)
    pset.addPrimitive(np.divide)
    # pset.addPrimitive(np.power)
    pset.addPrimitive(np.subtract)
    # pset.addPrimitive(np.floor_divide)
    # pset.addPrimitive(np.fmod)
    # pset.addPrimitive(np.mod)

    # pset.addPrimitive(np.sqrt)

    return pset


def test_pset(nvar: int) -> PrimitiveSet:

    pset = PrimitiveSet("TEST", nvar)

    pset.addPrimitive(np.add)
    pset.addPrimitive(np.multiply)
    pset.addPrimitive(np.subtract)

    return pset
