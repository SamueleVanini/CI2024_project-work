import copy
from random import Random
from typing import Callable, Iterable
from src.gp_random import PyRndGenerator
from src.toolbox.gp_objects import Primitive, PrimitiveSet, Terminal, Tree


##############
# GENERATION #
##############


def gen_half_and_half(pset: PrimitiveSet, min_height: int, max_height: int) -> Iterable[Primitive | Terminal]:
    gen: Random = PyRndGenerator().gen
    generation_func = gen.choice([gen_full_expr, gen_grow_expr])
    return Tree(generation_func(pset, min_height, max_height))


def gen_full_expr(pset: PrimitiveSet, min_height: int, max_height: int) -> Iterable[Primitive | Terminal]:
    condition = lambda depth, height: depth == height
    return _gen_expr(pset, min_height, max_height, condition)


def gen_grow_expr(pset: PrimitiveSet, min_height: int, max_height: int) -> Iterable[Primitive | Terminal]:
    gen = PyRndGenerator().gen
    condition = lambda depth, height: (depth == height) or (depth >= min_height and gen.random() < pset.terminal_ratio)
    return _gen_expr(pset, min_height, max_height, condition)


def _gen_expr(
    pset: PrimitiveSet, min_height: int, max_height: int, condition: Callable[[int, int], bool]
) -> Iterable[Primitive | Terminal]:
    expr = []
    gen: Random = PyRndGenerator().gen
    height = gen.randint(min_height, max_height)
    stack = [0]
    while len(stack) != 0:
        depth = stack.pop()
        if condition(depth, height):
            term = gen.choice(pset.terminals)
            term.sample_if_needed()
            expr.append(term)
        else:
            prim = gen.choice(pset.primitives)
            expr.append(prim)
            for _ in range(prim.arity):
                stack.append(depth + 1)
    return expr


#########
# XOVER #
#########


def xover(parrent1: Tree, parrent2: Tree) -> tuple[Tree, Tree]:

    gen: Random = PyRndGenerator().gen

    if len(parrent1) < 2 or len(parrent2) < 2:
        # No crossover on single node tree
        return parrent1, parrent2

    offspring1 = copy.deepcopy(parrent1)
    offspring2 = copy.deepcopy(parrent2)

    idx1 = gen.randrange(1, len(offspring1))
    idx2 = gen.randrange(1, len(offspring2))

    slice1 = offspring1.getSubTree(idx1)
    slice2 = offspring2.getSubTree(idx2)

    offspring1[slice1], offspring2[slice2] = offspring2[slice2], offspring1[slice1]

    return offspring1, offspring2


#############
# Mutations #
#############


def subtree_mut(
    individual: Tree, gen_func: Callable[[PrimitiveSet, int, int], Iterable[Terminal | Primitive]], pset: PrimitiveSet
) -> tuple[Tree]:
    # TODO take min_height and max_height out of the function
    min_height = 1
    max_height = 3
    gen: Random = PyRndGenerator().gen
    offspring = copy.deepcopy(individual)
    idx = gen.randrange(len(offspring))
    of_slice = offspring.getSubTree(idx)
    offspring[of_slice] = gen_func(pset, min_height, max_height)
    return (offspring,)


def point_mut(individual: Tree, pset: PrimitiveSet) -> tuple[Tree]:
    gen: Random = PyRndGenerator().gen
    offspring = copy.deepcopy(individual)
    idx = gen.randrange(len(individual))
    node: Primitive | Terminal = individual[idx]
    narity = node.arity
    if narity != 0:
        new_prim = gen.choice(pset.prim_dict[narity])
        offspring[idx] = new_prim
    else:
        new_term = gen.choice(pset.terminals)
        offspring[idx] = new_term

    return (offspring,)


def hoist_mut(individual: Tree) -> tuple[Tree]:
    gen: Random = PyRndGenerator().gen
    idx = gen.randrange(len(individual))
    indslice = individual.getSubTree(idx)
    offspring = Tree(copy.deepcopy(individual[indslice]))
    return (offspring,)


# TODO: investigate on Permutation Mutation
