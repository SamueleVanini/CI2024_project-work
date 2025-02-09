import copy
import operator
from random import Random

import numpy as np
from tqdm import tqdm
from src.toolbox.gp_bloat_control import ind_limit
from src.toolbox.gp_objects import ADFIndividual, Compiler, SimpleCompiler, Individual, PrimitiveSet, Tree
from src.toolbox.gp_expr_op import xover, point_mut
from src.gp_random import PyRndGenerator
from src.toolbox.gp_statistics import Statistics


xover = ind_limit(xover, measure=operator.attrgetter("height"), max_limit=10)
point_mut = ind_limit(point_mut, measure=operator.attrgetter("height"), max_limit=10)


def mu_comma_lambda(
    population: list[Individual],
    ngeneration: int,
    mu: int,
    lambd: int,
    mprob: float,
    cprob: float,
    pset: PrimitiveSet | list[PrimitiveSet],
    fitness_func,
    stat: Statistics,
) -> tuple[list[Individual], list[tuple[int, dict[str, float]]], Individual]:
    assert (mprob + cprob) == 1, "ERROR: the sum of mutation probability and crossover probability must be equal to 1"
    assert lambd > mu, "ERROR: lambd must be greater than mu"

    stats_history = []
    best = population[0]

    if isinstance(population[0], Tree):
        gen_of = gen_offsprings
    else:
        gen_of = adf_gen_offspring

    for ngen in tqdm(range(1, ngeneration + 1), desc="algo"):

        offsprings = gen_of(population, lambd, mprob, cprob, pset)  # type: ignore

        new_pop = []
        for of in offsprings:
            # of.fitness = fitness_func(of)
            fit = fitness_func(of)
            if fit != -1:
                of.fitness = fit
                new_pop.append(of)
        # offsprings.sort(key=lambda of: of.fitness)
        new_pop.sort(key=lambda of: of.fitness)

        if len(new_pop) < mu:
            for idx, _ in enumerate(range(mu - len(new_pop))):
                new_pop.append(population[idx])

        # population[:] = offsprings[:mu]
        population[:] = new_pop[:mu]

        if population[0].fitness < best.fitness:
            best = copy.deepcopy(population[0])

        stats = stat.compute_stats(population)
        stats_history.append((ngen, stats))

    return population, stats_history, best


def gen_offsprings(population: list[Tree], lambd: int, mprob: float, cprob: float, pset: PrimitiveSet) -> list[Tree]:
    offsprings = []
    noffsprings = 0
    gen: Random = PyRndGenerator().gen
    # TODO: for now implement uniform selection. Change it later to introduce new selection types
    while noffsprings < lambd:
        op_choice = gen.random()
        if op_choice < cprob:
            ind1, ind2 = gen.sample(population, 2)
            of1, of2 = xover(ind1, ind2)
            offsprings.append(of1)
            offsprings.append(of2)
            noffsprings += 2
        else:
            ind1 = gen.choice(population)
            offsprings.append(point_mut(ind1, pset)[0])
            noffsprings += 1
    return offsprings


def adf_gen_offspring(
    population: list[ADFIndividual], lambd: int, mprob: float, cprob: float, pset: list[PrimitiveSet]
):
    offsprings = []
    noffsprings = 0
    gen: Random = PyRndGenerator().gen
    # TODO: for now implement uniform selection. Change it later to introduce new selection types
    while noffsprings < lambd:

        par1, par2 = gen.sample(population, 2)
        of1 = []
        of2 = []
        crossed = False
        for tree1, tree2 in zip(par1, par2):
            if gen.random() < cprob:
                tof1, tof2 = xover(tree1, tree2)
                of1.append(tof1)
                of2.append(tof2)
                crossed = True
        if crossed:
            offsprings.append(ADFIndividual(of1))
            offsprings.append(ADFIndividual(of2))
            noffsprings += 2

        for ind in population:
            mutated = False
            of = []
            for idx, tree in enumerate(ind):
                if gen.random() < cprob:
                    of.append(point_mut(tree, pset[idx])[0])
                    mutated = True
            if mutated:
                offsprings.append(ADFIndividual(of))
                noffsprings += 1
    return offsprings


def mse(ind: Tree, compiler: Compiler, x, y):
    y_pred = []
    func, code = compiler.compile(ind)
    for xvar in x:
        try:
            y_pred.append(func(*xvar))
        except:
            # print(
            #     f"ERROR: the function {code} can't operate on the values {xvar} skipping rest of fitness computation and assign it value -1"
            # )
            return -1
    return 100 * np.square(y - y_pred).sum() / len(y)
