import copy
import operator
from random import Random

import numpy as np
from tqdm import tqdm
from src.toolbox.gp_bloat_control import ind_limit
from src.toolbox.gp_objects import (
    ADFIndividual,
    Compiler,
    Primitive,
    SimpleCompiler,
    Individual,
    PrimitiveSet,
    Terminal,
    Tree,
)
from src.toolbox.gp_expr_op import xover, point_mut, hoist_mut, subtree_mut
from src.gp_random import PyRndGenerator
from src.toolbox.gp_statistics import Statistics


xover = ind_limit(xover, measure=operator.attrgetter("height"), max_limit=10)
point_mut = ind_limit(point_mut, measure=operator.attrgetter("height"), max_limit=10)
# point_mut = ind_limit(hoist_mut, measure=operator.attrgetter("height"), max_limit=10)


def mu_comma_lambda(
    population: list[Individual],
    ngeneration: int,
    mu: int,
    gen_of,
    fitness_func,
    stat: Statistics,
) -> tuple[list[Individual], list[tuple[int, dict[str, float]]], Individual]:
    stats_history = []
    best = population[0]

    for ngen in tqdm(range(1, ngeneration + 1), desc="algo"):

        offsprings = gen_of(population)  # type: ignore

        new_pop = []
        for of in offsprings:
            fit = fitness_func(of)
            if fit != -1:
                of.fitness = fit
                new_pop.append(of)
        new_pop.sort(key=lambda of: of.fitness)

        if len(new_pop) < mu:
            for idx, _ in enumerate(range(mu - len(new_pop))):
                new_pop.append(population[idx])

        population[:] = new_pop[:mu]

        if population[0].fitness < best.fitness:
            best = copy.deepcopy(population[0])

        stats = stat.compute_stats(population)
        stats_history.append((ngen, stats))

    return population, stats_history, best


def mu_plus_lambda(
    population: list[Individual],
    ngeneration: int,
    mu: int,
    gen_off_func,
    fitness_func,
    stat: Statistics,
) -> tuple[list[Individual], list[tuple[int, dict[str, float]]], Individual]:

    stats_history = []
    best = population[0]

    for ngen in tqdm(range(1, ngeneration + 1), desc="algo"):

        offsprings = gen_off_func(population)  # type: ignore

        for of in offsprings:
            fit = fitness_func(of)
            if fit != -1:
                of.fitness = fit
            else:
                of.fitness = float("inf")

        population.extend(offsprings)
        population.sort(key=lambda ind: ind.fitness)

        population = population[:mu]

        if population[0].fitness < best.fitness:
            best = copy.deepcopy(population[0])
            print(f"\n {best.fitness}")

        stats = stat.compute_stats(population)
        stats_history.append((ngen, stats))

    return population, stats_history, best


def gen_offsprings(
    population: list[Tree], lambd: int, mprob: float, cprob: float, pset: PrimitiveSet, selection_func
) -> list[Tree]:
    offsprings = []
    noffsprings = 0
    gen: Random = PyRndGenerator().gen
    while noffsprings < lambd:
        op_choice = gen.random()
        if op_choice < cprob:
            ind1, ind2 = selection_func(population, nind=2)
            of1, of2 = xover(ind1, ind2)
            if ind_has_all_var(of1, pset):
                offsprings.append(of1)
                noffsprings += 1
            if ind_has_all_var(of2, pset):
                offsprings.append(of2)
                noffsprings += 1
        else:
            ind1 = selection_func(population, nind=1)[0]
            of = point_mut(ind1, pset)[0]
            if ind_has_all_var(of, pset):
                offsprings.append(of)
                noffsprings += 1
    return offsprings


def tournament_selection(population, tournament_size: int, nind: int):
    inds = []
    gen: Random = PyRndGenerator().gen
    for _ in range(nind):
        selected = gen.choices(population, k=tournament_size)
        inds.append(min(selected, key=lambda ind: ind.fitness))
    return inds


def uniform_selection(population, nind: int):
    gen: Random = PyRndGenerator().gen
    return gen.choices(population, k=nind)


def ind_has_all_var(ind, pset):
    vars = set(pset.arguments)
    var_found = set()
    for node in ind:
        if isinstance(node, Terminal) and node.is_symbolic:
            var_found.add(node.name)
    return vars == var_found


def custom_adf_gen_offspring(
    population: list[ADFIndividual], mprob: float, cprob: float, lambd: int, pset: list[PrimitiveSet]
):

    noffsprings = 0
    offsprings = []
    gen: Random = PyRndGenerator().gen

    while noffsprings < lambd:

        if gen.random() < cprob:

            par1, par2 = gen.sample(population, 2)

            of1 = copy.deepcopy(par1)
            of2 = copy.deepcopy(par2)

            prim_idx = gen.randrange(1, len(par1[0]))
            prim_name: str = par1[0][prim_idx].name

            if prim_name.startswith("ADF"):
                adf_idx = int(prim_name.split("_")[1])
                of1[adf_idx] = par2[adf_idx]
                of2[adf_idx] = par1[adf_idx]
                offsprings.append(of1)
                offsprings.append(of2)
                noffsprings += 2

        else:
            ind = gen.choice(population)
            of = copy.deepcopy(ind)
            prim_idx = gen.randrange(0, len(ind[0]))
            prim: Primitive = ind[0][prim_idx]
            prim_name: str = prim.name
            if prim_name.startswith("ADF"):
                adf_idx = int(prim_name.split("_")[1])
                of[adf_idx] = point_mut(ind[adf_idx], pset[adf_idx])[0]
                offsprings.append(of)
            else:
                narity = prim.arity
                if narity != 0:
                    new_prim = gen.choice(pset[0].prim_dict[narity])
                    of[0][prim_idx] = new_prim
                else:
                    new_term = gen.choice(pset[0].terminals)
                    of[0][prim_idx] = new_term

                offsprings.append(of)

            noffsprings += 1

    return offsprings


def adf_gen_offspring(
    population: list[ADFIndividual], lambd: int, mprob: float, cprob: float, pset: list[PrimitiveSet]
):
    offsprings = []
    noffsprings = 0
    gen: Random = PyRndGenerator().gen
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
            return -1
    return 100 * np.square(y - y_pred).sum() / len(y)
