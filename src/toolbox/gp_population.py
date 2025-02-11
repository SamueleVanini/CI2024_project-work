from typing import Any, Callable, Protocol
from src.toolbox.gp_algorithms import ind_has_all_var
from src.toolbox.gp_objects import ADFIndividual, Individual, PrimitiveSet, Tree


class IndividualBuilder(Protocol):

    def gen_ind(self) -> Individual: ...


# TODO: find correct type hint for partial function
def get_init_population(builder: IndividualBuilder, fit_func, nind: int = 30) -> list[Individual]:
    pop = []
    while nind > 0:
        ind = builder.gen_ind()
        fit = fit_func(ind)
        if fit != -1:
            ind.fitness = fit
            pop.append(ind)
            nind -= 1
    pop.sort(key=lambda ind: ind.fitness)
    return pop


class SimpleIndividualBuilder:

    def __init__(self, pset: PrimitiveSet, gen_func, gen_kwargs: dict[str, Any]) -> None:
        self.gen_func = gen_func
        self.pset = pset
        self.gen_kwargs = gen_kwargs

    def gen_ind(self) -> Individual:
        valid = False
        while not valid:
            ind = Tree(self.gen_func(pset=self.pset, **self.gen_kwargs))
            if ind_has_all_var(ind, self.pset):
                valid = True
        return ind


class ADFIndividualBuilder:

    # main must always be first

    def __init__(self, psets: list[PrimitiveSet], gen_funcs: list[Callable], gens_kwargs: list[dict[str, Any]]) -> None:
        self.gen_funcs = gen_funcs
        self.psets = psets
        self.gens_kwargs = gens_kwargs

    def gen_ind(self) -> Individual:
        ind = []
        for func, pset, gen_kwargs in zip(self.gen_funcs, self.psets, self.gens_kwargs):
            ind.append(Tree(func(pset=pset, **gen_kwargs)))
        return ADFIndividual(ind)
