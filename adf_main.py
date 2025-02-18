from random import Random
import numpy as np

from functools import partial
from src.gp_random import PyRndGenerator
from src.plots import plot_fitness_history, plot_func
from src.project_primitive import simple_pset, test_pset, complete_pset
from src.toolbox.gp_algorithms import custom_adf_gen_offspring, mu_comma_lambda, mse
from src.toolbox.gp_objects import ADFCompiler
from src.toolbox.gp_population import ADFIndividualBuilder, get_init_population
from src.toolbox.gp_expr_op import gen_half_and_half, gen_full_expr
from src.toolbox.gp_statistics import Statistics


problem = np.load("data/problem_3.npz")
x = problem["x"]
y = problem["y"]

NGEN = 30
LAMBD = 150
MU = 50
NVAR = x.shape[0]

gen: Random = PyRndGenerator().gen

pset0 = simple_pset(NVAR, "ADF_1")
pset0.addTerminal(partial(gen.randint, -10, 10), "rand101")
pset0.addTerminal(partial(gen.randint, -10, 10), "rand101")

pset1 = simple_pset(NVAR, "ADF_2")
pset0.addTerminal(partial(gen.randint, -10, 10), "rand101")
pset1.addADF(pset0)

pset2 = simple_pset(NVAR, "MAIN")
pset0.addTerminal(partial(gen.randint, -10, 10), "rand101")
pset2.addADF(pset1)
pset2.addADF(pset0)

psets = [pset2, pset1, pset0]
gen_funcs = [gen_half_and_half, gen_full_expr, gen_full_expr]
gen_kwargs = [
    {"min_height": 2, "max_height": 5},
    {"min_height": 1, "max_height": 3},
    {"min_height": 1, "max_height": 3},
]

compiler = ADFCompiler(psets)
ind_builder = ADFIndividualBuilder(psets=psets, gen_funcs=gen_funcs, gens_kwargs=gen_kwargs)

# generation_func = partial(gen_half_and_half, min_height=1, max_height=2)


fit_func = partial(mse, compiler=compiler, x=x.T, y=y)

pop = get_init_population(ind_builder, fit_func, nind=MU)

gen_off_func = partial(custom_adf_gen_offspring, mprob=0.05, cprob=0.95, lambd=LAMBD, pset=psets)

stat = Statistics(lambda x: x.fitness)
final_pop, stats, best = mu_comma_lambda(pop, NGEN, MU, gen_off_func, fit_func, stat)
print(stats[-1][1])
for tree in best:
    print(tree)
print(best.fitness)
plot_fitness_history(stats)
# plot_func(compiler.compile(final_pop[0])[0], x, y)  # type: ignore
