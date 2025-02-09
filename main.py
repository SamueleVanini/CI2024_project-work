from random import Random
import numpy as np

from functools import partial
from src.gp_random import PyRndGenerator
from src.plots import plot_fitness_history, plot_func
from src.project_primitive import simple_pset, test_pset
from src.toolbox.gp_algorithms import mu_comma_lambda, mse
from src.toolbox.gp_objects import PrimitiveSet, SimpleCompiler
from src.toolbox.gp_population import SimpleIndividualBuilder, get_init_population
from src.toolbox.gp_expr_op import gen_half_and_half
from src.toolbox.gp_statistics import Statistics


problem = np.load("data/problem_2.npz")
x = problem["x"]
y = problem["y"]

NGEN = 50
LAMBD = 300
MU = 100
NVAR = x.shape[0]

gen: Random = PyRndGenerator().gen

pset = PrimitiveSet("SIMPLE", NVAR)
pset.addPrimitive(np.sin)
pset.addPrimitive(np.cos)
pset.addPrimitive(np.pow)
pset.addPrimitive(np.add)
pset.addPrimitive(np.negative)
pset.addPrimitive(np.multiply)
pset.addPrimitive(np.subtract)
pset.addTerminal(partial(gen.randint, -4, 4), "rand101")
pset.addTerminal(partial(gen.randint, -4, 4), "rand101")
pset.addTerminal(partial(gen.randint, -4, 4), "rand101")
pset.addTerminal(partial(gen.randint, -4, 4), "rand101")


compiler = SimpleCompiler(pset)
ind_builder = SimpleIndividualBuilder(pset, gen_half_and_half, {"min_height": 1, "max_height": 3})
fit_func = partial(mse, compiler=compiler, x=x.T, y=y)

# generation_func = partial(gen_half_and_half, min_height=1, max_height=2)

pop = get_init_population(ind_builder, fit_func, nind=MU)
stat = Statistics(lambda x: x.fitness)

final_pop, stats, best = mu_comma_lambda(pop, NGEN, MU, LAMBD, 0.05, 0.95, pset, fit_func, stat)
print(stats[-1][1])
print(best)
plot_fitness_history(stats)
# plot_func(compiler.compile(final_pop[0])[0], x, y)
