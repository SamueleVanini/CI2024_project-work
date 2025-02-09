from functools import partial
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.exp_log import RunLogger
from src.runs import RUNS
from src.toolbox.gp_algorithms import mse, mu_comma_lambda
from src.toolbox.gp_expr_op import gen_half_and_half
from src.toolbox.gp_objects import PrimitiveSet, SimpleCompiler
from src.toolbox.gp_population import SimpleIndividualBuilder, get_init_population
from src.toolbox.gp_statistics import Statistics

logger = RunLogger(Path("runs"), "pr_2_bigger_tree")

for run in tqdm(RUNS, position=0, desc="runs"):

    logger.add_hyp(**run)

    for id_problem in tqdm(run["problems"], position=1, desc="problems"):

        problem = np.load(f"data/problem_{id_problem}.npz")
        x = problem["x"]
        y = problem["y"]
        nvar = x.shape[0]

        ngen = run["ngen"]
        lambd = run["lambd"]
        mu = run["mu"]
        mprob = run["mprob"]
        cprob = run["cprob"]
        min_height = run["min_height"]
        max_height = run["max_height"]

        pset = PrimitiveSet("SIMPLE", nvar)
        for primitive in run["primitives"]:
            pset.addPrimitive(primitive)
        for terminal, name in run["terminals"]:
            pset.addTerminal(terminal, name)

        compiler = SimpleCompiler(pset)
        ind_builder = SimpleIndividualBuilder(pset, gen_half_and_half, {"min_height": 1, "max_height": 3})
        fit_func = partial(mse, compiler=compiler, x=x.T, y=y)
        pop = get_init_population(ind_builder, fit_func, nind=mu)
        stat = Statistics(lambda x: x.fitness)

        final_pop, stats, best = mu_comma_lambda(pop, ngen, mu, lambd, mprob, cprob, pset, fit_func, stat)

        logger.set_problem(id_problem)
        logger.add_stats(stats)
        logger.add_champion(best)

    logger.commit()
