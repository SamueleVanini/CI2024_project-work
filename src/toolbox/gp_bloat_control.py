from functools import wraps
from random import Random
from src.gp_random import PyRndGenerator


def ind_limit(func, measure, max_limit):

    gen: Random = PyRndGenerator().gen

    @wraps(func)
    def wrapper(*args, **kwargs):
        offsprings = list(func(*args, **kwargs))
        for idx, of in enumerate(offsprings):
            if measure(of) > max_limit:
                parrent_choosen = gen.choice(args)
                offsprings[idx] = parrent_choosen
        return offsprings

    return wrapper
