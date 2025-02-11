import copy
import json
from pathlib import Path

from src.toolbox.gp_objects import Individual


class RunLogger:

    def __init__(self, path: Path, file_name: str | None = None) -> None:
        self.base_path = path
        self.current_file_idx = 0
        if file_name is None:
            self.base_file_name = "run"
        else:
            self.base_file_name = file_name
        self.results = {}
        self._idx_cur_prob = None

    def set_problem(self, idx_prob: int) -> None:
        self._idx_cur_prob = idx_prob
        self.results[f"problem_{idx_prob}"] = {}

    def add_stats(self, stats: list[tuple[int, dict[str, float]]]) -> None:
        self.results[f"problem_{self._idx_cur_prob}"]["trace"] = list()
        for ngen, stat in stats:
            stat["ngen"] = ngen
            self.results[f"problem_{self._idx_cur_prob}"]["trace"].append(stat)

    def add_champion(self, ind: Individual) -> None:
        self.results[f"problem_{self._idx_cur_prob}"]["champ"] = {"function": str(ind), "fitness": ind.fitness}

    def add_hyp(self, **params) -> None:
        hyp = copy.deepcopy(params)
        hyp["primitives"] = [prim.__name__ for prim in hyp["primitives"]]
        hyp["terminals"] = [term[1] for term in hyp["terminals"]]
        self.results["hyp"] = hyp
        self.results["hyp"]["gen_off_func"] = self.results["hyp"]["gen_off_func"].__name__
        self.results["hyp"]["selection_func"] = self.results["hyp"]["selection_func"].__name__

    def commit(self) -> None:
        full_path = self.base_path / (self.base_file_name + f"_{self.current_file_idx}.json")
        with open(full_path, "w") as f:
            json.dump(self.results, f)
        self.current_file_idx += 1
        self.results = {}
        self._idx_cur_prob = None
