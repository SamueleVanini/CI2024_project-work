from abc import ABC
from collections import defaultdict
import numbers
import sys
from typing import Callable, Iterable, Protocol, Self
import numpy as np
from numpy import ufunc


NumberType = float | int | numbers.Real | numbers.Number


class Primitive:

    __slots__ = ("name", "arity", "str_code")

    def __init__(self, name: str, arity: int) -> None:
        self.name = name
        self.arity = arity
        args = ", ".join(map("{{{0}}}".format, range(self.arity)))
        # str_code is formated like func_name({0}, {1}, {2})
        self.str_code = f"{name}({args})"

    def format(self, *args):
        return self.str_code.format(*args)


class PrimitiveSet:

    def __init__(self, name: str, nvariables: int) -> None:
        self.name = name
        self.nvariables = nvariables
        self.term_count = 0
        self.prims_count = 0
        self.primitives: list[Primitive] = []
        self.prim_dict: dict[int, list[Primitive]] = defaultdict(list)
        self.terminals: list[Terminal] = []
        self.arguments = []
        self.context = {"__builtins__": None}
        for idx in range(nvariables):
            arg_str = "x{idx}".format(idx=idx)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True)
            self.terminals.append(term)
            self.term_count += 1

    def addPrimitive(self, func: ufunc) -> None:
        name = func.__name__
        arity = func.nin
        prim = Primitive(name, arity)
        self.primitives.append(prim)
        self.prim_dict[arity].append(prim)
        self.prims_count += 1

    def addTerminal(self, value: NumberType | Callable[..., NumberType], name: str | None = None) -> None:
        term = Terminal(value, False, name)
        self.terminals.append(term)
        self.term_count += 1

    def addADF(self, adfSet: Self):
        prim = Primitive(adfSet.name, adfSet.nvariables)
        self.primitives.append(prim)
        self.prim_dict[adfSet.nvariables].append(prim)
        self.prims_count += 1

    @property
    def terminal_ratio(self) -> float:
        return self.term_count / (self.term_count + self.prims_count)


class Terminal:

    __slots__ = ("_value", "_func", "is_symbolic", "name", "format_class")

    def __init__(
        self, value: str | NumberType | Callable[..., NumberType], is_symbolic: bool, name: str | None = None
    ) -> None:
        if isinstance(value, Callable):
            self._func = value
            self._value = None
        else:
            self._value = value
            self._func = None
        if name is None:
            self.name = str(value)
        else:
            self.name = name
        self.is_symbolic = is_symbolic
        self.format_class = str if is_symbolic else repr

    @property
    def arity(self) -> int:
        return 0

    def sample_if_needed(self) -> None:
        if not self.is_symbolic and self._func is not None:
            self._value = self._func()

    def format(self) -> str:
        assert self._value is not None, "ERROR: a terminal value that is not symbolic does not have a value"
        return self.format_class(self._value)


class Individual(ABC):

    def __init__(self) -> None:
        self._fitness: float = -1

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = value


class Tree(list, Individual):

    def __init__(self, content: Iterable[Primitive | Terminal]):
        list.__init__(self, content)
        Individual.__init__(self)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice) and isinstance(value, Iterable):
            if key.start >= len(self):
                raise IndexError("Can't set and item in a position %s when tree has size %d" % (key, len(self)))
            arity_to_satisfy = value[0].arity
            for node in value[1:]:
                arity_to_satisfy += node.arity - 1
            if arity_to_satisfy != 0:
                raise ValueError("Invalid slice assignation, insertion of an incomplete subtree is not permitted")
        elif value.arity != self[key].arity:
            raise ValueError(
                "The arity of the replacing node (%d) and the current node (%d) does not match"
                % (value.arity, self[key].arity)
            )
        list.__setitem__(self, key, value)

    def __str__(self):
        """Return the expression in a human readable string."""
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        return string

    @property
    def height(self) -> int:
        stack = [0]
        max_height = 0
        for node in self:
            current_depth = stack.pop()
            max_height = max(current_depth, max_height)
            stack.extend([current_depth + 1] * node.arity)
        return max_height

    def getSubTree(self, index: int) -> slice:
        finish = index + 1
        arity_to_satisfy = self[index].arity
        while arity_to_satisfy > 0:
            arity_to_satisfy += self[finish].arity - 1
            finish += 1
        return slice(index, finish)


class ADFIndividual(list, Individual):

    def __init__(self, content: list[Tree]):
        list.__init__(self, content)
        Individual.__init__(self)


class Compiler(Protocol):

    def compile(self, expr: Tree | ADFIndividual): ...


class SimpleCompiler:

    def __init__(self, pset: PrimitiveSet) -> None:
        self.pset = pset

    def compile(self, expr: Tree):
        return self._compile(expr, self.pset)

    @staticmethod
    def _compile(expr: Tree, pset: PrimitiveSet):
        code = str(expr)
        if pset.nvariables > 0:
            args = ",".join(arg for arg in pset.arguments)
            code = "lambda {args}: {code}".format(code=code, args=args)
        try:
            symbol_table = np.__dict__.copy()
            symbol_table["__builtins__"] = None
            if len(pset.context) > 1:
                symbol_table.update(pset.context)
            return eval(code, symbol_table), code
        except MemoryError as me:
            _, _, traceback = sys.exc_info()
            raise MemoryError(
                "Error in tree evaluation :"
                " Python cannot evaluate a tree higher than 90. "
                "To avoid this problem, you should use bloat control on your "
                "operators."
                "Program will now abort."
            ).with_traceback(traceback)


class ADFCompiler:

    def __init__(self, psets: list[PrimitiveSet]) -> None:
        self.psets = psets

    def compile(self, expr: ADFIndividual):
        adf_context = {}
        func = None
        code = ""
        # for pset, subexpr in reversed(list(zip(self.psets, expr))):
        for pset, subexpr in zip(self.psets[::-1], expr[::-1]):
            pset.context.update(adf_context)
            func, code = SimpleCompiler._compile(subexpr, pset)
            adf_context.update({pset.name: func})
        return func, code
