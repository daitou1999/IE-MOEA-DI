import numpy as np
from pymoo.core.problem import Problem


class ZDT6(Problem):

    def __init__(self, n_var=10, n_obj=2):
        xl = np.zeros(n_var, dtype=np.float64)
        xu = np.ones(n_var, dtype=np.float64)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1 - np.exp(-4 * x[:, 0]) * np.sin(6 * np.pi * x[:, 0]) ** 6
        g = 1 + 9 * np.mean(x[:, 1:], axis=1) ** 0.25
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        out["F"] = np.column_stack([f1, f2])

    def get_optimum(self, n=100):
        minf1 = 0.280775
        f1 = np.linspace(minf1, 1, n)
        f2 = 1 - f1 ** 2
        return np.column_stack([f1, f2])

    def get_pf(self):
        return self.get_optimum(100)