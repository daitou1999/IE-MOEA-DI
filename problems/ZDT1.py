import numpy as np
from pymoo.core.problem import Problem


class ZDT1(Problem):

    def __init__(self, n_var=30, n_obj=2):
        xl = np.zeros(n_var, dtype=np.float64)
        xu = np.ones(n_var, dtype=np.float64)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        out["F"] = np.column_stack([f1, f2])

    def get_optimum(self, n=100):
        f1 = np.linspace(0, 1, n)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])

    def get_pf(self):
        return self.get_optimum(100)