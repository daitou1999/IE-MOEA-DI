import numpy as np
from pymoo.core.problem import Problem


class ZDT5(Problem):

    def __init__(self, n_var=None, n_obj=None):
        D_in = 80 if n_var is None else int(n_var)

        D = int(np.ceil(max(D_in - 30, 1) / 5.0) * 5 + 30)

        self.n_var = D
        self.n_obj = 2

        xl = np.zeros(D, dtype=float)
        xu = np.ones(D, dtype=float)

        super().__init__(n_var=D, n_obj=2, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, PopDec, out, *args, **kwargs):

        PopDec = np.asarray(PopDec, dtype=float)
        N, D = PopDec.shape

        n_u = 1 + (D - 30) // 5
        u = np.zeros((N, n_u), dtype=float)

        u[:, 0] = np.sum(PopDec[:, 0:30], axis=1)

        for i in range(2, n_u + 1):
            start = (i - 2) * 5 + 30
            end = start + 5
            u[:, i - 1] = np.sum(PopDec[:, start:end], axis=1)

        v = np.zeros_like(u)
        v[u < 5] = 2 + u[u < 5]
        v[u == 5] = 1

        f1 = 1 + u[:, 0]
        g = np.sum(v[:, 1:], axis=1)
        h = 1.0 / f1
        f2 = g * h

        out["F"] = np.column_stack([f1, f2])

    def get_optimum(self, N=None):
        R1 = np.arange(1, 32, dtype=float)
        R2 = (self.n_var - 30) / 5.0 / R1
        return np.column_stack([R1, R2])