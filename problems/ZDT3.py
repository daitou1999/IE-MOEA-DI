import numpy as np
from pymoo.core.problem import Problem


class ZDT3(Problem):

    def __init__(self, n_var=30, n_obj=2):
        xl = np.zeros(n_var, dtype=np.float64)
        xu = np.ones(n_var, dtype=np.float64)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        out["F"] = np.column_stack([f1, f2])

    @staticmethod
    def _nondominated_2d(F):
        idx = np.argsort(F[:, 0], kind="mergesort")
        Fs = F[idx]
        keep = np.zeros(len(Fs), dtype=bool)
        best_f2 = np.inf
        for i in range(len(Fs)):
            if Fs[i, 1] < best_f2:
                keep[i] = True
                best_f2 = Fs[i, 1]
        mask = np.zeros(len(F), dtype=bool)
        mask[idx] = keep
        return mask

    def get_optimum(self, n=1000):
        f1 = np.linspace(0, 1, n)
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        R = np.column_stack([f1, f2])
        return R[self._nondominated_2d(R)]

    def get_pf(self):
        f1 = np.linspace(0, 1, 100)
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        R = np.column_stack([f1, f2])
        mask = self._nondominated_2d(R)
        R[~mask, :] = np.nan
        return R