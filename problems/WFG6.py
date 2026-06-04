import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pymoo.core.problem import Problem


class WFG6(Problem):
    def __init__(self, n_obj=3, k=None, n_var=None):
        self.k = k if k is not None else (n_obj - 1)
        assert self.k % (n_obj - 1) == 0, "K must be a multiple of M-1"

        self.n_var = n_var if n_var is not None else self.k + 10
        self.l = self.n_var - self.k

        xl = np.zeros(self.n_var, dtype=np.float64)
        xu = np.arange(2, 2 * self.n_var + 2, 2, dtype=np.float64)
        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        n, _ = x.shape
        m, k, l = self.n_obj, self.k, self.l
        D_const = 1.0
        S = np.arange(2, 2 * m + 2, 2, dtype=np.float64)
        A = np.ones(m - 1, dtype=np.float64)

        z01 = x / self.xu[None, :]

        t1 = np.zeros((n, k + l), dtype=np.float64)
        t1[:, :k] = z01[:, :k]
        t1[:, k:] = self.s_linear(z01[:, k:], 0.35)

        t2 = np.zeros((n, m), dtype=np.float64)
        seg = k // (m - 1)
        for i in range(m - 1):
            t2[:, i] = self.r_nonsep(t1[:, i * seg:(i + 1) * seg], seg)

        sum_abs = np.zeros((n, 1), dtype=np.float64)
        for i in range(k, k + l - 1):
            for j in range(i + 1, k + l):
                sum_abs += np.abs(t1[:, [i]] - t1[:, [j]])
        t2[:, -1] = ((np.sum(t1[:, k:k + l], axis=1, keepdims=True) + 2 * sum_abs) /
                     (np.ceil(l / 2) * (1 + 2 * l - 2 * np.ceil(l / 2)))).ravel()

        x_wfg = np.zeros((n, m), dtype=np.float64)
        for i in range(m - 1):
            x_wfg[:, i] = np.maximum(t2[:, -1], A[i]) * (t2[:, i] - 0.5) + 0.5
        x_wfg[:, -1] = t2[:, -1]

        h = self.concave(x_wfg)
        out["F"] = D_const * x_wfg[:, [-1]] + S[None, :] * h

    @staticmethod
    def s_linear(y, A):
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)

    @staticmethod
    def r_nonsep(y, A):
        n, d = y.shape
        out = np.zeros(n, dtype=np.float64)
        for j in range(d):
            tmp = np.zeros(n, dtype=np.float64)
            for kk in range(A - 1):
                idx = (j + kk) % d
                tmp += np.abs(y[:, j] - y[:, idx])
            out += y[:, j] + tmp
        return out / ((d / A) * np.ceil(A / 2) * (1 + 2 * A - 2 * np.ceil(A / 2)))

    @staticmethod
    def concave(x):
        n = x.shape[0]
        cp = np.cumprod(np.hstack([np.ones((n, 1)), np.sin(x[:, :-1] * np.pi / 2)]), axis=1)
        return np.fliplr(cp) * np.hstack([np.ones((n, 1)), np.cos(x[:, -2::-1] * np.pi / 2)])