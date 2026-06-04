import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pymoo.core.problem import Problem


class WFG3(Problem):
    def __init__(self, n_obj=3, k=None, n_var=None):
        self.k = k if k is not None else (n_obj - 1)
        assert self.k % (n_obj - 1) == 0, "K must be a multiple of M-1"

        d0 = n_var if n_var is not None else self.k + 10
        self.n_var = int(np.ceil((d0 - self.k) / 2) * 2 + self.k)
        self.l = self.n_var - self.k

        xl = np.zeros(self.n_var, dtype=np.float64)
        xu = np.arange(2, 2 * self.n_var + 2, 2, dtype=np.float64)
        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        n, _ = x.shape
        m, k, l = self.n_obj, self.k, self.l
        D_const = 1.0
        S = np.arange(2, 2 * m + 2, 2, dtype=np.float64)
        A = np.hstack([1.0, np.zeros(m - 2)])

        z01 = x / self.xu[None, :]

        t1 = np.zeros((n, k + l), dtype=np.float64)
        t1[:, :k] = z01[:, :k]
        t1[:, k:] = self.s_linear(z01[:, k:], 0.35)

        t2 = np.zeros((n, k + l // 2), dtype=np.float64)
        t2[:, :k] = t1[:, :k]
        t2[:, k:] = (t1[:, k::2] + t1[:, k + 1::2] + 2 * np.abs(t1[:, k::2] - t1[:, k + 1::2])) / 3.0

        t3 = np.zeros((n, m), dtype=np.float64)
        seg = k // (m - 1)
        for i in range(m - 1):
            t3[:, i] = self.r_sum(t2[:, i * seg:(i + 1) * seg], np.ones(seg))
        t3[:, -1] = self.r_sum(t2[:, k:], np.ones(l // 2))

        x_wfg = np.zeros((n, m), dtype=np.float64)
        for i in range(m - 1):
            x_wfg[:, i] = np.maximum(t3[:, -1], A[i]) * (t3[:, i] - 0.5) + 0.5
        x_wfg[:, -1] = t3[:, -1]

        h = self.linear(x_wfg)
        out["F"] = D_const * x_wfg[:, [-1]] + S[None, :] * h

    @staticmethod
    def s_linear(y, A):
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)

    @staticmethod
    def r_sum(y, w):
        return np.sum(y * w[None, :], axis=1) / np.sum(w)

    @staticmethod
    def linear(x):
        n = x.shape[0]
        cp = np.cumprod(np.hstack([np.ones((n, 1)), x[:, :-1]]), axis=1)
        return np.fliplr(cp) * np.hstack([np.ones((n, 1)), 1 - x[:, -2::-1]])