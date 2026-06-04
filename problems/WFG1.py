import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pymoo.core.problem import Problem


class WFG1(Problem):
    def __init__(self, n_obj=3, k=None, n_var=None):
        self.k = k if k is not None else (n_obj - 1)
        assert self.k % (n_obj - 1) == 0, "K must be a multiple of M-1"

        self.n_var = n_var if n_var is not None else self.k + 10
        self.l = self.n_var - self.k

        xl = np.zeros(self.n_var, dtype=np.float64)
        xu = np.arange(2, 2 * self.n_var + 2, 2, dtype=np.float64)

        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        m, k, l = self.n_obj, self.k, self.l

        D_const = 1.0
        S = np.arange(2, 2 * m + 2, 2, dtype=np.float64)
        A = np.ones(m - 1, dtype=np.float64)

        z01 = x / self.xu[None, :]

        t1 = np.zeros_like(z01)
        t1[:, :k] = z01[:, :k]
        t1[:, k:] = self.s_linear(z01[:, k:], 0.35)

        t2 = np.zeros_like(z01)
        t2[:, :k] = t1[:, :k]
        t2[:, k:] = self.b_flat(t1[:, k:], 0.8, 0.75, 0.85)

        t3 = self.b_poly(t2, 0.02)

        t4 = np.zeros((n, m), dtype=np.float64)
        for i in range(m - 1):
            start = i * (k // (m - 1))
            end = (i + 1) * (k // (m - 1))
            w = np.arange(2 * (start + 1), 2 * end + 1, 2, dtype=np.float64)
            t4[:, i] = self.r_sum(t3[:, start:end], w)
        w_last = np.arange(2 * (k + 1), 2 * (k + l) + 1, 2, dtype=np.float64)
        t4[:, -1] = self.r_sum(t3[:, k:k + l], w_last)

        x_wfg = np.zeros((n, m), dtype=np.float64)
        for i in range(m - 1):
            x_wfg[:, i] = np.maximum(t4[:, -1], A[i]) * (t4[:, i] - 0.5) + 0.5
        x_wfg[:, -1] = t4[:, -1]

        h = self.convex(x_wfg)
        h[:, -1] = self.mixed(x_wfg)
        out["F"] = D_const * x_wfg[:, [-1]] + S[None, :] * h

    @staticmethod
    def s_linear(y, A):
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)

    @staticmethod
    def b_flat(y, A, B, C):
        out = A + np.minimum(0, np.floor(y - B)) * A * (B - y) / B \
              - np.minimum(0, np.floor(C - y)) * (1 - A) * (y - C) / (1 - C)
        return np.round(out * 1e4) / 1e4

    @staticmethod
    def b_poly(y, a):
        return y ** a

    @staticmethod
    def r_sum(y, w):
        return np.sum(y * w[None, :], axis=1) / np.sum(w)

    @staticmethod
    def convex(x):
        n = x.shape[0]
        cp = np.cumprod(np.hstack([np.ones((n, 1)), 1 - np.cos(x[:, :-1] * np.pi / 2)]), axis=1)
        left = np.fliplr(cp)
        right = np.hstack([np.ones((n, 1)), 1 - np.sin(x[:, -2::-1] * np.pi / 2)])
        return left * right

    @staticmethod
    def mixed(x):
        return 1 - x[:, 0] - np.cos(10 * np.pi * x[:, 0] + np.pi / 2) / (10 * np.pi)