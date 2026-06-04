import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pymoo.core.problem import Problem


class WFG9(Problem):

    def __init__(self, n_obj=3, k=None, n_var=None):
        self.k = k if k is not None else (n_obj - 1)
        assert self.k % (n_obj - 1) == 0, "Parameter K must be an integer multiple of M-1 (number of objectives minus one)"

        self.n_var = n_var if n_var is not None else self.k + 10
        self.l = self.n_var - self.k

        xl = np.zeros(self.n_var, dtype=np.float64)
        xu = np.arange(2, 2 * self.n_var + 2, 2, dtype=np.float64)

        super().__init__(
            n_var=self.n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            vtype=np.float64
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        m = self.n_obj
        k = self.k
        l = self.l

        D_const = 1
        S = np.arange(2, 2 * m + 2, 2, dtype=np.float64)
        A = np.ones(m - 1, dtype=np.float64)

        z01 = x / self.xu[None, :]

        t1 = np.zeros_like(z01)
        flipped_z01 = np.fliplr(z01)
        cumsum_flip = np.cumsum(flipped_z01, axis=1)
        flip_cumsum = np.fliplr(cumsum_flip)
        div_arr = np.arange(k + l - 1, -1, -1, dtype=np.float64)[None, :]
        Y = (flip_cumsum - z01) / div_arr

        base = 0.02 + (50 - 0.02) * (0.98 / 49.98 - (1 - 2 * Y[:, :-1]) *
                                     np.abs(np.floor(0.5 - Y[:, :-1]) + 0.98 / 49.98))
        t1[:, :-1] = z01[:, :-1] ** base
        t1[:, -1] = z01[:, -1]

        t2 = np.zeros_like(t1)
        t2[:, :k] = self.s_decept(t1[:, :k], 0.35, 0.001, 0.05)
        t2[:, k:] = self.s_multi(t1[:, k:], 30, 95, 0.35)

        t3 = np.zeros((n, m), dtype=np.float64)
        seg_len = k // (m - 1)

        for i in range(m - 1):
            start, end = i * seg_len, (i + 1) * seg_len
            t3[:, i] = self.r_nonsep(t2[:, start:end], seg_len)

        sum_abs = np.zeros((n, 1), dtype=np.float64)
        for i in range(k, k + l - 1):
            for j in range(i + 1, k + l):
                sum_abs += np.abs(t2[:, [i]] - t2[:, [j]])
        sum_t2 = np.sum(t2[:, k:], axis=1, keepdims=True)
        divisor = np.ceil(l / 2) * (1 + 2 * l - 2 * np.ceil(l / 2))
        t3[:, -1] = (sum_t2 + sum_abs * 2).flatten() / divisor

        x_wfg = np.zeros((n, m), dtype=np.float64)
        for i in range(m - 1):
            x_wfg[:, i] = np.maximum(t3[:, -1], A[i]) * (t3[:, i] - 0.5) + 0.5
        x_wfg[:, -1] = t3[:, -1]

        h = self.concave(x_wfg)
        out["F"] = D_const * x_wfg[:, [-1]] + S[None, :] * h

    def b_param(self, y, Y, A, B, C):
        return y ** (B + (C - B) * (A - (1 - 2 * Y) * np.abs(np.floor(0.5 - Y) + A)))

    def r_sum(self, y, w):
        return np.sum(y * w[None, :], axis=1) / np.sum(w)

    def s_decept(self, y, A, B, C):
        term1 = np.abs(y - A) - B
        term2 = np.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B)
        term3 = np.floor(A + B - y) * (1 - C + (1 - A - B) / B) / (1 - A - B)
        return 1 + term1 * (term2 + term3 + 1 / B)

    def s_multi(self, y, A, B, C):
        denom = np.floor(C - y) + C
        denom[denom == 0] = 1e-10
        abs_val = np.abs(y - C) / (2 * denom)
        cos_term = np.cos((4 * A + 2) * np.pi * (0.5 - abs_val))
        return (1 + cos_term + 4 * B * abs_val ** 2) / (B + 2)

    def r_nonsep(self, y, A):
        n, d = y.shape
        res = np.zeros(n, dtype=np.float64)
        for j in range(d):
            temp = np.zeros(n, dtype=np.float64)
            for k in range(A - 1):
                idx = np.mod(j + k, d)
                temp += np.abs(y[:, j] - y[:, idx])
            res += y[:, j] + temp
        divisor = (d / A) * np.ceil(A / 2) * (1 + 2 * A - 2 * np.ceil(A / 2))
        return res / divisor

    def concave(self, x):
        n, m = x.shape
        cumprod = np.cumprod(np.hstack([np.ones((n, 1)), np.sin(x[:, :-1] * np.pi / 2)]), axis=1)
        flip_cumprod = np.fliplr(cumprod)
        cos_part = np.hstack([np.ones((n, 1)), np.cos(x[:, -2::-1] * np.pi / 2)])
        return flip_cumprod * cos_part