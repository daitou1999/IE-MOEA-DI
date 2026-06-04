import numpy as np
from pymoo.util.ref_dirs.optimizer import Adam

class RieszEnergyInitialSampling:

    def __init__(self,
                 n_dim,
                 n_points,
                 xl=None,
                 xu=None,
                 n_max_iter=1000,
                 n_until_optimizer_reset=30,
                 norm_gradients=True,
                 verify_gradient=False,
                 precision=1e-5,
                 restarts=True,
                 X=None,
                 d=None,
                 callback=None,
                 verbose=False,
                 SEED=42,
                 **kwargs):
        self.rng = np.random.RandomState(SEED)
        self.n_dim = n_dim
        self.n_points = n_points


        if xl is None:
            xl = np.zeros(n_dim)
        if xu is None:
            xu = np.ones(n_dim)
        self.xl = np.asarray(xl)
        self.xu = np.asarray(xu)

        self.n_max_iter = n_max_iter
        self.n_max_not_improved = n_until_optimizer_reset
        self.X = X
        self.precision = precision
        self.verify_gradient = verify_gradient
        self.norm_gradients = norm_gradients
        self.d = d
        self.callback = callback
        self.restarts = restarts
        self.verbose = verbose


        if self.d is None:
            self.d = n_dim * 2

    def _step(self, optimizer, X, freeze=None):
        if freeze is None:
            freeze = np.full(len(X), False)
        free = np.logical_not(freeze)

        obj, grad, mutual_dist = calc_potential_energy_with_grad(X, self.d, return_mutual_dist=True)

        if self.verify_gradient:
            pass

        grad[freeze] = 0
        proj_grad = grad

        if self.norm_gradients:
            norm = np.linalg.norm(proj_grad, axis=1)
            proj_grad = proj_grad / max(norm.max(), 1e-24)

        X = optimizer.next(X, proj_grad)

        X[free] = np.clip(X[free], self.xl, self.xu)

        return X, obj

    def _solve(self, X, F=None):
        n_points = len(X)
        ret, obj = X, np.inf
        n_not_improved = 0

        freeze = np.full(len(X), False)

        if F is not None:
            X = np.vstack([X, F])
            freeze = np.concatenate([freeze, np.full(len(F), True)])

        if np.all(freeze):
            return X

        optimizer = Adam(alpha=0.005)

        if self.callback is not None:
            self.callback(self, X)

        for i in range(self.n_max_iter):
            _X, _obj = self._step(optimizer, X, freeze=freeze)

            if _obj < obj:
                ret, obj, n_not_improved = _X, _obj, 0
            else:
                n_not_improved += 1

            delta = np.sqrt((_X[:n_points] - X[:n_points]) ** 2).mean(axis=1).mean()

            if self.verbose:
                print(f"Iter {i}: Objective = {_obj:.6f}, Delta = {delta:.6e}")

            if delta < self.precision or np.isnan(_obj):
                break

            if self.restarts and n_not_improved > self.n_max_not_improved:
                optimizer = Adam(alpha=optimizer.alpha / 2)
                _X = ret
                n_not_improved = 0

            X = _X

            if self.callback is not None:
                self.callback(self, X)

        return ret[:n_points]

    def do(self, random_state=None):
        X = self.X

        if X is None:
            if random_state is None:
                random_state = self.rng
            X = random_state.rand(self.n_points, self.n_dim)
            X = X * (self.xu - self.xl) + self.xl

        X = self._solve(X)

        return X


def squared_dist(A, B):
    return ((A[:, None] - B[None, :]) ** 2).sum(axis=2)


def calc_potential_energy(A, d):
    n = len(A)
    i, j = np.triu_indices(n, 1)
    D = np.sqrt(squared_dist(A, A)[i, j])
    D = np.maximum(D, 1e-20)
    energy = np.log((1 / D ** d).mean())
    return energy


def calc_potential_energy_with_grad(x, d, return_mutual_dist=False):
    diff = (x[:, None] - x[None, :])
    dist = np.sqrt((diff ** 2).sum(axis=2))

    np.fill_diagonal(dist, np.inf)

    eps = 10 ** (-320 / (d + 2))
    mask = dist < eps
    dist[mask] = eps

    mutual_dist = dist[np.triu_indices(len(x), 1)]

    inv_dist_pow = 1 / mutual_dist ** d
    energy_sum = inv_dist_pow.sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy_sum)

    grad = (-d * diff) / (dist ** (d + 2))[..., None]
    grad = np.sum(grad, axis=1)
    grad /= energy_sum

    ret = [log_energy, grad]
    if return_mutual_dist:
        ret.append(mutual_dist)

    return tuple(ret)