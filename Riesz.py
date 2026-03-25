import numpy as np
from pymoo.util.ref_dirs.optimizer import Adam
import matplotlib.pyplot as plt


class RieszEnergyInitialSampling:
    """
    Riesz Energy-based Initial Sampling Class
    Generates uniformly distributed points within a box-constrained space by minimizing Riesz potential energy.
    The core idea is to minimize the repulsive potential energy between points to achieve uniform distribution.
    """

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
                 ):
        """
        Initialize the Riesz Energy Sampler

        Parameters:
        -----------
        n_dim : int
            Number of dimensions of the sampling space (e.g., 2 for 2D space, 3 for 3D space)
        n_points : int
            Number of sampling points to generate
        xl : np.ndarray (optional, default=None)
            Lower bound of the box constraint for each dimension. If None, defaults to 0 for all dimensions.
        xu : np.ndarray (optional, default=None)
            Upper bound of the box constraint for each dimension. If None, defaults to 1 for all dimensions.
        n_max_iter : int (optional, default=1000)
            Maximum number of optimization iterations
        n_until_optimizer_reset : int (optional, default=30)
            Number of non-improving iterations before resetting the optimizer
        norm_gradients : bool (optional, default=True)
            Whether to normalize gradients to unit norm (stabilizes optimization)
        verify_gradient : bool (optional, default=False)
            Whether to verify gradient correctness (e.g., with finite difference; implementation removed for simplicity)
        precision : float (optional, default=1e-5)
            Convergence threshold: optimization stops if average point movement is below this value
        restarts : bool (optional, default=True)
            Whether to enable optimizer restarts when progress stalls
        X : np.ndarray (optional, default=None)
            Initial point set (shape: [n_points, n_dim]). If None, random uniform sampling is used.
        d : int (optional, default=None)
            Exponent for Riesz energy calculation. If None, set to 2 * n_dim (empirical choice).
        callback : callable (optional, default=None)
            Callback function executed after each iteration (signature: callback(sampler, X))
        verbose : bool (optional, default=False)
            Whether to print optimization progress (objective value and delta per iteration)
        """

        self.n_dim = n_dim
        self.n_points = n_points

        # Set box constraints (default: [0, 1]^n_dim)
        if xl is None:
            xl = np.zeros(n_dim)
        if xu is None:
            xu = np.ones(n_dim)
        self.xl = np.asarray(xl)
        self.xu = np.asarray(xu)

        self.n_max_iter = n_max_iter
        self.n_max_not_improved = n_until_optimizer_reset  # Max non-improving steps before optimizer restart
        self.X = X  # Initial point set
        self.precision = precision  # Convergence precision threshold
        self.verify_gradient = verify_gradient  # Gradient verification flag
        self.norm_gradients = norm_gradients  # Gradient normalization flag
        self.d = d  # Riesz energy exponent
        self.callback = callback  # Iteration callback
        self.restarts = restarts  # Optimizer restart enable flag
        self.verbose = verbose  # Verbose output flag

        # Empirical setting: use squared dimension as energy exponent d
        if self.d is None:
            self.d = n_dim * 2

    def _step(self, optimizer, X, freeze=None):
        """
        Perform a single optimization step to minimize Riesz energy

        Parameters:
        -----------
        optimizer : Adam
            Adam optimizer instance for gradient descent
        X : np.ndarray
            Current point set (shape: [n_total_points, n_dim])
        freeze : np.ndarray (optional, default=None)
            Boolean array (shape: [n_total_points,]) indicating frozen points (no gradient update)

        Returns:
        --------
        X : np.ndarray
            Updated point set after gradient step and projection
        obj : float
            Current Riesz energy value (objective to minimize)
        """
        # Initialize freeze mask (all points free by default)
        if freeze is None:
            freeze = np.full(len(X), False)
        free = np.logical_not(freeze)  # Mask for non-frozen (optimizable) points

        # Calculate Riesz energy and gradient (core computation)
        obj, grad, mutual_dist = calc_potential_energy_with_grad(X, self.d, return_mutual_dist=True)

        # Verify gradient correctness (optional - implementation removed for simplicity)
        if self.verify_gradient:
            pass

        # Zero out gradients for frozen points (no update)
        grad[freeze] = 0
        proj_grad = grad  # Projected gradient (only free points have non-zero gradients)

        # Normalize gradients to unit norm (stabilizes gradient descent)
        if self.norm_gradients:
            norm = np.linalg.norm(proj_grad, axis=1)
            proj_grad = proj_grad / max(norm.max(), 1e-24)  # Avoid division by zero

        # Perform Adam gradient descent step
        X = optimizer.next(X, proj_grad)

        # Critical modification: project free points back to box constraint space [xl, xu]
        X[free] = np.clip(X[free], self.xl, self.xu)

        return X, obj

    def _solve(self, X, F=None):
        """
        Main optimization loop to minimize Riesz energy

        Parameters:
        -----------
        X : np.ndarray
            Initial point set (shape: [n_points, n_dim])
        F : np.ndarray (optional, default=None)
            Fixed (frozen) point set to include in the energy calculation (shape: [n_fixed, n_dim])

        Returns:
        --------
        ret : np.ndarray
            Optimized point set (shape: [n_points, n_dim]) with minimized Riesz energy
        """
        n_points = len(X)
        ret, obj = X, np.inf  # Initialize best solution and objective value
        n_not_improved = 0  # Counter for non-improving iterations

        # Initialize freeze mask (all initial points are free)
        freeze = np.full(len(X), False)

        # Add fixed points to the set and freeze them
        if F is not None:
            X = np.vstack([X, F])
            freeze = np.concatenate([freeze, np.full(len(F), True)])

        # Early return if all points are frozen (no optimization needed)
        if np.all(freeze):
            return X

        # Initialize Adam optimizer with learning rate 0.005
        optimizer = Adam(alpha=0.005)

        # Execute callback (if provided) before optimization starts
        if self.callback is not None:
            self.callback(self, X)

        # Main optimization iteration loop
        for i in range(self.n_max_iter):
            # Perform single optimization step
            _X, _obj = self._step(optimizer, X, freeze=freeze)

            # Update best solution if current objective is better
            if _obj < obj:
                ret, obj, n_not_improved = _X, _obj, 0
            else:
                n_not_improved += 1  # Increment non-improving counter

            # Calculate average point movement (delta) for convergence check
            delta = np.sqrt((_X[:n_points] - X[:n_points]) ** 2).mean(axis=1).mean()

            # Print progress if verbose mode is enabled
            if self.verbose:
                print(f"Iter {i}: Objective = {_obj:.6f}, Delta = {delta:.6e}")

            # Convergence check: stop if delta < precision or objective is NaN
            if delta < self.precision or np.isnan(_obj):
                break

            # Restart optimizer if progress stalls (reduce learning rate by half)
            if self.restarts and n_not_improved > self.n_max_not_improved:
                optimizer = Adam(alpha=optimizer.alpha / 2)
                _X = ret  # Reset to best solution so far
                n_not_improved = 0

            # Update current point set for next iteration
            X = _X

            # Execute callback after each iteration
            if self.callback is not None:
                self.callback(self, X)

        # Return only the original number of points (exclude fixed points)
        return ret[:n_points]

    def do(self, random_state=None):
        """
        Main entry point to execute Riesz energy-based sampling

        Parameters:
        -----------
        random_state : np.random.RandomState (optional, default=None)
            Random state for reproducible initial sampling

        Returns:
        --------
        X : np.ndarray
            Optimized uniformly distributed point set (shape: [n_points, n_dim])
        """
        X = self.X

        # Critical modification: generate random initial points if none provided
        if X is None:
            # Initialize random state if not provided
            if random_state is None:
                random_state = np.random.RandomState()
            # Uniform random sampling within box constraints [xl, xu]
            X = random_state.rand(self.n_points, self.n_dim)
            X = X * (self.xu - self.xl) + self.xl

        # Execute optimization to minimize Riesz energy
        X = self._solve(X)

        return X


# ---------------------------------------------------------------------------------------------------------
# Core Utility Functions for Riesz Energy and Gradient Calculation
# Note: Autograd dependency removed for simplicity (can be readded for gradient verification)
# ---------------------------------------------------------------------------------------------------------

def squared_dist(A, B):
    """
    Calculate squared Euclidean distance matrix between two point sets

    Parameters:
    -----------
    A : np.ndarray
        First point set (shape: [n_A, n_dim])
    B : np.ndarray
        Second point set (shape: [n_B, n_dim])

    Returns:
    --------
    dist : np.ndarray
        Squared distance matrix (shape: [n_A, n_B]) where dist[i,j] = ||A[i] - B[j]||²
    """
    return ((A[:, None] - B[None, :]) ** 2).sum(axis=2)


def calc_potential_energy(A, d):
    """
    Calculate Riesz potential energy for a point set

    The energy is defined as log(mean(1/||x_i - x_j||^d)) for all i < j (unique pairs)
    Minimizing this energy leads to uniform point distribution

    Parameters:
    -----------
    A : np.ndarray
        Point set (shape: [n_points, n_dim])
    d : int
        Riesz energy exponent

    Returns:
    --------
    energy : float
        Logarithmic Riesz potential energy value
    """
    n = len(A)
    # Get indices for upper triangular matrix (unique pairs i < j)
    i, j = np.triu_indices(n, 1)
    # Calculate Euclidean distances for unique pairs
    D = np.sqrt(squared_dist(A, A)[i, j])
    # Prevent division by zero (numerical stability)
    D = np.maximum(D, 1e-20)
    # Calculate log of mean inverse distance raised to power d
    energy = np.log((1 / D ** d).mean())
    return energy


def calc_potential_energy_with_grad(x, d, return_mutual_dist=False):
    """
    Calculate Riesz potential energy and its gradient with respect to each point

    Gradient formula: grad_k = -d * sum_{j≠k} (x_k - x_j) / ||x_k - x_j||^(d+2)
    Adjusted with chain rule for logarithmic mean energy

    Parameters:
    -----------
    x : np.ndarray
        Point set (shape: [n_points, n_dim])
    d : int
        Riesz energy exponent
    return_mutual_dist : bool (optional, default=False)
        Whether to return mutual distances for unique point pairs

    Returns:
    --------
    log_energy : float
        Logarithmic Riesz potential energy
    grad : np.ndarray
        Gradient of energy w.r.t. each point (shape: [n_points, n_dim])
    mutual_dist (optional) : np.ndarray
        Euclidean distances for unique point pairs (i < j)
    """
    # Calculate pairwise differences (shape: [n_points, n_points, n_dim])
    diff = (x[:, None] - x[None, :])
    # Calculate pairwise Euclidean distances (shape: [n_points, n_points])
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Fill diagonal with infinity (exclude self-distances)
    np.fill_diagonal(dist, np.inf)

    # Extreme numerical stability safeguard (prevent underflow/overflow)
    eps = 10 ** (-320 / (d + 2))
    mask = dist < eps
    dist[mask] = eps

    # Extract distances for unique pairs (upper triangular, i < j)
    mutual_dist = dist[np.triu_indices(len(x), 1)]

    # Calculate inverse distance raised to power d (repulsive potential)
    inv_dist_pow = 1 / mutual_dist ** d
    energy_sum = inv_dist_pow.sum()
    # Logarithmic energy (log(mean(inv_dist_pow)) = log(sum/len) = log(sum) - log(len))
    log_energy = -np.log(len(mutual_dist)) + np.log(energy_sum)

    # Calculate gradient of Riesz energy
    # Core gradient formula: grad_k = -d * sum_{j≠k} (x_k - x_j) / ||x_k - x_j||^(d+2)
    grad = (-d * diff) / (dist ** (d + 2))[..., None]
    grad = np.sum(grad, axis=1)
    # Apply chain rule for logarithmic mean (d/dx log(mean) = 1/mean * d/dx mean)
    grad /= energy_sum

    # Prepare return values
    ret = [log_energy, grad]
    if return_mutual_dist:
        ret.append(mutual_dist)

    return tuple(ret)