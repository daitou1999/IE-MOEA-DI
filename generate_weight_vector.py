import sys
import numpy as np
from typing import List, Tuple, Optional
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory


class WeightGenerator:

    def __init__(self, M: int, N: Optional[int] = None, T=15, n_max_iter=1000):
        """
        Initialize weight generator

        Parameters:
        ----------
        M : int
            Number of objectives
        N : int, optional
            Expected number of weight vectors. If H is provided, H will be used first
        """
        self.M = M
        self.N = N
        self.T = T
        self.n_max_iter = n_max_iter

    def generate_weights(self):
        # Instantiate factory class (custom parameters)
        factory = RieszEnergyReferenceDirectionFactory(
            n_dim=self.M,  # Dimension of reference points (number of objective functions)
            n_points=self.N,  # Generate reference points
            n_max_iter=self.n_max_iter,
            seed=42
        )
        # Increase recursion depth limit (e.g., set to 2000)
        sys.setrecursionlimit(2000)
        # Generate reference points
        ref_dirs = factory.do()
        # Distance matrix dist[i,j] = Euclidean distance between vector i and vector j
        dist = np.sqrt(((ref_dirs[:, np.newaxis] - ref_dirs) ** 2).sum(axis=2))
        # After sorting, the first index is itself (distance is 0), which conforms to MOEA/D standard (neighborhood can include itself)
        neighbors = np.argsort(dist, axis=1)[:, :self.T]

        return ref_dirs, neighbors