import sys
import numpy as np
from typing import List, Tuple, Optional
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory



class WeightGenerator:

    def __init__(self, M: int, N: Optional[int] = None, T=10, n_max_iter=1000):

        self.M = M
        self.N = N
        self.T = T
        self.n_max_iter = n_max_iter

    def generate_weights(self):
        factory = RieszEnergyReferenceDirectionFactory(
            n_dim=self.M,
            n_points=self.N,
            n_max_iter=self.n_max_iter,
            seed=42
        )
        sys.setrecursionlimit(2000)
        ref_dirs = factory.do()
        dist = np.sqrt(((ref_dirs[:, np.newaxis] - ref_dirs) ** 2).sum(axis=2))
        neighbors = np.argsort(dist, axis=1)[:, :self.T]

        return ref_dirs, neighbors
