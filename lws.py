import numpy as np


class MOEAD_LWS:
    def __init__(self, m, epsilon=1e-6):
        """
        Initialize parameters (strictly aligned with paper definitions)
        :param m: Number of objective functions
        :param epsilon: Minimal offset for utopian point (ε_i in paper)
        """
        self.m = m  # Number of objectives
        self.epsilon = epsilon  # Utopian point offset

        # Core parameters of the paper (need to call compute_weight_angles_and_thetas after external initialization)
        self.z_min = None  # Ideal point z_i^* (shape: (m,))
        self.z_nad = None  # Nadir point z_i^nad (shape: (m,))
        self.weights = None  # Weight vector set (shape: (pop_size, m))

        # LWS new core parameters
        self.theta_per_weight = None  # Hyper-cone apex angle Θ_i for each weight (shape: (pop_size,))
        self.weight_angle_matrix = None  # Angle matrix θ_ij^ww between weights (shape: (pop_size, pop_size))

    # -------------------------- Basic utility functions (keep correct parts) --------------------------
    def _normalize_fx(self, fx):
        """Paper formula (4): Normalize objective function values to [0,1]"""
        if self.z_min is None or self.z_nad is None:
            raise ValueError("Need to set ideal point z_min and nadir point z_nad first")
        denominator = self.z_nad - self.z_min
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)  # Numerical stability
        return (fx - self.z_min) / denominator

    def _calculate_utopian_point(self):
        """Paper definition: Utopian point z_i^u = z_i^* - ε_i"""
        if self.z_min is None:
            raise ValueError("Need to set ideal point z_min first")
        return self.z_min - self.epsilon

    def _normalize_weight(self, w):
        """Paper constraint: Normalize weight vectors (sum(w_i)=1 and w_i>0)"""
        w_clipped = np.clip(w, self.epsilon, None)  # Avoid zero weights
        return w_clipped / np.sum(w_clipped, axis=-1, keepdims=True)

    # -------------------------- LWS core improvement part --------------------------
    def compute_weight_angles_and_thetas(self):
        """
        Paper LWS core step 1:
        1. Calculate angle matrix θ_ij^ww between all weight vectors
        2. Calculate hyper-cone apex angle Θ_i = (average of angles of nearest m weights) for each weight w_i
        """
        if self.weights is None:
            raise ValueError("Need to set weight vector set self.weights first")

        pop_size = self.weights.shape[0]
        # 1. Normalize weights to unit vectors (for angle calculation)
        weights_unit = self.weights / np.linalg.norm(self.weights, axis=1, keepdims=True)

        # 2. Calculate pairwise angle matrix between weights (in radians)
        cos_angle_matrix = np.dot(weights_unit, weights_unit.T)
        cos_angle_matrix = np.clip(cos_angle_matrix, -1.0, 1.0)  # Avoid numerical errors
        self.weight_angle_matrix = np.arccos(cos_angle_matrix)

        # 3. Calculate Θ_i for each weight (paper formula: Θ_i = sum(nearest m angles)/m)
        self.theta_per_weight = np.zeros(pop_size)
        for i in range(pop_size):
            # Exclude self (angle is 0), take angles of nearest m weights
            sorted_angles = np.sort(self.weight_angle_matrix[i])
            nearest_m_angles = sorted_angles[1:self.m + 1]  # Skip the 0th (self)
            self.theta_per_weight[i] = np.mean(nearest_m_angles)

    def _hypercone_constraint_single_weight(self, fx_norm, w_j_norm, theta_j):
        """
        Paper LWS core step 2:
        For a single weight w_j, judge whether all solutions are within its hyper-cone (θ_ij^sw ≤ Θ_j)
        """
        # Normalize objective vectors of solutions to unit vectors
        fx_norm_unit = fx_norm / (np.linalg.norm(fx_norm, axis=1, keepdims=True) + self.epsilon)
        # Normalize weight vector to unit vector
        w_j_unit = w_j_norm / np.linalg.norm(w_j_norm)

        # Calculate cosine of the angle between solution and weight
        cos_theta = np.dot(fx_norm_unit, w_j_unit)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Angle ≤ Θ_j is equivalent to cosθ ≥ cos(Θ_j)
        cos_theta_j = np.cos(theta_j)
        return cos_theta >= cos_theta_j

    def compute_lws_matrix(self, fx):
        """
        Paper LWS core step 3:
        Build scalarization value matrix C_ij (LWS value of solution i for weight j)
        - In-cone solutions: Calculate standard WS values
        - Out-of-cone solutions: Set to infinity
        """
        pop_size = fx.shape[0]
        if self.theta_per_weight is None:
            raise ValueError("Need to call compute_weight_angles_and_thetas() first to calculate hyper-cone apex angles")

        # 1. Preprocessing: Normalize objective values, weights, utopian point
        fx_norm = self._normalize_fx(fx)
        weights_norm = self._normalize_weight(self.weights)
        z_u = self._calculate_utopian_point()
        z_u_norm = (z_u - self.z_min) / (self.z_nad - self.z_min)  # Normalize utopian point

        # 2. Initialize LWS matrix to infinity
        lws_matrix = np.full((pop_size, pop_size), np.inf)

        # 3. Iterate over each weight j, calculate LWS values of all solutions i
        for j in range(pop_size):
            w_j_norm = weights_norm[j]
            theta_j = self.theta_per_weight[j]

            # a. Filter solutions within the hyper-cone
            valid_mask = self._hypercone_constraint_single_weight(fx_norm, w_j_norm, theta_j)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            # b. Calculate standard WS values of in-cone solutions (paper formula: g^ws = sum( (f_i - z_i^u)/w_i )）
            fx_valid_norm = fx_norm[valid_indices]
            delta = fx_valid_norm - z_u_norm
            ws_values = np.sum(delta / w_j_norm, axis=1)

            # c. Fill LWS matrix
            lws_matrix[valid_indices, j] = ws_values

        return lws_matrix
