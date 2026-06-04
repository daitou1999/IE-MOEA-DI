import numpy as np


class MOEAD_LWS:
    def __init__(self, m, epsilon=1e-6):
        self.m = m
        self.epsilon = epsilon
        self.z_min = None
        self.z_nad = None
        self.weights = None

        self.theta_per_weight = None
        self.weight_angle_matrix = None

    def _normalize_fx(self, fx):
        if self.z_min is None or self.z_nad is None:
            raise ValueError("The ideal point z_min and nadir point z_nad need to be set first.")
        denominator = self.z_nad - self.z_min
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        return (fx - self.z_min) / denominator

    def _calculate_utopian_point(self):
        if self.z_min is None:
            raise ValueError("The ideal point z_min needs to be set first.")
        return self.z_min - self.epsilon

    def _normalize_weight(self, w):
        w_clipped = np.clip(w, self.epsilon, None)
        return w_clipped / np.sum(w_clipped, axis=-1, keepdims=True)

    def compute_weight_angles_and_thetas(self):
        if self.weights is None:
            raise ValueError("The weight vector set self.weights needs to be configured first.")

        pop_size = self.weights.shape[0]
        weights_unit = self.weights / np.linalg.norm(self.weights, axis=1, keepdims=True)

        cos_angle_matrix = np.dot(weights_unit, weights_unit.T)
        cos_angle_matrix = np.clip(cos_angle_matrix, -1.0, 1.0)
        self.weight_angle_matrix = np.arccos(cos_angle_matrix)

        self.theta_per_weight = np.zeros(pop_size)
        for i in range(pop_size):
            sorted_angles = np.sort(self.weight_angle_matrix[i])
            nearest_m_angles = sorted_angles[1:self.m + 1]
            self.theta_per_weight[i] = np.mean(nearest_m_angles)

    def _hypercone_constraint_single_weight(self, fx_norm, w_j_norm, theta_j):
        fx_norm_unit = fx_norm / (np.linalg.norm(fx_norm, axis=1, keepdims=True) + self.epsilon)
        w_j_unit = w_j_norm / np.linalg.norm(w_j_norm)

        cos_theta = np.dot(fx_norm_unit, w_j_unit)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        cos_theta_j = np.cos(theta_j)
        return cos_theta >= cos_theta_j

    def compute_lws_matrix(self, fx):
        pop_size = fx.shape[0]
        if self.theta_per_weight is None:
            raise ValueError("The `compute_weight_angles_and_thetas()` function needs to be called first to calculate the apex angles of the hypercones.")

        fx_norm = self._normalize_fx(fx)
        weights_norm = self._normalize_weight(self.weights)
        z_u = self._calculate_utopian_point()
        z_u_norm = (z_u - self.z_min) / (self.z_nad - self.z_min)

        lws_matrix = np.full((pop_size, pop_size), np.inf)

        for j in range(pop_size):
            w_j_norm = weights_norm[j]
            theta_j = self.theta_per_weight[j]

            valid_mask = self._hypercone_constraint_single_weight(fx_norm, w_j_norm, theta_j)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            fx_valid_norm = fx_norm[valid_indices]
            delta = fx_valid_norm - z_u_norm
            ws_values = np.sum(delta / w_j_norm, axis=1)

            lws_matrix[valid_indices, j] = ws_values

        return lws_matrix