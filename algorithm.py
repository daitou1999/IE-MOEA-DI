import copy
import numpy as np
from generate_weight_vector import WeightGenerator
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.util.nds.find_non_dominated import find_non_dominated
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.individual import Individual
from pymoo.operators.survival.rank_and_crowding.metrics import (
    calc_crowding_distance,  # Basic CD calculation function
    FunctionalDiversity      # Wrapper class (handles duplicate values/filtering)
)
from scipy.stats import iqr
from lws import MOEAD_LWS
from Riesz import RieszEnergyInitialSampling
from pymoo.functions import load_function


class MOEA:
    def __init__(self, MaxFEs, f_name, f_quantity, x_quantity, range_x, population_size, crossover_probability, crossover_probability_var, mutation_probability, mutation_probability_var, eta_c, eta_m, n_elites, obj_dir, pymoo_problem, platemo_problem, I_b, aaa, bbb):
        self.MaxFEs = MaxFEs
        self.f_name = f_name
        self.f_quantity = f_quantity
        self.x_quantity = x_quantity
        self.range_x = range_x
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.crossover_probability_var = crossover_probability_var
        self.mutation_probability = mutation_probability
        self.mutation_probability_var = mutation_probability_var
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.n_elites = n_elites
        self.obj_dir = obj_dir
        self.pymoo_problem = pymoo_problem
        self.platemo_problem = platemo_problem
        self.z_min = np.array([999999, 999999])
        self.z_max = np.array([-999999, -999999])
        w_w_generator = WeightGenerator(M=self.f_quantity, N=int(self.population_size))
        self.initial_w_w, self.neighbors = w_w_generator.generate_weights()
        self.w_w = copy.deepcopy(self.initial_w_w)
        self.epsilon = 1e-6
        self.I_b = I_b
        self.aaa = aaa
        self.bbb = bbb
        self.lws = MOEAD_LWS(m=self.f_quantity)

        self.calc_pcd = load_function("calc_pcd", _type="python")
        self.calc_mnn = load_function("calc_mnn")
        self.sbx = SBX(
            prob=self.crossover_probability,
            prob_var=self.crossover_probability_var,
            eta=self.eta_c,  # Distribution index, controls offspring distribution
            n_offsprings=1
        )
        self.pm = PM(
            prob=self.mutation_probability,
            prob_var = self.mutation_probability_var,
            eta=self.eta_m
        )

    def WS(self, population, fx, w_w_index=False, w_w_pd=False):  # WS objective
        if w_w_pd:
            w_w = self.w_w[w_w_index]
        else:
            w_w = self.w_w
        self.lws.weights = self.lws._normalize_weight(w_w)
        self.lws.compute_weight_angles_and_thetas()
        self.lws.z_min = self.z_min  # Example ideal point (needs dynamic update in practice)
        self.lws.z_nad = self.z_max  # Example nadir point (needs dynamic update in practice)
        lws_matrix = self.lws.compute_lws_matrix(fx)
        return np.min(lws_matrix, axis=0)

    def TC(self, population, fx, w_w_index=False, w_w_pd=False):  # TC objective
        if w_w_pd:
            w_w = self.w_w[w_w_index]
        else:
            w_w = self.w_w
        w_w = np.clip(w_w, self.epsilon, None)
        w_l2_norm = np.linalg.norm(w_w, ord=2, axis=1, keepdims=True)
        lambda_vec = w_w / w_l2_norm
        c2_tc = np.max((fx - self.z_min)/lambda_vec, axis=1)
        return c2_tc

    def PBI(self, population, fx, w_w_index=False, w_w_pd=False):  # PBI objective
        if w_w_pd:
            w_w = self.w_w[w_w_index]
        else:
            w_w = self.w_w
        a_a = 5  # Penalty parameter for PBI method

        # Calculate d1: distance along the weight direction to the reference point
        d1 = np.abs(np.sum((fx-self.z_min) * w_w , axis=1)) / np.linalg.norm(w_w , axis=1)
        # Calculate d2: distance perpendicular to the weight direction
        d2 = np.linalg.norm(fx - (self.z_min + d1[:, None] * w_w  / (np.linalg.norm(w_w , axis=1)[:, None])), axis=1)
        # Calculate PBI objective value, including penalty term
        c3_pbi = d1 + a_a * d2
        return c3_pbi

    def diversity_calculation_numpy(self, population):
        """
        Fully vectorized diversity calculation with Numpy:
        1. Efficient leave-one-out entropy (O(N) complexity)
        2. Adaptive binning (based on Freedman-Diaconis rule)
        3. Smooth fusion of entropy and distance
        Optimization points:
        - Remove more than 90% of Python loops, fully vectorized operations
        - Batch calculate histograms/bin assignments/entropy values
        - Construct one-hot matrix to batch process leave-one-out removal operations
        """
        n_ind, n_var = population.shape
        if n_ind <= 1:
            return np.ones(n_ind)

        # --- 1. Data normalization (vectorized) ---
        range_diff = self.range_x[:, 1] - self.range_x[:, 0] + self.epsilon
        norm_pop = (population - self.range_x[:, 0]) / range_diff

        # --- 2. Adaptive binning (fully vectorized FD rule calculation) ---
        # Calculate IQR for all dimensions at once (axis=0 calculates by column/dimension)
        data_iqr = iqr(norm_pop, axis=0)
        data_iqr = np.where(data_iqr < 1e-9, 1e-9, data_iqr)  # Avoid division by zero

        # Vectorized calculation of bin width and suggested bin number
        bw = 2 * data_iqr / (n_ind ** (1 / 3))
        bin_suggestions = np.maximum(2, np.ceil(1.0 / bw).astype(int))

        # Global bins take median and limit range
        bins = int(np.median(bin_suggestions))
        bins = np.clip(bins, 2, max(2, n_ind // 2))

        # --- 3. Batch calculate global histograms and bin assignments ---
        global_histograms = np.zeros((n_var, bins), dtype=int)  # (n_var, bins)
        bin_assignments = np.zeros((n_var, n_ind), dtype=int)  # (n_var, n_ind)

        # Dimension loop (histogram only supports 1D, this loop cannot be completely removed but is minimized)
        for d in range(n_var):
            hist, edges = np.histogram(norm_pop[:, d], bins=bins, range=(0, 1))
            global_histograms[d] = hist
            # Batch calculate bin assignments and correct boundaries
            assigns = np.clip(np.digitize(norm_pop[:, d], edges) - 1, 0, bins - 1)
            bin_assignments[d] = assigns

        # --- 4. Vectorized entropy calculation function (supports batch processing of hist matrices of any shape) ---
        def _entropy_vectorized(hist_matrix):
            """
            hist_matrix: Matrix of any shape with bins as the last dimension (e.g., (n_var, bins) or (n_var, n_ind, bins))
            Returns: Entropy values of corresponding dimensions (e.g., (n_var,) or (n_var, n_ind))
            """
            sum_hist = np.sum(hist_matrix, axis=-1, keepdims=True)
            prob = hist_matrix / sum_hist
            prob = np.where(prob > 0, prob, 1.0)  # Replace 0 probability terms with 1 (log(1)=0 does not affect results)
            return -np.sum(prob * np.log(prob), axis=-1)

        # --- 5. Vectorized leave-one-out entropy contribution calculation (core optimization) ---
        # 5.1 Global average entropy
        global_entropies = _entropy_vectorized(global_histograms)  # (n_var,)
        avg_global_ent = np.mean(global_entropies)

        # 5.2 Construct one-hot matrix to mark bin positions of each individual (n_var, n_ind, bins)
        one_hot = np.zeros((n_var, n_ind, bins), dtype=int)
        np.put_along_axis(one_hot, bin_assignments[:, :, None], 1, axis=-1)

        # 5.3 Batch calculate histograms after removing each individual
        hist_after_removal = global_histograms[:, None, :] - one_hot  # (n_var, n_ind, bins)
        hist_after_removal = np.maximum(hist_after_removal, 0)  # Prevent negative counts

        # 5.4 Batch calculate entropy after removal and take average
        ent_after_removal = _entropy_vectorized(hist_after_removal)  # (n_var, n_ind)
        avg_ent_after_removal = np.mean(ent_after_removal, axis=0)  # (n_ind,)

        # 5.5 Calculate final contribution value
        entropy_contrib = avg_global_ent - avg_ent_after_removal

    def evaluate_f(self, population):  # Calculate original fitness values
        pop_obj = self.platemo_problem.Evaluation(population)
        return pop_obj * self.obj_dir

    def initialize_population(self):  # Initialize population
        sampler = RieszEnergyInitialSampling(
            n_dim=self.x_quantity,
            n_points=self.population_size,
            xl=self.range_x[:, 0],
            xu=self.range_x[:, 1]
        )

        # Execute sampling
        initialize_population = sampler.do()
        np.random.shuffle(initialize_population)

        return initialize_population

    def run(self):
        FEs = 0
        population = self.initialize_population()  # Initial total population
        population_obj = self.evaluate_f(population)  # Evaluate fitness values

        self.z_min = np.min(population_obj, axis=0)
        self.z_max = np.max(population_obj, axis=0)
        FEs += self.population_size

        total_population, total_population_obj = self.c_pareto_front(population, population_obj)  # Non-dominated sorting

        n = 0
        while FEs + self.population_size <= self.MaxFEs:
            n_neighbors = self.neighbors.shape[0]
            shuffled_indices = np.random.permutation(n_neighbors)
            new_pop = []
            new_pop_obj = []
            elite_pop, elite_pop_obj = self.elite_selection(total_population, total_population_obj)
            for i in shuffled_indices:
                # Randomly select 1 neighborhood index
                chosen = np.random.choice(self.neighbors[i], 2)
                x1 = population[chosen[0]]
                x1_obj = population_obj[chosen[0]]

                rand = np.random.rand()
                if rand < self.I_b:
                    x2 = self.angle_competition(x1, x1_obj, elite_pop, elite_pop_obj)
                else:
                    x2 = population[chosen[1]]
                ind1 = Individual(X=x1)
                ind2 = Individual(X=x2)
                matings = self.sbx.do(self.pymoo_problem, [[ind1, ind2]])  # Crossover
                mutatings = self.pm.do(self.pymoo_problem, matings).get("X")[0] # Mutation

                mutatings_obj = self.evaluate_f(np.array([mutatings]))[0]  # Calculate objective values
                new_pop.append(mutatings)
                new_pop_obj.append(mutatings_obj)

                self.z_min = np.min(np.vstack((mutatings_obj, self.z_min)), axis=0) # Update z_min
                self.z_max = np.max(np.vstack((mutatings_obj, self.z_max)), axis=0)  # Update z_max

                offspring_1 = np.tile(mutatings, (self.neighbors.shape[1], 1))  # Expand
                offspring_1_obj = np.tile(mutatings_obj, (self.neighbors.shape[1], 1)) # Expand

                offspring_1_sobj_1 = self.WS(offspring_1, offspring_1_obj, w_w_index=self.neighbors[i], w_w_pd=True)  # Calculate indicator values
                offspring_1_sobj_2 = self.TC(offspring_1, offspring_1_obj, w_w_index=self.neighbors[i],
                                             w_w_pd=True)  # Calculate indicator values
                offspring_1_sobj_3 = self.PBI(offspring_1, offspring_1_obj, w_w_index=self.neighbors[i],
                                             w_w_pd=True)  # Calculate indicator values
                original_sobj_1 = self.WS(population[self.neighbors[i]],
                                                            population_obj[self.neighbors[i]],
                                                            w_w_index=self.neighbors[i], w_w_pd=True)
                original_sobj_2 = self.TC(population[self.neighbors[i]],
                                        population_obj[self.neighbors[i]],
                                        w_w_index=self.neighbors[i], w_w_pd=True)
                original_sobj_3 = self.PBI(population[self.neighbors[i]],
                                        population_obj[self.neighbors[i]],
                                        w_w_index=self.neighbors[i], w_w_pd=True)
                offspring_mat = np.column_stack([
                    offspring_1_sobj_1,
                    offspring_1_sobj_2,
                    offspring_1_sobj_3
                ])
                original_mat = np.column_stack([
                    original_sobj_1,
                    original_sobj_2,
                    original_sobj_3
                ])
                comparison_mat = original_mat >= offspring_mat
                # Sum by row (count the number of indicators satisfying the condition for each sample)
                counts = np.sum(comparison_mat, axis=1)

                pd = np.where(counts >= 2)[0]
                population[self.neighbors[i][pd]] = offspring_1[pd]
                population_obj[self.neighbors[i][pd]] = offspring_1_obj[pd]
            FEs += self.population_size
            total_population, total_population_obj = self.c_pareto_front(total_population, total_population_obj, new_pop, new_pop_obj, True)

            n += 1

        return total_population, total_population_obj

    '''------Repair-------'''
    def repair(self, arr):
        limits_arr = copy.deepcopy(self.range_x)
        if limits_arr.shape[0] == 1 and arr.shape[-1] != 1:  # Automatic broadcasting: single limit applied to all channels
            limits_arr = np.tile(limits_arr, (arr.shape[-1], 1))
        mins = limits_arr[:, 0]
        maxs = limits_arr[:, 1]
        new_shape = [1] * (arr.ndim - 1) + [mins.size]
        return np.clip(arr, mins.reshape(new_shape), maxs.reshape(new_shape))

    def c_pareto_front(self,original_population, original_obj, new_population=None, new_obj=None, new=False):
        if new:
            total_population = np.concatenate((original_population, new_population))  # Merge populations
            total_obj = np.concatenate((original_obj, new_obj))  # Merge population objective values
        else:
            total_population = copy.deepcopy(original_population)
            total_obj = copy.deepcopy(original_obj)
        layers = np.array(find_non_dominated(total_obj))
        total_population, return_index = np.unique(total_population[layers], return_index=True, axis=0)
        total_obj = total_obj[layers[return_index]]
        if self.bbb:
            while len(total_population) > self.population_size:
                if self.f_quantity == 2:
                    total_i = self.calc_pcd(np.copy(total_obj))
                else:
                    total_i = self.calc_mnn(np.copy(total_obj))
                top_max_indices = np.argsort(total_i)[::-1][:self.population_size]
                total_population = total_population[top_max_indices]
                total_obj = total_obj[top_max_indices]

        return total_population, total_obj

    def elite_selection(self, pop, pop_f):
        x_values = self.diversity_calculation_numpy(pop)

        sorted_indices = np.argsort(x_values)

        # Take the last indices and reverse the order (from largest to smallest)
        top_indices = sorted_indices[-self.n_elites:][::-1]

        return pop[top_indices], pop_f[top_indices]

    def angle_competition(self, X, Y, elite_X, elite_Y):
        winner_index = self.find_min_angle_index(Y-self.z_min, elite_Y-self.z_min)
        return elite_X[winner_index]

    def find_min_angle_index(self, arr_1d, arr_2d):
        # Calculate the norm (magnitude) of vectors
        norm_1d = np.linalg.norm(arr_1d)
        norm_2d = np.linalg.norm(arr_2d, axis=1)

        # Calculate cosine similarity: (A·B) / (|A| * |B|)
        dot_product = np.dot(arr_2d, arr_1d)  # Dot product
        cosine_similarity = dot_product / (norm_1d * norm_2d + self.epsilon)

        # Find the index with the maximum cosine similarity (i.e., the row with the smallest angle)
        min_angle_index = np.argmax(cosine_similarity)

        return min_angle_index

    def update_archive(self, pop, pop_obj, new_pop, new_pop_obj):
        # Ensure input is 2D array (prevent broadcast errors caused by 1D vectors)
        new_pop = np.atleast_2d(new_pop)
        new_pop_obj = np.atleast_2d(new_pop_obj)

        # Condition A: Whether new solutions dominate old solutions (new_obj <= old_obj and not all equal)
        cond_dominates_old = np.all(new_pop_obj <= pop_obj, axis=1) & \
                             ~np.all(new_pop_obj == pop_obj, axis=1)

        # Condition B: Whether old solutions dominate new solutions (old_obj <= new_obj and not all equal)
        cond_dominated_by_old = np.all(pop_obj <= new_pop_obj, axis=1) & \
                                ~np.all(pop_obj == new_pop_obj, axis=1)

        # Determine deletion indices and whether to add new solutions
        del_index = np.where(cond_dominates_old)[0]
        can_add = ~np.any(cond_dominated_by_old)

        pop = np.delete(pop, del_index, axis=0)
        pop_obj = np.delete(pop_obj, del_index, axis=0)

        if can_add:
            pop = np.vstack((pop, new_pop))
            pop_obj = np.vstack((pop_obj, new_pop_obj))

        return pop, pop_obj