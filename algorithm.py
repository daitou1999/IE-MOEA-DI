import copy
import numpy as np
from generate_weight_vector import WeightGenerator
from pymoo.util.nds.find_non_dominated import find_non_dominated
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.individual import Individual
from lws import MOEAD_LWS
from Riesz import RieszEnergyInitialSampling
from pymoo.functions import load_function
from scipy.stats import iqr
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.indicators.hv.approximate import ApproximateHypervolume

class MOEA:
    def __init__(self, MaxFEs, f_name, f_quantity, x_quantity, range_x, population_size, crossover_probability, crossover_probability_var, mutation_probability, mutation_probability_var, eta_c, eta_m, n_elites, obj_dir, pymoo_problem, I_b, aaa, bbb, SEED):
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
        self.z_min = np.array([999999, 999999])
        self.z_max = np.array([-999999, -999999])
        w_w_generator = WeightGenerator(M=self.f_quantity, N=int(self.population_size), T=n_elites)
        self.initial_w_w, self.neighbors = w_w_generator.generate_weights()
        self.w_w = copy.deepcopy(self.initial_w_w)
        self.epsilon = 1e-6
        self.I_b = I_b
        self.aaa = aaa
        self.bbb = bbb
        self.lws = MOEAD_LWS(m=self.f_quantity)
        self.SEED = SEED
        self.rng = np.random.default_rng(SEED)

        self.calc_pcd = load_function("calc_pcd", _type="python")
        self.calc_mnn = load_function("calc_mnn", _type="python")
        self.calc_2nn = load_function("calc_2nn", _type="python")
        self.sbx = SBX(
            prob=self.crossover_probability,
            prob_var=self.crossover_probability_var,
            eta=self.eta_c,
            n_offsprings=1
        )
        self.pm = PM(
            prob=self.mutation_probability,
            prob_var = self.mutation_probability_var,
            eta=self.eta_m
        )
    def WS(self, population, fx, w_w_index=False, w_w_pd=False):  # lws
        if w_w_pd:
            w_w = self.w_w[w_w_index]
        else:
            w_w = self.w_w
        self.lws.weights = self.lws._normalize_weight(w_w)
        self.lws.compute_weight_angles_and_thetas()
        self.lws.z_min = self.z_min
        self.lws.z_nad = self.z_max
        lws_matrix = self.lws.compute_lws_matrix(fx)
        return np.min(lws_matrix, axis=0)
    def TC(self, population, fx, w_w_index=False, w_w_pd=False):  # p-tch
        if w_w_pd:
            w_w = self.w_w[w_w_index]
        else:
            w_w = self.w_w
        w_w = np.clip(w_w, self.epsilon, None)
        w_l2_norm = np.linalg.norm(w_w, ord=2, axis=1, keepdims=True)
        lambda_vec = w_w / w_l2_norm
        c2_tc = np.max((fx - self.z_min)/lambda_vec, axis=1)
        return c2_tc

    def PBI(self, population, fx, w_w_index=False, w_w_pd=False):  # pbi
        if w_w_pd:
            w_w = self.w_w[w_w_index]
        else:
            w_w = self.w_w
        a_a = 5
        d1 = np.abs(np.sum((fx - self.z_min) * w_w, axis=1)) / np.linalg.norm(w_w, axis=1)
        d2 = np.linalg.norm(fx - (self.z_min + d1[:, None] * w_w / (np.linalg.norm(w_w, axis=1)[:, None])), axis=1)
        c3_pbi = d1 + a_a * d2
        return c3_pbi
    def diversity_calculation(self, population):
        n_ind, n_var = population.shape
        if n_ind <= 1:
            return np.ones(n_ind)

        range_diff = self.range_x[:, 1] - self.range_x[:, 0] + self.epsilon
        norm_pop = (population - self.range_x[:, 0]) / range_diff

        data_iqr = iqr(norm_pop, axis=0)
        zero_iqr_mask = data_iqr < 1e-9
        bin_suggestions = np.ones(n_var, dtype=int)

        if np.any(~zero_iqr_mask):
            bw = 2 * data_iqr[~zero_iqr_mask] / (n_ind ** (1 / 3))
            bin_suggestions[~zero_iqr_mask] = np.maximum(2, np.ceil(1.0 / bw).astype(int))

        bins = int(np.median(bin_suggestions))
        bins = np.clip(bins, 2, max(2, n_ind // 2))

        global_histograms = np.zeros((n_var, bins), dtype=int)
        bin_assignments = np.zeros((n_var, n_ind), dtype=int)

        for d in range(n_var):
            hist, edges = np.histogram(norm_pop[:, d], bins=bins, range=(0, 1))
            global_histograms[d] = hist
            assigns = np.clip(np.digitize(norm_pop[:, d], edges) - 1, 0, bins - 1)
            bin_assignments[d] = assigns

        def _entropy_vectorized(hist_matrix):
            sum_hist = np.sum(hist_matrix, axis=-1, keepdims=True)
            prob = hist_matrix / sum_hist
            prob = np.where(prob > 0, prob, 1.0)
            return -np.sum(prob * np.log(prob), axis=-1)

        global_entropies = _entropy_vectorized(global_histograms)
        avg_global_ent = np.mean(global_entropies)

        one_hot = np.zeros((n_var, n_ind, bins), dtype=int)
        np.put_along_axis(one_hot, bin_assignments[:, :, None], 1, axis=-1)

        hist_after_removal = global_histograms[:, None, :] - one_hot
        hist_after_removal = np.maximum(hist_after_removal, 0)

        ent_after_removal = _entropy_vectorized(hist_after_removal)
        avg_ent_after_removal = np.mean(ent_after_removal, axis=0)

        entropy_contrib = avg_global_ent - avg_ent_after_removal

        return entropy_contrib
    def evaluate_f(self, population):
        pop_obj = self.pymoo_problem.evaluate(population)
        return pop_obj * self.obj_dir

    def initialize_population(self):
        sampler = RieszEnergyInitialSampling(
            n_dim=self.x_quantity,
            n_points=self.population_size,
            xl=self.range_x[:, 0],
            xu=self.range_x[:, 1],
            SEED=self.SEED
        )
        initialize_population = sampler.do()
        self.rng.shuffle(initialize_population)
        self.rng = np.random.default_rng(self.SEED)
        return initialize_population

    def run(self):
        ps_histories = []
        pf_histories = []
        X_histories = []
        Y_histories = []
        FEs = 0
        population = self.initialize_population()
        population_obj = self.evaluate_f(population)
        self.z_min = np.min(population_obj, axis=0)
        self.z_max = np.max(population_obj, axis=0)
        FEs += self.population_size

        total_population, total_population_obj = self.c_pareto_front(population, population_obj)

        X_histories.append(population.tolist())
        Y_histories.append(population_obj.tolist())
        ps_histories.append(total_population.tolist())
        pf_histories.append(total_population_obj.tolist())

        n = 0
        while FEs + self.population_size <= self.MaxFEs:
            n_neighbors = self.neighbors.shape[0]
            shuffled_indices = np.random.permutation(n_neighbors)
            new_pop = []
            new_pop_obj = []
            for i in shuffled_indices:
                chosen = np.random.choice(self.neighbors[i], 2)
                x1 = population[chosen[0]]
                x1_obj = population_obj[chosen[0]]

                rand = np.random.rand()
                if rand < self.I_b:
                    elite_pop, elite_pop_obj = self.elite_selection(total_population, total_population_obj)
                    x2 = self.angle_competition(x1, x1_obj, elite_pop, elite_pop_obj)
                else:
                    x2 = population[chosen[1]]
                ind1 = Individual(X=x1)
                ind2 = Individual(X=x2)

                matings = self.sbx.do(self.pymoo_problem, [[ind1, ind2]], random_state=self.rng)
                mutatings = self.pm.do(self.pymoo_problem, matings, random_state=self.rng).get("X")[0]

                mutatings_obj = self.evaluate_f(np.array([mutatings]))[0]
                new_pop.append(mutatings)
                new_pop_obj.append(mutatings_obj)

                total_population, total_population_obj = self.update_archive(total_population, total_population_obj, mutatings, mutatings_obj)

                self.z_min = np.min(np.vstack((mutatings_obj, self.z_min)), axis=0)
                self.z_max = np.max(np.vstack((mutatings_obj, self.z_max)), axis=0)

                offspring_1 = np.tile(mutatings, (self.neighbors.shape[1], 1))
                offspring_1_obj = np.tile(mutatings_obj, (self.neighbors.shape[1], 1))


                offspring_1_sobj_1 = self.WS(offspring_1, offspring_1_obj, w_w_index=self.neighbors[i], w_w_pd=True)
                offspring_1_sobj_2 = self.TC(offspring_1, offspring_1_obj, w_w_index=self.neighbors[i],
                                             w_w_pd=True)
                offspring_1_sobj_3 = self.PBI(offspring_1, offspring_1_obj, w_w_index=self.neighbors[i],
                                             w_w_pd=True)
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
                comparison_mat = original_mat > offspring_mat
                counts = np.sum(comparison_mat, axis=1)

                pd = np.where(counts >= 2)[0]

                population[self.neighbors[i][pd]] = offspring_1[pd]
                population_obj[self.neighbors[i][pd]] = offspring_1_obj[pd]

            FEs += self.population_size
            if self.f_quantity > 3:
                hv_calc = ApproximateHypervolume(ref_point=self.z_max * 1.1,
                                                 random_state=np.random.RandomState(42))
            else:
                hv_calc = ExactHypervolume(ref_point=self.z_max * 1.1)
            hv_calc.add(total_population_obj)
            while len(total_population_obj) > self.population_size:
                min_idx = np.argmin(hv_calc.hvc)
                hv_calc.delete(min_idx)
                total_population_obj = np.delete(total_population_obj, min_idx, axis=0)
                total_population = np.delete(total_population, min_idx, axis=0)
            X_histories.append(population.tolist())
            Y_histories.append(population_obj.tolist())
            ps_histories.append(total_population.tolist())
            pf_histories.append(total_population_obj.tolist())
            n += 1

        return total_population, total_population_obj, X_histories, Y_histories, ps_histories, pf_histories

    def repair(self, arr):
        limits_arr = copy.deepcopy(self.range_x)
        if limits_arr.shape[0] == 1 and arr.shape[-1] != 1:
            limits_arr = np.tile(limits_arr, (arr.shape[-1], 1))
        mins = limits_arr[:, 0]
        maxs = limits_arr[:, 1]
        new_shape = [1] * (arr.ndim - 1) + [mins.size]
        return np.clip(arr, mins.reshape(new_shape), maxs.reshape(new_shape))

    def c_pareto_front(self,original_population, original_obj, new_population=None, new_obj=None, new=False):
        if new:
            total_population = np.concatenate((original_population, new_population))
            total_obj = np.concatenate((original_obj, new_obj))
        else:
            total_population = copy.deepcopy(original_population)
            total_obj = copy.deepcopy(original_obj)
        layers = np.array(find_non_dominated(total_obj))
        total_obj, return_index = np.unique(total_obj[layers], return_index=True, axis=0)
        total_population = total_population[layers[return_index]]
        return total_population, total_obj


    def elite_selection(self, pop, pop_f):
        x_values = self.diversity_calculation(pop)
        sorted_indices = np.argsort(x_values)
        top_indices = sorted_indices[-self.n_elites:][::-1]
        return pop[top_indices], pop_f[top_indices]

    def angle_competition(self, X, Y, elite_X, elite_Y):
        winner_index = self.find_min_angle_index(Y-self.z_min, elite_Y-self.z_min)
        return elite_X[winner_index]

    def find_min_angle_index(self, arr_1d, arr_2d):
        norm_1d = np.linalg.norm(arr_1d)
        norm_2d = np.linalg.norm(arr_2d, axis=1)
        dot_product = np.dot(arr_2d, arr_1d)
        cosine_similarity = dot_product / (norm_1d * norm_2d + self.epsilon)
        min_angle_index = np.argmax(cosine_similarity)

        return min_angle_index

    def update_archive(self, pop, pop_obj, new_pop, new_pop_obj):
        new_pop = np.atleast_2d(new_pop)
        new_pop_obj = np.atleast_2d(new_pop_obj)
        cond_dominates_old = np.all(new_pop_obj <= pop_obj, axis=1) & \
                             ~np.all(new_pop_obj == pop_obj, axis=1)
        cond_dominated_by_old = np.all(pop_obj <= new_pop_obj, axis=1) & \
                                ~np.all(pop_obj == new_pop_obj, axis=1)
        del_index = np.where(cond_dominates_old)[0]
        can_add = ~np.any(cond_dominated_by_old)

        pop = np.delete(pop, del_index, axis=0)
        pop_obj = np.delete(pop_obj, del_index, axis=0)

        if can_add:
            pop = np.vstack((pop, new_pop))
            pop_obj = np.vstack((pop_obj, new_pop_obj))

        return pop, pop_obj















