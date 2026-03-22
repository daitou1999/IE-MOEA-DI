from data import get_detailed_problem_info
from algorithm import MOEA
import numpy as np
from pf_calculation import get_platemo_pf
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv.exact import ExactHypervolume  # Exact calculation for low dimensions
from pymoo.indicators.hv.approximate import ApproximateHypervolume

PLATEMO_PATH = "C:/Users/16482/Desktop/Python2/PlatEMO-master/PlatEMO-master/PlatEMO"
algorithm_name = "IE-MOEA/DI"
problem_name = "WFG9_3"
n_var = 30
n_obj = 11
RUN_TIMES = 30

def main(problem_name, n_var, n_obj):
    # Solve test case
    f_quantity, x_quantity, range_x, pymoo_problem, platemo_problem, pemo = get_detailed_problem_info(PLATEMO_PATH, problem_name.upper().split('_', 1)[0], n_var, n_obj, 1)
    print(f"Current optimization: {f_quantity} objectives, {x_quantity} decision variables")
    MaxFEs = 10000
    population_size = 100
    crossover_probability = 1  # Individual crossover probability
    crossover_probability_var = 1
    mutation_probability = 1  # Mutation probability
    mutation_probability_var = 1 / x_quantity  # Individual mutation probability
    eta_c = eta_m = 20
    n_elites = 15
    obj_dir = 1  # Minimize 1 / Maximize -1
    I_b = 0.5  # Global crossover probability
    aaa = False  # Redundant
    bbb = True  # Whether to enable archive pruning
    igdplus_list = []
    gdplus_list = []
    hv_list = []
    pf = get_platemo_pf(
        eng=pemo.eng,
        problem_name=problem_name.upper().split('_', 1)[0],
        M=int(f_quantity),
        D=int(x_quantity),
        n_point=10000
    )
    for run_idx in range(1, RUN_TIMES + 1):
        print(f"===== Start {run_idx}/{RUN_TIMES} independent runs =====")
        Algorithm_MOEA = MOEA(MaxFEs, problem_name.upper().split('_', 1)[0], f_quantity, x_quantity, range_x,
                              population_size, crossover_probability, crossover_probability_var, mutation_probability,
                              mutation_probability_var, eta_c, eta_m, n_elites, obj_dir, pymoo_problem, platemo_problem, I_b, aaa, bbb)


        pareto_solutions, pareto_front = Algorithm_MOEA.run()

        igdplus = IGDPlus(pf=pf).do(pareto_front)
        igdplus_list.append(float(igdplus))
        gdplus = GDPlus(pf=pf).do(pareto_front)
        gdplus_list.append(float(gdplus))
        if f_quantity > 3:
            # High dimension: Monte Carlo approximate calculation
            hv_calc = ApproximateHypervolume(ref_point=np.max(pf, axis=0) * 1.1,
                                             random_state=np.random.RandomState(42))
            hv_calc.add(pareto_front)
            hv_value = hv_calc.hv
        else:
            # Low dimension: Exact calculation
            hv_calc = ExactHypervolume(ref_point=np.max(pf, axis=0) * 1.1)
            hv_calc.add(pareto_front)
            hv_value = hv_calc.hv
        hv_list.append(float(hv_value))
    print(f"===== All {RUN_TIMES} runs completed! =====")
    pemo.close()


if __name__ == "__main__":
    main(problem_name, n_var, n_obj)