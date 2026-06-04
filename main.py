import numpy as np
from data import get_detailed_problem_info
from algorithm import MOEA
from test_excel import read_excel_col_to_array
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.indicators.hv.approximate import ApproximateHypervolume

algorithm_name = "iemoeadi"
RUN_TIMES = 30

def main(problem_name, n_var, n_obj):
    f_quantity, x_quantity, range_x, pymoo_problem = get_detailed_problem_info(problem_name.upper().split('_', 1)[0], n_var, n_obj, 1)
    print(f"Optimization with {f_quantity} objectives and {x_quantity} decision variables")
    MaxFEs = 10000
    population_size = 100
    crossover_probability = 1
    crossover_probability_var = 1
    mutation_probability = 1
    mutation_probability_var = 1 / x_quantity
    eta_c = eta_m = 20
    n_elites = 10
    obj_dir = 1  # # 1 for minimization, -1 for maximization
    I_b = 0.5  # Global crossover probability
    aaa = False
    bbb = True
    igdplus_list = []
    gdplus_list = []
    hv_list = []
    file_name = f"./platemo_pf/{problem_name.lower()}.txt"
    with open(file_name, 'r', encoding='utf-8') as f:
        pf = np.array(eval(f.read()))
    total_X_histories = []
    total_Y_histories = []
    total_ps_histories = []
    total_pf_histories = []
    for run_idx in range(1, RUN_TIMES + 1):
        print(f"===== Start independent run {run_idx}/{RUN_TIMES} =====")
        SEED = run_idx
        np.random.seed(SEED)
        rng = np.random.default_rng(SEED)

        Algorithm_MOEA = MOEA(MaxFEs, problem_name.upper().split('_', 1)[0], f_quantity, x_quantity, range_x,
                              population_size, crossover_probability, crossover_probability_var, mutation_probability,
                              mutation_probability_var, eta_c, eta_m, n_elites, obj_dir, pymoo_problem, I_b, aaa, bbb, SEED)


        pareto_solutions, pareto_front, X_histories, Y_histories, ps_histories, pf_histories = Algorithm_MOEA.run()

        total_X_histories.append(X_histories)
        total_Y_histories.append(Y_histories)
        total_ps_histories.append(ps_histories)
        total_pf_histories.append(pf_histories)

        igdplus = IGDPlus(pf=pf).do(pareto_front)
        igdplus_list.append(float(igdplus))
        gdplus = GDPlus(pf=pf).do(pareto_front)
        gdplus_list.append(float(gdplus))

        if f_quantity > 3:
            hv_calc = ApproximateHypervolume(ref_point=np.max(pf, axis=0) * 1.1,
                                             random_state=np.random.RandomState(42))
            hv_calc.add(pareto_front)
            hv_value = hv_calc.hv
        else:
            hv_calc = ExactHypervolume(ref_point=np.max(pf, axis=0) * 1.1)
            hv_calc.add(pareto_front)
            hv_value = hv_calc.hv
        hv_list.append(float(hv_value))

        print(f"IGD+ of run {run_idx}: {igdplus}")
        print(f"GD+ of run {run_idx}: {gdplus}")
        print(f"HV of run {run_idx}: {hv_value}")
    print(f"===== All {RUN_TIMES} independent runs completed! =====")

problem_name_np = read_excel_col_to_array("p_parameter", "problem_name")
problem_var_np = read_excel_col_to_array("p_parameter", "n_var")
problem_obj_np = read_excel_col_to_array("p_parameter", "n_obj")

run_pd = False
for i in range(len(problem_name_np)):
    main(problem_name_np[i], problem_var_np[i], problem_obj_np[i])