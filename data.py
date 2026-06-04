import numpy as np
import importlib

def get_detailed_problem_info(problem_name, n_var, n_obj, d):
    try:
        module = importlib.import_module(f"problems.{problem_name.upper()}")
        pymoo_problem_1 = getattr(module, f"{problem_name.upper()}")
        pymoo_problem = pymoo_problem_1(n_obj=n_obj, n_var=n_var)
        lower = pymoo_problem.xl
        upper = pymoo_problem.xu
        if len(lower) == 1 or len(upper) == 1:
            lower = np.array(list(lower) * n_var)
            upper = np.array(list(upper) * n_var)
        bounds = np.column_stack((lower, upper))
        return n_obj, n_var, bounds, pymoo_problem

    except Exception as e:
        error_msg = f"Failed to retrieve information for problem '{problem_name}': {str(e)}"
        raise ValueError(error_msg)
