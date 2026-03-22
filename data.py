import numpy as np
import importlib
from pymoo.core.problem import Problem
from platemo_bridge import PlatEMO

def get_detailed_problem_info(path_to_platemo, problem_name, n_var, n_obj, d):
    """
    Get detailed information of pymoo problem instance

    Returns:
        dict: Dictionary containing various attributes of the problem
    """
    try:
        if d == 1:
            pemo = PlatEMO(platemo_path=path_to_platemo)
            platemo_problem = pemo.get_problem(problem_name, M=n_obj, D=n_var)
            lower, upper = platemo_problem.get_variable_range()
            if len(lower) == 1 or len(upper) == 1:
                lower = np.array(list(lower) * n_var)
                upper = np.array(list(upper) * n_var)
            bounds = np.column_stack((lower, upper))
        else:
            # Dynamically import module (functions.MMF1)
            module = importlib.import_module(f"functions.{problem_name}")
            # Step 2: Get the specified class (MMF1 class) from the imported module
            problem = getattr(module, problem_name)()
        pymoo_problem = Problem(n_obj=n_obj, n_var=n_var, xl=lower, xu=upper)
        return n_obj, n_var, bounds, pymoo_problem, platemo_problem, pemo

    except Exception as e:
        error_msg = f"Failed to get information for problem '{problem_name}': {str(e)}"
        raise ValueError(error_msg)