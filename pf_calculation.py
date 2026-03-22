import matlab.engine
import numpy as np

def get_platemo_pf(eng, problem_name: str, M: int, D: int, n_point: int) -> np.ndarray:
    """
    Python calls MATLAB PlatEMO to obtain PF reference points for the specified problem

    Parameters:
        problem_name: PlatEMO built-in problem name (e.g., 'DTLZ2', 'ZDT1', 'WFG1', 'SOP_F1')
        M: Number of objectives for the optimization problem
        D: Dimension of decision variables
        n_point: Number of PF reference points to generate
        platemo_path: Absolute path to the root directory of PlatEMO

    Returns:
        np.ndarray: PF reference point matrix with shape (n_point, M),
                    each row is a reference point, each column corresponds to an objective
    """
    try:
        # Corresponding to the instantiation method of the PROBLEM class in the PlatEMO manual
        problem_obj = eng.feval(problem_name, 'M', matlab.double([M]), 'D', matlab.double([D]), nargout=1)

        # Corresponding to PlatEMO manual: GetOptimum inputs the number of optimal values and outputs the optimal value set matrix
        pf_matrix = eng.feval('GetOptimum', problem_obj, matlab.double([n_point]), nargout=1)

        # Convert MATLAB matrix to Python numpy array
        pf_np = np.array(pf_matrix)
        print(f"PF reference points obtained successfully, matrix shape: {pf_np.shape}")

        return pf_np

    except Exception as e:
        raise RuntimeError(f"Failed to obtain PF reference points: {e}")

    finally:
        pass
