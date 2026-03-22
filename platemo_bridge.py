import matlab.engine
import numpy as np
from typing import Tuple, Optional, Union
import os
import sys


class PlatEMOProblem:
    """
    Python wrapper for PROBLEM class instances in PlatEMO
    """

    def __init__(self, eng, ml_problem_handle):
        self._eng = eng
        self._h = ml_problem_handle  # Handle of MATLAB PROBLEM object

        # Cache basic attributes to avoid frequent cross-process calls
        self.M = self._get_attr('M')
        self.D = self._get_attr('D')
        self.maxFE = self._get_attr('maxFE')
        self.encoding = np.array(self._get_attr('encoding')).flatten().astype(int)
        self.lower = np.array(self._get_attr('lower')).flatten()
        self.upper = np.array(self._get_attr('upper')).flatten()
        self.FE = 0

    def _get_attr(self, attr_name):
        """Internal method: Get MATLAB object attribute"""
        return self._eng.getfield(self._h, attr_name, nargout=1)

    def _reset_FE(self):
        """Reset evaluation count"""
        self._eng.setfield(self._h, 'FE', 0, nargout=0)
        self.FE = 0

    def get_variable_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get decision variable bounds"""
        return self.lower, self.upper

    def Initialization(self, N: int) -> np.ndarray:
        """
        Initialize population, corresponding to Problem.Initialization() in PlatEMO
        :param N: Population size
        :return: Decision variable matrix (N, D)
        """
        # Call MATLAB: PopDec = Problem.Initialization(N)
        ml_pop_dec = self._eng.Initialization(self._h, float(N), nargout=1)
        return np.array(ml_pop_dec)

    def Evaluation(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate solutions, corresponding to Problem.Evaluation() in PlatEMO
        :param x: Decision variables (N, D) or (D,)
        :return:
            PopObj: Objective value matrix (N, M)
            PopCon: Constraint violation value matrix (N, K)
        """
        # Data preprocessing
        # if isinstance(x, list):
        #     x = np.array(x)
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)

        # Convert to MATLAB double-precision matrix
        ml_x = matlab.double(x.tolist())

        # Call MATLAB: Population = Problem.Evaluation(PopDec)
        # Note: Evaluation in PlatEMO returns an array of SOLUTION objects
        # We need to extract obj and con on the MATLAB side, or directly call CalDec/CalObj/CalCon
        # For performance and stability, call underlying calculation functions directly here

        # # 1. Repair solutions (CalDec)
        # ml_x_fixed = self._eng.CalDec(self._h, ml_x, nargout=1)

        # # 2. Calculate objectives (CalObj)
        ml_obj = self._eng.CalObj(self._h, ml_x, nargout=1)

        # # Update evaluation count (record on Python side, and synchronize to MATLAB just in case)
        # self.FE += x.shape[0]
        # current_fe = self._get_attr('FE')
        # self._eng.setfield(self._h, 'FE', float(current_fe + x.shape[0]), nargout=0)

        # Convert back to Numpy
        PopObj = np.array(ml_obj)

        # Handle dimensions
        # if PopObj.ndim == 0: PopObj = PopObj.reshape(1, 1)

        return PopObj


class PlatEMO:
    """
    Main class of PlatEMO Python bridge
    """

    def __init__(self, platemo_path: str):
        """
        Initialize PlatEMO bridge
        :param platemo_path: Root directory path of the downloaded PlatEMO project
        """
        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()
        print("MATLAB Engine started successfully.")

        # Check path
        if not os.path.exists(platemo_path):
            raise FileNotFoundError(f"PlatEMO path not found: {platemo_path}")

        # Add PlatEMO to MATLAB path
        print(f"Loading PlatEMO from: {platemo_path}")
        self.eng.addpath(platemo_path, nargout=0)
        # Try to add subdirectories (recursively)
        try:
            self.eng.eval(f"addpath(genpath('{platemo_path}'));", nargout=0)
        except:
            pass  # Ignore possible warnings from genpath

    def get_problem(self, problem_name: str, M: int = None, D: int = None, **kwargs) -> PlatEMOProblem:
        """
        Core factory function: Get PlatEMO problem instance
        :param problem_name: Problem name (e.g., 'ZDT1', 'DTLZ2', 'SOP_F1', 'WFG4')
        :param M: Number of objectives (optional, supported by some problems)
        :param D: Number of decision variables (optional)
        :param kwargs: Other parameters, such as 'maxFE'
        :return: PlatEMOProblem instance
        """
        # Build parameter list
        params = []

        # Note: Instantiation of problem classes in PlatEMO is usually done through UserProblem or directly calling the class constructor
        # The most general way is to use the syntax of platemo.m, or directly feval the class name

        # Method: Call constructor through feval and pass parameters
        # Example: Pro = DTLZ2('M', 3, 'D', 12)

        # First try to instantiate the object directly
        try:
            # Build MATLAB key-value pair parameters
            ml_args = []
            if M is not None:
                ml_args.extend(['M', float(M)])
            if D is not None:
                ml_args.extend(['D', float(D)])

            # Call constructor
            # Syntax: eng.feval(ClassName, Arg1, Arg2, ...)
            ml_problem_handle = self.eng.feval(problem_name, *ml_args, nargout=1)

            # If there are attributes like maxFE, set them manually
            if 'maxFE' in kwargs:
                self.eng.setfield(ml_problem_handle, 'maxFE', float(kwargs['maxFE']), nargout=0)

            return PlatEMOProblem(self.eng, ml_problem_handle)

        except Exception as e:
            raise RuntimeError(f"Failed to create problem '{problem_name}'. Please ensure the problem exists in PlatEMO.\nMATLAB error: {str(e)}")

    def close(self):
        """Close MATLAB Engine"""
        if hasattr(self, 'eng'):
            self.eng.quit()
            print("MATLAB Engine closed.")