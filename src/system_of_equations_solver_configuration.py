import numpy as np
from determinant_solver_configuration import DeterminantSolverConfiguration


class BaseEquationSystemSolverConfiguration():

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_coefficients_and_constants(mat):
		if not isinstance(mat.arr, (tuple, list, np.ndarray)):
			raise ValueError("invalid type(mat.arr): {}".format(type(mat.arr)))
		arr = np.copy(
			mat.arr)
		size_of_shape = len(
			arr.shape)
		if size_of_shape != len(["number rows", "number columns"]):
			raise ValueError("invalid np.array(mat.arr).shape: {}".format(arr.shape))
		(number_rows, number_columns) = arr.shape
		if number_rows + 1 != number_columns:
			raise ValueError("mat should contain one more column than rows to represent the right-hand side of the equations in the system")
		coefficients = arr[:, :-1]
		constants = arr[:, -1]
		return coefficients, constants

class BaseEquationSystemSolverMethodsConfiguration(BaseEquationSystemSolverConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_solution_by_cramers_rule(coefficients, constants):
		det_coefficients = DeterminantSolverConfiguration().get_determinant(
			coefficients)
		if det_coefficients == 0:
			raise ValueError("det(coefficients) = 0 ==> no unique solution")
		variables = list()
		for index_at_column in range(coefficients.shape[1]):
			modified_coefficients = np.copy(
				coefficients)
			modified_coefficients[:, index_at_column] = np.copy(
				constants)
			det = DeterminantSolverConfiguration().get_determinant(
				modified_coefficients)
			variable = det / det_coefficients
			variables.append(
				variable)
		variables = np.array(
			variables)
		return variables

class EquationSystemSolverConfiguration(BaseEquationSystemSolverMethodsConfiguration):

	def __init__(self):
		super().__init__()

	def get_solved_system_of_equations(self, mat, method, is_test=False):
		coefficients, constants = self.get_coefficients_and_constants(
			mat=mat)
		solve_by_cramers_rule = lambda : self.get_solution_by_cramers_rule(
			coefficients=coefficients,
			constants=constants)
		mapping = {
			"cramers rule" : solve_by_cramers_rule,
			}
		if method not in mapping.keys():
			raise ValueError("invalid method: {}".format(method))
		get_solution = mapping[method]
		variables = get_solution()
		if is_test:
			solution_via_numpy = np.linalg.solve(
				coefficients,
				constants)
			is_approximately_equal = np.allclose(
				variables,
				solution_via_numpy)
			if not is_approximately_equal:
				raise ValueError("variables={} does not agree with solution_via_numpy={}".format(variables, solution_via_numpy))
		return coefficients, constants, variables

##