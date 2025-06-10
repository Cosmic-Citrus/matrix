import numpy as np
from matrix_configuration import MatrixConfiguration


if __name__ == "__main__":

	# coefficients = np.array([
	# 	[1, 1, -1],
	# 	[3, -2, 1],
	# 	[1, 3, -2]],
	# 	dtype=int)
	# constants = np.array([
	# 	6,
	# 	-5,
	# 	14])
	coefficients = np.array([
		[100, 1000, 1],
		[3, -2, -0.1 * np.exp(5)],
		[25, -25, -100]],
		dtype=float)
	constants = np.array([
		np.sqrt(np.pi),
		-15,
		7],
		dtype=float)
	augment = np.column_stack((
		coefficients,
		constants))
	# augment = np.array([
	# 	[1, 10, 1 + 1j, 1000],
	# 	[4 * 1j, 0, -20, 1j],
	# 	[3, 2, 1, 1, 25],
	# 	[1j, 3 - 1j, 1, 25]],
	# 	dtype=int)

	mat = MatrixConfiguration(
		augment,
		is_real=True)
	_coefficients, _constants, solution_variables = mat.get_solved_system_of_equations(
		method="cramers rule",
		is_test=True)
	
	print("\n .. COEFFICIENTS (shape={}):\n{}\n".format(
		_coefficients.shape,
		_coefficients))
	print("\n .. CONSTANTS (shape={}):\n{}\n".format(
		_constants.shape,
		_constants))
	print("\n .. AUGMENT (shape={}):\n{}\n".format(
		augment.shape,
		augment))
	print("\n .. SOLUTION VARIABLES (shape={}):\n{}\n".format(
		solution_variables.shape,
		solution_variables))

##