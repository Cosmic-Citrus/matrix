import numpy as np
from matrix_configuration import MatrixConfiguration


if __name__ == "__main__":

	identity_mat = MatrixConfiguration(
		np.eye(
			3,
			dtype=int),
		is_real=True)

	number_rows = number_columns = 5
	chronological_mat = MatrixConfiguration(
		np.reshape(
			np.arange(
				number_rows * number_columns,
				dtype=int),
			(number_rows, number_columns)),
		is_real=True)
	
	custom_mat = MatrixConfiguration(
		np.array([
			[np.pi, np.pi, 1, 2, 3],
			[np.exp(1), 3, 100, 0, 1],
			[1, 4, 2, 3, 0],
			[0, 1, 2, 1, 1],
			[-100, -1000, 100, 0, 1]]),
		is_real=True)

	complex_mat = MatrixConfiguration(
		custom_mat.arr * 1j,
		is_real=False)

	matrices = (
		identity_mat,
		chronological_mat,
		custom_mat,
		complex_mat)

	for mat in matrices:
		det = mat.get_determinant(
			is_test=True)
		print("\n .. MATRIX (shape={}):\n{}\n".format(
			mat.shape,
			mat))
		print("\n .. det(MATRIX):\n{}\n".format(
			det))

##