import numpy as np
from matrix_configuration import MatrixConfiguration


if __name__ == "__main__":

	number_first_rows, number_first_columns = 4, 7
	first_mat = MatrixConfiguration(
		np.reshape(
			np.arange(
				number_first_rows * number_first_columns,
				dtype=int),
			(number_first_rows, number_first_columns)),
		is_real=True)

	number_second_rows, number_second_columns = 5, 4
	second_mat = MatrixConfiguration(
		np.reshape(
			100 * np.arange(
				number_second_rows * number_second_columns,
				dtype=int),
			(number_second_rows, number_second_columns)),
		is_real=True)
	complex_mat = MatrixConfiguration(
		second_mat.arr * 1j,
		is_real=False)
	matrix_product_between_first_and_second = first_mat @ second_mat
	matrix_product_between_first_and_complex = first_mat @ complex_mat
	print("\n .. FIRST MATRIX (shape={}):\n{}\n".format(
		first_mat.shape,
		first_mat))
	print("\n .. SECOND MATRIX (shape={}):\n{}\n".format(
		second_mat.shape,
		second_mat))
	print("\n .. COMPLEX MATRIX (shape={}):\n{}\n".format(
		complex_mat.shape,
		complex_mat))
	print("\n .. MATRIX PRODUCT between FIRST AND SECOND (shape={}):\n{}\n".format(
		matrix_product_between_first_and_second.shape,
		matrix_product_between_first_and_second))
	print("\n .. MATRIX PRODUCT between FIRST AND COMPLEX (shape={}):\n{}\n".format(
		matrix_product_between_first_and_complex.shape,
		matrix_product_between_first_and_complex))

##