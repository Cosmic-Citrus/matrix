import numpy as np
from determinant_solver_configuration import DeterminantSolverConfiguration
from system_of_equations_solver_configuration import EquationSystemSolverConfiguration


class BaseMatrixConfiguration():

	def __init__(self):
		super().__init__()
		self._arr = None
		self._shape = None
		self._is_real = None
		self._is_complex = None
		self._is_square = None
		self._is_invertible = None
		self._is_hermitian = None
		self._is_anti_hermitian = None

	@property
	def arr(self):
		return self._arr

	@property
	def shape(self):
		return self._shape
	
	@property
	def is_real(self):
		return self._is_real
	
	@property
	def is_complex(self):
		return self._is_complex
	
	@property
	def is_square(self):
		return self._is_square

	@property
	def is_invertible(self):
		return self._is_invertible
	
	@property
	def is_hermitian(self):
		return self._is_hermitian

	@property
	def is_anti_hermitian(self):
		return self._is_anti_hermitian
	
	@staticmethod
	def get_autocorrected_arr(arr, is_real):

		def get_dtype(arr, is_real):
			if not isinstance(is_real, bool):
				raise ValueError("invalid type(is_real): {}".format(type(is_real)))
			if is_real:
				is_int = True
				for element in arr:
					if not isinstance(element, int):
						is_int = False
						break
				if is_int:
					dtype = int
				else:
					dtype = float
			else:
				dtype = complex
			return dtype

		def get_modified_arr(arr, dtype):
			if isinstance(arr, (tuple, list, np.ndarray)):
				modified_arr = np.array(
					arr,
					dtype=dtype)
			else:
				raise ValueError("invalid type(arr): {}".format(type(arr)))
			size_of_shape = len(
				modified_arr.shape)
			if size_of_shape == 0:
				raise ValueError("invalid np.array(arr).shape: {}".format(modified_arr.shape))
			elif size_of_shape == 1:
				modified_arr = np.reshape(
					modified_arr,
					(modified_arr.shape[0], 1))
			elif size_of_shape != len(["number rows", "number columns"]):
				raise ValueError("invalid np.array(arr).shape: {}".format(modified_arr.shape))
			return modified_arr

		dtype = get_dtype(
			arr=arr,
			is_real=is_real)
		modified_arr = get_modified_arr(
			arr=arr,
			dtype=dtype)
		return modified_arr

	@staticmethod
	def get_invertible_status(modified_arr):
		(number_rows, number_columns) = modified_arr.shape
		if number_rows == number_columns:
			det = np.linalg.det(
				modified_arr)
			is_invertible = (
				det != 0)
		else:
			is_invertible = False
		return is_invertible

	@staticmethod
	def get_hermitian_status(modified_arr):
		conjugate_arr = np.conjugate(
			modified_arr)
		adjoint_arr = conjugate_arr.T
		is_hermitian = np.array_equal(
			modified_arr,
			adjoint_arr)
		return is_hermitian

	@staticmethod
	def get_anti_hermitian_status(modified_arr):
		conjugate_arr = np.conjugate(
			modified_arr)
		adjoint_arr = conjugate_arr.T
		is_anti_hermitian = np.array_equal(
			adjoint_arr,
			-1 * modified_arr)
		return is_anti_hermitian

	def get_real_matrix(self):
		real_mat = MatrixConfiguration(
			np.real(
				self.arr),
			is_real=True)
		return real_mat

	def get_imaginary_matrix(self):
		imaginary_mat = MatrixConfiguration(
			1j * np.imag(
				self.arr),
			is_real=False)
		return imaginary_mat

class BaseMatrixMethodsConfiguration(BaseMatrixConfiguration):

	def __init__(self):
		super().__init__()

	def multiply_matrix(self, other_mat, is_test=False):
		if not isinstance(other_mat, MatrixConfiguration):
			raise ValueError("invalid type(other_mat): {}".format(type(other_mat)))
		if self.arr.shape[0] != other_mat.arr.shape[1]:
			raise ValueError("self.arr.shape={} is not compatible with other_mat.arr.shape={}".format(self.arr.shape, other_mat.shape))
		if self.arr.dtype == other_mat.arr.dtype:
			dtype = self.arr.dtype
		else:
			if (self.is_complex or other_mat.is_complex):
				dtype = complex
			else:
				dtype = float
		shape = (
			self.arr.shape[0],
			other_mat.arr.shape[1])
		product_value = np.full(
			fill_value=0,
			shape=shape,
			dtype=dtype)
		for index_at_primary_row in range(self.arr.shape[0]):
			for index_at_other_column in range(other_mat.arr.shape[1]):
				for index_at_primary_column in range(self.arr.shape[0]):
					product_value[index_at_primary_row, index_at_other_column] += (self.arr[index_at_primary_row, index_at_primary_column] * other_mat.arr[index_at_primary_column, index_at_other_column])
		mat = MatrixConfiguration(
			arr=product_value)
		return mat

	def get_determinant(self, cofactor_by="row", index_at_non_cofactor=0, is_test=False):
		if not self.is_square:
			raise ValueError("non-square matrix does not have determinant")
		det = DeterminantSolverConfiguration().get_determinant(
			self,
			cofactor_by=cofactor_by,
			index_at_non_cofactor=index_at_non_cofactor,
			is_test=is_test)
		return det

	def get_solved_system_of_equations(self, method="cramers rule", is_test=False):
		solver = EquationSystemSolverConfiguration()
		coefficients, constants, variables = solver.get_solved_system_of_equations(
			mat=self,
			method=method,
			is_test=is_test)
		coefficients_mat = MatrixConfiguration(
			coefficients,
			is_real=self.is_real)
		constants_mat = MatrixConfiguration(
			constants,
			is_real=self.is_real)
		variables_mat = MatrixConfiguration(
			variables,
			is_real=self.is_real)
		return coefficients_mat, constants_mat, variables_mat

	def get_inverse(self, is_test=False):
		raise ValueError("not yet implemented")
		# if not self.is_square:
		# 	raise ValueError("non-square matrix is not invertible")
		# if not self.is_invertible:
		# 	raise ValueError("matrix is not invertible")
		# det = self.get_determinant()
		# adjoint = self.get_adjoint()
		# if self.is_real:
		# 	tmp_inverse_mat = adjoint / det
		# 	inverse_mat = MatrixConfiguration(
		# 		tmp_inverse_mat.arr,
		# 		is_real=True)
		# else:
		# 	inverse_mat = adjoint / det
		# if is_test:
		# 	inverse_via_numpy = np.linalg.inv(
		# 		self.arr)
		# 	# is_inverse_elements_only_real = np.all(
		# 	# 	np.isreal(
		# 	# 		inverse_mat))
		# 	# is_numpy_elements_only_real = np.all(
		# 	# 	np.isreal(
		# 	# 		inverse_via_numpy))
		# 	# if is_inverse_elements_only_real:
		# 	# 	modified_inverse_arr = np.real(
		# 	# 		inverse_mat.arr)
		# 	# else:
		# 	# 	modified_inverse_arr = np.copy(
		# 	# 		inverse_mat.arr)
		# 	is_equal = np.allclose(
		# 		inverse_mat.arr,
		# 		inverse_via_numpy,
		# 		rtol=1e-10,
		# 		atol=1e-12)
		# 	if not is_equal:
		# 		raise ValueError("inverse_mat={} does not agree with inverse_via_numpy={}".format(inverse_mat, inverse_via_numpy))
		return inverse_mat

	def get_adjoint(self):
		complex_conjugate_mat = self.get_complex_conjugate()
		adjoint_mat = complex_conjugate_mat.get_transpose()
		return adjoint_mat

	def get_complex_conjugate(self):
		complex_conjugate_mat = MatrixConfiguration(
			np.conjugate(
				self.arr))
		return complex_conjugate_mat

	def get_transpose(self, is_test=False):
		new_shape = (
			self.shape[1],
			self.shape[0])
		new_arr = np.full(
			fill_value=0,
			shape=new_shape,
			dtype=self.arr.dtype)
		old_arr = np.copy(
			self.arr)
		for index_at_row in range(self.shape[0]):
			for index_at_column in range(self.shape[1]):
				new_arr[index_at_column, index_at_row] = old_arr[index_at_row, index_at_column]
		matrix = MatrixConfiguration(
			new_arr)
		if is_test:
			transpose_via_numpy = np.transpose(
				self.arr)
			if not np.array_equal(new_arr, transpose_via_numpy):
				raise ValueError("transpose={} does not agree with transpose_via_numpy={}".format(new_arr, transpose_via_numpy))
		return matrix

	def get_row_echelon(self, is_test=False):
		raise ValueError("not yet implemented")
		row_echelon_mat, augment_mat = ...
		return row_echelon_mat, augment_mat

	def get_reduced_row_echelon(self, is_test=False):
		raise ValueError("not yet implemented")
		reduced_row_echelon_mat, augment_mat = ...
		return reduced_row_echelon_mat, augment_mat

	def get_eigenvalues_and_eigenvectors(self, method, is_test=False):
		raise ValueError("not yet implemented")
		eigenvalues, eigenvectors = ...
		return eigenvalues, eigenvectors

class MatrixMethodsConfiguration(BaseMatrixMethodsConfiguration):

	def __init__(self):
		super().__init__()

	def initialize_matrix(self, arr, is_real):
		modified_arr = self.get_autocorrected_arr(
			arr=arr,
			is_real=is_real)
		is_complex = np.invert(
			is_real)
		shape = modified_arr.shape
		is_square = (modified_arr.shape[0] == modified_arr.shape[1])
		is_invertible = self.get_invertible_status(
			modified_arr=modified_arr)
		is_hermitian = self.get_hermitian_status(
			modified_arr=modified_arr)
		is_anti_hermitian = self.get_anti_hermitian_status(
			modified_arr=modified_arr)
		self._arr = modified_arr
		self._shape = shape
		self._is_real = is_real
		self._is_complex = is_complex
		self._is_square = is_square
		self._is_invertible = is_invertible
		self._is_hermitian = is_hermitian
		self._is_anti_hermitian = is_anti_hermitian

class MatrixConfiguration(MatrixMethodsConfiguration):

	def __init__(self, arr, is_real=False):
		super().__init__()
		self.initialize_matrix(
			arr=arr,
			is_real=is_real)

	def __call__(self):
		return self.arr

	def __str__(self):
		s = str(
			self.arr)
		return s

	def __repr__(self):
		mat = f"MatrixConfiguration({self.arr})"
		return mat

	def __len__(self):
		length = self.shape[0]
		return length

	def __iter__(self):
		for r in range(self.shape[0]):
			for c in range(self.shape[1]):
				yield self.arr[r][c]

	def __next__(self):
		raise ValueError("not yet implemented")
		...
		return

	def __index__(self, loc):
		raise ValueError("not yet implemented")
		...
		return

	def __contains__(self, other_value):
		if isinstance(other_value, (int, float)):
			is_contained = (
				other_value in self.arr.flatten())
		elif isinstance(other_value, (tuple, list, np.ndarray, MatrixConfiguration)):
			if isinstance(other_value, MatrixConfiguration):
				modified_other_value = other_value
			else:
				modified_other_value = MatrixConfiguration(
					other_value)
			raise ValueError("not yet implemented")
			...
		else:
			raise ValueError("invalid type(other_value): {}".format(type(other_value)))
		return is_contained

	def __eq__(self, other_value):
		raise ValueError("not yet implemented")
		...
		return

	def __ne__(self, other_value):
		raise ValueError("not yet implemented")
		...
		return

	def __lt__(self, other_value):
		raise ValueError("not yet implemented")
		...
		return

	def __gt__(self, other_value):
		raise ValueError("not yet implemented")
		...
		return

	def __le__(self, other_value):
		raise ValueError("not yet implemented")
		...
		return

	def __ge__(self, other_value):
		raise ValueError("not yet implemented")
		...
		return

	def __abs__(self):
		mat = MatrixConfiguration(
			arr=np.abs(
				self.arr))
		return mat

	def __add__(self, other_value):
		mat = MatrixConfiguration(
			self.arr + other_value)
		return mat

	def __sub__(self, other_value):
		mat = MatrixConfiguration(
			self.arr - other_value)
		return mat

	def __mul__(self, other_value):
		mat = MatrixConfiguration(
			self.arr * other_value)
		return mat

	def __truediv__(self, other_value):
		mat = MatrixConfiguration(
			self.arr / other_value)
		return mat

	def __floordiv__(self, other_value):
		mat = MatrixConfiguration(
			self.arr // other_value)
		return mat

	def __matmul__(self, other_value):
		mat = self.multiply_matrix(
			other_mat=other_value,
			is_test=True)
		return mat

	def __pow__(self, other_value):
		if isinstance(other_value, (int, float)):
			if (float(other_value) == int(other_value)) and (other_value >= 0):
				if other_value == 0:
					...
					raise ValueError("not yet implemented")
				elif other_value == 1:
					mat = self.mat
				else:
					mat = self.mat @ self.mat
					for _ in range(other_value - 1):
						mat = self.mat @ mat
			else:
				...
				raise ValueError("not yet implemented")
		else:
			raise ValueError("not yet implemented")
		return mat

	def __iadd__(self, other_value):
		self._mat = self.mat + other_value
		return self.mat

	def __isub__(self, other_value):
		self._mat = self.mat - other_value
		return self.mat

	def __imul__(self, other_value):
		self._mat = self.mat * other_value
		return self.mat

	def __itruediv__(self, other_value):
		self._mat = self.mat / other_value
		return self.mat

	def __ifloordiv__(self, other_value):
		self._mat = self.mat // other_value
		return self.mat

	def __imatmul__(self, other_value):
		self._mat = self.mat @ other_value
		return self.mat

	def __ipow__(self, other_value):
		self._mat = self.mat ** other_value
		return self.mat

##