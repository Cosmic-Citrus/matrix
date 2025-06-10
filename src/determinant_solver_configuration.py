import numpy as np


class BaseDeterminantSolverConfiguration():

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_modified_arr(mat):
		if isinstance(mat, np.ndarray):
			arr = np.copy(
				mat)
		elif isinstance(mat.arr, (tuple, list, np.ndarray)):
			arr = np.copy(
				mat.arr)
		else:
			raise ValueError("invalid type(mat.arr): {}".format(type(mat.arr)))
		size_of_shape = len(
			arr.shape)
		if size_of_shape != len(["number rows", "number columns"]):
			raise ValueError("invalid np.array(mat.arr).shape: {}".format(arr.shape))
		(number_rows, number_columns) = arr.shape
		if number_rows != number_columns:
			raise ValueError("mat is not a square matrix")
		return arr

	@staticmethod
	def verify_inputs(arr, cofactor_by, index_at_non_cofactor, is_test):
		if cofactor_by not in ("row", "column"):
			raise ValueError("invalid cofactor_by: {}".format(cofactor_by))
		if not isinstance(is_test, bool):
			raise ValueError("invalid type(is_test): {}".format(type(is_test)))
		if not isinstance(index_at_non_cofactor, int):
			raise ValueError("invalid type(index_at_non_cofactor): {}".format(type(index_at_non_cofactor)))
		if index_at_non_cofactor > 0:
			if index_at_non_cofactor >= arr.shape[0]:
				raise ValueError("invalid index_at_non_cofactor: {}".format(index_at_non_cofactor))
			if (index_at_non_cofactor < 0) and (abs(index_at_non_cofactor) > arr.shape[0]):
				raise ValueError("invalid index_at_non_cofactor: {}".format(index_at_non_cofactor))

class DeterminantSolverMethodsConfiguration(BaseDeterminantSolverConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_determinant_of_two_by_two(arr):
		value_11 = arr[0, 0]
		value_12 = arr[0, 1]
		value_21 = arr[1, 0]
		value_22 = arr[1, 1]
		det = (value_11 * value_22) - (value_12 * value_21)
		return det

	def get_determinant_by_recursive_method(self, arr, cofactor_by, index_at_non_cofactor):

		def get_sign_value(index_at_cofactor):
			if index_at_cofactor % 2 == 0:
				sign_value = 1
			else:
				sign_value = -1
			return sign_value

		def get_cell_value(arr, cofactor_by, index_at_cofactor, index_at_non_cofactor):
			if cofactor_by == "row":
				cell_value = arr[index_at_cofactor, index_at_non_cofactor]
			else: # elif cofactor_by == "column":
				cell_value = arr[index_at_non_cofactor, index_at_cofactor]
			return cell_value

		def get_inner_matrix(arr, cofactor_by, index_at_cofactor, index_at_non_cofactor):
			if cofactor_by == "row":
				sub_arr = np.delete(
					arr,
					index_at_cofactor,
					axis=0)
				sub_arr = np.delete(
					sub_arr,
					index_at_non_cofactor,
					axis=1)
			else: # elif cofactor_by == "column":
				sub_arr = np.delete(
					arr,
					index_at_non_cofactor,
					axis=0)
				sub_arr = np.delete(
					sub_arr,
					index_at_cofactor,
					axis=1)
			return sub_arr

		if arr.shape == (1, 1):
			pass
		elif arr.shape == (2, 2):
			det = self.get_determinant_of_two_by_two(
				arr=arr)
		else:
			det = 0
			for index_at_cofactor in range(arr.shape[0]):
				sign_value = get_sign_value(
					index_at_cofactor=index_at_cofactor)
				cell_value = get_cell_value(
					arr=arr,
					cofactor_by=cofactor_by,
					index_at_cofactor=index_at_cofactor,
					index_at_non_cofactor=index_at_non_cofactor)
				sub_arr = get_inner_matrix(
					arr=arr,
					cofactor_by=cofactor_by,
					index_at_cofactor=index_at_cofactor,
					index_at_non_cofactor=index_at_non_cofactor)
				inner_det = self.get_determinant_by_recursive_method(
					arr=sub_arr,
					cofactor_by=cofactor_by,
					index_at_non_cofactor=index_at_non_cofactor)
				det += (sign_value * cell_value * inner_det)
		return det

class DeterminantSolverConfiguration(DeterminantSolverMethodsConfiguration):

	def __init__(self):
		super().__init__()

	def get_determinant(self, mat, cofactor_by="row", index_at_non_cofactor=0, is_test=False):
		arr = self.get_modified_arr(
			mat=mat)
		self.verify_inputs(
			arr=arr,
			cofactor_by=cofactor_by,
			index_at_non_cofactor=index_at_non_cofactor,
			is_test=is_test)
		det = self.get_determinant_by_recursive_method(
			arr=arr,
			cofactor_by=cofactor_by,
			index_at_non_cofactor=index_at_non_cofactor)
		if is_test:
			det_via_numpy = np.linalg.det(
				arr)
			is_approximately_equal = np.isclose(
				[det],
				[det_via_numpy])
			if not is_approximately_equal:
				raise ValueError("det={} does not agree with det_via_numpy={}".format(det, det_via_numpy))
		return det

##