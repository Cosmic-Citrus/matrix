import numpy as np


class BaseInverseConfiguration():

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
	def get_lower_upper_triangles(arr):
		if np.isreal(arr):
			dtype = float
		else:
			dtype = complex
		lower_triangle = np.full(
			fill_value=0,
			shape=arr.shape,
			dtype=dtype)
		upper_triangle = np.full(
			fill_value=0,
			shape=arr.shape,
			dtype=dtype)
		...
		return lower_triangle, upper_triangle


class InverseConfiguration(BaseInverseConfiguration):

	def __init__(self):
		super().__init__()

	def get_inverse(self, mat):
		arr = self.get_modified_arr(
			mat=mat)
		lower_triangle, upper_triangle = self.get_lower_upper_triangles(
			arr=arr)
		inverse = np.full(
			fill_value=0,
			shape=arr.shape,
			dtype=lower_triangle.dtype)
		...



##