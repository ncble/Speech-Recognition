import numpy as np

class Objective_Function():
	"""
	This wrapper aims to save all the points/values evaluated by solver.

	Input:
		fun: Black-box function
		save_to: [path_to_points, path_to_values]  e.g. ["./SVM_points.txt", "./SVM_values.txt"]

	"""

	def __init__(self, fun, isBO = False, save_to=None):
		self.fun = fun
		self.history_f = []
		self.fbest = np.inf
		self.history_fbest = []
		self.isBO = isBO
		self.save_to = save_to
		self.count = 0
		
	def __call__(self, x):
		if self.isBO:
			x = np.array(x)
		value = self.fun(x)
		self.count = self.count+1
		if self.save_to is not None:
			assert len(self.save_to) == 2, "save_to should be a length 2 list/array [path_to_points, path_to_values]"
			np.savetxt(open(self.save_to[0], "ab"), x.reshape(1,-1))
			np.savetxt(open(self.save_to[1], "ab"), np.array(value).reshape(1,1))

		# self.history_f.append(value)
		# self.fbest = min(self.fbest, value)
		# self.history_fbest.append(self.fbest)
		return value