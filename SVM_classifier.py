# _*_ coding: utf-8 _*_
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## SVM model
from sklearn.model_selection import train_test_split
# from random import shuffle  
# np.random.shuffle
from sklearn.utils import shuffle  # create and return a copy
from sklearn import svm
from sklearn.preprocessing import normalize

##### Hyperparameters tuing solvers ####
from PRS import PRS
from skopt import gp_minimize
from wrapper import Objective_Function

try:
	from DFO_src import dfo_tr
	import dill # To read DFO's result # pip install dill
except:
	print("Please install dill (pip install dill)")
try:
	import cma
except:
	print("Please install CMA (pip install CMA)")



class SVM_Model(object):
	"""docstring for SVM_Model"""
	def __init__(self, filename_X, filename_Y):
		# super(SVM_Model, self).__init__()
		'''
		Load data and split -> train, validate, test
		'''
		self.data_X = np.load(filename_X)
		self.data_Y = np.load(filename_Y)
		self.shape = self.data_X.shape
		self.n_features = np.product(self.data_X.shape[1:]) # The features size
		self.X_train = None
		self.X_val = None
		self.X_test = None

		self.y_train = None
		self.y_val = None
		self.y_test = None

	def preprocessing(self, split_rate = [0.8, 0.1, 0.1], cut = None):
		print("Start preprocessing...")
		assert sum(split_rate) == 1.0, "Sum of split rate should be 1.0 !"
		self.data_X = self.data_X.reshape((self.shape[0], self.shape[1]*self.shape[2]))
		self.data_X, self.data_Y = shuffle(self.data_X, self.data_Y)
		if cut is not None:
			self.data_X = self.data_X[:cut]
			self.data_Y = self.data_Y[:cut]
		
		# Normalization: center and reduce (var ~= 1) # TODO
		# moy = np.mean(self.data_X, axis = 0)
		# self.data_X = (self.data_X-moy)
		# x_max = np.maximum(self.data_X)
		# self.data_X = self.data_X/x_max
		#################################
		self.data_X = normalize(self.data_X, norm = "l2")  # TODO to check


		self.data_Y = np.argmax(self.data_Y, axis = 1)

		if len(split_rate) == 3:

			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_X, self.data_Y, test_size=split_rate[-1], random_state=17)
			validation_size = split_rate[1]/(split_rate[0] + split_rate[1])
			self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size =validation_size, random_state=19)
		elif len(split_rate) == 2:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_X, self.data_Y, test_size=split_rate[-1], random_state=17)
		else:
			raise ValueError("'split_rate' should be an list/array of length 2 or 3. e.g. [0.7, 0.2, 0.1] means train/val/test set.")

		print("Preprocessing done.")

	def EvaluateSVM(self, x, evaluate_on = "test", bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):

		"""
		Loss function to be minimize using DFO-TR algorithm (hyperparameters' tuning). 
		"""
		if x is None:
			C = 1.0
			gamma = 1./self.n_features
		else:
			gamma, C = self.scale2real(x, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
		
		clf = svm.SVC(gamma=gamma, C=C, max_iter=-1) 
		clf.fit(self.X_train, self.y_train)

		if evaluate_on == "val":
			Loss_val = -clf.score(self.X_val, self.y_val)
			print("Loss on the validation set: {} with (gamma: {}, C: {}).".format(Loss_val, gamma, C))
			return Loss_val
		elif evaluate_on == "test":
			Loss_test = -clf.score(self.X_test, self.y_test)
			print("Loss on the test set: {} with (gamma: {}, C: {}).".format(Loss_test, gamma, C))
			return Loss_test
		else:
			raise ValueError("'evaluate_on' should be 'test' or 'val'.")
		

	def linear_transform(self, x, bounds_old, log_scale = False, to_real = False):
		"""
		Transform a scalar to another search space.

		Input: 
			x: a scalar
			bounds_old: list or np.array of form [a, b]

		return:
			scaled x
		"""
		# 

		if log_scale:
			a, b = np.log10(bounds_old)
			m, k = 10./(b-a), -5*(b+a)/(b-a)
			
			if to_real:
				return pow(10,(x-k)/m) # gamma
			else:
				x = np.log10(x)	
				return x*m + k
		else:
			a, b = bounds_old
			m, k = 10./(b-a), -5*(b+a)/(b-a)
			if to_real:
				return (x-k)/m # C
			else:
				return x*m + k


	def scale2real(self, x, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		"""
		Input:
			x: a length 2 np.array or list in the searching space [-5, 5]^2
		return:
			gamma, C
		"""
		x = x.ravel()
		gamma = self.linear_transform(x[0], bounds_gamma, log_scale = True, to_real = True)
		C = self.linear_transform(x[1], bounds_C, log_scale = False, to_real = True)+0.000001
		return gamma, C

	def scale_x(self, x, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		"""
		Input:
			x: a length 2 np.array (gamma, C)
		return:
			(y[0], y[1]) in the searching space [-5, 5]^2
		"""

		y = np.zeros(2)

		y[0] = self.linear_transform(x[0], bounds_gamma, log_scale = True, to_real = False)
		y[1] = self.linear_transform(x[1], bounds_C, log_scale = False, to_real = False)
		return y

	# def Fine_tune_SVM_with_DFO(self, x_initial, evaluate_on = "test", restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
	# 	"""
	# 	Searching space: [-5,5] * [-5,5] 


	# 	"""
		
	# 	if evaluate_on == "test":
	# 		eval_set = self.X_test
	# 	elif evaluate_on == "val":
	# 		eval_set = self.X_val
	# 	else:
	# 		raise ValueError("'evaluate_on' should be 'test' or 'val'.")

	# 	sample_size = len(self.X_train) + len(eval_set)

	# 	if not os.path.exists("./hp_tuning_data/dfo_data_lu/"):
	# 		os.makedirs("./hp_tuning_data/dfo_data_lu/")
	# 	with open("./hp_tuning_data/dfo_data_lu/config.txt", "wb") as file:
	# 		message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
	# 		gamma_init , C_init = self.scale2real(x_initial, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
	# 		message = message+"Initial point of x: {}  (real values: (gamma = {}, C = {}))".format(x_initial, gamma_init, C_init)
	# 		file.write(message)
		
	# 	res = dfo_tr.dfo_tr(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), x_initial)
	# 	print(res.x)

	# def Fine_tune_SVM_with_CMA_ES(self, x_initial, sigma0, evaluate_on = "test", restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
	# 	"""
	# 	Searching space: ???
		
	# 	TODO 

	# 	"""
		

	# 	save_path = ["./hp_tuning_data/cma_data/SVM.txt", "./hp_tuning_data/cma_data/SVM_value.txt"] 
	# 	bb_fun = Objective_Function(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), save_to=save_path)

	# 	if evaluate_on == "test":
	# 		eval_set = self.X_test
	# 	elif evaluate_on == "val":
	# 		eval_set = self.X_val
	# 	else:
	# 		raise ValueError("'evaluate_on' should be 'test' or 'val'.")

	# 	sample_size = len(self.X_train) + len(eval_set)
		

	# 	if not os.path.exists("./hp_tuning_data/cma_data/"):
	# 		os.makedirs("./hp_tuning_data/cma_data/")
	# 	with open("./hp_tuning_data/cma_data/config.txt", "wb") as file:
	# 		message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
	# 		gamma_init , C_init = self.scale2real(x_initial, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
	# 		message = message+"Initial point of x: {}  (real values: (gamma = {}, C = {}))".format(x_initial, gamma_init, C_init)
	# 		file.write(message)
		
	# 	res = cma.fmin(bb_fun, x_initial, sigma0, options={'bounds': [[-5.000001,-5.000001], [5,5]]})
	# 	cma.plot()
	# 	plt.savefig("./hp_tuning_data/cma_data/cma_plot.png")
	# 	print("="*50)
	# 	print("Best point found [gamma, C]: {}".format(self.scale2real(np.array(res[0]), bounds_gamma=bounds_gamma, bounds_C=bounds_C)))
	# 	print("CMA all done.")
        
	# def Fine_tune_SVM_with_PRS(self, n_evals = 100, evaluate_on = "test", restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
	# 	save_path = ["./hp_tuning_data/prs_data/SVM.txt", "./hp_tuning_data/prs_data/SVM_value.txt"] 
	# 	bb_fun = Objective_Function(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), save_to=save_path)

	# 	if evaluate_on == "test":
	# 		eval_set = self.X_test
	# 	elif evaluate_on == "val":
	# 		eval_set = self.X_val
	# 	else:
	# 		raise ValueError("'evaluate_on' should be 'test' or 'val'.")

	# 	sample_size = len(self.X_train) + len(eval_set)
		

	# 	if not os.path.exists("./hp_tuning_data/prs_data/"):
	# 		os.makedirs("./hp_tuning_data/prs_data/")
	# 	with open("./hp_tuning_data/prs_data/config.txt", "wb") as file:
	# 		message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
	# 		file.write(message)
		
	# 	res = PRS(bb_fun, 2, n_evals, bounds = [-5,5])
	# 	print("="*50)
	# 	print("Best point found [gamma, C]: {}".format(self.scale2real(np.array(res[0]), bounds_gamma=bounds_gamma, bounds_C=bounds_C)))
	# 	print("PRS all done.")
	
	# def Fine_tune_SVM_with_BO(self, n_calls = 20, evaluate_on = "test", restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
	# 	save_path = ["./hp_tuning_data/bo_data/SVM.txt", "./hp_tuning_data/bo_data/SVM_value.txt"] 
	# 	bb_fun = Objective_Function(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), save_to=save_path, isBO = True)

	# 	if evaluate_on == "test":
	# 		eval_set = self.X_test
	# 	elif evaluate_on == "val":
	# 		eval_set = self.X_val
	# 	else:
	# 		raise ValueError("'evaluate_on' should be 'test' or 'val'.")

	# 	sample_size = len(self.X_train) + len(eval_set)
		

	# 	if not os.path.exists("./hp_tuning_data/bo_data/"):
	# 		os.makedirs("./hp_tuning_data/bo_data/")
	# 	with open("./hp_tuning_data/bo_data/config.txt", "wb") as file:
	# 		message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
	# 		file.write(message)
		
	# 	res = gp_minimize(bb_fun, [(-5,5)]*2, n_calls = n_calls)
	# 	print("="*50)
	# 	print("Best point found [gamma, C]: {}".format(self.scale2real(np.array(res.x), bounds_gamma=bounds_gamma, bounds_C=bounds_C)))
	# 	print("Bayesian optimization all done.")

	def Fine_Tune_SVM(self, optimizer = "DFO", evaluate_on = "test", x_initial = None, sigma0 = None, options_cma = {'bounds': [[-5.000001,-5.000001], [5,5]]}, n_evals = 100,
					n_calls = 20, restart = 0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		"""
		Fine tuning for SVM
		Parameters:
			- optimizer : str ('DFO', 'CMA', 'PRS', 'BO')
			- x_initial : np array, for DFO and CMA
			- sigma0 : float, for CMA
			- n_evals : int, for PRS
			- n_calls : int, for BO
			- bounds_gamma : np array, bounds for gamma
			- bounds_C : np array, bounds for C
		"""
		save_path = ["./hp_tuning_data/" + optimizer + "_data/SVM.txt", "./hp_tuning_data/" + optimizer + "_data/SVM_value.txt"] 
		if optimizer == "BO":
			bb_fun = Objective_Function(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), save_to=save_path, isBO = True)
		else:
			bb_fun = Objective_Function(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), save_to=save_path, isBO = False)

		if evaluate_on == "test":
			eval_set = self.X_test
		elif evaluate_on == "val":
			eval_set = self.X_val
		else:
			raise ValueError("'evaluate_on' should be 'test' or 'val'.")

		sample_size = len(self.X_train) + len(eval_set)
		

		if not os.path.exists("./hp_tuning_data/" + optimizer + "_data/"):
			os.makedirs("./hp_tuning_data/" + optimizer + "_data/")
		with open("./hp_tuning_data/" + optimizer + "_data/config.txt", "wb") as file:
			message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
			if optimizer in ["DFO","CMA"]:
				# In thet case that a start point is needed by a solver
				gamma_init , C_init = self.scale2real(x_initial, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
				message = message + "Initial point of x: {}  (real values: (gamma = {}, C = {}))".format(x_initial, gamma_init, C_init)
			file.write(message)

		if optimizer == "DFO":
			# res = dfo_tr.dfo_tr(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), x_initial)
			res = dfo_tr.dfo_tr(bb_fun, x_initial)
			with open("./hp_tuning_data/" + optimizer + "_data/result.txt", "wb") as file:
				dill.dump(res, file)
			res = res.x
		elif optimizer == "CMA":
			res = cma.fmin(bb_fun, x_initial, sigma0, options=options_cma)[0]
			cma.plot()
			plt.savefig("./hp_tuning_data/CMA_data/CMA_plot.png")
		elif optimizer == "PRS":
			res = PRS(bb_fun, 2, n_evals, bounds = [-5,5])[0]
		elif optimizer == "BO":
			res = gp_minimize(bb_fun, [(-5,5)]*2, n_calls = n_calls)
			res = res.x
		else:
			print("Unknown optimizer : should be DFO, CMA, PRS or BO")
			return
		print("Best point found [gamma, C]: {}".format(self.scale2real(np.array(res), bounds_gamma=bounds_gamma, bounds_C=bounds_C)))
		with open("./hp_tuning_data/" + optimizer + "_data/final_result.txt", "wb") as file:
			message = "Best point found [gamma, C]: {}\n".format(self.scale2real(np.array(res), bounds_gamma=bounds_gamma, bounds_C=bounds_C))
			message = message + "Total function evaluations: {}".format(bb_fun.count)
			file.write(message)
		return
    

if __name__ == "__main__":
	print("Start...")
	filename_X = "preprocessed/input.npy"
	filename_Y = "preprocessed/output.npy"
	# model = NN_Model(filename_X, filename_Y)
	# print(model.X_train)
	# print("build model")
	# model.build_model()
	# print("train")
	# model.train()
	# print("test")
	# model.test()

	DEBUG = False
	from time import time
	BOUNDS_gamma = np.array([1e-2, 100.0])
	BOUNDS_C = np.array([0, 1000.0])

	obj = SVM_Model(filename_X, filename_Y)
	# obj.preprocessing(cut = 2000, split_rate = [0.7, 0.3]) # For fine-tuning purpose
	obj.preprocessing(cut = 2000, split_rate = [0.8, 0.2]) # For fine-tuning purpose
	# obj.preprocessing(cut = 1000, split_rate = [0.7, 0.3]) # For fine-tuning purpose
	# obj.preprocessing(cut = None, split_rate = [0.9, 0.1]) # Test on hole data set !
	# obj.preprocessing(cut = None, split_rate = [0.8, 0.2]) # Test on hole data set !
	
	# x_initial = None (default setting) gamma ~= 0.000372 (1/n_features), C = 1.0 
	# x_initial = obj.scale_x([22.6, 31.5], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # 80%, 79%  (dfo_data_lu)
	# x_initial = obj.scale_x([10.1637, 825.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # 77.45%  (cma_data0)
	# x_initial = obj.scale_x([7.77, 453.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # 76%  (cma_data1)
	# x_initial = obj.scale_x([40., 100.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # 

	#============================ After silence cut ============================
	# Start point 1
	x_initial = obj.scale_x([10., 300.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)

	# x_initial = obj.scale_x([6.45, 178.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # improved preprocessing (cut) 86% !!!
	# x_initial = obj.scale_x([15, 265.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # 84%
	# x_initial = obj.scale_x([4.7, 720.], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # 81.09% DFO_data0 
	# x_initial = obj.scale_x([7.0, 551.1], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # % DFO_data1 
	# x_initial = obj.scale_x([15.65, 74.7], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # % DFO_data2 
	# x_initial = obj.scale_x([3.879, 669.5837], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # % CMA_data0 
	# x_initial = obj.scale_x([15.01775, 359.156], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # % CMA_data1
	print("Initial point: {}".format(x_initial))
	if DEBUG:
		print(dfo_tr.constraint_shift(x_initial))
		print(obj.scale2real(dfo_tr.constraint_shift(x_initial), bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))
		print(obj.scale2real(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))

	st = time()
	# Old commands
	# obj.Fine_tune_SVM_with_PRS(bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	# obj.Fine_tune_SVM_with_BO(bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	# obj.Fine_tune_SVM_with_DFO(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	# obj.Fine_tune_SVM_with_CMA_ES(x_initial, 5.0, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)



	# print("Loss: {}".format(obj.EvaluateSVM(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)))
	# Default setting
	# print("Loss: {}".format(obj.EvaluateSVM(None, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))) 

	# obj.Fine_Tune_SVM(optimizer = "DFO", x_initial = x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	# obj.Fine_Tune_SVM(optimizer = "CMA", x_initial = x_initial, sigma0 = 5.0, options_cma = {'bounds': [[-5.000001,-5.000001], [5,5]], 'popsize': 15}, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	# obj.Fine_Tune_SVM(optimizer = "PRS", n_evals = 100, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	obj.Fine_Tune_SVM(optimizer = "BO", n_calls = 100, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	print("Total elapsed time {}".format(time()-st))

	
	#np.isfinite(obj.X_train).any()
	# First experiment:
	# gamma = 0.04, C = 10
	# Loss: -0.302272727273
	# Total elapsed time 88.071696043

	##### Transform a points to real searching space [gamma, C] ######
	# print(obj.scale2real(np.array([2.51762961118158301, 3.252057151340544294]), bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))
	# print(obj.scale2real(np.array([2.226888578906029981, -4.692889484117070964e-01]), bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))
	


	########## Read DFO fine-tuning result ############
	# Please install dill by (pip install dill)
	# with open("./dfo_data_lu/result.txt", "rb") as file:
	# 	res = dill.load(file)
	# print("Argmin of function:")
	# print(res.x)
	# print(obj.scale2real(res.x, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))
	# print("Best value of function (minimum):")
	# print(res.fun)
	# print("Iteration of algorithm:")
	# print(res.iteration)
	# print(res.iter_suc)
	# print("Number of function evaluation:")
	# print(res.func_eval)
	# print("Final delta:")
	# print(res.delta)