# _*_ coding: utf-8 _*_
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Input


## SVM model
from sklearn.model_selection import train_test_split
# from random import shuffle  
# np.random.shuffle
from sklearn.utils import shuffle  # create and return a copy
from sklearn import svm
from sklearn.preprocessing import normalize

##### Hyperparameters tuing solvers ####

try:
	from DFO_src import dfo_tr
	import dill # To read DFO's result # pip install dill
except:
	print("Please install dill (pip install dill)")
try:
	import cma
except:
	print("Please install CMA (pip install CMA)")



class NN_Model:
	def __init__(self, filename_X, filename_Y):
		'''
		Load data and split -> train, validate, test
		'''
		self.data_X = np.load(filename_X)
		self.data_Y = np.load(filename_Y)
		self.shape = self.data_X.shape

		self.X_train = None
		self.X_val = None
		self.X_test = None

		self.y_train = None
		self.y_val = None
		self.y_test = None
		
		# Default parameters
		self.activation = 'relu'  # activation function
		self.optimizer = 'adam'  # optimization
		self.loss = 'categorical_crossentropy' # Cost function
		self.batch_size = 10 # Batch size 
		self.epochs = 10 # Epochs
		self.layers = [100,100]

	def preprocessing(self, split_rate = [0.8, 0.1, 0.1], cut = None):
		print("Start preprocessing...")
		self.data_X = self.data_X.reshape((self.shape[0], self.shape[1]*self.shape[2]))
		self.data_X, self.data_Y = shuffle(self.data_X, self.data_Y)
		if cut is not None:
			self.data_X = self.data_X[:cut]
			self.data_Y = self.data_Y[:cut]
		
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


	def set_hyper_params(self, activation = 'relu', optimizer = 'adam', loss = 'categorical_crossentropy',
						 batch_size = 30, epochs = 5, layers = [100]):
		self.activation = activation  # activation function
		self.optimizer = optimizer  # optimization
		self.loss = loss # Cost function
		self.batch_size = int(batch_size) # Batch size 
		self.epochs = int(epochs) # Epochs
		self.layers = [int(x) for x in layers]

	def build_model(self):
		# Initialize model
		self.model = Sequential()
		# Define NN structure
		inputs = Input(shape=(self.shape[1]*self.shape[2],))
		x = Dense(self.layers[0], activation=self.activation)(inputs)    # First hidden layer
		for k in range(1, len(self.layers)):
			x = Dense(self.layers[k], activation = self.activation)(x)      # Other hidden layers
		predictions = Dense(self.data_Y.shape[1], activation='softmax')(x)
		self.model = Model(inputs=inputs, outputs=predictions)
		self.model.compile(optimizer=self.optimizer, loss=self.loss,metrics=['accuracy']) 
	
	def train(self, mode = 'validation'):
		if mode == 'validation':
			self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)
		elif mode == 'test':
			self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)
		else:
			self.model.fit(self.data_X, self.data_Y, epochs=self.epochs, batch_size=self.batch_size)
	
	def test(self, mode = 'validation'):
		if mode == 'validation':
			score = self.model.evaluate(self.X_val, self.y_val, batch_size=self.batch_size)
			print("Score : " + str(score[1]))
		else:
			return 0


class Objective_Function():
	def __init__(self, fun, isBO = False, save_to=None):
		self.fun = fun
		self.history_f = []
		self.fbest = np.inf
		self.history_fbest = []
		self.isBO = isBO
		
	def __call__(self, x):
		if self.isBO:
			x = np.array(x)
		value = self.fun(x)
		if save_to is not None:
			assert len(save_to) == 2
			np.savetxt(open(save_to[0], "ab"), x.reshape(1,-1))
			np.savetxt(open(save_to[1], "ab"), np.array(value).reshape(1,1))

		# self.history_f.append(value)
		# self.fbest = min(self.fbest, value)
		# self.history_fbest.append(self.fbest)
		return value


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

	def Fine_tune_SVM_with_DFO(self, x_initial, evaluate_on = "test", restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		"""
		Searching space: [-5,5] * [-5,5] 


		"""
		
		if evaluate_on == "test":
			eval_set = self.X_test
		elif evaluate_on == "val":
			eval_set = self.X_val
		else:
			raise ValueError("'evaluate_on' should be 'test' or 'val'.")

		sample_size = len(self.X_train) + len(eval_set)

		if not os.path.exists("./hp_tuning_data/dfo_data_lu/"):
			os.makedirs("./hp_tuning_data/dfo_data_lu/")
		with open("./hp_tuning_data/dfo_data_lu/config.txt", "wb") as file:
			message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
			gamma_init , C_init = self.scale2real(x_initial, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
			message = message+"Initial point of x: {}  (real values: (gamma = {}, C = {}))".format(x_initial, gamma_init, C_init)
			file.write(message)
		
		dfo_tr.dfo_tr(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), x_initial)


	def Fine_tune_SVM_with_CMA_ES(self, x_initial, sigma0, evaluate_on = "test", restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		"""
		Searching space: ???
		
		"""
		save_path = ["./hp_tuning_data/cma_data/SVM.txt", "./hp_tuning_data/cma_data/SVM_value.txt"] 
		bb_fun = Objective_Function(lambda x: self.EvaluateSVM(x, evaluate_on = evaluate_on, bounds_gamma=bounds_gamma, bounds_C=bounds_C), save_to=save_path)

		if evaluate_on == "test":
			eval_set = self.X_test
		elif evaluate_on == "val":
			eval_set = self.X_val
		else:
			raise ValueError("'evaluate_on' should be 'test' or 'val'.")

		sample_size = len(self.X_train) + len(eval_set)
		

		if not os.path.exists("./hp_tuning_data/cma_data/"):
			os.makedirs("./hp_tuning_data/cma_data/")
		with open("./hp_tuning_data/cma_data/config.txt", "wb") as file:
			message = "Sample size = {} (train: {}, {}: {}).\n".format(sample_size, len(self.X_train), evaluate_on, len(eval_set))
			gamma_init , C_init = self.scale2real(x_initial, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
			message = message+"Initial point of x: {}  (real values: (gamma = {}, C = {}))".format(x_initial, gamma_init, C_init)
			file.write(message)
		
		cma.fmin(bb_fun, x_initial, sigma0)
		cma.plot()
		plt.savefig("./hp_tuning_data/cma_data/cma_plot.png")


if __name__ == "__main__":
	print("Start reading")
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

	obj = SVM_Model(filename_X, filename_Y)
	# obj.preprocessing(cut = 2000, split_rate = [0.8, 0.1, 0.1]) # For fine-tuning purpose
	obj.preprocessing(cut = 500, split_rate = [0.7, 0.3]) # For fine-tuning purpose
	# obj.preprocessing(cut = None, split_rate = [0.9, 0.1]) # Test on hole data set !
	BOUNDS_gamma = np.array([1e-2, 100.0])
	BOUNDS_C = np.array([0, 1000.0])

	x_initial = obj.scale_x([22.6, 31.5], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # [0.04, 10]
	print(x_initial)
	if DEBUG:
		print(dfo_tr.constraint_shift(x_initial))
		print(obj.scale2real(dfo_tr.constraint_shift(x_initial), bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))
		print(obj.scale2real(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C))

	st = time()
	# obj.Fine_tune_SVM_with_DFO(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	# print("Loss: {}".format(obj.EvaluateSVM(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)))
	# obj.Fine_tune_SVM_with_CMA_ES(x_initial, 5.0, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	print("Total elapsed time {}".format(time()-st))


	#np.isfinite(obj.X_train).any()
	# First experiment:
	# gamma = 0.04, C = 10
	# Loss: -0.302272727273
	# Total elapsed time 88.071696043

	def draw_surface_level(fun, centre = np.zeros(2), taille = 1.0, message = None):

		plt.figure(figsize=(5, 3))
		axes = plt.gca()
		if message:
			plt.title(message)
		x_min, y_min = (centre - taille)#[0,0], (centre - taille)[0,1]  #X[:, 0].min()
		x_max, y_max = (centre + taille)#[0,0], (centre + taille)[0,1]     #X[:, 0].max()
		
		axes.set_xlim([x_min,x_max])
		axes.set_ylim([y_min,y_max])
		XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

		Z = fun(np.c_[XX.ravel(), YY.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(XX.shape)
		plt.pcolormesh(XX, YY, Z, cmap = plt.cm.jet)
		# CS = plt.contour(XX, YY, Z, cmap = plt.cm.jet)
		# plt.clabel(CS, fmt='%2.1f', colors='b', fontsize=14)
		plt.show()

	def slope(x):

		tengent = -5
		# return np.sum(tengent*x, axis = 1)
		return np.sum(tengent*x)

	# cma.CMAOptions()
	# draw_surface_level(slope)

	CMA_cls = cma.constraints_handler.BoundaryHandlerBase([np.array([-5,-5]), np.array([5,6])])
	print(CMA_cls)
	cma.fmin(cma.ff.sphere, np.ones(2), 0.1, options = {"BoundaryHandler":None})
	
	# print CMA_cls.has_bounds()
	print CMA_cls.get_bounds('lower', 2)
	print CMA_cls.get_bounds('upper', 2)
	# import ipdb; ipdb.set_trace()
	# cma.plot()
	plt.savefig("./test_cma_plot.png")

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

