# _*_ coding: utf-8 _*_
import os
import numpy as np
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
from DFO_src import dfo_tr

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

	def EvaluateSVM(self, x, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):

		###### Linear kernel  ######
		# clf = svm.LinearSVC(C=1.) # 0.9122 with whole dataset !!
		
		###### Gaussian kernel  ######
		### The first value is "accuracy on 7000 (10% of data): 6000 train 1000 test" 
		### The second value is "accuracy on whole dataset: 60000 train 10000 test" 

		# clf = svm.SVC(gamma=0.0413, C=78.22,max_iter=-1) # 0.973, 0.9851 with whole dataset !! 
		# clf = svm.SVC(gamma=0.024154, C=632.649,max_iter=-1) # 0.971, 0.9848 with whole dataset !! 
		# clf = svm.SVC(gamma=0.02845, C=521.59,max_iter=-1) # 0.967, 0.985 0. with whole dataset !!  
		# clf = svm.SVC(gamma=0.028, C=100,max_iter=-1) # 0.976, 0.9884 with whole dataset !!  
		# clf = svm.SVC(gamma=0.024, C=100,max_iter=-1) # 0.963, 0.9864 with whole dataset !!  
		# clf = svm.SVC(gamma=0.04, C=70,max_iter=-1) # 0.966, 0.987 with whole dataset !! 
		
		"""
		Loss function to be minimize usgin DFO-TR algorithm (hyperparameters' tuning). 
		"""
		gamma, C = self.scale2real(x, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
		

		clf = svm.SVC(gamma=gamma, C=C, max_iter=-1) 
		clf.fit(self.X_train, self.y_train)
		Loss_val = -clf.score(self.X_val, self.y_val)
		print("Loss on the validation set: {} with (gamma: {}, C: {}).".format(Loss_val, gamma, C))
		return Loss_val

	def linear_transform(self, x, bounds_old, log_scale = False, to_real = False):
		# x is a scalar

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
		x = x.ravel()
		# gamma = pow(10, (x[0]-5)/2.)  # x[0] \in [-5,5]  # gamma = 1e-5 ~ 0.
		# C = (x[1]+5.0001)*100 # x[1] \in [-5,5] # C = 0. ~ 1000.

		# a, b = np.log10(bounds_gamma)
		# m, k = 10./(b-a), -5*(b+a)/(b-a)
		# gamma = (x[0]-k)/m
		# C = (x[1]-k)/m +0.00001

		gamma = self.linear_transform(x[0], bounds_gamma, log_scale = True, to_real = True)
		C = self.linear_transform(x[1], bounds_C, log_scale = False, to_real = True)+0.000001
		return gamma, C

	def scale_x(self, x, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		# y = np.zeros(2)
		# y[0] = np.log10(x[0])*2+5 # gamma
		# y[1] = x[1]/100.-5.0

		# a, b = np.log10(bounds_gamma)
		# m, k = 10./(b-a), -5*(b+a)/(b-a)

		y = np.zeros(2)
		# y[0] = x[0]*m + k 
		# y[1] = x[1]*m + k
		y[0] = self.linear_transform(x[0], bounds_gamma, log_scale = True, to_real = False)
		y[1] = self.linear_transform(x[1], bounds_C, log_scale = False, to_real = False)
		return y

	def Fine_tune_SVM_with_DFO(self, x_initial, restart=0, bounds_gamma=np.array([1e-5, 1.0]), bounds_C=np.array([0, 500.0])):
		# [-5,5] * [-5,5] 
 
		sample_size = len(self.X_train) + len(self.X_val)

		if not os.path.exists("./dfo_data_lu/"):
			os.mkdir("./dfo_data_lu/")
		with open("./dfo_data_lu/config.txt", "wb") as file:
			message = "Sample size = {} (train: {}, test: {}).\n".format(sample_size, len(self.X_train), len(self.X_val))
			gamma_init , C_init = self.scale2real(x_initial, bounds_gamma=bounds_gamma, bounds_C=bounds_C)
			message = message+"Initial point of x: {}  (real values: (gamma = {}, C = {}))".format(x_initial, gamma_init, C_init)
			file.write(message)
		
		dfo_tr.dfo_tr(lambda x: self.EvaluateSVM(x, bounds_gamma=bounds_gamma, bounds_C=bounds_C), x_initial)




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
	obj.preprocessing(cut = 2000, split_rate = [0.8,0.1,0.1])
	BOUNDS_gamma = np.array([1e-2, 100.0])
	BOUNDS_C = np.array([0, 1000.0])

	x_initial = obj.scale_x([11., 82], bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C) # [0.04, 10]
	print(x_initial)
	if DEBUG:
		print(dfo_tr.constraint_shift(x_initial))
		print(obj.scale2real(dfo_tr.constraint_shift(x_initial), bounds_gamma=np.array([1e-5, 10.0]), bounds_C=np.array([0, 500.0])))
		print(obj.scale2real(x_initial, bounds_gamma=np.array([1e-5, 10.0]), bounds_C=np.array([0, 500.0])))

	st = time()
	# obj.Fine_tune_SVM_with_DFO(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)
	print("Loss: {}".format(obj.EvaluateSVM(x_initial, bounds_gamma=BOUNDS_gamma, bounds_C=BOUNDS_C)))
	print("Total elapsed time {}".format(time()-st))


	#np.isfinite(obj.X_train).any()


	# First experiment:
	# gamma = 0.04, C = 10
	# Loss: -0.302272727273
	# Total elapsed time 88.071696043

