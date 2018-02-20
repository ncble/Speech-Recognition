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

