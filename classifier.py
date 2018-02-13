# _*_ coding: utf-8 _*_
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input

class NN_Model:
    def __init__(self, filename_X, filename_Y):
        '''
        Load data and split -> train, validate, test
        '''
        self.data_X = np.load(filename_X)
        self.data_Y = np.load(filename_Y)
        self.shape = self.data_X.shape
        self.data_X = self.data_X.reshape((self.shape[0], self.shape[1]*self.shape[2]))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_X, self.data_Y, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size = 0.2)
        
        # Default parameters
        self.activation = 'relu'  # activation function
        self.optimizer = 'adam'  # optimization
        self.loss = 'categorical_crossentropy' # Cost function
        self.batch_size = 10 # Batch size 
        self.epochs = 10 # Epochs
        self.layers = [100,100]

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

if __name__ == "__main__":
    print("Start reading")
    filename_X = "preprocessed/input.npy"
    filename_Y = "preprocessed/output.npy"
    model = NN_Model(filename_X, filename_Y)
    print(model.X_train)
    print("build model")
    model.build_model()
    print("train")
    model.train()
    print("test")
    model.test()


