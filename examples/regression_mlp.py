"""Trains and evaluates a simple MLP to calculate the product of N numbers"""
from __future__ import print_function
from __future__ import unicode_literals
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

np.random.seed(1)
# Number of inputs to calculate the product of
N_inputs = 2
# Number of training data examples
N_examples = 10000
# Number of hidden neurons to use
N_hidden = 64

# Generate uniformly distributed random training data
min_X, max_X = -5.0, 5.0
X_train = np.random.uniform(min_X, max_X, size=(N_examples, N_inputs))
Y_train = np.prod(X_train, axis=1)
min_Y, max_Y = min(Y_train), max(Y_train)

# Normalize training data between -0.5 and +0.5
# Normalization improves prediction accuracy and reduces required training time
X_train = (X_train-min_X)/(max_X-min_X)-0.5
Y_train = (Y_train-min_Y)/(max_Y-min_Y)-0.5

model = Sequential()
model.add(Dense(N_hidden, input_dim=N_inputs, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adamax')
model.fit(X_train, Y_train, nb_epoch=50, validation_split=0.2)
          
def predict(X):
    """Returns predicted product of the elements in each row of X"""
    # Normalize input data between -0.5 and +0.5
    X = np.atleast_2d(X)
    X = (X-min_X)/(max_X-min_X)-0.5
    # Predict the sum of the input data
    Y = model.predict(X).flatten()
    # Reverse the normalization of the output data
    Y = (Y+0.5)*(max_Y-min_Y) + min_Y
    return Y

# Display the result of some test cases
N_test = 10
X_test = np.random.uniform(min_X, max_X, size=(N_test, N_inputs))
Y_test = predict(X_test)
for i in range(N_test):
    print('product({:}) = {:.4f} \u2248 {:.4f}'.format(X_test[i,:],
                                                       np.prod(X_test[i,:]),
                                                       Y_test[i]))
