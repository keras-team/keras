# p8.py Hidden Layers


import math
from datetime import datetime

import cd
import numpy as np

start_time = datetime.now()
np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = (
            0.1 * np.random.normal(loc=0, scale=1.0, size=(n_inputs, n_neurons))
        )  # creates a random set of weights that is size n_inputs by n_neurons (float values that are normalized around zero, and have standard deviation of 'scale')

        self.biases = np.zeros(
            (1, n_neurons)
        )  # 1, n_neurons ensures the shape of zeros matrix created can multiply with n_neurons matrix (this step initializes the bias matrix to zero)

    def forward(self, inputs):
        self.output = (
            np.dot(inputs, self.weights) + self.biases
        )  # Takes dot product of input matrix with weights matrix and adds bias

    # def backward(self, adj_type):

    # self.weights = self.weights +


# simple but effective for non-output layers
class Activation_ReLU:  # Linear rectified activation function. This takes everything from outputted dot products and if it is 0 or above makes it the output. Otherwise it is 0
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# this is a useful but more expensive activation function. it creates a probability distribution outa stuff
class Activation_Softmax:  # Softmax activation function. This tunrs them all into e^inputs, then normalizes. Better activation function when you don't want to lose negative values to rectified linear
    def forward(self, inputs):
        exp_inputs = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # raises e^ of all inputs in matrix after subtracting largest value in each row to make sure they don't get to big

        # probabilites of each row?
        self.output = (
            exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        )  # normalizes e^inputs over the sum of all inputs in the same row. The other auguments make sure of that


class Loss:
    def calculate(self, output, y):
        samlpe_losses = self.forward(output, y)
        data_loss = np.mean(samlpe_losses)
        return data_loss


class Loss_CategoricalCrossentropy(
    Loss
):  # catigorical cross entry with '1 hot vector'
    def forward(
        self, y_pred, y_true
    ):  # takes the predicted values and the correct values for 'class perdiction'
        samples = len(y_pred)  # length of predictions, so how many rows it has
        print(samples)

        y_pred_clipped = np.clip(
            y_pred, 1e-7, 1 - 1e-7
        )  # clips out any possible 0 in probability scale

        # one hot vector fancy dot product thing [0,1,0],[1,0,0]. Fancy way of choosing the class in the prediction data you care about. If the y_true says a given class is the true one, the probabilites of generated for the other classes are multiplied by 0 and disregarded
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(
            correct_confidences
        )  # now that all the useless predictions are gone, you get a nice and simple summation

        return negative_log_likelihoods


X, y = cd.create_data(
    100, 3
)  # creates a set of random spiral data in a 2d array. Example is 100 readings from sensors. 3 classes (call em sensor types in example, 3 types of sensors). 2 sensors of each type, 100 readings from each.

# initialize 1 layer of size 2,5
dense1 = Layer_Dense(
    2, 10
)  # must be size 2 for the inuts because our sin() and cos() array has 2 data points per class


activation1 = Activation_ReLU()


dense2 = Layer_Dense(
    10, 3
)  # 3 output layers because 3 classes. This is important


activation2 = Activation_Softmax()


dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print(loss)


print(np.mean(np.mean(activation2.output, axis=0)))

predictions = np.argmax(activation2.output, axis=1)
accuracy = np.mean(predictions == y)

print("acc:", accuracy)


time_elapsed = datetime.now() - start_time
print(f"Elapsed time: {time_elapsed} seconds")
