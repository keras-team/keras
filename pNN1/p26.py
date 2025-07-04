import numpy as np
import tensorflow
from tensorflow import keras

from keras.layers import Dense
from keras.models import Sequential

# Assume you have your data loaded into x (input features) and y_train (target labels)
# X should have a shape like (number_of_samples, 44)
# y should have a shape like (number_of_samples,) with values 0 or 1 for binary classification

# Example of dummy data generation (replace with your actual data)
num_serialized_unts = 800  # serialized units through manufacturing process

"""make sure to come back to the dataset and weed-out serialized units that
have made multiple runs through process if applicable"""

"""
Make sure to normalize inputs based on process variable specs. / mean of training data
Not all n of the process variables are gonna be on the same scale.
Maybe it's redundant, the 'input layer' might do it for me who knows?
"""

pv = 44  # process variables per manufacturing run

oc = 2  # of outcome characteristics (in this case is present or is not)

"""
again maybe will have to split this into another tensor to include multiple
runs with same serial number (unit is reworked and put back through process)
"""


np.random.seed(0)  # Random seed for consistent results?

"""
No idea how random stuff is determined (it can't be that random)
"""

X = np.random.random((num_serialized_unts, pv))
# Random data created for n serialized units, with n process variables per unit

y = np.random.randint(oc, size=num_serialized_unts)
# Random outcome characteristics with size n serialized units

"""
Probably need to remember that sigmoid function is only good for binary outcomes
"""

print(X, y)


# Building the Sequential model
model = Sequential()

# Input layer: The first layer needs to know the input shape
model.add(keras.Input(shape=(pv,)))
# Input layer that matches shape of input data

model.add(Dense(num_serialized_unts, activation="relu"))
# Hidden layer with n units, uses rectified linear activation

"""
Don't ask me why the unit number on the next layer is the number of serialized
units. It just feels right...
"""

# You can add more hidden layers as needed
# model.add(Dense(32, activation='relu'))

# Output layer: For binary classification, use a single neuron with sigmoid activation
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(
    X,
    y,
    epochs=100,
)  # batch_size=32 # Adjust epochs and batch_size as needed

# You can evaluate the model on test data if available
# loss, accuracy = model.evaluate(X, y)
# print('Test loss:', loss)
# print('Test accuracy:', accuracy)
