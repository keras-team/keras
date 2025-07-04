import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Assume you have your data loaded into x_train (input features) and y_train (target labels)
# x_train should have a shape like (number_of_samples, 44)
# y_train should have a shape like (number_of_samples,) with values 0 or 1 for binary classification

# Example of dummy data generation (replace with your actual data)
num_samples = 1000
num_features = 44
X = np.random.random((num_samples, num_features))
y = np.random.randint(2, size=num_samples)

# Building the Sequential model
model = Sequential()

# Input layer: The first layer needs to know the input shape
model.add(Dense(64, activation='relu', input_shape=(num_features,))) # Example hidden layer with 64 neurons

# You can add more hidden layers as needed
# model.add(Dense(32, activation='relu'))

# Output layer: For binary classification, use a single neuron with sigmoid activation
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32) # Adjust epochs and batch_size as needed

# You can evaluate the model on test data if available
# loss, accuracy = model.evaluate(x_test, y_test)
# print('Test loss:', loss)
# print('Test accuracy:', accuracy)
