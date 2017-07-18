"""
An implementation of sequence to sequence learning for string reversing

Input: keras
Output: sarek

The string contains lower-case letters. The length of the string is set before learning.
The code is maintainable by changing the hyperparameters, such as the number of training items, letters and maximum length
"""

import numpy as np
from random import choice
import operator

from keras.models import Sequential
from keras import layers
from string import ascii_lowercase


class GenerateData(object):
    """
    Data generation class. It takes the characters which it will be used to generate the strings.
    The characters are considered to be the vocabulary of the training and testing data

    For each character, we assign an index which will be used to set a 1-hot encoding vector for each character
    """

    def __init__(self, chars):
        self.__chars = chars
        self.__char2idx = {}
        self.__idx2char = {}

        self.__encode()

    def __encode(self):
        """
        Set a unique integer identifier to each character
        """

        for i, char in enumerate(self.__chars):
            self.__char2idx[char] = i
            self.__idx2char[i] = char

    def encode(self, string):
        """
        Encode a given string to 1-hot encoder matrix. Each row represents a 1-hot encoding character.

        For example: abc

        [
            [1, 0, 0]
            [0, 1, 0]
            [0, 0, 1]
        ]

        Assuming that a, b and c are the whole vocab
        """

        encoded = np.zeros((len(string), len(self.__chars)))

        for i, c in enumerate(string):
            encoded[i, self.__char2idx[c]] = 1

        return encoded

    def decode(self, indices):
        """
        Revert the indices of 1-hot to the original characters.

        For example: [0 1 2]

        The function shall return abc
        """

        return ''.join(list(map(lambda v: self.__idx2char[v], indices)))

    def generate_data(self, batch_size, max_length):
        """
        Generates the data for training. The data is structured into a 3D array.
        Rows indicate the number of data that will be generated
        and columns are the 1-hot encoding matrices that represent the string
        """

        x = np.zeros((batch_size, max_length, len(self.__chars)), dtype=np.int)
        y = np.zeros((batch_size, max_length, len(self.__chars)), dtype=np.int)

        for i in range(0, batch_size):
            original_string = ''.join(choice(self.__chars) for j in range(max_length))
            reverse_string = original_string[::-1]
            x[i] = self.encode(original_string)
            y[i] = self.encode(reverse_string)

        return x, y


chars = ascii_lowercase  # 26 characters

DATA_SIZE = 4000
VOCAB_SIZE = len(chars)
HIDDEN_LAYER_SIZE = 100
BATCH_SIZE = 128
MAX_LENGTH = 5
LAYERS = 1

generator = GenerateData(chars)
x, y = generator.generate_data(DATA_SIZE, MAX_LENGTH)

model = Sequential()

# Feeding the hidden layer with input. The input shape is MAX_LENGTH * VOCAB_SIZE which is the same dim of each string
# The output of the hidden layer is a vector representation (100) of the string
model.add(layers.SimpleRNN(HIDDEN_LAYER_SIZE, input_shape=(MAX_LENGTH, VOCAB_SIZE)))

# Here, we tell the output layer to expect a MAX_LENGTH timesteps of vectors. Note that, it could be different from
# the input dim.
model.add(layers.RepeatVector(MAX_LENGTH))

# Adding multiple hidden layers according to the number of hidden layers specified by us
for l in range(0, LAYERS):
    model.add(layers.LSTM(HIDDEN_LAYER_SIZE, return_sequences=True))

# For each timesteps, we shall have a dense layer so that we could be able to use a Softmax activation for classification
model.add(layers.TimeDistributed(layers.Dense(VOCAB_SIZE)))

# Apply Softmax on the vectors
model.add(layers.Activation('softmax'))

# Compile the model with specifying the the loss function and the optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary to know the Neural Network structure and its trainable and non-trainable parameters
model.summary()

# Begin training the data
model.fit(x, y, batch_size=BATCH_SIZE, epochs=50)


n_test = BATCH_SIZE
test_x, test_y = generator.generate_data(n_test, MAX_LENGTH)

predictions = model.predict_classes(test_x)

count_correct = 0

for batch in range(0, len(predictions)):
    predicted_string = generator.decode(predictions[batch])
    true_string_mat = test_y[batch]
    true_indices = []

    for vec in true_string_mat:
        index, value = max(enumerate(vec), key=operator.itemgetter(1))
        true_indices.append(index)

    true_string = generator.decode(true_indices)

    if true_string == predicted_string:
        count_correct += 1
    else:
        print(true_string)
        print(predicted_string)

print("\n\nTest Accuracy: " + str((float(count_correct) / float(n_test)) * 100) + "%")
