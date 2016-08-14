"""
This is an example of using Hierarchical RNN (HRNN) to predict MNIST digits.

HRNNs can learn across multiple levels of boundaries from a complex sequence.
In text processing, a normal RNN can only learn the relationship of words in
one sentence. HRNNs can also learn the relationships between sentences.
Usually, the first recurrent layer of an HRNN encodes a sentence(word vectors)
into a  sentence vector. The second recurrent layer then encodes a sequence of
sentence vectors(encoded by the first layer) into a document vector. This
document vector is considered to preserve both the word-level and
sentence-level structure of the context.

Paper references:
"A Hierarchical Neural Autoencoder for Paragraphs and Documents"
(https://web.stanford.edu/~jurafsky/pubs/P15-1107.pdf) encodes paragraphs
and documents with HRNN. Results have shown that HRNN outperforms standard
RNNs and may play some role in more sophisticated generation tasks like
summarization or question answering.

"Hierarchical recurrent neural network for skeleton based action recognition"
(http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714) achieved
state-of-the-art results on skeleton based action recogntion with 3 levels
of bidirectional HRNN combind with fully connected layers.

In the below MNIST example the first LSTM layer first encodes every
column of pixels(28,1) to a column vector(128,). The second LSTM
layer encodes then these 28 column vectors(28,128) to a image vector
representing the whole image. A final Dense layer is added for prediction.

After 5 epochs: train acc: 0.9858, val acc: 0.9864

PS: Modified from mnist_irnn.py
"""
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epochs = 5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to 4D for Hierarchical RNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Define model
pixel_hidden = 128
col_hidden = 128

row, col, pixel = X_train.shape[1:]

# 4D input
x = Input(shape=(row, col, pixel))

# Encodes a col of pixels, uses TimeDistributed Wrapper
encoded_pixels = TimeDistributed(LSTM(output_dim=pixel_hidden))(x)

# Encodes columns of encoded pixels
encoded_columns = LSTM(col_hidden)(encoded_pixels)

prediction = Dense(nb_classes, activation='softmax')(encoded_columns)

model = Model(input=[x], output=[prediction])

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)

print('Hierarchical RNN test loss:', scores[0])
print('Hierarchical RNN test accuracy:', scores[1])
