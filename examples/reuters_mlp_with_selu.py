'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.noise import AlphaDropout
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
epochs = 25
plot = True

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building relu model...')
model_relu = Sequential()
model_relu.add(Dense(512, input_shape=(max_words,)))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(num_classes))
model_relu.add(Activation('softmax'))

model_relu.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

history_relu = model_relu.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_split=0.1)


score = model_relu.evaluate(x_test, y_test,
                            batch_size=batch_size, verbose=1)

print('RELU model results')
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Building selu model...')
model_selu = Sequential()
model_selu.add(Dense(512, input_shape=(max_words,)))
model_selu.add(Activation('selu'))
model_selu.add(AlphaDropout(0.5))
model_selu.add(Dense(num_classes))
model_selu.add(Activation('softmax'))

model_selu.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

history_selu = model_selu.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_split=0.1)
score = model_selu.evaluate(x_test, y_test,
                            batch_size=batch_size, verbose=1)

print('SELU model results')
print('Test score:', score[0])
print('Test accuracy:', score[1])

if plot:
    import matplotlib.pyplot as plt
    plt.plot(range(epochs), history_relu.history['val_loss'], 'g-', label='RELU Val Loss')
    plt.plot(range(epochs), history_selu.history['val_loss'], 'r-', label='SELU Val Loss')
    plt.plot(range(epochs), history_relu.history['loss'], 'g--', label='RELU Loss')
    plt.plot(range(epochs), history_selu.history['loss'], 'r--', label='SELU Loss')
    plt.legend()
    plt.show()

