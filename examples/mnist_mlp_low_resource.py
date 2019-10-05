'''Trains a simple deep NN on the MNIST dataset.
Gets to 92% test accuracy after 2 epochs on low resources
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint,Callback
from keras.utils import np_utils, print_summary

def preprocess_labels(y):
    labels = np_utils.to_categorical(y)
    return labels

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_norm = x_train / 127.5 - 1.
x_test_norm = x_test / 127.5 - 1.

y_train = preprocess_labels(y_train)
y_test = preprocess_labels(y_test)

def keras_model(image_x, image_y):
    num_of_classes = 10
    model = Sequential()
    model.add(Flatten(input_shape=(image_x, image_y, 1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "mnist_low_resources.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list
    
model, callbacks_list = keras_model(28, 28)
print_summary(model)
model.fit(x_train_norm, y_train, validation_data=(x_test_norm, y_test), epochs=2, batch_size=64,
              callbacks=callbacks_list)
              
score = model.evaluate(x_test_norm, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
    
