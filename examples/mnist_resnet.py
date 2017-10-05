''' Trains a 6-layer ResNet on the MNIST dataset.
    Gets to 99.4% test accuracy in 50 epochs
    4sec per epoch on GTX 1080
'''

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.optimizers import SGD
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
import numpy as np

batch_size = 128
num_classes = 10
epochs = 50
dropout = 0.4

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

xin = keras.layers.Input(shape=input_shape)
x = xin

for i in range(3):
    y = Conv2D(filters=16, kernel_size=3, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=16, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)
    x = keras.layers.pooling.MaxPooling2D(2)(x)

y = keras.layers.Flatten()(x)
y = Dropout(0.4)(y)
yout = Dense(num_classes, activation='softmax')(y)
        
model = Model(inputs=[xin], outputs=[yout])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-1),\
        metrics=['accuracy']) 
model.summary()

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, \
        shuffle=True, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
