'''Trains a LeNet-5 convolutional neural network on the MNIST dataset.

It follows Yann LeCun, et al.
"Gradient-based learning applied to document recognition" (1998) 
http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

but with more current activations, initialization, optimizer,
no trainable parameters in maxpooling layers, and no sparse connectivity in C3.

Gets to 99.1% test accuracy after 12 epochs
2 seconds per epoch on Titan X Maxwell GPU
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), padding='same',
                 activation='relu',
                 name='C1',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', name='C3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='S4'))
model.add(Flatten())
model.add(Dense(120, activation='relu', name='C5'))
model.add(Dense(84, name='F6'))
model.add(Dense(num_classes, activation='softmax', name='OUTPUT'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
