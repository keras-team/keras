import numpy as np
import random
import theano

from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from matplotlib import pyplot as plt
from matplotlib import animation

nb_classes = 10
batch_size = 128
nb_epoch = 1

max_train_samples = 5000
max_test_samples = 1

np.random.seed(1337)

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,1,28,28)[:max_train_samples]
X_train = X_train.astype("float32")
X_train /= 255

X_test = X_test.reshape(-1,1,28,28)[:max_test_samples]
X_test = X_test.astype("float32")
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]

class DrawWeight(Callback):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def on_train_begin(self):
        self.imgs = []
        # self.test = theano.function([self.model.get_input()], self.model.layers[1].get_output(train=False))
        self.test = theano.function([self.model.get_input()], self.model.layers[10].get_output(train=False))

    def on_batch_end(self, batch, indices, loss, accuracy, val_loss, val_acc):
        if batch % 1 == 0:
            # img = self.ax.imshow(self.test(X_test)[0,0,:,:], interpolation='nearest')
            img = self.ax.imshow(self.test(X_test)[0,:].reshape(16,16), interpolation='nearest')
            self.imgs.append([img])

    def on_train_end(self):
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=10, blit=False, repeat_delay=1000)
        plt.show()

# model = Sequential()
# model.add(Dense(784, 50))
# model.add(Activation('relu'))
# model.add(Dense(50, 10))
# model.add(Activation('softmax'))

model = Sequential()
model.add(Convolution2D(32, 1, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64*8*8, 256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Fit the model
draw_weights = DrawWeight()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[draw_weights])