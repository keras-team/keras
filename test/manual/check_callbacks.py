import numpy as np
import random
import theano

from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import keras.callbacks as cbks

from matplotlib import pyplot as plt
from matplotlib import animation

##############################
# model DrawActivations test #
##############################

print('Running DrawActivations test')

nb_classes = 10
batch_size = 128
nb_epoch = 10

max_train_samples = 512
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

class Frames(object):
    def __init__(self, n_plots=16):
        self._n_frames = 0
        self._framedata = []
        self._titles = []
        for i in range(n_plots):
            self._framedata.append([])

    def add_frame(self, i, frame):
        self._framedata[i].append(frame)

    def set_title(self, title):
        self._titles.append(title)

class SubplotTimedAnimation(animation.TimedAnimation):

    def __init__(self, fig, frames, grid=(4, 4), interval=10, blit=False, **kwargs):
        self.n_plots = grid[0] * grid[1]
        self.axes = [fig.add_subplot(grid[0], grid[1], i + 1) for i in range(self.n_plots)]
        for axis in self.axes:
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
        self.frames = frames
        self.imgs = [self.axes[i].imshow(frames._framedata[i][0], interpolation='nearest', cmap='bone') for i in range(self.n_plots)]
        self.title = fig.suptitle('')
        super(SubplotTimedAnimation, self).__init__(fig, interval=interval, blit=blit, **kwargs)

    def _draw_frame(self, j):
        for i in range(self.n_plots):
            self.imgs[i].set_data(self.frames._framedata[i][j])
        if len(self.frames._titles) > j:
            self.title.set_text(self.frames._titles[j])
        self._drawn_artists = self.imgs

    def new_frame_seq(self):
        return iter(range(len(self.frames._framedata[0])))

    def _init_draw(self):
        for img in self.imgs:
            img.set_data([[]])

def combine_imgs(imgs, grid=(1,1)):
    n_imgs, img_h, img_w = imgs.shape
    if n_imgs != grid[0] * grid[1]:
        raise ValueError()
    combined = np.zeros((grid[0] * img_h, grid[1] * img_w))
    for i in range(grid[0]):
        for j in range(grid[1]):
            combined[img_h*i:img_h*(i+1),img_w*j:img_w*(j+1)] = imgs[grid[0] * i + j]
    return combined

class DrawActivations(Callback):
    def __init__(self, figsize):
        self.fig = plt.figure(figsize=figsize)

    def on_train_begin(self, logs={}):
        self.imgs = Frames(n_plots=5)

        layers_0_ids = np.random.choice(32, 16, replace=False)
        self.test_layer0 = theano.function([self.model.get_input()], self.model.layers[1].get_output(train=False)[0, layers_0_ids])

        layers_1_ids = np.random.choice(64, 36, replace=False)
        self.test_layer1 = theano.function([self.model.get_input()], self.model.layers[5].get_output(train=False)[0, layers_1_ids])

        self.test_layer2 = theano.function([self.model.get_input()], self.model.layers[10].get_output(train=False)[0])

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        if batch % 5 == 0:
            self.imgs.add_frame(0, X_test[0,0])
            self.imgs.add_frame(1, combine_imgs(self.test_layer0(X_test), grid=(4, 4)))
            self.imgs.add_frame(2, combine_imgs(self.test_layer1(X_test), grid=(6, 6)))
            self.imgs.add_frame(3, self.test_layer2(X_test).reshape((16,16)))
            self.imgs.add_frame(4, self.model._predict(X_test)[0].reshape((1,10)))
            self.imgs.set_title('Epoch #%d - Batch #%d' % (self.epoch, batch))

    def on_train_end(self, logs={}):
        anim = SubplotTimedAnimation(self.fig, self.imgs, grid=(1,5), interval=10, blit=False, repeat_delay=1000)
        # anim.save('test_gif.gif', fps=15, writer='imagemagick')
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

model.add(Dense(256, 10, W_regularizer = l2(0.1)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Fit the model
draw_weights = DrawActivations(figsize=(5.4, 1.35))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[draw_weights])


##########################
# model checkpoint tests #
##########################

print('Running ModelCheckpoint test')

nb_classes = 10
batch_size = 128
nb_epoch = 20

# small sample size to overfit on training data
max_train_samples = 50
max_test_samples = 1000

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)[:max_train_samples]
X_test = X_test.reshape(10000,784)[:max_test_samples]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]


# Create a slightly larger network than required to test best validation save only
model = Sequential()
model.add(Dense(784, 500))
model.add(Activation('relu'))
model.add(Dense(500, 10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# test file location
path = "/tmp"
filename = "model_weights.hdf5"
import os
f = os.path.join(path, filename)

print("Test model checkpointer")
# only store best validation model in checkpointer
checkpointer = cbks.ModelCheckpoint(filepath=f, verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_test, Y_test), callbacks =[checkpointer])

if not os.path.isfile(f):
    raise Exception("Model weights were not saved to %s" % (f))

print("Test model checkpointer without validation data")
import warnings
warnings.filterwarnings('error')
try:
    # this should issue a warning
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, callbacks =[checkpointer])
except:
    print("Tests passed")
    import sys
    sys.exit(0)

raise Exception("Modelcheckpoint tests did not pass")

