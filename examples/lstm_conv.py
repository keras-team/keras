from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.recurrent_convolutional import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from pylab import *

# We create a layer whose take movies as input
# of shape (time, width, height, channel) and that return a movie
# with identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   input_shape=(None, 40, 40, 1),
                   border_mode="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                      kernel_dim3=3, activation='sigmoid',
                      border_mode="same", dim_ordering="tf"))

seq.compile(loss="binary_crossentropy", optimizer="adadelta")


# Generating artificial data:
# We are going to create a movie with
# square of size one or two by two pixels moving linearly
# trought time. For convenience we first create
# a movie with bigger width and height, and at the end
# we cut it to 40x40

time = 15
row = 80
col = 80
filters = 1
training = 1200
train = np.zeros((training, time, row, col, 1), dtype=np.float)
gt = np.zeros((training, time, row, col, 1), dtype=np.float)

for i in range(training):

    # add from 3 to 7 moving squares
    n = np.random.randint(3, 8)

    for j in range(n):
        # Initial position
        xstart = np.random.randint(20, 60)
        ystart = np.random.randint(20, 60)
        # Direction of motion
        directionx = np.random.randint(0, 3) - 1
        directiony = np.random.randint(0, 3) - 1

        # Size of the square
        w = np.random.randint(2, 4)

        for t in range(time):
            x_shift = xstart + directionx * t
            y_shift = ystart + directiony * t
            train[i, t, x_shift - w: x_shift + w,
                  y_shift - w: y_shift + w, 0] += 1

            # Make it more robust by adding noise.
            # The idea is that if during predict time,
            # the value of the pixel is not exactly one,
            # we need to train the network to be robust and stille
            # consider it is a pixel belonging to a square.
            if np.random.randint(0, 2):
                noise_f = (-1)**np.random.randint(0, 2)
                train[i, t, x_shift - w - 1: x_shift + w + 1,
                      y_shift - w - 1: y_shift + w + 1, 0] += noise_f * 0.1

            # Shitf the ground truth by 1
            x_shift = xstart + directionx * (t + 1)
            y_shift = ystart + directiony * (t + 1)
            gt[i, t, x_shift - w: x_shift + w,
               y_shift - w: y_shift + w, 0] += 1

# Cut to a forty's sized window
train = train[::, ::, 20:60, 20:60, ::]
gt = gt[::, ::, 20:60, 20:60, ::]
train[train >= 1] = 1
gt[gt >= 1] = 1

# Train the network
seq.fit(train[:1000], gt[:1000], batch_size=10,
        nb_epoch=300, validation_split=0.05)

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1004
track = train[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = train[which][::, ::, ::, ::]
for i in range(15):
    fig = figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, "Predictions !", fontsize=20, color="w")
    else:
        ax.text(1, 3, "Inital trajectory", fontsize=20)

    toplot = track[i, ::, ::, 0]

    imshow(toplot)
    ax = fig.add_subplot(122)
    text(1, 3, "Ground truth", fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = gt[which][i - 1, ::, ::, 0]

    imshow(toplot)
    savefig("%i_animate.png" % (i + 1))
