'''
Pai Peng (pengpai_sh@163.com)

An implementation of  "DisturbLabel" which randomly replaces a part of labels as incorrect
values in each iteration, i.e. adding noises on the loss layer.

"DisturbLabel: Regularizing CNN on the Loss Layer." http://bigml.cs.tsinghua.edu.cn/~lingxi/PDFs/Xie_CVPR16_DisturbLabel.pdf

We are trying to reproduce the results presented in Section 4.3. "Train a LeNet on the MNIST dataset,
with only 1% (600) and 10% (6000) training samples, we obtain 10.92% and 2.83% error rates on the
original testing set, respectively, which are dramatic compared to 0.86% when the network is trained
on the complete set. DisturbLabel significantly decreases the error rates to 6.38% and 1.89%, respectively.
As a reference, the error rate on the complete trainig set is further decreased to 0.66% by the
 DisturbLabel."

My results of error rates (LeNet without Dropout) with 1% and 10% training data is: 7.83% and 2.31%.
The DisturbLabel will reduce them to 5.93% and 1.76% with alpha=0.1, respectively. When we use the
whole training samples, the error rate is 0.69%, while DisturbLabel decreases it to 0.61% (alpha=0.05)
and 0.58%(alpha=0.075), 0.54% (alpha=0.1), 0.56% (alpha=0.15), 0.62% (alpha=0.2). Coperation with
Dropout (drop_prob=0.5) further decreases error rate to 0.47% (alpha=0.1) while it is reported as
0.33% in the paper.

'''

import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.datasets import mnist
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.cross_validation import train_test_split

def disturb_label(y, alpha = 0.1):
    '''
    Disturb the label for a training data.
    Randomly generate the disturbed label from a Multinoulli distribution P(alpha), which is defined
    as follows:
        Suppose that c is the ground-truth label (i.e. y_c = 1, y_i = 0), then
            p_c = 1 - (C - 1) / C * alpha, p_i = 1 / C * alpha
        where C is the number of classes.

    Input:
        y:  (numpy.array or list) training label, one-hot vector, e.g. [0,1,0, ..., 0]
    alpha:  disturb probability, usually a small number (10% - 20%), note that alpha=0 means no disturb

    Output:
        y_disturb (also a one-hot vector, numpy.array)
    '''

    nbr_class_C = len(y)
    label_c = list(y).index(1)
    prob = [1. / nbr_class_C * alpha] * nbr_class_C
    prob[label_c] = 1. - (nbr_class_C - 1.) / nbr_class_C * alpha
    y_disturb = np.random.multinomial(n = 1, pvals = prob)

    return y_disturb


def generator_disturb_label(X, Y, batch_size = 32, shuffle = False, alpha = 0.1):
    '''
    A generator that yields a batch of (data, label), while
    label is distrubed.

    Input:
        X     : numpy.array for data
        Y     : numpy.array for labels, each row is a one-hot vector
      shuffle : wheter shuffle X and Y
    batch_size: batch size
      alpha   : disturb probability

    Output:
        (X_batch, Y_batch_disturbed)
    '''

    N = X.shape[0]

    if shuffle:
        indices = np.random.permutation(N)
        X = X[indices]
        Y = Y[indices]

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = X[current_index: current_index + current_batch_size]
        Y_batch = Y[current_index: current_index + current_batch_size].copy()

        # each sample in the current batch is distrubed independently
        for i in range(Y_batch.shape[0]):
            Y_batch[i] = disturb_label(Y_batch[i], alpha = alpha)

        yield (X_batch, Y_batch)

def get_LeNet():
    '''
    Define the LeNet model.
    '''
    model = Sequential()

    model.add(Convolution2D(32, 5, 5,
                            border_mode='valid',
                            input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    # output shape: 24x24

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # output shape: 12x12

    model.add(Convolution2D(64, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    # output shape: 8x8

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # output shape: 4x4

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

if __name__ =='__main__':
    learning_rate = 0.01
    batch_size = 128
    nb_epoch = 30
    train_proportion = 1.

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    nbr_train = X_train.shape[0]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # perpixel mean substracted
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    if train_proportion < 1.:
        X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size = train_proportion,
                                                  random_state = 2016)

    model = get_LeNet()
    model.summary()

    optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    # autosave best Model
    best_model_file = "./lenet_mnist_weights.h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

    for i in range(3):
        if i != 0:
            # devide the learning rate by 10 for two times
            lr_old = K.get_value(optimizer.lr)
            K.set_value(optimizer.lr, 0.1 * lr_old)
            print('Changing learning rate from %f to %f' % (lr_old, K.get_value(optimizer.lr)))

        # model training
        # model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch,
        #           verbose = 1, validation_data = (X_test, Y_test), callbacks = [best_model])

        model.fit_generator(generator_disturb_label(X_train, Y_train, batch_size = batch_size,
                            shuffle = False, alpha = 0.1),
                            samples_per_epoch = X_train.shape[0], nb_epoch = nb_epoch,
                            verbose = 1, validation_data = (X_test, Y_test),
                            callbacks = [best_model,
                            TensorBoard(log_dir='/tmp/disturb', histogram_freq=1)])

    print('loading best model...')
    model.load_weights(best_model_file)
    score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
