'''MNIST dataset with TensorFlow TFRecords.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
'''
import os
import copy
import time

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras import callbacks as cbks
from keras import optimizers, objectives
from keras import metrics as metrics_module

from tensorflow.contrib.learn.python.learn.datasets.mnist import load_mnist

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend for the time being, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


def cnn_layers(x_train_input):
    x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x_train_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_train_out = Dense(classes,
                        activation='softmax',
                        name='x_train_out')(x)
    return x_train_out


sess = tf.Session()
K.set_session(sess)

batch_size = 256
batch_shape = [batch_size, 28, 28, 1]
steps_per_epoch = 1000
epochs = 12
classes = 10
capacity = 10000
min_after_dequeue = 3000

data = load_mnist()
x_train_batch, y_train_batch = tf.train.shuffle_batch(
    tensors=[data.train.images, data.train.labels.astype(np.int32)],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    enqueue_many=True,
    num_threads=4)


x_train_batch = tf.cast(x_train_batch, tf.float32)
x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)

y_train_batch = tf.cast(y_train_batch, tf.int32)
y_train_batch = tf.one_hot(y_train_batch, classes)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

x_train_in = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
x_train_out = cnn_layers(x_train_in)
y_train_in = Input(tensor=y_train_batch, batch_shape=y_batch_shape, name='y_labels')
cce = categorical_crossentropy(y_train_batch, x_train_out)
train_model = Model(inputs=[x_train_in], outputs=[x_train_out])

train_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
train_model.summary()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
train_model.fit(None, y_train_in, batch_size=None,
                epochs=epochs, steps_per_epoch=steps_per_epoch)
train_model.save_weights('saved_wt.h5')

coord.request_stop()
coord.join(threads)
K.clear_session()

# Second Session, pure Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
x_test_inp = Input(batch_shape=(None,) + (X_test.shape[1:]))
test_out = cnn_layers(x_test_inp)
test_model = Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights('saved_wt.h5')
test_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(X_test, np_utils.to_categorical(y_test), classes)
print('\nTest accuracy: {0}'.format(acc))
