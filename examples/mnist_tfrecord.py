'''MNIST dataset with TensorFlow TFRecords.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
'''
import os
import copy
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D
from keras.callbacks import EarlyStopping
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
from keras import callbacks as cbks
from keras import optimizers, objectives
from keras import metrics as metrics_module

from keras.datasets import mnist

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend for the time being, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


def images_to_tfrecord(images, labels, filename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """ Save data into TFRecord """
    if not os.path.isfile(filename):
        num_examples = images.shape[0]

        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print 'tfrecord already exists'


def read_and_decode(filename, one_hot=True, classes=None, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img.set_shape([28 * 28])
    img = tf.reshape(img, [28, 28, 1])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)
    if one_hot and classes:
        label = tf.one_hot(label, classes)

    x_train_batch, y_train_batch = K.tf.train.shuffle_batch(
        [img, label],
        batch_size=batch_size,
        capacity=2000,
        min_after_dequeue=1000,
        num_threads=32)  # set the number of threads here

    return x_train_batch, y_train_batch


def read_and_decode_recordinput(tf_glob, one_hot=True, classes=None, is_train=None, batch_size=1000):
    """ Return tensor to read from TFRecord """

    record_input = data_flow_ops.RecordInput(tf_glob, batch_size=batch_size)
    records_op = record_input.get_yield_op()
    records_op = tf.split(records_op, batch_size, 0)
    records_op = [tf.reshape(record, []) for record in records_op]

    imgs = []
    labels = []
    for serialized_example in records_op:
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                           })
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        img.set_shape([28 * 28])
        img = tf.reshape(img, [1, 28, 28, 1])

        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

        label = tf.cast(features['label'], tf.int32)
        if one_hot and classes:
            label = tf.one_hot(label, classes)

        imgs.append(img)
        labels.append(label)

    imgs = tf.concat(imgs, 0)
    labels = tf.concat(labels, 0)
    return imgs, labels


def save_mnist_as_tfrecord():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    images_to_tfrecord(images=X_train, labels=y_train, filename='train.mnist.tfrecord')
    images_to_tfrecord(images=X_test, labels=y_test, filename='test.mnist.tfrecord')


def cnn_layers(x_train_input):
    con1 = Conv2D(32, (3, 3), activation='relu',
                  strides=(2, 2), padding='valid')(x_train_input)
    con2 = Conv2D(32, (3, 3), activation='relu',
                  strides=(2, 2))(con1)
    fla1 = Flatten()(con2)
    den1 = Dense(128, activation='relu')(fla1)
    x_train_out = Dense(classes,
                        activation='softmax',
                        name='x_train_out')(den1)
    return x_train_out


def create_cnn_model(x_train_batch, y_train_batch, x_batch_shape, y_batch_shape):
    x_train_input = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
    x_train_out = cnn_layers(x_train_input)
    # Workaround until _is_placeholder can be deduced automatically
    x_train_out._is_placeholder = False
    y_train_in_out = Input(tensor=y_train_batch, batch_shape=y_batch_shape, name='y_labels')
    return Model(inputs=[x_train_input, y_train_in_out], outputs=[x_train_out, y_train_in_out])


sess = tf.Session()
K.set_session(sess)

save_mnist_as_tfrecord()

batch_size = 1000
epochs = 300
classes = 10

x_train_batch, y_train_batch = read_and_decode_recordinput(
    'train.mnist.tfrecord',
    one_hot=True,
    classes=classes,
    is_train=True,
    batch_size=batch_size)


x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

train_model = create_cnn_model(x_train_batch,
                               y_train_batch,
                               x_batch_shape,
                               y_batch_shape)

train_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

train_model.summary()
train_model.fit(batch_size=batch_size, epochs=epochs)
train_model.save_weights('saved_wt.h5')

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
print '\nTest accuracy: {0}'.format(acc)
