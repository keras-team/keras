'''Optimized MNIST dataset with TFRecords, the standard TensorFlow data format.

TFRecord is a data format supported throughout TensorFlow.
For a straightforward usage example see mnist_tfrecord.py.
This example demonstrates how to write and read TFRecord data using
Input Tensors in a way that is better optimized for high performance
on large datasets with Keras.

Gets to 99.25% test accuracy after 12 epochs
(there is still some margin for parameter tuning).
'''
import os
import copy
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras import backend as K
from keras.models import Model
from keras import layers
from keras.utils.generic_utils import Progbar

from keras.datasets import mnist

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend for the time being, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


def images_to_tfrecord(images, labels, filename):
    """Write images and their labels to a TFRecord file.

    # Arguments
        images: A numpy array or list of image data with
            shape (images, rows, cols, depth).
        labels: A numpy array of labels, with one for each image.
        filename: Path and name for the output dataset TFRecord file.
            An example is `'path/to/mnist_train.tfrecord'`
    """
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
        print('tfrecord %s already exists' % filename)


def read_and_decode_recordinput(tf_glob, one_hot=True,
                                classes=None,
                                batch_shape=None, parallelism=1):
    """Get a TF Tensor that supplies shuffled batches of images.

    # Arguments
        tf_glob: File path for selecting one or more tfrecord files.
            Examples are `'path/to/data.tfrecord'` and `'path/to/*.tfrecord'`.
        one_hot: Use one hot encoding for labels, also known as categorical.
        batch_shape: Specify the desired image batch shape, where the first
            entry is the batch size. MNIST might be (128, 28, 28, 1).
        parallelism: The number of threads to use for loading new data.
            A reasonable value is the number of logical cores on your processor.
    """
    if batch_shape is None:
        batch_shape = [1000, 28, 28, 1]
    print 'Creating graph for loading %s TFRecords...' % tf_glob
    with tf.variable_scope("TFRecords"):
        record_input = data_flow_ops.RecordInput(
            tf_glob, batch_size=batch_shape[0], parallelism=parallelism)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_shape[0], 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        progbar = Progbar(len(records_op))

        images = []
        labels = []
        for i, serialized_example in enumerate(records_op):
            progbar.update(i)
            with tf.variable_scope("parse_images", reuse=True):
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'label': tf.FixedLenFeature([], tf.int64),
                        'image_raw': tf.FixedLenFeature([], tf.string),
                    })
                img = tf.decode_raw(features['image_raw'], tf.uint8)
                img.set_shape(batch_shape[1] * batch_shape[2])
                img = tf.reshape(img, [1] + batch_shape[1:])

                img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

                label = tf.cast(features['label'], tf.int32)
                if one_hot and classes:
                    label = tf.one_hot(label, classes)

                images.append(img)
                labels.append(label)

        images = tf.parallel_stack(images, 0)
        labels = tf.parallel_stack(labels, 0)
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=batch_shape)

        return images, labels


def save_mnist_as_tfrecord():
    """Save one tfrecord file for each of the train and test mnist datasets.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    images_to_tfrecord(images=x_train, labels=y_train, filename='train.mnist.tfrecord')
    images_to_tfrecord(images=x_test, labels=y_test, filename='test.mnist.tfrecord')


def cnn_layers(x_train_input):
    """Create the CNN layers for use with either numpy inputs or tensor inputs.
    """
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x_train_input)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_train_out = layers.Dense(classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return x_train_out


sess = K.get_session()

save_mnist_as_tfrecord()

batch_size = 100
batch_shape = [batch_size, 28, 28, 1]
epochs = 12
steps_per_epoch = 1000
classes = 10
parallelism = 10

x_train_batch, y_train_batch = read_and_decode_recordinput(
    'train.mnist.tfrecord',
    one_hot=True,
    classes=classes,
    batch_shape=batch_shape,
    parallelism=parallelism)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

# The input tensors are provided directly into the Model network.
# The network is fixed once it is initialized, so it must be
# reconstructed every time a new input data source is needed.
# This is substantially different from typical
# Keras numpy array inputs, and is more like TensorFlow.
x_train_in = layers.Input(tensor=x_train_batch, batch_shape=x_batch_shape)
x_train_out = cnn_layers(x_train_in)
y_train_in = layers.Input(tensor=y_train_batch, batch_shape=y_batch_shape, name='y_labels')
train_model = Model(inputs=[x_train_in], outputs=[x_train_out])
train_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# The input data was created with x_train_input,
# so only the label data needs to be provided.
train_model.fit(y=y_train_in,
                batch_size=None,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch)
train_model.save_weights('saved_wt.h5')

K.clear_session()

# Second Session, test data
x_test_batch, y_test_batch = read_and_decode_recordinput(
    'test.mnist.tfrecord',
    one_hot=True,
    classes=classes,
    batch_shape=batch_shape,
    parallelism=parallelism)

x_batch_shape = x_test_batch.get_shape().as_list()
y_batch_shape = y_test_batch.get_shape().as_list()

# Create a completely new network for new input data.
x_test_in = layers.Input(tensor=x_test_batch, batch_shape=x_batch_shape)
x_test_out = cnn_layers(x_test_in)
y_test_in = layers.Input(tensor=y_test_batch, batch_shape=y_batch_shape, name='y_labels')
test_model = Model(inputs=[x_test_in], outputs=[x_test_out])
test_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
test_model.load_weights('saved_wt.h5')
test_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Take steps for each element of validation data.
validation_samples = 10000
evaluate_steps = validation_samples / batch_size
loss, acc = test_model.evaluate(y=y_test_in, steps=evaluate_steps)
print('\nTest accuracy: {0}'.format(np.mean(acc)))
