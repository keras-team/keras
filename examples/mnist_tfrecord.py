'''MNIST dataset with TensorFlow TFRecords.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
'''
import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils.generic_utils import Progbar
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
        print('tfrecord %s already exists' % filename)


def read_and_decode_recordinput(tf_glob, one_hot=True, classes=None, is_train=None, batch_shape=[128, 28, 28, 1]):
    """ Return tensor to read from TFRecord """
    print 'Creating graph for loading %s TFRecords...' % tf_glob
    with tf.variable_scope("TFRecords"):
        record_input = data_flow_ops.RecordInput(tf_glob, batch_size=batch_shape[0])
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

        # StagingArea will store tensors
        # across multiple steps to
        # speed up execution
        images_shape = images.get_shape()
        labels_shape = labels.get_shape()
        copy_stage = data_flow_ops.StagingArea(
            [tf.float32, tf.float32],
            shapes=[images_shape, labels_shape])
        copy_stage_op = copy_stage.put(
            [images, labels])
        staged_images, staged_labels = copy_stage.get()

        return images, labels


def save_mnist_as_tfrecord():
    print('Loading MNIST data...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print('Writing TFRecords...')
    images_to_tfrecord(images=X_train, labels=y_train, filename='train.mnist.tfrecord')
    images_to_tfrecord(images=X_test, labels=y_test, filename='test.mnist.tfrecord')


def count_tfrecord_entries(filename):
    c = 0
    for record in tf.python_io.tf_record_iterator(filename):
        c += 1
    return c


def cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


batch_size = 128
num_classes = 10
epochs = 12
batch_shape = [batch_size, 28, 28, 1]

save_mnist_as_tfrecord()

train_length = count_tfrecord_entries('train.mnist.tfrecord')
train_steps = train_length / batch_size
print('%d training records, %d batches' % (train_length, train_steps))
test_length = count_tfrecord_entries('test.mnist.tfrecord')
test_steps = test_length / batch_size
print('%d training records, %d batches' % (test_length, test_steps))

x_train_batch, y_train_batch = read_and_decode_recordinput(
    'train.mnist.tfrecord',
    one_hot=True,
    classes=num_classes,
    is_train=True,
    batch_shape=batch_shape)

x_test_batch, y_test_batch = read_and_decode_recordinput(
    'test.mnist.tfrecord',
    one_hot=True,
    classes=num_classes,
    is_train=True,
    batch_shape=batch_shape)

x_batch_shape = x_train_batch.get_shape().as_list()

model = cnn_model(x_batch_shape[1:], num_classes=num_classes)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_batch, y_train_batch,
          steps_per_epoch=train_steps,
          validation_data=(x_test_batch, y_test_batch),
          validation_steps=test_steps,
          epochs=epochs)
