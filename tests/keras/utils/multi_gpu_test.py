"""These tests are not meant to be run on CI.
"""
from __future__ import print_function

import keras
from keras import backend as K
from keras.utils import multi_gpu_model

import numpy as np
import pytest
import time
import tempfile
import tensorflow as tf
from keras.utils.test_utils import keras_test
from keras.preprocessing.image import ImageDataGenerator


pytestmark = pytest.mark.skipif(K.backend() != 'tensorflow',
                                reason='Requires TF.')
if K.backend() == 'tensorflow':
    available_devices = keras.utils.multi_gpu_utils._get_available_devices()
    available_devices = [keras.utils.multi_gpu_utils._normalize_device_name(name)
                         for name in available_devices]
    pytestmark = pytest.mark.skipif('/gpu:7' not in available_devices,
                                    reason='Requires 8 GPUs.')


@keras_test
def test_multi_gpu_simple_model():
    print('####### test simple model')
    num_samples = 1000
    input_dim = 10
    output_dim = 1
    hidden_dim = 10
    gpus = 8
    target_gpu_id = [0, 2, 4]
    epochs = 2
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_dim,
                                 input_shape=(input_dim,)))
    model.add(keras.layers.Dense(output_dim))

    x = np.random.random((num_samples, input_dim))
    y = np.random.random((num_samples, output_dim))

    parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(loss='mse', optimizer='rmsprop')
    parallel_model.fit(x, y, epochs=epochs)

    parallel_model = multi_gpu_model(model, gpus=target_gpu_id)
    parallel_model.compile(loss='mse', optimizer='rmsprop')
    parallel_model.fit(x, y, epochs=epochs)


@keras_test
def test_multi_gpu_multi_io_model():
    print('####### test multi-io model')
    num_samples = 1000
    input_dim_a = 10
    input_dim_b = 5
    output_dim_a = 1
    output_dim_b = 2
    hidden_dim = 10
    gpus = 8
    target_gpu_id = [0, 2, 4]
    epochs = 2

    input_a = keras.Input((input_dim_a,))
    input_b = keras.Input((input_dim_b,))
    a = keras.layers.Dense(hidden_dim)(input_a)
    b = keras.layers.Dense(hidden_dim)(input_b)
    c = keras.layers.concatenate([a, b])
    output_a = keras.layers.Dense(output_dim_a)(c)
    output_b = keras.layers.Dense(output_dim_b)(c)
    model = keras.models.Model([input_a, input_b], [output_a, output_b])

    a_x = np.random.random((num_samples, input_dim_a))
    b_x = np.random.random((num_samples, input_dim_b))
    a_y = np.random.random((num_samples, output_dim_a))
    b_y = np.random.random((num_samples, output_dim_b))

    parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(loss='mse', optimizer='rmsprop')
    parallel_model.fit([a_x, b_x], [a_y, b_y], epochs=epochs)

    parallel_model = multi_gpu_model(model, gpus=target_gpu_id)
    parallel_model.compile(loss='mse', optimizer='rmsprop')
    parallel_model.fit([a_x, b_x], [a_y, b_y], epochs=epochs)


@keras_test
def test_multi_gpu_invalid_devices():
    input_shape = (1000, 10)
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10,
                                 activation='relu',
                                 input_shape=input_shape[1:]))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    x = np.random.random(input_shape)
    y = np.random.random((input_shape[0], 1))
    with pytest.raises(ValueError):
        parallel_model = multi_gpu_model(model, gpus=10)
        parallel_model.compile(loss='mse', optimizer='rmsprop')
        parallel_model.fit(x, y, epochs=2)

    with pytest.raises(ValueError):
        parallel_model = multi_gpu_model(model, gpus=[0, 2, 4, 6, 8])
        parallel_model.compile(loss='mse', optimizer='rmsprop')
        parallel_model.fit(x, y, epochs=2)

    with pytest.raises(ValueError):
        parallel_model = multi_gpu_model(model, gpus=1)
        parallel_model.compile(loss='mse', optimizer='rmsprop')
        parallel_model.fit(x, y, epochs=2)

    with pytest.raises(ValueError):
        parallel_model = multi_gpu_model(model, gpus=[0])
        parallel_model.compile(loss='mse', optimizer='rmsprop')
        parallel_model.fit(x, y, epochs=2)


@keras_test
def test_serialization():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(3,
                                 input_shape=(4,)))
    model.add(keras.layers.Dense(4))

    x = np.random.random((100, 4))
    y = np.random.random((100, 4))

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='mse', optimizer='rmsprop')
    parallel_model.fit(x, y, epochs=1)

    ref_output = parallel_model.predict(x)

    _, fname = tempfile.mkstemp('.h5')
    parallel_model.save(fname)

    K.clear_session()
    parallel_model = keras.models.load_model(fname)
    output = parallel_model.predict(x)
    np.testing.assert_allclose(ref_output, output, atol=1e-5)


def multi_gpu_application_np_array_benchmark():
    print('####### Xception benchmark - np i/o')
    model_cls = keras.applications.Xception

    num_samples = 1000
    height = 224
    width = 224
    num_classes = 1000
    epochs = 4
    batch_size = 40
    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))

    # Baseline
    model = model_cls(weights=None,
                      input_shape=(height, width, 3),
                      classes=num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')

    # Training
    start_time = time.time()
    model.fit(x, y, epochs=epochs)
    total_time = time.time() - start_time
    print('baseline training:', total_time)

    # Inference
    start_time = time.time()
    model.predict(x)
    total_time = time.time() - start_time
    print('baseline inference:', total_time)

    for i in range(2, 9, 2):
        K.clear_session()
        with tf.device('/cpu:0'):
            model = model_cls(weights=None,
                              input_shape=(height, width, 3),
                              classes=num_classes)
        parallel_model = multi_gpu_model(model, gpus=i)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')

        start_time = time.time()
        parallel_model.fit(x, y, epochs=epochs, batch_size=batch_size)
        total_time = time.time() - start_time
        print('%d gpus training:' % i, total_time)

        # Inference
        start_time = time.time()
        parallel_model.predict(x, batch_size=batch_size)
        total_time = time.time() - start_time
        print('%d gpus inference:' % i, total_time)


def multi_gpu_application_folder_generator_benchmark():
    """Before running this test:

    wget https://s3.amazonaws.com/img-datasets/cats_and_dogs_small.zip
    unzip cats_and_dogs_small.zip
    """
    print('####### Xception benchmark - folder generator i/o')
    model_cls = keras.applications.Xception

    height = 150
    width = 150
    num_classes = 2
    epochs = 3
    steps_per_epoch = 100
    batch_size = 64

    # Baseline
    model = model_cls(weights=None,
                      input_shape=(height, width, 3),
                      classes=num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_dir = '/home/ubuntu/cats_and_dogs_small/train'  # Change this
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

    # Training
    start_time = time.time()
    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        workers=4)
    total_time = time.time() - start_time
    print('baseline training:', total_time)

    for i in range(2, 9):
        K.clear_session()
        with tf.device('/cpu:0'):
            model = model_cls(weights=None,
                              input_shape=(height, width, 3),
                              classes=num_classes)
        parallel_model = multi_gpu_model(model, gpus=i)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')

        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='categorical')

        start_time = time.time()
        parallel_model.fit_generator(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            workers=4 * i)
        total_time = time.time() - start_time
        print('%d gpus training:' % i, total_time)


if __name__ == '__main__':
    pytest.main([__file__])
