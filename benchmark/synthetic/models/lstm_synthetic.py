"""Benchmark a lstm model on Synthetic dataset
Credit:
Script modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/lstm_benchmark.py
"""

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

from models import timehistory
from data_generator import generate_text_input_data

if keras.backend.backend() != 'mxnet' and \
        keras.backend.backend() != 'tensorflow':
    raise NotImplementedError(
        'This benchmark script only support MXNet and TensorFlow backend')

if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf


def crossentropy_from_logits(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class LstmBenchmark:

    def __init__(self):
        self.test_name = "lstm"
        self.sample_type = "text"
        self.total_time = 0
        self.batch_size = 128
        self.epochs = 4
        self.num_samples = 50000

    def run_benchmark(self, gpus=0, inference=False, use_dataset_tensors=False):
        print("Running model ", self.test_name)
        keras.backend.set_learning_phase(True)

        input_dim_1 = 40
        input_dim_2 = 60

        input_shape = (self.num_samples, input_dim_1, 60)
        x_train, y_train = generate_text_input_data(input_shape)
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        # build the model: a single LSTM
        model = Sequential()
        model.add(
            LSTM(128, input_shape=(input_dim_1, input_dim_2), unroll=True))

        optimizer = RMSprop(lr=0.01)

        if use_dataset_tensors:
            # Create the dataset and its associated one-shot iterator.
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(10000)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()

            # Model creation using tensors from the get_next() graph node.
            inputs, targets = iterator.get_next()

        if use_dataset_tensors:
            input_tensor = keras.layers.Input(tensor=inputs)
            model.add(Dense(input_dim_2))
            predictions = model(input_tensor)
            model = keras.models.Model(input_tensor, predictions)
        else:
            model.add(Dense(input_dim_2, activation='softmax'))

        # use multi gpu model for more than 1 gpu
        if (keras.backend.backend() == 'tensorflow' or
                keras.backend.backend() == 'mxnet') and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        if use_dataset_tensors:
            model.compile(loss=crossentropy_from_logits,
                          optimizer=optimizer,
                          metrics=['accuracy'],
                          target_tensors=[targets])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        time_callback = timehistory.TimeHistory()

        if use_dataset_tensors:
            model.fit(epochs=self.epochs, steps_per_epoch=15,
                      callbacks=[time_callback])
        else:
            model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      callbacks=[time_callback])

        self.total_time = 0
        for i in range(1, self.epochs):
            self.total_time += time_callback.times[i]

        if keras.backend.backend() == "tensorflow":
            keras.backend.clear_session()
