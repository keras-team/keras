"""Benchmark a lstm model
Original Model from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
Benchmark for a LSTM model.
Script template modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/lstm_benchmark.py
"""

from __future__ import print_function
import random
import sys
import keras
import numpy as np

from keras.utils import multi_gpu_model
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from models import timehistory, dataset_utils

if keras.backend.backend() != 'mxnet' and \
        keras.backend.backend() != 'tensorflow':
    raise NotImplementedError(
        'This benchmark script only support MXNet and TensorFlow backend')

if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf


def crossentropy_from_logits(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class LstmBenchmark:

    def __init__(self, dataset_name=None):
        self.test_name = "lstm_text_generation"
        self.sample_type = "text"
        self.total_time = 0
        self.batch_size = 128
        self.epochs = 60
        self.dataset_name = dataset_name

    def run_benchmark(self, gpus=0, inference=False, use_dataset_tensors=False):
        print("Running model ", self.test_name)
        keras.backend.set_learning_phase(True)

        text = dataset_utils.get_dataset(self.dataset_name)
        print('corpus length:', len(text))

        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        maxlen = 40
        step = 3
        input_dim_1 = maxlen
        input_dim_2 = len(chars)
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x_train = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y_train = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x_train[i, t, char_indices[char]] = 1
            y_train[i, char_indices[next_chars[i]]] = 1

        # build the model: a single LSTM
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars)), unroll=True))

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
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer)

        time_callback = timehistory.TimeHistory()

        def sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        def on_epoch_end(epoch, logs):
            # Function invoked at end of each epoch. Prints generated text.
            print()
            print('----- Generating text after Epoch: %d' % epoch)

            start_index = random.randint(0, len(text) - maxlen - 1)
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print('----- diversity:', diversity)

                generated = ''
                sentence = text[start_index: start_index + maxlen]
                generated += sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(400):
                    x_pred = np.zeros((32, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        if inference:
            callback = print_callback
        else:
            callback = time_callback

        if use_dataset_tensors:
            model.fit(epochs=self.epochs, steps_per_epoch=15,
                      callbacks=[callback])
        else:
            model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      callbacks=[callback])

        if keras.backend.backend() == "tensorflow":
            keras.backend.clear_session()
