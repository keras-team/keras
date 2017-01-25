from __future__ import print_function
import numpy as np
np.random.seed(1337)
import pytest
import string

from keras.utils.test_utils import get_test_data, keras_test
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import TimeDistributedDense
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Embedding


@keras_test
def test_temporal_classification():
    '''
    Classify temporal sequences of float numbers
    of length 3 into 2 classes using
    single layer of GRU units and softmax applied
    to the last activations of the units
    '''
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=500,
                                                         input_shape=(3, 5),
                                                         classification=True,
                                                         nb_class=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(GRU(y_train.shape[-1],
                  input_shape=(X_train.shape[1], X_train.shape[2]),
                  activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=0)
    assert(history.history['val_acc'][-1] >= 0.8)


@keras_test
def test_temporal_regression():
    '''
    Predict float numbers (regression) based on sequences
    of float numbers of length 3 using a single layer of GRU units
    '''
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=400,
                                                         input_shape=(3, 5),
                                                         output_shape=(2,),
                                                         classification=False)
    model = Sequential()
    model.add(GRU(y_train.shape[-1],
                  input_shape=(X_train.shape[1], X_train.shape[2])))
    model.compile(loss='hinge', optimizer='adam')
    history = model.fit(X_train, y_train, nb_epoch=5, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert(history.history['val_loss'][-1] < 1.)


@keras_test
def test_sequence_to_sequence():
    '''
    Apply a same Dense layer for each element of time dimension of the input
    and make predictions of the output sequence elements.
    This does not make use of the temporal structure of the sequence
    (see TimeDistributedDense for more details)
    '''
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         output_shape=(3, 5),
                                                         classification=False)

    model = Sequential()
    model.add(TimeDistributedDense(y_train.shape[-1],
                                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.compile(loss='hinge', optimizer='rmsprop')
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert(history.history['val_loss'][-1] < 0.8)


@keras_test
def test_stacked_lstm_char_prediction():
    '''
    Learn alphabetical char sequence with stacked LSTM.
    Predict the whole alphabet based on the first two letters ('ab' -> 'ab...z')
    See non-toy example in examples/lstm_text_generation.py
    '''
    # generate alphabet: http://stackoverflow.com/questions/16060899/alphabet-range-python
    alphabet = string.ascii_lowercase
    number_of_chars = len(alphabet)

    # generate char sequences of length 'sequence_length' out of alphabet and store the next char as label (e.g. 'ab'->'c')
    sequence_length = 2
    sentences = [alphabet[i: i + sequence_length] for i in range(len(alphabet) - sequence_length)]
    next_chars = [alphabet[i + sequence_length] for i in range(len(alphabet) - sequence_length)]

    # Transform sequences and labels into 'one-hot' encoding
    X = np.zeros((len(sentences), sequence_length, number_of_chars), dtype=np.bool)
    y = np.zeros((len(sentences), number_of_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, ord(char) - ord('a')] = 1
        y[i, ord(next_chars[i]) - ord('a')] = 1

    # learn the alphabet with stacked LSTM
    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(sequence_length, number_of_chars)),
        LSTM(16, return_sequences=False),
        Dense(number_of_chars, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, batch_size=1, nb_epoch=60, verbose=1)

    # prime the model with 'ab' sequence and let it generate the learned alphabet
    sentence = alphabet[:sequence_length]
    generated = sentence
    for iteration in range(number_of_chars - sequence_length):
        x = np.zeros((1, sequence_length, number_of_chars))
        for t, char in enumerate(sentence):
            x[0, t, ord(char) - ord('a')] = 1.
        preds = model.predict(x, verbose=0)[0]
        next_char = chr(np.argmax(preds) + ord('a'))
        generated += next_char
        sentence = sentence[1:] + next_char

    # check that it did generate the alphabet correctly
    assert(generated == alphabet)


@keras_test
def test_masked_temporal():
    '''
    Confirm that even with masking on both inputs and outputs, cross-entropies are
    of the expected scale.

    In this task, there are variable length inputs of integers from 1-9, and a random
    subset of unmasked outputs. Each of these outputs has a 50% probability of being
    the input number unchanged, and a 50% probability of being 2*input%10.

    The ground-truth best cross-entropy loss should, then be -log(0.5) = 0.69

    '''
    model = Sequential()
    model.add(Embedding(10, 20, mask_zero=True, input_length=20))
    model.add(TimeDistributedDense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  sample_weight_mode='temporal')

    X = np.random.random_integers(1, 9, (50000, 20))
    for rowi in range(X.shape[0]):
        padding = np.random.random_integers(X.shape[1] / 2)
        X[rowi, :padding] = 0

    # 50% of the time the correct output is the input.
    # The other 50% of the time it's 2 * input % 10
    y = (X * np.random.random_integers(1, 2, X.shape)) % 10
    Y = np.zeros((y.size, 10), dtype='int32')
    for i, target in enumerate(y.flat):
        Y[i, target] = 1
    Y = Y.reshape(y.shape + (10,))

    # Mask 50% of the outputs via sample weights
    sample_weight = np.random.random_integers(0, 1, y.shape)
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    print('sample_weight shape:', Y.shape)

    history = model.fit(X, Y, validation_split=0.05,
                        sample_weight=None,
                        verbose=1, nb_epoch=2)
    ground_truth = -np.log(0.5)
    assert(np.abs(history.history['val_loss'][-1] - ground_truth) < 0.06)

if __name__ == '__main__':
    pytest.main([__file__])
