from __future__ import print_function
import numpy as np
import pytest
import string

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dropout, Dense
from keras.layers.recurrent import GRU, LSTM
from keras.utils.np_utils import to_categorical


def test_temporal_classification():
    '''
    Classify temporal sequences of float numbers of length 3 into 2 classes using
    single layer of GRU units and softmax applied to the last activations of the units
    '''
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         classification=True,
                                                         nb_class=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(GRU(y_train.shape[-1],
                  input_shape=(X_train.shape[1], X_train.shape[2]),
                  activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    history = model.fit(X_train, y_train, nb_epoch=5, batch_size=16,
                        validation_data=(X_test, y_test),
                        show_accuracy=True, verbose=0)
    assert(history.history['val_acc'][-1] > 0.9)


def test_temporal_regression():
    '''
    Predict float numbers (regression) based on sequences of float numbers of length 3 using
    single layer of GRU units
    '''
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         output_shape=(2,),
                                                         classification=False)
    model = Sequential()
    model.add(GRU(y_train.shape[-1],
                  input_shape=(X_train.shape[1], X_train.shape[2])))
    model.compile(loss='hinge', optimizer='adam')
    history = model.fit(X_train, y_train, nb_epoch=5, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert(history.history['val_loss'][-1] < 0.75)


def test_sequence_to_sequence():
    '''
    Apply a same Dense layer for each element of time dimension of the input
    and make predictions of the output sequence elements.
    This does not make use of the temporal structure of the sequence
    (see TimeDistributedDense for more details)
    '''
    np.random.seed(1337)
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


def test_stacked_lstm_char_prediction():
    '''
    Learn alphabetical char sequence with stacked LSTM.
    Predict the whole alphabet based on the first two letters ('ab' -> 'ab...z')
    See non-toy example in examples/lstm_text_generation.py
    '''
    np.random.seed(1336)
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
            X[i, t, ord(char)-ord('a')] = 1
        y[i, ord(next_chars[i])-ord('a')] = 1

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
    for iteration in range(number_of_chars-sequence_length):
        x = np.zeros((1, sequence_length, number_of_chars))
        for t, char in enumerate(sentence):
            x[0, t, ord(char) - ord('a')] = 1.
        preds = model.predict(x, verbose=0)[0]
        next_char = chr(np.argmax(preds) + ord('a'))
        generated += next_char
        sentence = sentence[1:] + next_char

    # check that it did generate the alphabet correctly
    assert(generated == alphabet)


if __name__ == '__main__':
    pytest.main([__file__])
