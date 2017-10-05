from __future__ import print_function
import numpy as np
import pytest
import string

from keras.utils.test_utils import get_test_data, keras_test
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import layers, optimizers
import keras.backend as K
import keras


@keras_test
def test_temporal_classification():
    '''
    Classify temporal sequences of float numbers
    of length 3 into 2 classes using
    single layer of GRU units and softmax applied
    to the last activations of the units
    '''
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=200,
                                                         num_test=20,
                                                         input_shape=(3, 4),
                                                         classification=True,
                                                         num_classes=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(layers.GRU(8,
                         input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(layers.Dense(y_train.shape[-1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=4, batch_size=10,
                        validation_data=(x_test, y_test),
                        verbose=0)
    assert(history.history['acc'][-1] >= 0.8)
    config = model.get_config()
    model = Sequential.from_config(config)


@keras_test
def test_temporal_classification_functional():
    '''
    Classify temporal sequences of float numbers
    of length 3 into 2 classes using
    single layer of GRU units and softmax applied
    to the last activations of the units
    '''
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=200,
                                                         num_test=20,
                                                         input_shape=(3, 4),
                                                         classification=True,
                                                         num_classes=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = layers.SimpleRNN(8)(inputs)
    outputs = layers.Dense(y_train.shape[-1], activation='softmax')(x)
    model = keras.models.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=4, batch_size=10,
                        validation_data=(x_test, y_test),
                        verbose=0)
    assert(history.history['acc'][-1] >= 0.8)


@keras_test
def test_temporal_regression():
    '''
    Predict float numbers (regression) based on sequences
    of float numbers of length 3 using a single layer of GRU units
    '''
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=200,
                                                         num_test=20,
                                                         input_shape=(3, 5),
                                                         output_shape=(2,),
                                                         classification=False)
    model = Sequential()
    model.add(layers.LSTM(y_train.shape[-1],
                          input_shape=(x_train.shape[1], x_train.shape[2])))
    model.compile(loss='hinge', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=5, batch_size=16,
                        validation_data=(x_test, y_test), verbose=0)
    assert(history.history['loss'][-1] < 1.)


@keras_test
def test_3d_to_3d():
    '''
    Apply a same Dense layer for each element of time dimension of the input
    and make predictions of the output sequence elements.
    This does not make use of the temporal structure of the sequence
    (see TimeDistributedDense for more details)
    '''
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=100,
                                                         num_test=20,
                                                         input_shape=(3, 5),
                                                         output_shape=(3, 5),
                                                         classification=False)

    model = Sequential()
    model.add(layers.TimeDistributed(
        layers.Dense(y_train.shape[-1]), input_shape=(x_train.shape[1], x_train.shape[2])))
    model.compile(loss='hinge', optimizer='rmsprop')
    history = model.fit(x_train, y_train, epochs=20, batch_size=16,
                        validation_data=(x_test, y_test), verbose=0)
    assert(history.history['loss'][-1] < 1.)


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
    x = np.zeros((len(sentences), sequence_length, number_of_chars), dtype=np.bool)
    y = np.zeros((len(sentences), number_of_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, ord(char) - ord('a')] = 1
        y[i, ord(next_chars[i]) - ord('a')] = 1

    # learn the alphabet with stacked LSTM
    model = Sequential([
        layers.LSTM(16, return_sequences=True, input_shape=(sequence_length, number_of_chars)),
        layers.LSTM(16, return_sequences=False),
        layers.Dense(number_of_chars, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x, y, batch_size=1, epochs=60, verbose=1)

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
    model.add(layers.Embedding(10, 10, mask_zero=True))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')

    x = np.random.random_integers(1, 9, (20000, 10))
    for rowi in range(x.shape[0]):
        padding = np.random.random_integers(x.shape[1] / 2)
        x[rowi, :padding] = 0

    # 50% of the time the correct output is the input.
    # The other 50% of the time it's 2 * input % 10
    y = (x * np.random.random_integers(1, 2, x.shape)) % 10
    ys = np.zeros((y.size, 10), dtype='int32')
    for i, target in enumerate(y.flat):
        ys[i, target] = 1
    ys = ys.reshape(y.shape + (10,))

    history = model.fit(x, ys, validation_split=0.05, batch_size=10,
                        verbose=0, epochs=3)
    ground_truth = -np.log(0.5)
    assert(np.abs(history.history['loss'][-1] - ground_truth) < 0.06)


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow backend')
@keras_test
def test_embedding_with_clipnorm():
    model = Sequential()
    model.add(layers.Embedding(input_dim=1, output_dim=1))
    model.compile(optimizer=optimizers.SGD(clipnorm=0.1), loss='mse')
    model.fit(np.array([[0]]), np.array([[[0.5]]]), epochs=1)

if __name__ == '__main__':
    pytest.main([__file__])
