from __future__ import print_function
import pytest

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras import layers
import keras
from keras.utils.np_utils import to_categorical

num_classes = 2


def test_vector_classification():
    '''
    Classify random float vectors into 2 classes with logistic regression
    using 2 layer neural network with ReLU hidden units.
    '''
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=500,
                                                         num_test=200,
                                                         input_shape=(20,),
                                                         classification=True,
                                                         num_classes=num_classes)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Test with Sequential API
    model = Sequential([
        layers.Dense(16, input_shape=(x_train.shape[-1],), activation='relu'),
        layers.Dense(8),
        layers.Activation('relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=15, batch_size=16,
                        validation_data=(x_test, y_test),
                        verbose=0)
    assert(history.history['val_acc'][-1] > 0.8)
    config = model.get_config()
    model = Sequential.from_config(config)


def test_vector_classification_functional():
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=500,
                                                         num_test=200,
                                                         input_shape=(20,),
                                                         classification=True,
                                                         num_classes=num_classes)
    # Test with functional API
    inputs = layers.Input(shape=(x_train.shape[-1],))
    x = layers.Dense(16, activation=keras.activations.relu)(inputs)
    x = layers.Dense(8)(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs, outputs)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=15, batch_size=16,
                        validation_data=(x_test, y_test),
                        verbose=0)
    assert(history.history['val_acc'][-1] > 0.8)


def test_vector_regression():
    '''
    Perform float data prediction (regression) using 2 layer MLP
    with tanh and sigmoid activations.
    '''
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=500,
                                                         num_test=200,
                                                         input_shape=(20,),
                                                         output_shape=(num_classes,),
                                                         classification=False)

    model = Sequential([
        layers.Dense(16, input_shape=(x_train.shape[-1],), activation='tanh'),
        layers.Dense(num_classes)
    ])

    model.compile(loss='hinge', optimizer='adagrad')
    history = model.fit(x_train, y_train, epochs=20, batch_size=16,
                        validation_data=(x_test, y_test), verbose=0)
    assert (history.history['val_loss'][-1] < 0.9)


if __name__ == '__main__':
    pytest.main([__file__])
