from __future__ import print_function
import pytest
import numpy as np

from keras.utils import test_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical


def get_test_data():
    np.random.seed(1337)
    (x_train, y_train), _ = test_utils.get_test_data(num_train=1000,
                                                     num_test=200,
                                                     input_shape=(10,),
                                                     classification=True,
                                                     num_classes=2)
    y_train = to_categorical(y_train)
    return x_train, y_train


def get_model(input_dim, num_hidden, output_dim):
    model = Sequential()
    model.add(Dense(num_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


def _test_optimizer(optimizer, target=0.75):
    x_train, y_train = get_test_data()
    model = get_model(x_train.shape[1], 10, y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    assert history.history['acc'][-1] >= target
    config = optimizers.serialize(optimizer)
    optim = optimizers.deserialize(config)
    new_config = optimizers.serialize(optim)
    new_config['class_name'] = new_config['class_name'].lower()
    assert config == new_config


def test_sgd():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    _test_optimizer(sgd)


def test_rmsprop():
    _test_optimizer(optimizers.RMSprop())
    _test_optimizer(optimizers.RMSprop(decay=1e-3))


def test_adagrad():
    _test_optimizer(optimizers.Adagrad())
    _test_optimizer(optimizers.Adagrad(decay=1e-3))


def test_adadelta():
    _test_optimizer(optimizers.Adadelta(), target=0.6)
    _test_optimizer(optimizers.Adadelta(decay=1e-3), target=0.6)


def test_adam():
    _test_optimizer(optimizers.Adam())
    _test_optimizer(optimizers.Adam(decay=1e-3))


def test_adamax():
    _test_optimizer(optimizers.Adamax())
    _test_optimizer(optimizers.Adamax(decay=1e-3))


def test_nadam():
    _test_optimizer(optimizers.Nadam())


def test_clipnorm():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=0.5)
    _test_optimizer(sgd)


def test_clipvalue():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    _test_optimizer(sgd)


if __name__ == '__main__':
    pytest.main([__file__])
