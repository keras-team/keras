from __future__ import print_function
import pytest

from keras.utils.test_utils import get_test_data
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical


(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000,
                                                     nb_test=200,
                                                     input_shape=(10,),
                                                     classification=True,
                                                     nb_class=2)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def get_model(input_dim, nb_hidden, output_dim):
    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


def _test_optimizer(optimizer, target=0.89):
    model = get_model(X_train.shape[1], 10, y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16,
                        validation_data=(X_test, y_test), verbose=2)
    config = optimizer.get_config()
    assert type(config) == dict
    assert history.history['val_acc'][-1] >= target


def test_sgd():
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    _test_optimizer(sgd)


def test_rmsprop():
    _test_optimizer(RMSprop())


def test_adagrad():
    _test_optimizer(Adagrad())


def test_adadelta():
    _test_optimizer(Adadelta())


def test_adam():
    _test_optimizer(Adam())


def test_adamax():
    _test_optimizer(Adamax())


def test_nadam():
    _test_optimizer(Nadam())


if __name__ == '__main__':
    pytest.main([__file__])
