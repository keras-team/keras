import pytest
import numpy as np

from keras.utils.test_utils import get_test_data
from keras.utils import np_utils
from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

np.random.seed(1337)

input_dim = 10
nb_class = 3
batch_size = 32
nb_epoch = 1
verbosity = 0
optim = 'adam'
loss = 'categorical_crossentropy'


(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=400,
                                                     nb_test=200,
                                                     input_shape=(input_dim,),
                                                     classification=True,
                                                     nb_class=nb_class)
y_train = np_utils.to_categorical(y_train, nb_classes=nb_class)
y_test = np_utils.to_categorical(y_test, nb_classes=nb_class)


(X_train_reg, y_train_reg), (X_test_reg, y_test_reg) = get_test_data(nb_train=400,
                                                                     nb_test=200,
                                                                     input_shape=(input_dim,),
                                                                     classification=False,
                                                                     nb_class=1,
                                                                     output_shape=(1,))


@pytest.mark.skipif(K._BACKEND=='tensorflow', reason="currently not working with TensorFlow")
def test_keras_classifier():
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    sklearn_clf = KerasClassifier(model, optimizer=optim, loss=loss,
                                  train_batch_size=batch_size,
                                  test_batch_size=batch_size,
                                  nb_epoch=nb_epoch)
    sklearn_clf.fit(X_train, y_train)
    sklearn_clf.score(X_test, y_test)


@pytest.mark.skipif(K._BACKEND=='tensorflow', reason="currently not working with TensorFlow")
def test_keras_regressor():
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    sklearn_regressor = KerasRegressor(model, optimizer=optim, loss=loss,
                                       train_batch_size=batch_size,
                                       test_batch_size=batch_size,
                                       nb_epoch=nb_epoch)
    sklearn_regressor.fit(X_train_reg, y_train_reg)
    sklearn_regressor.score(X_test_reg, y_test_reg)
