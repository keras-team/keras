from __future__ import absolute_import
from __future__ import print_function
import pytest

from keras.models import Model
from keras.layers import Dense, Input
from keras.utils.test_utils import keras_test
from keras import backend as K
from keras.backend import theano_backend as KTH
from keras.backend import tensorflow_backend as KTF

import scipy.sparse as sparse
import numpy as np
np.random.seed(1337)


input_dim = 16
nb_hidden = 8
nb_class = 4
batch_size = 32
nb_epoch = 1


def do_sparse():
    return K == KTF or KTH.th_sparse_module


@keras_test
def test_sparse_mlp():
    if not do_sparse():
        return

    input = Input(batch_shape=(None, input_dim), sparse=True)
    hidden = Dense(nb_hidden, activation='relu')(input)
    hidden = Dense(nb_hidden, activation='relu')(hidden)
    predictions = Dense(nb_class, activation='sigmoid')(hidden)
    model = Model(input=[input], output=predictions)
    model.compile(loss='mse', optimizer='sgd')
    x = sparse.rand(batch_size, input_dim, density=0.1, format='csr')
    y = np.random.random((batch_size, nb_class))
    model.fit(x, y, nb_epoch=1)
