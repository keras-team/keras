import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.constraints import unitnorm
from keras import backend as K


X1 = np.array([[1], [2]], dtype='int32')
W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype='float32')


def test_unitnorm_constraint():
    lookup = Sequential()
    lookup.add(Embedding(3, 2, weights=[W1],
                         W_constraint=unitnorm(),
                         input_length=1))
    lookup.add(Flatten())
    lookup.add(Dense(1))
    lookup.add(Activation('sigmoid'))
    lookup.compile(loss='binary_crossentropy', optimizer='sgd',
                   class_mode='binary')
    lookup.train_on_batch(X1, np.array([[1], [0]], dtype='int32'))
    norm = np.linalg.norm(K.get_value(lookup.params[0]), axis=1)
    assert_allclose(norm, np.ones_like(norm).astype('float32'), rtol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__])
