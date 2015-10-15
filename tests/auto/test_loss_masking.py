import numpy as np
np.random.seed(1337)

import unittest
from keras.models import Sequential, weighted_objective
from keras.layers.core import TimeDistributedDense, Masking
from keras import objectives
import theano


class TestLossMasking(unittest.TestCase):
    def test_loss_masking(self):
        X = np.array(
            [[[1, 1], [2, 1], [3, 1], [5, 5]],
             [[1, 5], [5, 0], [0, 0], [0, 0]]], dtype=np.int32)
        model = Sequential()
        model.add(Masking(mask_value=0, input_shape=(None, 2)))
        model.add(TimeDistributedDense(1, init='one'))
        model.compile(loss='mse', optimizer='sgd')
        y = model.predict(X)
        loss = model.fit(X, 4*y, nb_epoch=1, batch_size=2, verbose=1).history['loss'][0]
        assert loss == 285.

    def test_loss_masking_time(self):
        theano.config.mode = 'FAST_COMPILE'
        weighted_loss = weighted_objective(objectives.get('categorical_crossentropy'))
        shape = (3, 4, 2)
        X = np.arange(24).reshape(shape)
        Y = 2 * X

        weights = np.ones((3, 4, 1))  # Normally the trailing 1 is added by standardize_weights
        weights[0, 0] = 0
        mask = np.ones((3, 4))
        mask[1, 0] = 0

        out = weighted_loss(X, Y, weights, mask).eval()
        weights[0, 0] = 1e-9  # so that nonzero() doesn't remove this weight
        out2 = weighted_loss(X, Y, weights, mask).eval()
        print(out)
        print(out2)
        assert abs(out - out2) < 1e-8


if __name__ == '__main__':
    print('Test loss masking')
    unittest.main()
