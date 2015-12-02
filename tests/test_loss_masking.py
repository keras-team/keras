import numpy as np
np.random.seed(1337)

import unittest
from keras.models import Sequential, weighted_objective
from keras.layers.core import TimeDistributedDense, Masking
from keras import objectives
from keras import backend as K


class TestMasking(unittest.TestCase):
    def test_masking(self):
        if K._BACKEND == 'tensorflow':
            # skip this test for TF: not supported yet
            return
        X = np.array(
            [[[1, 1], [2, 1], [3, 1], [5, 5]],
             [[1, 5], [5, 0], [0, 0], [0, 0]]], dtype=np.int32)
        model = Sequential()
        model.add(Masking(mask_value=0, input_shape=(None, 2)))
        model.add(TimeDistributedDense(1, init='one'))
        model.compile(loss='mse', optimizer='sgd')
        y = model.predict(X)
        history = model.fit(X, 4 * y, nb_epoch=1, batch_size=2, verbose=1)
        assert history.history['loss'][0] == 285.

    def test_loss_masking(self):
        weighted_loss = weighted_objective(objectives.get('mae'))
        shape = (3, 4, 2)
        X = np.arange(24).reshape(shape)
        Y = 2 * X

        # Normally the trailing 1 is added by standardize_weights
        weights = np.ones((3,))
        mask = np.ones((3, 4))
        mask[1, 0] = 0

        out = K.eval(weighted_loss(K.variable(X),
                                   K.variable(Y),
                                   K.variable(weights),
                                   K.variable(mask)))
        print(out)


if __name__ == '__main__':
    print('Test masking')
    unittest.main()
