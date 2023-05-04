import numpy as np

from keras_core import backend
from keras_core import constraints
from keras_core import optimizers
from keras_core import testing


class OptimizerTest(testing.TestCase):
    def test_constraints_are_applied(self):
        v = backend.Variable(np.random.random((2, 2)) - 1.0)
        v.constraint = constraints.NonNeg()
        optimizer = optimizers.SGD(learning_rate=0.0001)
        grad = backend.numpy.zeros((2, 2))
        optimizer.apply_gradients([(grad, v)])
        self.assertAlmostEqual(np.min(v), 0.0)
