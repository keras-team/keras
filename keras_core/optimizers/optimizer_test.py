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

    def test_get_method(self):
        obj = optimizers.get("sgd")
        self.assertIsInstance(obj, optimizers.SGD)
        obj = optimizers.get("adamw")
        self.assertIsInstance(obj, optimizers.AdamW)

        obj = optimizers.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            optimizers.get("typo")

    def test_static_loss_scaling(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) * 1024.0
        optimizer = optimizers.SGD(learning_rate=1.0, loss_scale_factor=1024.0)
        optimizer.apply_gradients([(grads, v)])
        self.assertEqual(optimizer.scale_loss(1.0), 1024.0)
        self.assertAllClose(v, [[0.0, 0.0], [0.0, 0.0]])

    def test_set_weights(self):
        x = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer_1 = optimizers.Adam()
        grads = backend.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        optimizer_1.apply_gradients(zip([grads], [x]))
        optimizer_2 = optimizers.Adam()
        with self.assertRaisesRegex(ValueError, "You are calling*"):
            optimizer_2.set_weights(optimizer_1.variables)
        optimizer_2.build([x])
        optimizer_2.set_weights(optimizer_1.variables)
        for i in range(len(optimizer_1.variables)):
            self.assertAllClose(
                optimizer_1.variables[i],
                optimizer_2.variables[i],
            )
