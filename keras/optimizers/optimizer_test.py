import numpy as np

from keras import backend
from keras import constraints
from keras import optimizers
from keras import testing


class OptimizerTest(testing.TestCase):
    def test_iterations_counter(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.Adam(learning_rate=1.0)
        self.assertAllClose(optimizer.iterations, 0)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(optimizer.iterations, 1)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(optimizer.iterations, 2)

    def test_ema(self):
        v = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
        grads = backend.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.SGD(
            learning_rate=1.0,
            use_ema=True,
            ema_momentum=0.9,
            ema_overwrite_frequency=3,
        )
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[2.0, 3.0], [4.0, 5.0]])
        self.assertAllClose(
            optimizer._model_variables_moving_average[0],
            [[2.0, 3.0], [4.0, 5.0]],
        )
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.71, 2.71], [3.71, 4.71]])

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

    def test_gradient_accumulation(self):
        v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        grads = backend.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        optimizer = optimizers.SGD(
            learning_rate=1.0, gradient_accumulation_steps=3
        )
        self.assertEqual(optimizer.gradient_accumulation_steps, 3)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[1.0, 1.0], [1.0, 1.0]]
        )
        self.assertAllClose(optimizer.iterations, 1)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[2.0, 2.0], [2.0, 2.0]]
        )
        self.assertAllClose(optimizer.iterations, 2)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[0.0, 1.0], [2.0, 3.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[0.0, 0.0], [0.0, 0.0]]
        )
        self.assertAllClose(optimizer.iterations, 3)
        optimizer.apply_gradients([(grads, v)])
        self.assertAllClose(v, [[0.0, 1.0], [2.0, 3.0]])
        self.assertAllClose(
            optimizer._accumulated_gradients[0], [[1.0, 1.0], [1.0, 1.0]]
        )
        self.assertAllClose(optimizer.iterations, 4)
