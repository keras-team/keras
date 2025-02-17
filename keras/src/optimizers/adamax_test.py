# flake8: noqa


import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.adamax import Adamax


class AdamaxTest(testing.TestCase):
    def test_config(self):
        optimizer = Adamax(
            learning_rate=0.5,
            beta_1=0.8,
            beta_2=0.95,
            epsilon=1e-5,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = Adamax(learning_rate=0.5)
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(vars, [0.5, 1.5, 2.5, 3.5], rtol=1e-4, atol=1e-4)

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            ops.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = Adamax(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = Adamax(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = Adamax(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        optimizer = Adamax(
            learning_rate=0.2, beta_1=0.85, beta_2=0.95, epsilon=1e-6
        )

        x = backend.Variable(np.ones([10]))
        grads = ops.arange(0.1, 1.1, 0.1)
        first_grads = ops.full((10,), 0.01)

        # fmt: off
        golden = np.array(
            [[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [0.6827, 0.6873, 0.6888, 0.6896, 0.6901, 0.6904, 0.6906, 0.6908, 0.6909, 0.691],
            [0.5333, 0.5407, 0.5431, 0.5444, 0.5451, 0.5456, 0.546, 0.5462, 0.5464, 0.5466],
            [0.368, 0.3773, 0.3804, 0.382, 0.3829, 0.3835, 0.384, 0.3843, 0.3846, 0.3848],
            [0.1933, 0.204, 0.2076, 0.2094, 0.2105, 0.2112, 0.2117, 0.2121, 0.2124, 0.2126]]
        )
        # fmt: on

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        optimizer = Adamax(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Adamax(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
