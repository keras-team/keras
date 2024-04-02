# flake8: noqa


import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.adagrad import Adagrad


class AdagradTest(testing.TestCase):
    def test_config(self):
        optimizer = Adagrad(
            learning_rate=0.5,
            initial_accumulator_value=0.2,
            epsilon=1e-5,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = Adagrad(learning_rate=0.5)
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(
            vars, [0.5233, 1.5007, 2.5005, 3.5061], rtol=1e-4, atol=1e-4
        )

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            ops.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = Adagrad(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = Adagrad(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = Adagrad(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        optimizer = Adagrad(
            learning_rate=0.2, initial_accumulator_value=0.3, epsilon=1e-6
        )

        x = backend.Variable(np.ones([10]))
        grads = ops.arange(0.1, 1.1, 0.1)
        first_grads = ops.full((10,), 0.01)

        # fmt: off
        golden = np.array(
            [[0.9963, 0.9963, 0.9963, 0.9963, 0.9963, 0.9963, 0.9963, 0.9963, 0.9963, 0.9963],
            [0.9604, 0.9278, 0.9003, 0.8784, 0.8615, 0.8487, 0.8388, 0.8313, 0.8255, 0.8209],
            [0.9251, 0.8629, 0.8137, 0.7768, 0.7497, 0.7298, 0.7151, 0.704, 0.6956, 0.6891],
            [0.8903, 0.8012, 0.7342, 0.6862, 0.6521, 0.6277, 0.6099, 0.5967, 0.5867, 0.579],
            [0.856, 0.7422, 0.6604, 0.6037, 0.5644, 0.5367, 0.5168, 0.5021, 0.491, 0.4825]]
        )
        # fmt: on

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        optimizer = Adagrad(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Adagrad(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
