# flake8: noqa

import numpy as np

from keras_core import backend
from keras_core import operations as ops
from keras_core import testing
from keras_core.optimizers.sgd import SGD


class SGDTest(testing.TestCase):
    def test_config(self):
        optimizer = SGD(
            learning_rate=0.5,
            momentum=0.06,
            nesterov=True,
            weight_decay=0.004,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = SGD(learning_rate=0.5)
        grads = np.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(vars, [0.5, -1.0, -0.5, 3.0], rtol=1e-4, atol=1e-4)

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            np.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = SGD(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = SGD(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = SGD(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        optimizer = SGD(nesterov=True)

        x = backend.Variable(np.ones([10]))
        grads = np.arange(0.1, 1.1, 0.1)
        first_grads = np.full((10,), 0.01)

        # fmt: off
        golden = np.array(
            [[0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999,
            0.9999, 0.9999], [0.9989, 0.9979, 0.9969, 0.9959, 0.9949, 0.9939,
            0.9929, 0.9919, 0.9909, 0.9899], [0.9979, 0.9959, 0.9939, 0.9919,
            0.9899, 0.9879, 0.9859, 0.9839, 0.9819, 0.9799], [0.9969, 0.9939,
            0.9909, 0.9879, 0.9849, 0.9819, 0.9789, 0.9759, 0.9729, 0.9699],
            [0.9959, 0.9919, 0.9879, 0.9839, 0.9799, 0.9759, 0.9719, 0.9679,
            0.9639, 0.9599]]
        )
        # fmt: on

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))
