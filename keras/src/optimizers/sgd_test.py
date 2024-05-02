# flake8: noqa

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.sgd import SGD


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
        self.assertEqual(len(optimizer.variables), 2)
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.build([vars])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(vars, [0.5, -1.0, -0.5, 3.0], rtol=1e-4, atol=1e-4)
        self.assertEqual(len(optimizer.variables), 2)
        self.assertEqual(optimizer.variables[0], 1)
        self.assertEqual(optimizer.variables[1], 0.5)

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            ops.zeros(()),
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
        grads = ops.arange(0.1, 1.1, 0.1)
        first_grads = ops.full((10,), 0.01)

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

    def test_clip_norm(self):
        optimizer = SGD(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = SGD(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
