# flake8: noqa


import numpy as np

from keras import backend
from keras import testing
from keras.optimizers.adafactor import Adafactor


class AdafactorTest(testing.TestCase):
    def test_config(self):
        optimizer = Adafactor(
            learning_rate=0.5,
            beta_2_decay=-0.65,
            epsilon_1=1e-15,
            epsilon_2=1e-4,
            clip_threshold=0.9,
            relative_step=False,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step_1d(self):
        optimizer = Adafactor(learning_rate=0.5)
        grads = np.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(
            vars, [-0.3693, 0.6307, 1.6307, 2.6307], rtol=1e-4, atol=1e-4
        )

    def test_single_step_2d(self):
        optimizer = Adafactor(learning_rate=0.5)
        grads = np.array([[1.0, 6.0], [7.0, 2.0]])
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(
            vars, [[0.7007, -0.0081], [1.2492, 3.4407]], rtol=1e-4, atol=1e-4
        )

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            np.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = Adafactor(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = Adafactor(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = Adafactor(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        optimizer = Adafactor(
            learning_rate=0.5,
            beta_2_decay=-0.65,
            epsilon_1=1e-15,
            epsilon_2=1e-4,
            clip_threshold=0.9,
            relative_step=False,
        )

        x = backend.Variable(np.ones([10]))
        grads = np.arange(0.1, 1.1, 0.1)
        first_grads = np.full((10,), 0.01)

        # fmt: off
        golden = np.array(
            [[0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
            [0.3031, 0.3026, 0.3025, 0.3024, 0.3024, 0.3024, 0.3024, 0.3024, 0.3024, 0.3024],
            [0.1671, 0.1665, 0.1663, 0.1663, 0.1663, 0.1663, 0.1663, 0.1663, 0.1663, 0.1663],
            [0.0923, 0.0916, 0.0915, 0.0914, 0.0914, 0.0914, 0.0914, 0.0914, 0.0914, 0.0914],
            [0.0554, 0.0548, 0.0546, 0.0546, 0.0546, 0.0546, 0.0546, 0.0545, 0.0545, 0.0545]]
        )
        # fmt: on

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        optimizer = Adafactor(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Adafactor(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
