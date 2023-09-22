# flake8: noqa


import numpy as np

from keras import backend
from keras import testing
from keras.optimizers.ftrl import Ftrl


class FtrlTest(testing.TestCase):
    def test_config(self):
        optimizer = Ftrl(
            learning_rate=0.05,
            learning_rate_power=-0.2,
            initial_accumulator_value=0.4,
            l1_regularization_strength=0.05,
            l2_regularization_strength=0.15,
            l2_shrinkage_regularization_strength=0.01,
            beta=0.3,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = Ftrl(learning_rate=0.5)
        grads = np.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(
            vars, [0.2218, 1.3954, 2.3651, 2.8814], rtol=1e-4, atol=1e-4
        )

    def test_correctness_with_golden(self):
        optimizer = Ftrl(
            learning_rate=0.05,
            learning_rate_power=-0.2,
            initial_accumulator_value=0.4,
            l1_regularization_strength=0.05,
            l2_regularization_strength=0.15,
            l2_shrinkage_regularization_strength=0.01,
            beta=0.3,
        )

        x = backend.Variable(np.ones([10]))
        grads = np.arange(0.1, 1.1, 0.1)
        first_grads = np.full((10,), 0.01)

        # fmt: off
        golden = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0034, -0.0077, -0.0118, -0.0157, -0.0194, -0.023, -0.0263, -0.0294, -0.0325, -0.0354],
            [-0.0078, -0.0162, -0.0242, -0.0317, -0.0387, -0.0454, -0.0516, -0.0575, -0.0631, -0.0685],
            [-0.0121, -0.0246, -0.0363, -0.0472, -0.0573, -0.0668, -0.0757, -0.0842, -0.0922, -0.0999],
            [-0.0164, -0.0328, -0.0481, -0.0623, -0.0753, -0.0875, -0.099, -0.1098, -0.1201, -0.1299]]
        )
        # fmt: on

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        optimizer = Ftrl(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Ftrl(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
