# flake8: noqa

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.eager import context

from keras import backend
from keras import testing
from keras.optimizers.sgd import SGD


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The distribute test can only run with TF backend.",
)
class OptimizerDistributeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Need at least 2 devices for distribution related tests.
        cpus = tf.config.list_physical_devices("CPU")
        context._reset_context()
        tf.config.set_logical_device_configuration(
            cpus[0],
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration(),
            ],
        )
        self.strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

    def test_config(self):
        with self.strategy.scope():
            optimizer = SGD(
                learning_rate=0.5,
                momentum=0.06,
                nesterov=True,
                weight_decay=0.004,
            )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        with self.strategy.scope():
            optimizer = SGD(
                learning_rate=0.5,
                momentum=0.06,
            )
            grads = tf.constant([1.0, 6.0, 7.0, 2.0])
            vars = backend.Variable([1.0, 2.0, 3.0, 4.0])

            self.strategy.run(
                lambda: optimizer.apply_gradients(zip([grads], [vars]))
            )
            self.assertAllClose(
                vars, [0.5, -1.0, -0.5, 3.0], rtol=1e-4, atol=1e-4
            )

    def test_weight_decay(self):
        with self.strategy.scope():
            grads, var1, var2, var3 = (
                tf.zeros(()),
                backend.Variable(2.0),
                backend.Variable(3.0, name="exclude"),
                backend.Variable(4.0),
            )
            optimizer_1 = SGD(learning_rate=1.0, weight_decay=0.004)
            self.strategy.run(
                lambda: optimizer_1.apply_gradients(zip([grads], [var1]))
            )

            optimizer_2 = SGD(learning_rate=1.0, weight_decay=0.004)

            def opt2_run():
                optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
                optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

            self.strategy.run(opt2_run)

            optimizer_3 = SGD(learning_rate=1.0, weight_decay=0.004)

            def opt3_run():
                optimizer_3.exclude_from_weight_decay(var_list=[var3])
                optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

            self.strategy.run(opt3_run)

        self.assertAlmostEqual(var1.numpy(), 1.9760959)
        self.assertAlmostEqual(var2.numpy(), 3.0)
        self.assertAlmostEqual(var3.numpy(), 4.0)

    def test_correctness_with_golden(self):
        with self.strategy.scope():
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

        self.strategy.run(
            lambda: optimizer.apply_gradients(zip([first_grads], [x]))
        )
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            self.strategy.run(
                lambda: optimizer.apply_gradients(zip([grads], [x]))
            )

    def test_clip_norm(self):
        with self.strategy.scope():
            optimizer = SGD(clipnorm=1)
            grad = [np.array([100.0, 100.0])]
            clipped_grad = optimizer._clip_gradients(grad)
            self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        with self.strategy.scope():
            optimizer = SGD(clipvalue=1)
            grad = [np.array([100.0, 100.0])]
            clipped_grad = optimizer._clip_gradients(grad)
            self.assertAllClose(clipped_grad[0], [1.0, 1.0])

    def test_stateless_not_supported(self):
        optimizer = SGD(learning_rate=0.5)
        grads = [np.array([1.0, 6.0, 7.0, 2.0])]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        with self.assertRaisesRegex(ValueError, "not supported"):
            optimizer.stateless_apply(optimizer.variables, grads, vars)

    def test_ema(self):
        with self.strategy.scope():
            v = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
            grads = backend.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
            optimizer = SGD(
                learning_rate=1.0,
                use_ema=True,
                ema_momentum=0.9,
                ema_overwrite_frequency=3,
            )
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[2.0, 3.0], [4.0, 5.0]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[2.0, 3.0], [4.0, 5.0]],  # initialized after first step
            )
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[1.9, 2.9], [3.9, 4.9]],
            )
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            # Variables were overwritten with EMA
            self.assertAllClose(v, [[1.71, 2.71], [3.71, 4.71]])
            self.assertAllClose(
                optimizer._model_variables_moving_average[0],
                [[1.71, 2.71], [3.71, 4.71]],
            )

    def test_gradient_accumulation(self):
        with self.strategy.scope():
            v = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
            grads = backend.convert_to_tensor([[1.0, 1.0], [2.0, 2.0]])
            optimizer = SGD(learning_rate=1.0, gradient_accumulation_steps=3)
            self.assertEqual(optimizer.gradient_accumulation_steps, 3)
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._accumulated_gradients[0], [[1.0, 1.0], [2.0, 2.0]]
            )
            self.assertAllClose(optimizer.iterations, 1)
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[1.0, 2.0], [3.0, 4.0]])
            self.assertAllClose(
                optimizer._accumulated_gradients[0], [[2.0, 2.0], [4.0, 4.0]]
            )
            self.assertAllClose(optimizer.iterations, 2)
            self.strategy.run(lambda: optimizer.apply_gradients([(grads, v)]))
            self.assertAllClose(v, [[0.0, 1.0], [1.0, 2.0]])
            self.assertAllClose(
                optimizer._accumulated_gradients[0], [[0.0, 0.0], [0.0, 0.0]]
            )
            self.assertAllClose(optimizer.iterations, 3)
