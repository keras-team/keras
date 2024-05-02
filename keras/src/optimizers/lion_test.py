import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.lion import Lion


class LionTest(testing.TestCase):
    def test_config(self):
        optimizer = Lion(
            learning_rate=0.5,
            beta_1=0.5,
            beta_2=0.67,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = Lion(learning_rate=0.5)
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
        optimizer_1 = Lion(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = Lion(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = Lion(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        optimizer = Lion()

        x = backend.Variable(np.ones([10]))
        grads = ops.arange(0.1, 1.1, 0.1)
        first_grads = ops.full((10,), 0.01)

        golden = np.tile(
            [[0.999], [0.998], [0.997], [0.996], [0.995]],
            (1, 10),
        )

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        optimizer = Lion(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Lion(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])

    @pytest.mark.requires_trainable_backend
    def test_ema(self):
        # TODO: test correctness
        model = keras.Sequential([keras.layers.Dense(10)])
        model.compile(optimizer=Lion(use_ema=True), loss="mse")
        x = keras.ops.zeros((1, 5))
        y = keras.ops.zeros((1, 10))
        model.fit(x, y)
