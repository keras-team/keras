# flake8: noqa


import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.adamw import AdamW


class AdamWTest(testing.TestCase):
    def test_config(self):
        optimizer = AdamW(
            learning_rate=0.5,
            weight_decay=0.008,
            beta_1=0.5,
            beta_2=0.67,
            epsilon=1e-5,
            amsgrad=True,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = AdamW(learning_rate=0.5)
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(
            vars, [0.4980, 1.4960, 2.494, 3.492], rtol=1e-4, atol=1e-4
        )

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            ops.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = AdamW(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = AdamW(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = AdamW(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        optimizer = AdamW(learning_rate=1.0, weight_decay=0.5, epsilon=2)

        x = backend.Variable(np.ones([10]))
        grads = ops.arange(0.1, 1.1, 0.1)
        first_grads = ops.full((10,), 0.01)

        # fmt: off
        golden = np.array(
            [[0.4998, 0.4998, 0.4998, 0.4998, 0.4998, 0.4998, 0.4998, 0.4998, 0.4998, 0.4998],
            [0.2486, 0.2475, 0.2463, 0.2451, 0.244, 0.2428, 0.2417, 0.2405, 0.2394, 0.2382],
            [0.1223, 0.1198, 0.1174, 0.1149, 0.1124, 0.11, 0.1075, 0.1051, 0.1027, 0.1003],
            [0.0586, 0.0549, 0.0512, 0.0475, 0.0439, 0.0402, 0.0366, 0.033, 0.0294, 0.0258],
            [0.0263, 0.0215, 0.0167, 0.012, 0.0073, 0.0026, -0.0021, -0.0067, -0.0113, -0.0159]]
        )
        # fmt: on

        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=5e-4, atol=5e-4)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        optimizer = AdamW(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = AdamW(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
