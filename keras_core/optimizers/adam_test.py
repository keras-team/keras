import numpy as np

from keras_core import backend
from keras_core import operations as ops
from keras_core import testing
from keras_core.optimizers.adam import Adam


class AdamTest(testing.TestCase):
    def test_config(self):
        optimizer = Adam(
            learning_rate=0.5,
            beta_1=0.5,
            beta_2=0.67,
            epsilon=1e-5,
            amsgrad=True,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = Adam(learning_rate=0.5)
        grads = np.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(vars, [0.5, 1.5, 2.5, 3.5], rtol=1e-4, atol=1e-4)

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            np.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = Adam(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = Adam(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = Adam(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_clip_norm(self):
        # TODO: implement clip_gradients, then uncomment
        pass

    #     optimizer = Adam(clipnorm=1)
    #     grad = [np.array([100.0, 100.0])]
    #     clipped_grad = optimizer._clip_gradients(grad)
    #     self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        # TODO: implement clip_gradients, then uncomment
        pass

    #     optimizer = Adam(clipvalue=1)
    #     grad = [np.array([100.0, 100.0])]
    #     clipped_grad = optimizer._clip_gradients(grad)
    #     self.assertAllClose(clipped_grad[0], [1.0, 1.0])
