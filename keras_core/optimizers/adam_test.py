from keras_core import backend
from keras_core import operations as ops
from keras_core import testing
from keras_core.optimizers.adam import Adam
import numpy as np

class AdamTest(testing.TestCase):
    def test_config(self):
        # TODO
        pass

    def test_weight_decay(self):
        grads, var1, var2, var3 = (
            np.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
            backend.Variable(2.0),
        )
        optimizer_1 = Adam(learning_rate=1, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = Adam(learning_rate=1, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        optimizer_3 = Adam(learning_rate=1, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))

        self.assertAlmostEqual(var1, 1.9760959)
        self.assertAlmostEqual(var2, 2.0)
        self.assertAlmostEqual(var3, 2.0)

    def test_clip_norm(self):
        optimizer = Adam(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Adam(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])

