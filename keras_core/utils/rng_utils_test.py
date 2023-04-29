import numpy as np
import tensorflow as tf

import keras_core
from keras_core.testing import test_case
from keras_core.utils import rng_utils


class TestRandomSeedSetting(test_case.TestCase):
    def test_set_random_seed(self):
        def get_model_output():
            model = keras_core.Sequential(
                [
                    keras_core.layers.Dense(10),
                    keras_core.layers.Dropout(0.5),
                    keras_core.layers.Dense(10),
                ]
            )
            x = np.random.random((32, 10)).astype("float32")
            ds = tf.data.Dataset.from_tensor_slices(x).shuffle(32).batch(16)
            return model.predict(ds)

        rng_utils.set_random_seed(42)
        y1 = get_model_output()
        rng_utils.set_random_seed(42)
        y2 = get_model_output()
        self.assertAllClose(y1, y2)
