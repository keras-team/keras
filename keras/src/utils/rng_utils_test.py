import numpy as np

import keras
from keras.src import backend
from keras.src.testing import test_case
from keras.src.utils import rng_utils


class TestRandomSeedSetting(test_case.TestCase):
    def test_set_random_seed_with_seed_generator(self):
        def get_model_output():
            model = keras.Sequential(
                [
                    keras.layers.Dense(10),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(10),
                ]
            )
            x = np.random.random((32, 10)).astype("float32")
            return model.predict(x, batch_size=16)

        rng_utils.set_random_seed(42)
        y1 = get_model_output()

        # Second call should produce different results.
        y2 = get_model_output()
        self.assertNotAllClose(y1, y2)

        # Re-seeding should produce the same results as the first time.
        rng_utils.set_random_seed(42)
        y3 = get_model_output()
        self.assertAllClose(y1, y3)

        # Re-seeding with a different seed should produce different results.
        rng_utils.set_random_seed(1337)
        y4 = get_model_output()
        self.assertNotAllClose(y1, y4)

    def test_set_random_seed_with_global_seed_generator(self):
        rng_utils.set_random_seed(42)
        y1 = backend.random.randint((32, 10), minval=0, maxval=1000)

        # Second call should produce different results.
        y2 = backend.random.randint((32, 10), minval=0, maxval=1000)
        self.assertNotAllClose(y1, y2)

        # Re-seeding should produce the same results as the first time.
        rng_utils.set_random_seed(42)
        y3 = backend.random.randint((32, 10), minval=0, maxval=1000)
        self.assertAllClose(y1, y3)

        # Re-seeding with a different seed should produce different results.
        rng_utils.set_random_seed(1337)
        y4 = backend.random.randint((32, 10), minval=0, maxval=1000)
        self.assertNotAllClose(y1, y4)
