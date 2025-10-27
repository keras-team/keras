import numpy as np

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.losses.lpips import LPIPS, lpips


def _tiny_feature_model():
    inp = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    y = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    return models.Model(inp, [x, y])


class LPIPSTest(testing.TestCase):
    def test_identical_images_zero(self):
        fm = _tiny_feature_model()
        loss = LPIPS(feature_model=fm, reduction=None)
        x = np.random.RandomState(0).rand(2, 32, 32, 3).astype("float32")
        y = x.copy()
        out = loss(x, y)
        # Exactly zero can be achieved with identical inputs
        self.assertAllClose(out, np.zeros((2,), dtype=np.float32), atol=1e-6)

    def test_basic_increase_with_noise(self):
        fm = _tiny_feature_model()
        x = np.zeros((2, 16, 16, 3), dtype="float32")
        y = np.zeros((2, 16, 16, 3), dtype="float32")
        # Add small noise to y
        y[0] += 0.1
        # Functional API
        d = lpips(x, y, feature_model=fm)
        self.assertTrue(d.shape == (2,))
        self.assertGreater(d[0], d[1])

    def test_reduction(self):
        fm = _tiny_feature_model()
        loss = LPIPS(feature_model=fm, reduction="sum")
        x = np.random.RandomState(1).rand(4, 8, 8, 3).astype("float32")
        y = np.random.RandomState(2).rand(4, 8, 8, 3).astype("float32")
        out = loss(x, y)
        # Scalar reduction
        self.assertEqual(out.shape, ())
