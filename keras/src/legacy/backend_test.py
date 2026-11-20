import numpy as np
import pytest

from keras.src import backend
from keras.src import testing
from keras.src.legacy import backend as legacy_backend


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Legacy backend requires TensorFlow.",
)
class LegacyBinaryCrossentropyTest(testing.TestCase):
    def test_matching_shapes(self):
        target = np.array([[1.0, 0.0, 1.0]], "float32")
        output = np.array([[0.9, 0.2, 0.6]], "float32")
        result = legacy_backend.binary_crossentropy(target, output)
        expected = -(
            target * np.log(output) + (1 - target) * np.log(1 - output)
        )
        self.assertAllClose(result, expected, atol=1e-5)

    def test_matching_shapes_from_logits(self):
        target = np.array([[1.0, 0.0, 1.0]], "float32")
        logits = np.array([[2.0, -1.0, 0.5]], "float32")
        result = legacy_backend.binary_crossentropy(
            target, logits, from_logits=True
        )
        probs = 1.0 / (1.0 + np.exp(-logits))
        expected = -(target * np.log(probs) + (1 - target) * np.log(1 - probs))
        self.assertAllClose(result, expected, atol=1e-5)

    def test_rank_mismatch_raises(self):
        target = np.ones((3,), "float32")
        output = np.full((1, 3), 0.5, "float32")
        with self.assertRaisesRegex(ValueError, "must have the same rank"):
            legacy_backend.binary_crossentropy(target, output)

    def test_shape_mismatch_raises(self):
        target = np.ones((2, 1), "float32")
        output = np.full((2, 3), 0.5, "float32")
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            legacy_backend.binary_crossentropy(target, output)

    def test_shape_mismatch_raises_from_logits(self):
        target = np.ones((2, 1), "float32")
        output = np.full((2, 3), 0.5, "float32")
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            legacy_backend.binary_crossentropy(target, output, from_logits=True)
