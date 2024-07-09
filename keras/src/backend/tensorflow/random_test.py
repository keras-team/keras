import pytest
import tensorflow as tf

from keras.src import backend
from keras.src.backend.tensorflow import random
from keras.src.testing import TestCase


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Only applies to TensorFlow random ops.",
)
class TFRandomTest(TestCase):

    def test_categorical(self):
        inputs = tf.ones([2, 3], dtype="float32")
        outputs = random.categorical(inputs, 2, seed=42)
        expected = tf.constant([[0, 2], [1, 0]])
        self.assertAllClose(outputs, expected)

    def test_categorical_seed_cast(self):
        inputs = tf.ones([2, 3], dtype="float32")
        seed = tf.int32.max + 1000
        outputs_mod = random.categorical(inputs, 2, seed=seed)
        outputs_nomod = random.categorical(inputs, 2, seed=1001)
        self.assertAllClose(outputs_mod, outputs_nomod)
