import os

import numpy as np
from absl.testing import parameterized

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf

import keras
from keras.src import backend
from keras.src import testing
from keras.src.utils import backend_utils


def dict_input_fn(inputs):
    outputs = dict(inputs)
    outputs["x"] = inputs["x"][:, 0]
    outputs["y"] = inputs["y"] + 1
    return outputs


def list_input_fn(inputs):
    outputs = [x**2 for x in inputs]
    return outputs


class NestedTest(testing.TestCase):
    def setUp(self):
        self.xs = {
            "x": tf.convert_to_tensor(
                np.random.rand(4, 100, 3), dtype=tf.float32
            ),
            "y": tf.convert_to_tensor(
                np.random.randint(0, 10, size=(4, 1)), dtype=tf.int32
            ),
        }

    def test_dict_input_fn_outputs(self):
        ys = keras.ops.map(dict_input_fn, self.xs)
        self.assertEqual(ys["x"].shape, (4, 100))
        self.assertTrue((ys["y"] == self.xs["y"] + 1).numpy().all())

    def test_list_input_fn_outputs(self):
        xs = [
            tf.convert_to_tensor(np.random.rand(4, 100, 3), dtype=tf.float32),
            tf.convert_to_tensor(
                np.random.randint(0, 10, size=(4, 1)), dtype=tf.int32
            ),
        ]
        ys = keras.ops.map(list_input_fn, xs)
        for x, y in zip(xs, ys):
            self.assertTrue((y == x**2).numpy().all())
