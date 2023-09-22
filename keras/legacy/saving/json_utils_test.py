import enum

import pytest

from keras import backend
from keras import testing
from keras.legacy.saving import json_utils

if backend.backend() == "tensorflow":
    import tensorflow as tf


class JsonUtilsTestAllBackends(testing.TestCase):
    def test_encode_decode_tuple(self):
        metadata = {"key1": (3, 5), "key2": [(1, (3, 4)), (1,)]}
        string = json_utils.Encoder().encode(metadata)
        loaded = json_utils.decode(string)

        self.assertEqual(set(loaded.keys()), {"key1", "key2"})
        self.assertAllEqual(loaded["key1"], (3, 5))
        self.assertAllEqual(loaded["key2"], [(1, (3, 4)), (1,)])

    def test_encode_decode_enum(self):
        class Enum(enum.Enum):
            CLASS_A = "a"
            CLASS_B = "b"

        config = {"key": Enum.CLASS_A, "key2": Enum.CLASS_B}
        string = json_utils.Encoder().encode(config)
        loaded = json_utils.decode(string)
        self.assertAllEqual({"key": "a", "key2": "b"}, loaded)

    def test_encode_decode_bytes(self):
        b_string = b"abc"
        json_string = json_utils.Encoder().encode(b_string)
        loaded = json_utils.decode(json_string)
        self.assertAllEqual(b_string, loaded)


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="These JSON serialization tests are specific to TF components.",
)
class JsonUtilsTestTF(testing.TestCase):
    def test_encode_decode_tensor_shape(self):
        metadata = {
            "key1": tf.TensorShape(None),
            "key2": [tf.TensorShape([None]), tf.TensorShape([3, None, 5])],
        }
        string = json_utils.Encoder().encode(metadata)
        loaded = json_utils.decode(string)

        self.assertEqual(set(loaded.keys()), {"key1", "key2"})
        self.assertEqual(loaded["key1"].rank, None)
        self.assertAllEqual(loaded["key2"][0].as_list(), [None])
        self.assertAllEqual(loaded["key2"][1].as_list(), [3, None, 5])

    def test_encode_decode_type_spec(self):
        spec = tf.TensorSpec((1, 5), tf.float32)
        string = json_utils.Encoder().encode(spec)
        loaded = json_utils.decode(string)
        self.assertEqual(spec, loaded)

        invalid_type_spec = {
            "class_name": "TypeSpec",
            "type_spec": "Invalid Type",
            "serialized": None,
        }
        string = json_utils.Encoder().encode(invalid_type_spec)
        with self.assertRaisesRegexp(
            ValueError, "No TypeSpec has been registered"
        ):
            loaded = json_utils.decode(string)

    def test_encode_decode_ragged_tensor(self):
        x = tf.ragged.constant([[1.0, 2.0], [3.0]])
        string = json_utils.Encoder().encode(x)
        loaded = json_utils.decode(string)
        self.assertAllClose(loaded.values, x.values)

    def test_encode_decode_extension_type_tensor(self):
        class MaskedTensor(tf.experimental.ExtensionType):
            __name__ = "MaskedTensor"
            values: tf.Tensor
            mask: tf.Tensor

        x = MaskedTensor(
            values=[[1, 2, 3], [4, 5, 6]],
            mask=[[True, True, False], [True, False, True]],
        )
        string = json_utils.Encoder().encode(x)
        loaded = json_utils.decode(string)
        self.assertAllClose(loaded.values, x.values)
        self.assertAllClose(loaded.mask, x.mask)
