# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for FeatureSpace utility."""

import os

import tensorflow.compat.v2 as tf

import keras
from keras import layers
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import feature_space


@test_utils.run_v2_only
class FeatureSpaceTest(test_combinations.TestCase):
    def _get_train_data_dict(
        self, as_dataset=False, as_tf_tensors=False, as_labeled_dataset=False
    ):
        data = {
            "float_1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "float_2": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "float_3": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "string_1": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "string_2": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "int_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "int_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "int_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        if as_dataset:
            return tf.data.Dataset.from_tensor_slices(data)
        elif as_tf_tensors:
            return tf.nest.map_structure(tf.convert_to_tensor, data)
        elif as_labeled_dataset:
            labels = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1]
            return tf.data.Dataset.from_tensor_slices((data, labels))
        return data

    def test_basic_usage(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "string_1": "string_categorical",
                "string_2": "string_hashed",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("float_3", "string_1"), ("string_2", "int_2")],
            output_mode="concat",
        )
        # Test unbatched adapt
        fs.adapt(self._get_train_data_dict(as_dataset=True))
        # Test batched adapt
        fs.adapt(self._get_train_data_dict(as_dataset=True).batch(4))

        # Test unbatched call on raw data
        data = {
            key: value[0] for key, value in self._get_train_data_dict().items()
        }
        out = fs(data)
        self.assertEqual(out.shape.as_list(), [195])

        # Test unbatched call on TF tensors
        data = self._get_train_data_dict(as_tf_tensors=True)
        data = {key: value[0] for key, value in data.items()}
        out = fs(data)
        self.assertEqual(out.shape.as_list(), [195])

        # Test batched call on raw data
        out = fs(self._get_train_data_dict())
        self.assertEqual(out.shape.as_list(), [10, 195])

        # Test batched call on TF tensors
        out = fs(self._get_train_data_dict(as_tf_tensors=True))
        self.assertEqual(out.shape.as_list(), [10, 195])

    def test_output_mode_dict(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "string_1": "string_categorical",
                "string_2": "string_hashed",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("float_3", "string_1"), ("string_2", "int_2")],
            output_mode="dict",
        )
        fs.adapt(self._get_train_data_dict(as_dataset=True))

        # Test unbatched call on raw data
        data = {
            key: value[0] for key, value in self._get_train_data_dict().items()
        }
        out = fs(data)
        self.assertIsInstance(out, dict)
        self.assertLen(out, 10)
        self.assertEqual(out["string_1"].shape.as_list(), [11])
        self.assertEqual(out["int_2"].shape.as_list(), [32])
        self.assertEqual(out["string_2_X_int_2"].shape.as_list(), [32])

        # Test batched call on raw data
        out = fs(self._get_train_data_dict())
        self.assertIsInstance(out, dict)
        self.assertLen(out, 10)
        self.assertEqual(out["string_1"].shape.as_list(), [10, 11])
        self.assertEqual(out["int_2"].shape.as_list(), [10, 32])
        self.assertEqual(out["string_2_X_int_2"].shape.as_list(), [10, 32])

        # Test batched call on TF tensors
        out = fs(self._get_train_data_dict(as_tf_tensors=True))
        self.assertIsInstance(out, dict)
        self.assertLen(out, 10)
        self.assertEqual(out["string_1"].shape.as_list(), [10, 11])
        self.assertEqual(out["int_2"].shape.as_list(), [10, 32])
        self.assertEqual(out["string_2_X_int_2"].shape.as_list(), [10, 32])

    def test_output_mode_dict_of_ints(self):
        cls = feature_space.FeatureSpace
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "string_1": cls.string_categorical(output_mode="int"),
                "string_2": cls.string_hashed(num_bins=32, output_mode="int"),
                "int_1": cls.integer_categorical(output_mode="int"),
                "int_2": cls.integer_hashed(num_bins=32, output_mode="int"),
                "int_3": cls.integer_categorical(output_mode="int"),
            },
            crosses=[
                cls.cross(
                    ("float_3", "string_1"), output_mode="int", crossing_dim=32
                ),
                cls.cross(
                    ("string_2", "int_2"), output_mode="int", crossing_dim=32
                ),
            ],
            output_mode="dict",
        )
        fs.adapt(self._get_train_data_dict(as_dataset=True))
        data = {
            key: value[0] for key, value in self._get_train_data_dict().items()
        }
        out = fs(data)
        self.assertIsInstance(out, dict)
        self.assertLen(out, 10)
        self.assertEqual(out["string_1"].shape.as_list(), [1])
        self.assertEqual(out["string_1"].dtype.name, "int64")
        self.assertEqual(out["int_2"].shape.as_list(), [1])
        self.assertEqual(out["int_2"].dtype.name, "int64")
        self.assertEqual(out["string_2_X_int_2"].shape.as_list(), [1])
        self.assertEqual(out["string_2_X_int_2"].dtype.name, "int64")

    def test_functional_api_sync_processing(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "string_1": "string_categorical",
                "string_2": "string_hashed",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("float_3", "string_1"), ("string_2", "int_2")],
            output_mode="concat",
        )
        fs.adapt(self._get_train_data_dict(as_dataset=True))
        inputs = fs.get_inputs()
        features = fs.get_encoded_features()
        outputs = layers.Dense(1)(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile("adam", "mse")
        ds = self._get_train_data_dict(as_labeled_dataset=True)
        model.fit(ds.batch(4))
        model.evaluate(ds.batch(4))
        ds = self._get_train_data_dict(as_dataset=True)
        model.predict(ds.batch(4))

    def test_tf_data_async_processing(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "string_1": "string_categorical",
                "string_2": "string_hashed",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("float_3", "string_1"), ("string_2", "int_2")],
            output_mode="concat",
        )
        fs.adapt(self._get_train_data_dict(as_dataset=True))
        features = fs.get_encoded_features()
        outputs = layers.Dense(1)(features)
        model = keras.Model(inputs=features, outputs=outputs)
        model.compile("adam", "mse")
        ds = self._get_train_data_dict(as_labeled_dataset=True)
        # Try map before batch
        ds = ds.map(lambda x, y: (fs(x), y))
        model.fit(ds.batch(4))
        # Try map after batch
        ds = self._get_train_data_dict(as_labeled_dataset=True)
        ds = ds.batch(4)
        ds = ds.map(lambda x, y: (fs(x), y))
        model.evaluate(ds)
        ds = self._get_train_data_dict(as_dataset=True)
        ds = ds.map(fs)
        model.predict(ds.batch(4))

    def test_advanced_usage(self):
        cls = feature_space.FeatureSpace
        fs = feature_space.FeatureSpace(
            features={
                "float_1": cls.float(),
                "float_2": cls.float_normalized(),
                "float_3": cls.float_discretized(num_bins=3),
                "string_1": cls.string_categorical(max_tokens=5),
                "string_2": cls.string_hashed(num_bins=32),
                "int_1": cls.integer_categorical(
                    max_tokens=5, num_oov_indices=2
                ),
                "int_2": cls.integer_hashed(num_bins=32),
                "int_3": cls.integer_categorical(max_tokens=5),
            },
            crosses=[
                cls.cross(("float_3", "string_1"), crossing_dim=32),
                cls.cross(("string_2", "int_2"), crossing_dim=32),
            ],
            output_mode="concat",
        )
        fs.adapt(self._get_train_data_dict(as_dataset=True))
        data = {
            key: value[0] for key, value in self._get_train_data_dict().items()
        }
        out = fs(data)
        self.assertEqual(out.shape.as_list(), [148])

    def test_manual_kpl(self):
        data = {
            "text": ["1st string", "2nd string", "3rd string"],
        }
        cls = feature_space.FeatureSpace

        # Test with a tf-idf TextVectorization layer
        tv = layers.TextVectorization(output_mode="tf_idf")
        fs = feature_space.FeatureSpace(
            features={
                "text": cls.feature(
                    preprocessor=tv, dtype="string", output_mode="float"
                ),
            },
            output_mode="concat",
        )
        fs.adapt(tf.data.Dataset.from_tensor_slices(data))
        out = fs(data)
        self.assertEqual(out.shape.as_list(), [3, 5])

    def test_no_adapt(self):
        data = {
            "int_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        fs = feature_space.FeatureSpace(
            {
                "int_1": "integer_hashed",
            },
            output_mode="concat",
        )
        out = fs(data)
        self.assertEqual(out.shape.as_list(), [10, 32])

    def test_saving(self):
        cls = feature_space.FeatureSpace
        fs = feature_space.FeatureSpace(
            features={
                "float_1": cls.float(),
                "float_2": cls.float_normalized(),
                "float_3": cls.float_discretized(num_bins=3),
                "string_1": cls.string_categorical(max_tokens=5),
                "string_2": cls.string_hashed(num_bins=32),
                "int_1": cls.integer_categorical(
                    max_tokens=5, num_oov_indices=2
                ),
                "int_2": cls.integer_hashed(num_bins=32),
                "int_3": cls.integer_categorical(max_tokens=5),
            },
            crosses=[
                cls.cross(("float_3", "string_1"), crossing_dim=32),
                cls.cross(("string_2", "int_2"), crossing_dim=32),
            ],
            output_mode="concat",
        )
        fs.adapt(self._get_train_data_dict(as_dataset=True))
        data = {
            key: value[0] for key, value in self._get_train_data_dict().items()
        }
        ref_out = fs(data)

        temp_filepath = os.path.join(self.get_temp_dir(), "fs.keras")
        fs.save(temp_filepath)
        fs = keras.models.load_model(temp_filepath)

        # Save again immediately after loading to test idempotency
        temp_filepath = os.path.join(self.get_temp_dir(), "fs2.keras")
        fs.save(temp_filepath)

        # Test correctness of the first saved FS
        out = fs(data)
        self.assertAllClose(out, ref_out)

        inputs = fs.get_inputs()
        outputs = fs.get_encoded_features()
        model = keras.Model(inputs=inputs, outputs=outputs)
        ds = self._get_train_data_dict(as_dataset=True)
        out = model.predict(ds.batch(4))
        self.assertAllClose(out[0], ref_out)

        # Test correctness of the re-saved FS
        fs = keras.models.load_model(temp_filepath)
        out = fs(data)
        self.assertAllClose(out, ref_out)

    def test_errors(self):
        # Test no features
        with self.assertRaisesRegex(ValueError, "cannot be None or empty"):
            feature_space.FeatureSpace(features={})
        # Test no crossing dim
        with self.assertRaisesRegex(ValueError, "`crossing_dim`"):
            feature_space.FeatureSpace(
                features={
                    "f1": "integer_categorical",
                    "f2": "integer_categorical",
                },
                crosses=[("f1", "f2")],
                crossing_dim=None,
            )
        # Test wrong cross feature name
        with self.assertRaisesRegex(ValueError, "should be present in "):
            feature_space.FeatureSpace(
                features={
                    "f1": "integer_categorical",
                    "f2": "integer_categorical",
                },
                crosses=[("f1", "unknown")],
                crossing_dim=32,
            )
        # Test wrong output mode
        with self.assertRaisesRegex(ValueError, "for argument `output_mode`"):
            feature_space.FeatureSpace(
                features={
                    "f1": "integer_categorical",
                    "f2": "integer_categorical",
                },
                output_mode="unknown",
            )
        # Test call before adapt
        with self.assertRaisesRegex(ValueError, "You need to call `.adapt"):
            fs = feature_space.FeatureSpace(
                features={
                    "f1": "integer_categorical",
                    "f2": "integer_categorical",
                }
            )
            fs({"f1": [0], "f2": [0]})
        # Test get_encoded_features before adapt
        with self.assertRaisesRegex(ValueError, "You need to call `.adapt"):
            fs = feature_space.FeatureSpace(
                features={
                    "f1": "integer_categorical",
                    "f2": "integer_categorical",
                }
            )
            fs.get_encoded_features()


if __name__ == "__main__":
    tf.test.main()
