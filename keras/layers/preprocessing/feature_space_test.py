import os

import pytest
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import models
from keras import ops
from keras import testing
from keras.layers.preprocessing import feature_space
from keras.saving import saving_api


class FeatureSpaceTest(testing.TestCase):
    def _get_train_data_dict(
        self,
        as_dataset=False,
        as_tensors=False,
        as_labeled_dataset=False,
        include_strings=True,
    ):
        data = {
            "float_1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "float_2": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "float_3": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "int_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "int_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "int_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        if include_strings:
            data["string_1"] = [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]
            data["string_2"] = [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]

        if as_dataset:
            return tf_data.Dataset.from_tensor_slices(data)
        elif as_tensors:
            return {
                key: ops.convert_to_tensor(value) for key, value in data.items()
            }
        elif as_labeled_dataset:
            labels = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1]
            return tf_data.Dataset.from_tensor_slices((data, labels))
        return data

    def test_basic_usage_no_strings(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("int_1", "int_2"), ("int_2", "int_3")],
            output_mode="concat",
        )
        # Test unbatched adapt
        fs.adapt(
            self._get_train_data_dict(as_dataset=True, include_strings=False)
        )
        # Test batched adapt
        fs.adapt(
            self._get_train_data_dict(
                as_dataset=True, include_strings=False
            ).batch(4)
        )

        # Test unbatched call on raw data
        data = {
            key: value[0]
            for key, value in self._get_train_data_dict(
                include_strings=False
            ).items()
        }
        out = fs(data)
        out_dim = 152
        self.assertEqual(out.shape, (out_dim,))

        # Test unbatched call on backend tensors
        data = self._get_train_data_dict(as_tensors=True, include_strings=False)
        data = {key: value[0] for key, value in data.items()}
        out = fs(data)
        self.assertEqual(out.shape, (out_dim,))

        # Test batched call on raw data
        out = fs(self._get_train_data_dict(include_strings=False))
        self.assertEqual(out.shape, (10, out_dim))

        # Test batched call on backend tensors
        out = fs(
            self._get_train_data_dict(as_tensors=True, include_strings=False)
        )
        self.assertEqual(out.shape, (10, out_dim))

    def test_output_mode_dict_no_strings(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("int_1", "int_2")],
            output_mode="dict",
        )
        fs.adapt(
            self._get_train_data_dict(as_dataset=True, include_strings=False)
        )

        # Test unbatched call on raw data
        data = {
            key: value[0]
            for key, value in self._get_train_data_dict(
                include_strings=False
            ).items()
        }
        out = fs(data)
        self.assertIsInstance(out, dict)
        self.assertLen(out, 7)
        self.assertEqual(out["int_2"].shape, (32,))
        self.assertEqual(out["int_1_X_int_2"].shape, (32,))

        # Test batched call on raw data
        out = fs(self._get_train_data_dict(include_strings=False))
        self.assertIsInstance(out, dict)
        self.assertLen(out, 7)
        self.assertEqual(out["int_2"].shape, (10, 32))

        # Test batched call on backend tensors
        out = fs(
            self._get_train_data_dict(as_tensors=True, include_strings=False)
        )
        self.assertIsInstance(out, dict)
        self.assertLen(out, 7)
        self.assertEqual(out["int_2"].shape, (10, 32))

    def test_output_mode_dict_of_ints_no_strings(self):
        cls = feature_space.FeatureSpace
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "int_1": cls.integer_categorical(output_mode="int"),
                "int_2": cls.integer_hashed(num_bins=32, output_mode="int"),
                "int_3": cls.integer_categorical(output_mode="int"),
            },
            crosses=[
                cls.cross(
                    ("int_1", "int_2"), output_mode="int", crossing_dim=32
                ),
            ],
            output_mode="dict",
        )
        fs.adapt(
            self._get_train_data_dict(as_dataset=True, include_strings=False)
        )
        data = {
            key: value[0]
            for key, value in self._get_train_data_dict(
                include_strings=False
            ).items()
        }
        out = fs(data)
        self.assertIsInstance(out, dict)
        self.assertLen(out, 7)
        self.assertEqual(out["int_2"].shape, (1,))
        self.assertTrue(
            backend.standardize_dtype(out["int_2"].dtype).startswith("int")
        )
        self.assertEqual(out["int_1_X_int_2"].shape, (1,))
        self.assertTrue(
            backend.standardize_dtype(out["int_1_X_int_2"].dtype).startswith(
                "int"
            )
        )

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
        out_dim = 195
        self.assertEqual(out.shape, (out_dim,))

        # Test unbatched call on tensors
        if backend.backend() == "tensorflow":
            data = self._get_train_data_dict(as_tensors=True)
            data = {key: value[0] for key, value in data.items()}
            out = fs(data)
            self.assertEqual(out.shape, (out_dim,))

        # Test batched call on raw data
        out = fs(self._get_train_data_dict())
        self.assertEqual(out.shape, (10, out_dim))

        # Test batched call on tensors
        if backend.backend() == "tensorflow":
            out = fs(self._get_train_data_dict(as_tensors=True))
            self.assertEqual(out.shape, (10, out_dim))

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
        self.assertEqual(out["string_1"].shape, (11,))
        self.assertEqual(out["int_2"].shape, (32,))
        self.assertEqual(out["string_2_X_int_2"].shape, (32,))

        # Test batched call on raw data
        out = fs(self._get_train_data_dict())
        self.assertIsInstance(out, dict)
        self.assertLen(out, 10)
        self.assertEqual(out["string_1"].shape, (10, 11))
        self.assertEqual(out["int_2"].shape, (10, 32))
        self.assertEqual(out["string_2_X_int_2"].shape, (10, 32))

        # Test batched call on tensors
        if backend.backend == "tensorflow":
            out = fs(self._get_train_data_dict(as_tensors=True))
            self.assertIsInstance(out, dict)
            self.assertLen(out, 10)
            self.assertEqual(out["string_1"].shape, (10, 11))
            self.assertEqual(out["int_2"].shape, (10, 32))
            self.assertEqual(out["string_2_X_int_2"].shape, (10, 32))

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
        self.assertEqual(out["string_1"].shape, (1,))
        self.assertTrue(
            backend.standardize_dtype(out["string_1"].dtype).startswith("int")
        )
        self.assertEqual(out["int_2"].shape, (1,))
        self.assertTrue(
            backend.standardize_dtype(out["int_2"].dtype).startswith("int")
        )
        self.assertEqual(out["string_2_X_int_2"].shape, (1,))
        self.assertTrue(
            backend.standardize_dtype(out["string_2_X_int_2"].dtype).startswith(
                "int"
            )
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires string dtype."
    )
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
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile("adam", "mse")
        ds = self._get_train_data_dict(as_labeled_dataset=True)
        model.fit(ds.batch(4))
        model.evaluate(ds.batch(4))
        ds = self._get_train_data_dict(as_dataset=True)
        model.predict(ds.batch(4))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TODO: debug it"
    )
    def test_tf_data_async_processing(self):
        fs = feature_space.FeatureSpace(
            features={
                "float_1": "float",
                "float_2": "float_normalized",
                "float_3": "float_discretized",
                "int_1": "integer_categorical",
                "int_2": "integer_hashed",
                "int_3": "integer_categorical",
            },
            crosses=[("float_3", "int_1"), ("int_1", "int_2")],
            output_mode="concat",
        )
        fs.adapt(
            self._get_train_data_dict(as_dataset=True, include_strings=False)
        )
        features = fs.get_encoded_features()
        outputs = layers.Dense(1)(features)
        model = models.Model(inputs=features, outputs=outputs)
        model.compile("adam", "mse")
        ds = self._get_train_data_dict(
            as_labeled_dataset=True, include_strings=False
        )
        # Try map before batch
        ds = ds.map(lambda x, y: (fs(x), y))
        model.fit(ds.batch(4))
        # Try map after batch
        ds = self._get_train_data_dict(
            as_labeled_dataset=True, include_strings=False
        )
        ds = ds.batch(4)
        ds = ds.map(lambda x, y: (fs(x), y))
        model.evaluate(ds)
        ds = self._get_train_data_dict(as_dataset=True, include_strings=False)
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
        self.assertEqual(out.shape, (148,))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TODO: debug it"
    )
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
        fs.adapt(tf_data.Dataset.from_tensor_slices(data))
        out = fs(data)
        self.assertEqual(out.shape, [3, 5])

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
        self.assertEqual(tuple(out.shape), (10, 32))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TODO: debug it"
    )
    def test_saving(self):
        # Torch GPU: `model.predict(ds.batch(4))` fails on device placement
        # JAX GPU: out[0] and ref_out don't match. May be concat feature order?
        cls = feature_space.FeatureSpace
        fs = feature_space.FeatureSpace(
            features={
                "float_1": cls.float(),
                "float_2": cls.float_normalized(),
                "float_3": cls.float_discretized(num_bins=3),
                "int_1": cls.integer_categorical(
                    max_tokens=5, num_oov_indices=2
                ),
                "int_2": cls.integer_hashed(num_bins=32),
                "int_3": cls.integer_categorical(max_tokens=5),
            },
            crosses=[
                cls.cross(("float_3", "int_1"), crossing_dim=32),
                cls.cross(("int_1", "int_2"), crossing_dim=32),
            ],
            output_mode="concat",
        )
        fs.adapt(
            self._get_train_data_dict(as_dataset=True, include_strings=False)
        )
        data = {
            key: value[0]
            for key, value in self._get_train_data_dict(
                include_strings=False
            ).items()
        }
        ref_out = fs(data)

        temp_filepath = os.path.join(self.get_temp_dir(), "fs.keras")
        fs.save(temp_filepath)
        fs = saving_api.load_model(temp_filepath)

        # Save again immediately after loading to test idempotency
        temp_filepath = os.path.join(self.get_temp_dir(), "fs2.keras")
        fs.save(temp_filepath)

        # Test correctness of the first saved FS
        out = fs(data)
        self.assertAllClose(out, ref_out)

        inputs = fs.get_inputs()
        outputs = fs.get_encoded_features()
        model = models.Model(inputs=inputs, outputs=outputs)
        ds = self._get_train_data_dict(as_dataset=True, include_strings=False)
        out = model.predict(ds.batch(4))
        self.assertAllClose(out[0], ref_out)

        # Test correctness of the re-saved FS
        fs = saving_api.load_model(temp_filepath)
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
