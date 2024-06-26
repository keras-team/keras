import numpy as np
import pytest

from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src.dtype_policies.dtype_policy import dtype_policy
from keras.src.dtype_policies.dtype_policy import set_dtype_policy
from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap


class DTypePolicyMapTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self._global_dtype_policy = dtype_policy()

    def tearDown(self):
        super().tearDown()
        set_dtype_policy(self._global_dtype_policy)

    @pytest.mark.requires_trainable_backend
    def test_basic_usage(self):
        # Create a subclass that might contain mixing dtype policies for
        # sublayers.
        # It is important to ensure that `dtype` is passed to sublayers and
        # that each sublayer has a unique `name`.
        @saving.register_keras_serializable()
        class Subclass(layers.Layer):
            def __init__(self, dtype=None, name="subclass", **kwargs):
                super().__init__(dtype=dtype, name=name, **kwargs)
                self.dense = layers.Dense(8, dtype=dtype, name=f"{name}_dense")
                self.bn = layers.BatchNormalization(
                    dtype=dtype, name=f"{name}_bn"
                )
                self.relu = layers.ReLU(dtype=dtype, name=f"{name}_relu")

            def call(self, inputs, training=None):
                return self.relu(self.bn(self.dense(inputs), training=training))

            def get_config(self):
                # Typically, we only need to record the quantized policy for
                # `DTypePolicyMap`
                config = super().get_config()
                dtype_policy_map = DTypePolicyMap()
                for layer in self._flatten_layers():
                    if layer.quantization_mode is not None:
                        dtype_policy_map[layer.path] = layer.dtype_policy
                if len(dtype_policy_map) > 0:
                    config.update({"dtype": dtype_policy_map})
                return config

        # Instantiate the model
        inputs = layers.Input([4])
        outputs = Subclass()(inputs)
        model = models.Model(inputs, outputs)

        # Quantize the model to make mixing of dtype policies in sublayers
        model.quantize("int8")
        for layer in model._flatten_layers():
            if isinstance(layer, layers.Dense):
                self.assertEqual(
                    layer.dtype_policy,
                    dtype_policies.QuantizedDTypePolicy("int8"),
                )
            elif isinstance(layer, layers.BatchNormalization):
                self.assertEqual(
                    layer.dtype_policy, dtype_policies.DTypePolicy()
                )
            elif isinstance(layer, layers.ReLU):
                self.assertEqual(
                    layer.dtype_policy, dtype_policies.DTypePolicy()
                )

        # Verify the output after saving and loading
        x = np.random.uniform(size=[16, 4])
        temp_dir = self.get_temp_dir()
        y = model(x, training=False)
        model.save(f"{temp_dir}/model.keras")
        reloaded_model = saving.load_model(f"{temp_dir}/model.keras")
        reloaded_y = reloaded_model(x, training=False)
        self.assertAllClose(y, reloaded_y)

    def test_add(self):
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )
        dtype_policy_map["layer/dense_2"] = (
            dtype_policies.QuantizedFloat8DTypePolicy("float8", "mixed_float16")
        )

        self.assertLen(dtype_policy_map, 3)

        policy = dtype_policy_map["layer/dense_0"]
        self.assertIsInstance(policy, dtype_policies.DTypePolicy)
        self.assertEqual(policy.name, "bfloat16")

        policy = dtype_policy_map["layer/dense_1"]
        self.assertIsInstance(policy, dtype_policies.QuantizedDTypePolicy)
        self.assertEqual(policy._source_name, "mixed_bfloat16")
        self.assertEqual(policy.quantization_mode, "int8")

        policy = dtype_policy_map["layer/dense_2"]
        self.assertIsInstance(policy, dtype_policies.QuantizedFloat8DTypePolicy)
        self.assertEqual(policy._source_name, "mixed_float16")
        self.assertEqual(policy.quantization_mode, "float8")

        with self.assertRaisesRegex(
            ValueError, "layer/dense_0 already exist in the DTypePolicyMap"
        ):
            dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
                "float32"
            )

        with self.assertRaisesRegex(
            ValueError, "Cannot interpret the assigned value."
        ):
            dtype_policy_map["layer/dense_3"] = 123

    def test_get(self):
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )
        dtype_policy_map["layer/dense_2"] = (
            dtype_policies.QuantizedFloat8DTypePolicy("float8", "mixed_float16")
        )

        self.assertEqual(
            dtype_policy_map["layer/dense_0"],
            dtype_policies.DTypePolicy("bfloat16"),
        )
        self.assertEqual(
            dtype_policy_map["layer/dense_1"],
            dtype_policies.QuantizedDTypePolicy("int8", "mixed_bfloat16"),
        )
        self.assertEqual(
            dtype_policy_map["layer/dense_2"],
            dtype_policies.QuantizedFloat8DTypePolicy(
                "float8", "mixed_float16"
            ),
        )

        self.assertNotEqual(
            dtype_policy_map["layer/dense_2"],
            dtype_policies.QuantizedFloat8DTypePolicy("float8", "bfloat16"),
        )

        # No hit
        self.assertEqual(
            dtype_policy_map["layer/batch_normalization"],
            dtype_policy_map.default_policy,
        )

        # It will cause a ValueError in the case of one-to-many.
        dtype_policy_map["dense"] = dtype_policies.DTypePolicy("float32")
        dtype_policy_map["dense_1"] = dtype_policies.DTypePolicy("float32")
        with self.assertRaisesRegex(
            ValueError, "Path 'dense_10' matches multiple dtype policy"
        ):
            dtype_policy_map["dense_10"]

    def test_delete(self):
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )

        self.assertEqual(
            dtype_policy_map.pop("layer/dense_0"),
            dtype_policies.DTypePolicy("bfloat16"),
        )
        with self.assertRaises(KeyError):
            dtype_policy_map.pop("layer/dense_0")

        # Test `del`, causing no hit
        del dtype_policy_map["layer/dense_1"]
        self.assertEqual(
            dtype_policy_map["layer/dense_1"], dtype_policy_map.default_policy
        )

        self.assertLen(dtype_policy_map, 0)

    def test_len(self):
        dtype_policy_map = DTypePolicyMap()
        self.assertLen(dtype_policy_map, 0)

        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )
        self.assertLen(dtype_policy_map, 2)

    def test_iter(self):
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )

        self.assertEqual(
            list(dtype_policy_map.keys()), ["layer/dense_0", "layer/dense_1"]
        )

        keys = []
        values = []
        for k, v in dtype_policy_map.items():
            keys.append(k)
            values.append(v)
        self.assertEqual(keys, ["layer/dense_0", "layer/dense_1"])
        self.assertEqual(
            values,
            [
                dtype_policies.DTypePolicy("bfloat16"),
                dtype_policies.QuantizedDTypePolicy("int8", "mixed_bfloat16"),
            ],
        )

    def test_in(self):
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )

        self.assertTrue("layer/dense_0" in dtype_policy_map)
        self.assertTrue("layer/dense_1" in dtype_policy_map)
        self.assertFalse("layer/dense_2" in dtype_policy_map)

    def test_default_policy(self):
        # Test default_policy is set to `"float32"`
        dtype_policy_map = DTypePolicyMap(default_policy="mixed_bfloat16")
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "mixed_bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )
        config = dtype_policy_map.get_config()
        dtype_policy_map = DTypePolicyMap.from_config(config)
        self.assertEqual(
            dtype_policy_map["layer/dense_0"],
            dtype_policies.DTypePolicy("mixed_bfloat16"),
        )
        self.assertEqual(
            dtype_policy_map["layer/dense_1"],
            dtype_policies.QuantizedDTypePolicy("int8", "mixed_bfloat16"),
        )
        # No hit, defers to `dtype_policy_map.default_policy`
        self.assertEqual(
            dtype_policy_map["layer/dense_2"], dtype_policy_map.default_policy
        )

        # Test that default_policy defers to `keras.config.dtype_policy()`
        # during loading
        set_dtype_policy("bfloat16")
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "mixed_bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )
        config = dtype_policy_map.get_config()
        dtype_policy_map = DTypePolicyMap.from_config(config)
        self.assertEqual(
            dtype_policy_map["layer/dense_0"],
            dtype_policies.DTypePolicy("bfloat16"),
        )
        self.assertEqual(
            dtype_policy_map["layer/dense_1"],
            dtype_policies.QuantizedDTypePolicy("int8", "bfloat16"),
        )
        # No hit, defers to `dtype_policy_map.default_policy` which is
        # `keras.config.dtype_policy()`
        self.assertEqual(
            dtype_policy_map["layer/dense_2"], dtype_policy_map.default_policy
        )
        self.assertEqual(
            dtype_policy_map["layer/dense_2"], dtype_policies.get("bfloat16")
        )

    def test_serialization(self):
        dtype_policy_map = DTypePolicyMap(default_policy="mixed_bfloat16")
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "mixed_bfloat16"
        )
        dtype_policy_map["layer/dense_1"] = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )

        config = dtype_policies.serialize(dtype_policy_map)
        reloaded_dtype_policy_map = dtype_policies.deserialize(config)
        self.assertEqual(
            dtype_policy_map.default_policy,
            reloaded_dtype_policy_map.default_policy,
        )
        for k, v in dtype_policy_map.items():
            self.assertEqual(reloaded_dtype_policy_map[k], v)

        # Test that config remains intact during deserialization
        config = dtype_policy_map.get_config()
        original_config = config.copy()
        DTypePolicyMap.from_config(config)
        self.assertDictEqual(config, original_config)

    def test_repr(self):
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["layer/dense_0"] = dtype_policies.DTypePolicy(
            "mixed_bfloat16"
        )
        repr_str = repr(dtype_policy_map)
        self.assertTrue("DTypePolicyMap" in repr_str)
        self.assertTrue("default_policy" in repr_str)
        self.assertTrue(
            "mapping=[('layer/dense_0', 'mixed_bfloat16')]" in repr_str
        )

    def test_invalid_policy_map(self):
        with self.assertRaisesRegex(
            TypeError, "If specified, `policy_map` must be a dict."
        ):
            DTypePolicyMap(policy_map=123)

        with self.assertRaisesRegex(
            TypeError, "If specified, `policy_map` must be a dict."
        ):
            DTypePolicyMap(
                policy_map=dtype_policies.DTypePolicy("mixed_bfloat16")
            )
