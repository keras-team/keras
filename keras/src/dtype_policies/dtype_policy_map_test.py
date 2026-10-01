import copy
import json

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
from keras.src.saving import serialization_lib


def _serialize_by_blanking_policies(dtype_policy_map):
    """Reference for `DTypePolicyMap.get_config` with no default policy.

    It removes the sources from the policies themselves and then serializes
    them, so call it on a copy only.
    """
    for policy in dtype_policy_map._policy_map.values():
        if isinstance(policy, dtype_policies.QuantizedDTypePolicy):
            policy._name = None
            policy._source_name = None
        elif isinstance(policy, dtype_policies.DTypePolicy):
            policy._name = None
    return {
        "default_policy": dtype_policy_map._default_policy_arg,
        "policy_map": serialization_lib.serialize_keras_object(
            dtype_policy_map._policy_map
        ),
    }


@pytest.mark.skipif(testing.jax_uses_gpu(), reason="Leads to core dumps on CI")
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
        # 1. Setup
        bfloat16_policy = dtype_policies.DTypePolicy("bfloat16")
        int8_policy = dtype_policies.QuantizedDTypePolicy(
            "int8", "mixed_bfloat16"
        )
        float32_policy = dtype_policies.DTypePolicy("float32")
        float16_policy = dtype_policies.DTypePolicy("float16")

        policy_map = DTypePolicyMap()
        # Policy for an exact layer path
        policy_map["model/encoder/layer_0/dense"] = bfloat16_policy
        # Policy for a layer that is also a prefix of another layer's name
        policy_map["model/encoder/attention/query"] = int8_policy
        # Regex policies for entire scopes MUST include wildcards
        policy_map["model/decoder/.*"] = float32_policy
        policy_map["model/decoder/attention/.*"] = float16_policy

        # 2. Test exact match
        self.assertEqual(
            policy_map["model/encoder/layer_0/dense"], bfloat16_policy
        )
        self.assertEqual(
            policy_map["model/encoder/attention/query"], int8_policy
        )

        # 3. Test successful regex fallback (explicit wildcard)
        # "model/decoder/.*" should match its children.
        self.assertEqual(policy_map["model/decoder/layer_0"], float32_policy)

        # 4. Test that partial matches are ignored
        # The exact key "model/encoder/attention/query" should not match
        # "model/encoder/attention/query_norm" without a wildcard.
        self.assertEqual(
            policy_map["model/encoder/attention/query_norm"],
            policy_map.default_policy,
        )
        # A plain key "model/decoder" will not match "model/decoder/layer_0"
        policy_map["model/decoder"] = bfloat16_policy  # Add exact key
        self.assertEqual(policy_map["model/decoder/layer_0"], float32_policy)
        # Still matches the more general regex
        self.assertEqual(policy_map["model/decoder"], bfloat16_policy)

        # 5. Test no match
        self.assertEqual(
            policy_map["model/embedding"], policy_map.default_policy
        )

        # 6. Test multiple regex matches causing a ValueError
        # "model/decoder/attention/output" matches two regex patterns:
        # - "model/decoder/.*"
        # - "model/decoder/attention/.*"
        with self.assertRaisesRegex(
            ValueError,
            "Path 'model/decoder/attention/output' matches multiple "
            "dtype policy",
        ):
            _ = policy_map["model/decoder/attention/output"]

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

    def test_get_config_keeps_the_policies_in_the_map(self):
        # With no default policy, the serialized entries carry no source.
        # The policies in the map keep their names, sources and equality.
        policies = {
            "plain": dtype_policies.DTypePolicy("mixed_bfloat16"),
            "int8": dtype_policies.QuantizedDTypePolicy(
                "int8", "mixed_bfloat16"
            ),
            "int4": dtype_policies.get("int4/32_from_bfloat16"),
            "float8": dtype_policies.QuantizedFloat8DTypePolicy(
                "float8", "mixed_float16", amax_history_length=16
            ),
            "gptq": dtype_policies.get("gptq/4/128_from_float32"),
            "awq": dtype_policies.get("awq/4/64_from_mixed_bfloat16"),
        }
        nested = DTypePolicyMap(default_policy="float16")
        nested["inner"] = dtype_policies.get("int8_from_mixed_bfloat16")
        dtype_policy_map = DTypePolicyMap()
        for key, policy in policies.items():
            dtype_policy_map[key] = policy
        dtype_policy_map["nested"] = nested
        dtype_policy_map["int8_again"] = policies["int8"]
        originals = copy.deepcopy(policies)
        reference = copy.deepcopy(dtype_policy_map)

        config = dtype_policy_map.get_config()

        for key, policy in policies.items():
            self.assertIs(dtype_policy_map[key], policy)
            self.assertEqual(policy.name, originals[key].name)
            self.assertEqual(policy.get_config(), originals[key].get_config())
            self.assertEqual(policy, originals[key])
        self.assertEqual(nested["inner"].name, "int8_from_mixed_bfloat16")
        other = dtype_policies.QuantizedDTypePolicy("int8", "float32")
        other_map = DTypePolicyMap()
        other_map["dense"] = other
        other_map.get_config()
        self.assertNotEqual(policies["int8"], other)
        # The serialized output is byte-identical to the reference.
        expected = _serialize_by_blanking_policies(reference)
        self.assertEqual(json.dumps(config), json.dumps(expected))
        self.assertEqual(
            json.dumps(dtype_policy_map.get_config()), json.dumps(config)
        )

    def test_get_config_keeps_the_layer_policy(self):
        # A map that holds a layer's own policy, as in the class docstring,
        # leaves that policy unchanged when it is serialized.
        layer = layers.Dense(4, dtype="int8_from_mixed_bfloat16")
        dtype_policy_map = DTypePolicyMap()
        dtype_policy_map["dense"] = layer.dtype_policy
        dtype_policy_map.get_config()

        self.assertEqual(layer.dtype_policy.name, "int8_from_mixed_bfloat16")
        self.assertEqual(
            layer.get_config()["dtype"]["config"]["source_name"],
            "mixed_bfloat16",
        )
        clone = layers.Dense.from_config(layer.get_config())
        self.assertEqual(clone.compute_dtype, "bfloat16")

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
