import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.dtype_policies.dtype_policy import QUANTIZATION_MODES
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.quantization_config import QuantizationConfig


class StrategyRegistryTest(testing.TestCase):
    def test_builtin_modes_match_public_tuple(self):
        # The registration order is observable (validation error messages
        # render the registered-names tuple), so it must stay identical to
        # the public QUANTIZATION_MODES constant.
        self.assertEqual(
            strategy_registry.registered_modes(), QUANTIZATION_MODES
        )
        for name in QUANTIZATION_MODES:
            self.assertIsNotNone(strategy_registry.get_strategy(name))

    def test_unknown_mode(self):
        self.assertIsNone(strategy_registry.get_strategy("bogus"))
        self.assertFalse(strategy_registry.is_registered("bogus"))

    def test_register_requires_name(self):
        class Nameless(strategy_registry.QuantizationStrategy):
            requires_config = True

        with self.assertRaisesRegex(ValueError, "non-empty string `name`"):
            strategy_registry.register_quantization_strategy(Nameless)

    def test_register_rejects_duplicates(self):
        class Duplicate(strategy_registry.QuantizationStrategy):
            name = "int8"
            requires_config = True

        with self.assertRaisesRegex(ValueError, "already registered"):
            strategy_registry.register_quantization_strategy(Duplicate)

    @parameterized.named_parameters(
        ("existing_builtin_is_prefix", "int42"),
        ("new_is_prefix_of_builtin", "in"),
    )
    def test_register_rejects_builtin_prefix_collisions(self, name):
        # Built-in mode names are routed by `str.startswith` over policy
        # strings, so no mode name may share a prefix with a built-in.
        colliding_name = name

        class Colliding(strategy_registry.QuantizationStrategy):
            name = colliding_name
            requires_config = True

        with self.assertRaisesRegex(ValueError, "collides"):
            strategy_registry.register_quantization_strategy(Colliding)

    def test_register_allows_custom_prefix_overlap(self):
        # Externally registered modes match only their exact grammar
        # (name, name + "/", name + "_from_"), so two custom modes may
        # share a prefix without ambiguity.
        class Custom(strategy_registry.QuantizationStrategy):
            name = "custom"
            requires_config = True

        class CustomTwo(strategy_registry.QuantizationStrategy):
            name = "custom2"
            requires_config = True

        strategy_registry.register_quantization_strategy(Custom)
        try:
            strategy_registry.register_quantization_strategy(CustomTwo)
            policy = dtype_policies.get("custom2_from_float32")
            self.assertEqual(policy.quantization_mode, "custom2")
        finally:
            strategy_registry.unregister_quantization_strategy("custom")
            strategy_registry.unregister_quantization_strategy("custom2")

    @parameterized.named_parameters(
        ("slash", "my/mode", "must not contain"),
        ("from_separator", "my_from_mode", "must not contain"),
        ("standard_dtype", "float32", "conflicts with a standard dtype"),
        ("mixed_policy", "mixed_custom", "conflicts with a standard dtype"),
    )
    def test_register_rejects_reserved_names(self, name, error):
        # Names containing the policy-grammar separators or shadowing a
        # standard dtype / mixed-precision policy would break ordinary
        # policy-string parsing.
        reserved_name = name

        class Reserved(strategy_registry.QuantizationStrategy):
            name = reserved_name
            requires_config = True

        with self.assertRaisesRegex(ValueError, error):
            strategy_registry.register_quantization_strategy(Reserved)

    def test_registered_name_does_not_capture_ordinary_policies(self):
        # Policy strings are routed by mode name, but only through the
        # quantized grammar (bare name, name + "/", name + "_from_"). A
        # registered mode whose name prefixes ordinary policy strings (like
        # "mixed" prefixing "mixed_bfloat16") must not hijack them.
        class MixedMode(strategy_registry.QuantizationStrategy):
            name = "mixed"
            requires_config = True

        strategy_registry.register_quantization_strategy(MixedMode)
        try:
            policy = dtype_policies.get("mixed_bfloat16")
            self.assertIsNone(policy.quantization_mode)
            self.assertEqual(policy.compute_dtype, "bfloat16")
        finally:
            strategy_registry.unregister_quantization_strategy("mixed")

    def test_register_as_decorator_keeps_the_class(self):
        # Registering returns its argument, so a decorated strategy stays
        # a class and can still be subclassed.
        @strategy_registry.register_quantization_strategy
        class Decorated(strategy_registry.QuantizationStrategy):
            name = "decorated"
            requires_config = True

        try:
            self.assertIsInstance(Decorated, type)
            self.assertIsNotNone(strategy_registry.get_strategy("decorated"))

            class Sub(Decorated):
                name = "decorated_sub"

            self.assertIsInstance(Sub, type)
        finally:
            strategy_registry.unregister_quantization_strategy("decorated")

    def test_register_requires_config_source(self):
        # A mode must be able to produce a config: via config_cls, via
        # requires_config (explicit config mandatory), or by overriding
        # default_config. Registration fails otherwise, not first use.
        class NoConfig(strategy_registry.QuantizationStrategy):
            name = "noconfig"

        with self.assertRaisesRegex(ValueError, "must define `config_cls`"):
            strategy_registry.register_quantization_strategy(NoConfig)


class PolicyCodecCorpusTest(testing.TestCase):
    """Every historical policy-string form parses and round-trips."""

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", "int8_from_float32", "int8", {}),
        (
            "int8_mixed",
            "int8_from_mixed_bfloat16",
            "int8_from_mixed_bfloat16",
            "int8",
            {},
        ),
        (
            "int4_legacy_bare",
            "int4_from_float32",
            "int4_from_float32",
            "int4",
            {},
        ),
        (
            "int4_grouped",
            "int4/128_from_float32",
            "int4/128_from_float32",
            "int4",
            {"block_size": 128},
        ),
        (
            "int4_per_channel",
            "int4/-1_from_float32",
            "int4/-1_from_float32",
            "int4",
            {"block_size": -1},
        ),
        (
            "int4_legacy_none_block",
            "int4/None_from_float32",
            "int4/-1_from_float32",
            "int4",
            {"block_size": -1},
        ),
        (
            "float8",
            "float8_from_float32",
            "float8_from_float32",
            "float8",
            {},
        ),
        (
            "ternary",
            "ternary_from_float32",
            "ternary_from_float32",
            "ternary",
            {},
        ),
        (
            "gptq",
            "gptq/4/128_from_float32",
            "gptq/4/128_from_float32",
            "gptq",
            {"weight_bits": 4, "group_size": 128},
        ),
        (
            "gptq_whole_tensor",
            "gptq/2/-1_from_bfloat16",
            "gptq/2/-1_from_bfloat16",
            "gptq",
            {"weight_bits": 2, "group_size": -1},
        ),
        (
            "gptq_mixed",
            "gptq/8/32_from_mixed_bfloat16",
            "gptq/8/32_from_mixed_bfloat16",
            "gptq",
            {"weight_bits": 8, "group_size": 32},
        ),
        (
            "awq",
            "awq/4/128_from_float32",
            "awq/4/128_from_float32",
            "awq",
            {"weight_bits": 4, "group_size": 128},
        ),
        (
            "awq_per_channel",
            "awq/4/-1_from_float32",
            "awq/4/-1_from_float32",
            "awq",
            {"weight_bits": 4, "group_size": -1},
        ),
    )
    def test_policy_string_corpus(
        self, policy_str, expected_name, expected_mode, expected_params
    ):
        policy = dtype_policies.get(policy_str)
        self.assertEqual(policy.name, expected_name)
        self.assertEqual(policy.quantization_mode, expected_mode)
        for attr, value in expected_params.items():
            self.assertEqual(getattr(policy, attr), value)
        # Serialization round-trip preserves the resolved policy.
        revived = dtype_policies.deserialize(dtype_policies.serialize(policy))
        self.assertEqual(revived.name, expected_name)
        for attr, value in expected_params.items():
            self.assertEqual(getattr(revived, attr), value)

    @parameterized.named_parameters(
        ("no_source", "int8"),
        ("int4_zero_block", "int4/0_from_float32"),
        ("int4_garbage_block", "int4/abc_from_float32"),
        ("gptq_bad_bits", "gptq/5/128_from_float32"),
        ("gptq_missing_group", "gptq/4_from_float32"),
        ("awq_bad_bits", "awq/8/128_from_float32"),
        ("unknown_mode", "int7_from_float32"),
    )
    def test_invalid_policy_strings(self, policy_str):
        with self.assertRaises(ValueError):
            dtype_policies.get(policy_str)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", "QuantizedDTypePolicy"),
        ("int4_legacy_bare", "int4_from_float32", "QuantizedDTypePolicy"),
        ("int4_grouped", "int4/128_from_float32", "Int4DTypePolicy"),
        ("float8", "float8_from_float32", "QuantizedFloat8DTypePolicy"),
        ("ternary", "ternary_from_float32", "QuantizedDTypePolicy"),
        ("gptq", "gptq/4/128_from_float32", "GPTQDTypePolicy"),
        ("awq", "awq/4/128_from_float32", "AWQDTypePolicy"),
    )
    def test_policy_string_class(self, policy_str, class_name):
        policy = dtype_policies.get(policy_str)
        self.assertEqual(type(policy).__name__, class_name)
        self.assertEqual(
            dtype_policies.serialize(policy)["class_name"], class_name
        )


class ToyModeConfig(QuantizationConfig):
    """Config for the toy float16-storage mode used in the test below."""

    def __init__(self):
        super().__init__(None, None)

    @property
    def mode(self):
        return "demo_half"

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


class ToyHalfStrategy(strategy_registry.QuantizationStrategy):
    """A toy quantization mode: store the kernel in float16.

    This is the "add a mode is a registration, not a treasure hunt"
    demonstration: the strategy implements build/call/quantize directly
    against the layer's geometry (`_quantization_geometry()`), so no
    layer class needs editing and no dispatch chain exists to extend.
    """

    name = "demo_half"
    config_cls = ToyModeConfig

    def supports_layer(self, layer):
        return isinstance(layer, layers.Dense)

    def build(self, layer, input_shape, config):
        del config
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=input_shape,
            initializer="zeros",
            dtype="float16",
            trainable=False,
        )

    def quantize(self, layer, config):
        kernel_shape = layer._quantization_geometry().weight_shape
        kernel_value = ops.cast(layer._kernel, "float16")
        del layer._kernel
        layer.quantized_build(kernel_shape, self.name, config)
        layer._kernel.assign(kernel_value)

    def call(self, layer, inputs, training=None):
        x = ops.matmul(inputs, ops.cast(layer._kernel, layer.compute_dtype))
        if layer.bias is not None:
            x = ops.add(x, layer.bias)
        if layer.activation is not None:
            x = layer.activation(x)
        return x


class ToyModeRegistrationTest(testing.TestCase):
    """End-to-end test that a new mode is just a registry entry."""

    def setUp(self):
        super().setUp()
        strategy_registry.register_quantization_strategy(ToyHalfStrategy)

    def tearDown(self):
        strategy_registry.unregister_quantization_strategy("demo_half")
        super().tearDown()

    def test_toy_mode_end_to_end(self):
        layer = layers.Dense(units=3)
        layer.build((None, 4))
        reference_kernel = ops.convert_to_numpy(layer._kernel)

        layer.quantize("demo_half")

        # The kernel is now stored in float16 and the policy is named after
        # the mode, all through the generic machinery.
        self.assertEqual(
            backend.standardize_dtype(layer._kernel.dtype), "float16"
        )
        self.assertEqual(layer.dtype_policy.name, "demo_half_from_float32")
        self.assertEqual(layer.quantization_mode, "demo_half")
        self.assertTrue(layer._is_quantized)

        # The quantized forward pass dispatches through the strategy.
        x = np.random.uniform(-1, 1, size=(2, 4)).astype("float32")
        y = ops.convert_to_numpy(layer(x))
        expected = x @ reference_kernel.astype("float16").astype(
            "float32"
        ) + ops.convert_to_numpy(layer.bias)
        self.assertAllClose(
            y, expected, atol=1e-3, tpu_atol=1e-2, tpu_rtol=5e-2
        )

    def test_toy_mode_through_model_quantize(self):
        model = models.Sequential(
            [layers.Input((4,)), layers.Dense(3, name="target")]
        )
        report = model.quantize("demo_half", verbose=False)
        self.assertEqual(
            model.get_layer("target").dtype_policy.name,
            "demo_half_from_float32",
        )
        self.assertIn(
            "target", "".join(path for path, _, _ in report.quantized)
        )

    def test_toy_mode_rejects_unsupported_layer(self):
        # `supports_layer` only claims Dense; an Embedding must be skipped
        # with NotImplementedError, like any unsupported (layer, mode) pair.
        layer = layers.Embedding(5, 3)
        layer.build()
        with self.assertRaises(NotImplementedError):
            layer.quantize("demo_half")


class ToyNoneConfig(ToyModeConfig):
    @property
    def mode(self):
        return "demo_none"


class ToyUnsupportedStrategy(ToyHalfStrategy):
    """A registered mode that claims no layer at all."""

    name = "demo_none"
    config_cls = ToyNoneConfig

    def supports_layer(self, layer):
        return False


class QuantizeTransactionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        strategy_registry.register_quantization_strategy(ToyUnsupportedStrategy)

    def tearDown(self):
        strategy_registry.unregister_quantization_strategy("demo_none")
        super().tearDown()

    def test_unsupported_mode_leaves_layer_untouched(self):
        # An unsupported (layer, mode) pair must be rejected before any
        # state is mutated, so `Model.quantize` records the layer as
        # skipped and the layer stays fully usable and quantizable.
        layer = layers.Dense(4)
        layer.build((None, 8))
        with self.assertRaises(NotImplementedError):
            layer.quantize("demo_none", config=ToyNoneConfig())
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(getattr(layer, "_is_quantized", False))
        self.assertIsNone(layer.quantization_mode)
        # The layer is still float and still quantizable.
        layer.quantize("int8")
        self.assertEqual(layer.quantization_mode, "int8")
