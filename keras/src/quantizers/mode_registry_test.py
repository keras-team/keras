from absl.testing import parameterized

from keras.src import dtype_policies
from keras.src import testing
from keras.src.dtype_policies.dtype_policy import QUANTIZATION_MODES
from keras.src.quantizers import mode_registry


class ModeRegistryTest(testing.TestCase):
    def test_builtin_modes_match_public_tuple(self):
        # The registration order is observable (validation error messages
        # render the registered-names tuple), so it must stay identical to
        # the public QUANTIZATION_MODES constant.
        self.assertEqual(
            mode_registry.registered_mode_names(), QUANTIZATION_MODES
        )
        for name in QUANTIZATION_MODES:
            self.assertIsNotNone(mode_registry.get_mode(name))

    def test_unknown_mode(self):
        self.assertIsNone(mode_registry.get_mode("bogus"))
        self.assertFalse(mode_registry.is_registered("bogus"))

    def test_register_requires_name(self):
        class Nameless(mode_registry.QuantizationMode):
            requires_config = True

        with self.assertRaisesRegex(ValueError, "non-empty string `name`"):
            mode_registry.register_quantization_mode(Nameless)

    def test_register_rejects_duplicates(self):
        class Duplicate(mode_registry.QuantizationMode):
            name = "int8"
            requires_config = True

        with self.assertRaisesRegex(ValueError, "already registered"):
            mode_registry.register_quantization_mode(Duplicate)

    @parameterized.named_parameters(
        ("existing_builtin_is_prefix", "int42"),
        ("new_is_prefix_of_builtin", "in"),
    )
    def test_register_rejects_builtin_prefix_collisions(self, name):
        # Built-in mode names are routed by `str.startswith` over policy
        # strings, so no mode name may share a prefix with a built-in.
        colliding_name = name

        class Colliding(mode_registry.QuantizationMode):
            name = colliding_name
            requires_config = True

        with self.assertRaisesRegex(ValueError, "collides"):
            mode_registry.register_quantization_mode(Colliding)

    def test_register_allows_custom_prefix_overlap(self):
        # Externally registered modes match only their exact grammar
        # (name, name + "/", name + "_from_"), so two custom modes may
        # share a prefix without ambiguity.
        class Custom(mode_registry.QuantizationMode):
            name = "custom"
            requires_config = True

        class CustomTwo(mode_registry.QuantizationMode):
            name = "custom2"
            requires_config = True

        mode_registry.register_quantization_mode(Custom)
        try:
            mode_registry.register_quantization_mode(CustomTwo)
            policy = dtype_policies.get("custom2_from_float32")
            self.assertEqual(policy.quantization_mode, "custom2")
        finally:
            mode_registry.unregister_quantization_mode("custom")
            mode_registry.unregister_quantization_mode("custom2")

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

        class Reserved(mode_registry.QuantizationMode):
            name = reserved_name
            requires_config = True

        with self.assertRaisesRegex(ValueError, error):
            mode_registry.register_quantization_mode(Reserved)

    def test_registered_name_does_not_capture_ordinary_policies(self):
        # Policy strings are routed by mode name, but only through the
        # quantized grammar (bare name, name + "/", name + "_from_"). A
        # registered mode whose name prefixes ordinary policy strings (like
        # "mixed" prefixing "mixed_bfloat16") must not hijack them.
        class MixedMode(mode_registry.QuantizationMode):
            name = "mixed"
            requires_config = True

        mode_registry.register_quantization_mode(MixedMode)
        try:
            policy = dtype_policies.get("mixed_bfloat16")
            self.assertIsNone(policy.quantization_mode)
            self.assertEqual(policy.compute_dtype, "bfloat16")
        finally:
            mode_registry.unregister_quantization_mode("mixed")

    def test_register_as_decorator_keeps_the_class(self):
        # Registering returns its argument, so a decorated descriptor stays
        # a class and can still be subclassed.
        @mode_registry.register_quantization_mode
        class Decorated(mode_registry.QuantizationMode):
            name = "decorated"
            requires_config = True

        try:
            self.assertIsInstance(Decorated, type)
            self.assertIsNotNone(mode_registry.get_mode("decorated"))

            class Sub(Decorated):
                name = "decorated_sub"

            self.assertIsInstance(Sub, type)
        finally:
            mode_registry.unregister_quantization_mode("decorated")

    def test_register_requires_config_source(self):
        # A mode must be able to produce a config: via config_cls, via
        # requires_config (explicit config mandatory), or by overriding
        # default_config. Registration fails otherwise, not first use.
        class NoConfig(mode_registry.QuantizationMode):
            name = "noconfig"

        with self.assertRaisesRegex(ValueError, "must define `config_cls`"):
            mode_registry.register_quantization_mode(NoConfig)


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
