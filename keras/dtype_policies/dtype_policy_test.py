from keras.dtype_policies.dtype_policy import DTypePolicy
from keras.dtype_policies.dtype_policy import FloatDTypePolicy
from keras.dtype_policies.dtype_policy import QuantizedDTypePolicy
from keras.dtype_policies.dtype_policy import dtype_policy
from keras.dtype_policies.dtype_policy import set_dtype_policy
from keras.testing import test_case


class DTypePolicyTest(test_case.TestCase):
    def test_initialization_valid_name(self):
        """Test initialization with a valid name."""
        policy = DTypePolicy("mixed_float16")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_initialization_invalid_name(self):
        """Test initialization with an invalid name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            DTypePolicy("invalid_name")

    def test_initialization_non_string_name(self):
        """Test initialization with a non-string name."""
        with self.assertRaisesRegex(TypeError, "'name' must be a string"):
            DTypePolicy(123)

    def test_properties_mixed_float16(self):
        """Test properties for 'mixed_float16'."""
        policy = DTypePolicy("mixed_float16")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_properties_mixed_bfloat16(self):
        """Test properties for 'mixed_bfloat16'."""
        policy = DTypePolicy("mixed_bfloat16")
        self.assertEqual(policy.compute_dtype, "bfloat16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_initialization_with_invalid_name_behaviour(self):
        """Test initialization behavior with an invalid name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            DTypePolicy("invalid_name")

    def test_properties(self):
        """Test variable_dtype, compute_dtype, and name properties."""
        policy = DTypePolicy("mixed_float16")
        self.assertEqual(policy.variable_dtype, "float32")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.name, "mixed_float16")

    def test_repr(self):
        """Test __repr__ method."""
        policy = DTypePolicy("mixed_float16")
        self.assertEqual(repr(policy), '<FloatDTypePolicy "mixed_float16">')

    def test_get_config_from_config(self):
        """Test get_config and from_config methods."""
        policy = DTypePolicy("mixed_float16")
        config = policy.get_config()
        self.assertEqual(config, {"name": "mixed_float16"})

        new_policy = DTypePolicy.from_config(config)
        self.assertEqual(new_policy.name, "mixed_float16")


class FloatDTypePolicyTest(test_case.TestCase):
    def test_initialization_valid_name(self):
        """Test initialization with a valid name."""
        policy = FloatDTypePolicy("mixed_float16")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_initialization_invalid_name(self):
        """Test initialization with an invalid name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            FloatDTypePolicy("invalid_name")

    def test_initialization_non_string_name(self):
        """Test initialization with a non-string name."""
        with self.assertRaisesRegex(TypeError, "'name' must be a string"):
            FloatDTypePolicy(123)

    def test_properties_mixed_float16(self):
        """Test properties for 'mixed_float16'."""
        policy = FloatDTypePolicy("mixed_float16")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_properties_mixed_bfloat16(self):
        """Test properties for 'mixed_bfloat16'."""
        policy = FloatDTypePolicy("mixed_bfloat16")
        self.assertEqual(policy.compute_dtype, "bfloat16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_initialization_with_invalid_name_behaviour(self):
        """Test initialization behavior with an invalid name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            FloatDTypePolicy("invalid_name")

    def test_properties(self):
        """Test variable_dtype, compute_dtype, and name properties."""
        policy = FloatDTypePolicy("mixed_float16")
        self.assertEqual(policy.variable_dtype, "float32")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.name, "mixed_float16")

    def test_properties_uint8(self):
        """Test properties for 'uint8'."""
        policy = FloatDTypePolicy("uint8")
        self.assertEqual(policy.compute_dtype, "uint8")
        self.assertEqual(policy.variable_dtype, "uint8")
        self.assertEqual(policy.name, "uint8")

    def test_repr(self):
        """Test __repr__ method."""
        policy = FloatDTypePolicy("mixed_float16")
        self.assertEqual(repr(policy), '<FloatDTypePolicy "mixed_float16">')

    def test_get_config_from_config(self):
        """Test get_config and from_config methods."""
        policy = FloatDTypePolicy("mixed_float16")
        config = policy.get_config()
        self.assertEqual(config, {"name": "mixed_float16"})

        new_policy = FloatDTypePolicy.from_config(config)
        self.assertEqual(new_policy.name, "mixed_float16")


class QuantizedDTypePolicyTest(test_case.TestCase):
    def test_initialization_valid_name(self):
        """Test initialization with a valid name."""
        policy = QuantizedDTypePolicy("int8_from_mixed_float16")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_initialization_invalid_name(self):
        """Test initialization with an invalid name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            QuantizedDTypePolicy("invalid_name")

    def test_initialization_non_string_name(self):
        """Test initialization with a non-string name."""
        with self.assertRaisesRegex(TypeError, "'name' must be a string"):
            QuantizedDTypePolicy(123)

    def test_properties_mixed_float16(self):
        """Test properties for 'mixed_float16'."""
        policy = QuantizedDTypePolicy("int8_from_mixed_float16")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_properties_mixed_bfloat16(self):
        """Test properties for 'mixed_bfloat16'."""
        policy = QuantizedDTypePolicy("int8_from_mixed_bfloat16")
        self.assertEqual(policy.compute_dtype, "bfloat16")
        self.assertEqual(policy.variable_dtype, "float32")

    def test_initialization_with_invalid_name_behaviour(self):
        """Test initialization behavior with an invalid name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            QuantizedDTypePolicy("invalid_name")

    def test_properties(self):
        """Test variable_dtype, compute_dtype, and name properties."""
        policy = QuantizedDTypePolicy("int8_from_mixed_float16")
        self.assertEqual(policy.variable_dtype, "float32")
        self.assertEqual(policy.compute_dtype, "float16")
        self.assertEqual(policy.name, "int8_from_mixed_float16")

    def test_repr(self):
        """Test __repr__ method."""
        policy = QuantizedDTypePolicy("int8_from_mixed_float16")
        self.assertEqual(
            repr(policy), '<QuantizedDTypePolicy "int8_from_mixed_float16">'
        )

    def test_get_config_from_config(self):
        """Test get_config and from_config methods."""
        policy = QuantizedDTypePolicy("int8_from_mixed_float16")
        config = policy.get_config()
        self.assertEqual(config, {"name": "int8_from_mixed_float16"})

        new_policy = QuantizedDTypePolicy.from_config(config)
        self.assertEqual(new_policy.name, "int8_from_mixed_float16")


class DTypePolicyGlobalFunctionsTest(test_case.TestCase):
    def setUp(self):
        """Reset the global dtype policy before each test."""
        set_dtype_policy("float32")

    def test_set_dtype_policy_valid_string(self):
        """Test set_dtype_policy with a valid string."""
        set_dtype_policy("mixed_float16")
        policy = dtype_policy()
        self.assertEqual(policy.name, "mixed_float16")

    def test_set_dtype_policy_valid_string_quantized(self):
        """Test set_dtype_policy with a valid string."""
        set_dtype_policy("int8_from_mixed_float16")
        policy = dtype_policy()
        self.assertEqual(policy.name, "int8_from_mixed_float16")

    def test_set_dtype_policy_valid_policy(self):
        """Test set_dtype_policy with a valid FloatDTypePolicy object."""
        policy_obj = FloatDTypePolicy("mixed_float16")
        set_dtype_policy(policy_obj)
        policy = dtype_policy()
        self.assertEqual(policy.name, "mixed_float16")

    def test_set_dtype_policy_valid_policy_quantized(self):
        """Test set_dtype_policy with a valid FloatDTypePolicy object."""
        policy_obj = QuantizedDTypePolicy("int8_from_mixed_float16")
        set_dtype_policy(policy_obj)
        policy = dtype_policy()
        self.assertEqual(policy.name, "int8_from_mixed_float16")

    def test_set_dtype_policy_invalid(self):
        """Test set_dtype_policy with an invalid input."""
        with self.assertRaisesRegex(ValueError, "Invalid `policy` argument"):
            set_dtype_policy(12345)

    def test_dtype_policy_default(self):
        """Test dtype_policy default value."""
        policy = dtype_policy()
        self.assertEqual(policy.name, "float32")


class FloatDTypePolicyEdgeCasesTest(test_case.TestCase):
    def test_empty_name(self):
        """Test initialization with an empty name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            FloatDTypePolicy("")

    def test_special_character_name(self):
        """Test initialization with special characters in the name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            FloatDTypePolicy("@mixed_float16!")

    def test_very_long_name(self):
        """Test initialization with a very long name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            FloatDTypePolicy("mixed_float16" * 100)

    def test_almost_valid_name(self):
        """Test initialization with a name close to a valid one."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            FloatDTypePolicy("mixed_float15")


class QuantizedDTypePolicyEdgeCasesTest(test_case.TestCase):
    def test_empty_name(self):
        """Test initialization with an empty name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            QuantizedDTypePolicy("")

    def test_special_character_name(self):
        """Test initialization with special characters in the name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            QuantizedDTypePolicy("@int8_from_mixed_float16!")

    def test_very_long_name(self):
        """Test initialization with a very long name."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            QuantizedDTypePolicy("int8_from_mixed_float16" * 100)

    def test_almost_valid_name(self):
        """Test initialization with a name close to a valid one."""
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            QuantizedDTypePolicy("int7_from_mixed_float16")


class DTypePolicyGlobalFunctionsEdgeCasesTest(test_case.TestCase):
    def setUp(self):
        """Reset the global dtype policy before each test."""
        set_dtype_policy("float32")

    def test_set_policy_multiple_times(self):
        """Test setting the policy multiple times in a row."""
        set_dtype_policy("mixed_float16")
        policy = dtype_policy()
        self.assertEqual(policy.name, "mixed_float16")

        set_dtype_policy("float32")
        policy = dtype_policy()
        self.assertEqual(policy.name, "float32")

    def test_set_policy_none(self):
        """Test setting the policy to None."""
        with self.assertRaisesRegex(ValueError, "Invalid `policy` argument"):
            set_dtype_policy(None)
