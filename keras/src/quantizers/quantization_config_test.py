import os

from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantization_config import validate_and_resolve_config
from keras.src.quantizers.quantizers import AbsMaxQuantizer


class QuantizationConfigTest(testing.TestCase):
    def test_base_quantization_config(self):
        config = QuantizationConfig()
        with self.assertRaises(NotImplementedError):
            _ = config.mode

    def test_int8_quantization_config_valid(self):
        config = Int8QuantizationConfig()
        self.assertEqual(config.mode, "int8")
        self.assertIsNone(config.weight_quantizer)

        # Valid weight quantizer
        q = AbsMaxQuantizer(axis=0, value_range=(-127, 127))
        config = Int8QuantizationConfig(weight_quantizer=q)
        self.assertEqual(config.weight_quantizer, q)

    def test_int8_quantization_config_invalid(self):
        # Invalid value_range
        with self.assertRaisesRegex(ValueError, "value_range"):
            AbsMaxQuantizer(axis=0, value_range=(-256, 256))

    def test_int4_quantization_config_valid(self):
        config = Int4QuantizationConfig()
        self.assertEqual(config.mode, "int4")
        self.assertIsNone(config.weight_quantizer)

        # Valid weight quantizer
        q = AbsMaxQuantizer(axis=0, value_range=(-8, 7))
        config = Int4QuantizationConfig(weight_quantizer=q)
        self.assertEqual(config.weight_quantizer, q)

    def test_int4_quantization_config_invalid(self):
        # Invalid value_range
        q = AbsMaxQuantizer(axis=0, value_range=(-127, 127))
        with self.assertRaisesRegex(ValueError, "value_range"):
            Int4QuantizationConfig(weight_quantizer=q)

    def test_quantization_config_serialization(self):
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=0),
            activation_quantizer=AbsMaxQuantizer(axis=-1),
        )
        serialized = config.get_config()
        deserialized = Int8QuantizationConfig.from_config(serialized)
        self.assertIsInstance(deserialized, Int8QuantizationConfig)
        self.assertIsInstance(deserialized.weight_quantizer, AbsMaxQuantizer)
        self.assertIsInstance(
            deserialized.activation_quantizer, AbsMaxQuantizer
        )
        self.assertEqual(deserialized.weight_quantizer.axis, (0,))
        self.assertEqual(deserialized.activation_quantizer.axis, (-1,))

    def test_validate_and_resolve_config(self):
        # 1. String mode
        config = validate_and_resolve_config("int8", None)
        self.assertIsInstance(config, Int8QuantizationConfig)
        self.assertEqual(config.mode, "int8")

        config = validate_and_resolve_config("int4", None)
        self.assertIsInstance(config, Int4QuantizationConfig)
        self.assertEqual(config.mode, "int4")

        # 2. Config object
        config_in = Int8QuantizationConfig()
        config_out = validate_and_resolve_config(None, config_in)
        self.assertIs(config_out, config_in)

        # 3. Mode + Config (matching)
        config_in = Int8QuantizationConfig()
        config_out = validate_and_resolve_config("int8", config_in)
        self.assertIs(config_out, config_in)

        # 4. Mode + Config (mismatch)
        config_in = Int8QuantizationConfig()
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            validate_and_resolve_config("int4", config_in)

        # 5. Invalid mode
        with self.assertRaisesRegex(ValueError, "Invalid quantization mode"):
            validate_and_resolve_config("invalid_mode", None)

        # 6. GPTQ without config
        with self.assertRaisesRegex(ValueError, "must pass a `GPTQConfig`"):
            validate_and_resolve_config("gptq", None)

        # 7. Contradictory config
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            validate_and_resolve_config("gptq", Int8QuantizationConfig())

        # 8. GPTQ with invalid config type (but correct mode)
        class FakeGPTQConfig(QuantizationConfig):
            @property
            def mode(self):
                return "gptq"

        with self.assertRaisesRegex(ValueError, "requires a valid `config`"):
            validate_and_resolve_config("gptq", FakeGPTQConfig())

    def test_int8_quantization_config_output_dtype_mismatch(self):
        # Invalid output_dtype
        q = AbsMaxQuantizer(
            axis=0, value_range=(-127, 127), output_dtype="int16"
        )
        with self.assertRaisesRegex(ValueError, "output_dtype='int8'"):
            Int8QuantizationConfig(weight_quantizer=q)

    def test_int4_quantization_config_output_dtype_mismatch(self):
        # Invalid output_dtype
        q = AbsMaxQuantizer(axis=0, value_range=(-8, 7), output_dtype="int16")
        with self.assertRaisesRegex(ValueError, "output_dtype='int8'"):
            Int4QuantizationConfig(weight_quantizer=q)

    def test_model_save_and_load(self):
        """
        Test custom quantizer serialization for model save and load.
        """
        # Setup
        weight_range = (-100, 100)
        custom_quantizer = AbsMaxQuantizer(axis=0, value_range=weight_range)
        config = Int8QuantizationConfig(
            weight_quantizer=custom_quantizer,
            activation_quantizer=None,
        )

        layer = layers.Dense(10)
        layer.build((None, 5))
        layer.quantize("int8", config=config)

        model = models.Sequential([layer])
        model.build((None, 5))

        # Save to temp file
        filepath = os.path.join(self.get_temp_dir(), "quantized_model.keras")
        model.save(filepath)

        # Load back
        loaded_model = saving.load_model(filepath)

        # Verify
        loaded_layer = loaded_model.layers[0]
        self.assertIsInstance(
            loaded_layer.quantization_config, Int8QuantizationConfig
        )

        quantizer = loaded_layer.quantization_config.weight_quantizer
        self.assertIsInstance(quantizer, AbsMaxQuantizer)
        self.assertEqual(quantizer.axis, (0,))
        self.assertAllEqual(quantizer.value_range, weight_range)
        self.assertIsNone(loaded_layer.quantization_config.activation_quantizer)
        self.assertTrue(loaded_layer._is_quantized)
