from keras.src import testing
from keras.src.backend import config


class FloatxTest(testing.TestCase):
    def test_default_floatx(self):
        self.assertEqual(config.floatx(), "float32")

    def test_set_floatx_float64(self):
        original = config.floatx()
        try:
            config.set_floatx("float64")
            self.assertEqual(config.floatx(), "float64")
        finally:
            config.set_floatx(original)

    def test_set_floatx_float16(self):
        original = config.floatx()
        try:
            config.set_floatx("float16")
            self.assertEqual(config.floatx(), "float16")
        finally:
            config.set_floatx(original)

    def test_set_floatx_bfloat16(self):
        original = config.floatx()
        try:
            config.set_floatx("bfloat16")
            self.assertEqual(config.floatx(), "bfloat16")
        finally:
            config.set_floatx(original)

    def test_set_floatx_invalid_raises(self):
        with self.assertRaises(ValueError):
            config.set_floatx("int32")

    def test_set_floatx_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            config.set_floatx("invalid")


class EpsilonTest(testing.TestCase):
    def test_default_epsilon(self):
        self.assertEqual(config.epsilon(), 1e-7)

    def test_set_epsilon(self):
        original = config.epsilon()
        try:
            config.set_epsilon(1e-5)
            self.assertEqual(config.epsilon(), 1e-5)
        finally:
            config.set_epsilon(original)

    def test_set_epsilon_very_small(self):
        original = config.epsilon()
        try:
            config.set_epsilon(1e-12)
            self.assertEqual(config.epsilon(), 1e-12)
        finally:
            config.set_epsilon(original)


class ImageDataFormatTest(testing.TestCase):
    def test_default_image_data_format(self):
        self.assertEqual(config.image_data_format(), "channels_last")

    def test_set_channels_first(self):
        original = config.image_data_format()
        try:
            config.set_image_data_format("channels_first")
            self.assertEqual(config.image_data_format(), "channels_first")
        finally:
            config.set_image_data_format(original)

    def test_set_channels_last(self):
        config.set_image_data_format("channels_last")
        self.assertEqual(config.image_data_format(), "channels_last")

    def test_invalid_data_format_raises(self):
        with self.assertRaises(ValueError):
            config.set_image_data_format("invalid")

    def test_case_insensitive(self):
        original = config.image_data_format()
        try:
            config.set_image_data_format("CHANNELS_FIRST")
            self.assertEqual(config.image_data_format(), "channels_first")
        finally:
            config.set_image_data_format(original)


class StandardizeDataFormatTest(testing.TestCase):
    def test_none_returns_default(self):
        result = config.standardize_data_format(None)
        self.assertEqual(result, config.image_data_format())

    def test_channels_first(self):
        result = config.standardize_data_format("channels_first")
        self.assertEqual(result, "channels_first")

    def test_channels_last(self):
        result = config.standardize_data_format("channels_last")
        self.assertEqual(result, "channels_last")

    def test_case_insensitive(self):
        result = config.standardize_data_format("CHANNELS_LAST")
        self.assertEqual(result, "channels_last")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            config.standardize_data_format("invalid")


class FlashAttentionTest(testing.TestCase):
    def test_enable_flash_attention(self):
        config.enable_flash_attention()
        result = config.is_flash_attention_enabled()
        # When enabled, returns None (not False)
        self.assertIsNone(result)

    def test_disable_flash_attention(self):
        config.disable_flash_attention()
        self.assertFalse(config.is_flash_attention_enabled())
        # Re-enable
        config.enable_flash_attention()

    def test_toggle_flash_attention(self):
        config.enable_flash_attention()
        result_enabled = config.is_flash_attention_enabled()
        config.disable_flash_attention()
        result_disabled = config.is_flash_attention_enabled()
        self.assertIsNone(result_enabled)
        self.assertFalse(result_disabled)
        config.enable_flash_attention()


class KerasHomeTest(testing.TestCase):
    def test_keras_home_returns_string(self):
        home = config.keras_home()
        self.assertIsInstance(home, str)
        self.assertTrue(len(home) > 0)
