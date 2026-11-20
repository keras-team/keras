from absl.testing import parameterized

from keras.src import testing
from keras.src.backend import config
from keras.src.backend.common.standardization import standardize_data_format


class StandardizeDataFormatTest(testing.TestCase):
    @parameterized.named_parameters(
        ("channels_first", "channels_first", "channels_first"),
        ("channels_last", "channels_last", "channels_last"),
        ("channels_first_upper", "CHANNELS_FIRST", "channels_first"),
        ("channels_last_mixed", "Channels_Last", "channels_last"),
    )
    def test_valid_data_format(self, data_format, expected):
        self.assertEqual(standardize_data_format(data_format), expected)

    @parameterized.named_parameters(
        ("channels_first", "channels_first"),
        ("channels_last", "channels_last"),
    )
    def test_none_uses_config_default(self, default_format):
        self.addCleanup(
            config.set_image_data_format, config.image_data_format()
        )
        config.set_image_data_format(default_format)
        self.assertEqual(standardize_data_format(None), default_format)

    def test_invalid_data_format_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"must be one of \{'channels_first', 'channels_last'\}"
        ):
            standardize_data_format("last_channels")
