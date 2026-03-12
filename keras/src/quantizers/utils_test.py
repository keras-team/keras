from absl.testing import parameterized

from keras.src import layers
from keras.src import testing
from keras.src.quantizers import utils


class UtilsTest(testing.TestCase):
    @parameterized.named_parameters(
        ("none_filter", None, "dense", True),
        ("regex_match", "dense", "dense_1", True),
        ("regex_no_match", "conv", "dense_1", False),
        ("list_match", ["dense", "conv"], "dense_1", True),
        ("list_no_match", ["conv", "pool"], "dense_1", False),
        ("callable_match", lambda l: "dense" in l.name, "dense_1", True),
        ("callable_no_match", lambda l: "conv" in l.name, "dense_1", False),
    )
    def test_should_quantize_layer(self, filters, layer_name, expected):
        layer = layers.Layer(name=layer_name)
        self.assertEqual(utils.should_quantize_layer(layer, filters), expected)
