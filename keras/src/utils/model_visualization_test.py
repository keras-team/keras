from keras.src import layers
from keras.src.testing import test_case
from keras.src.utils import model_visualization


class ModelVisualizationTest(test_case.TestCase):
    def test_make_layer_label_escapes_html_metacharacters(self):
        # Layer names only disallow "/", so "<", ">", "&" and '"' are valid and
        # must be entity-escaped before being placed in the Graphviz HTML-like
        # label. Otherwise a name can close the label and inject DOT attributes.
        name = 'bad>, tooltip="x'
        layer = layers.Dense(1, name=name)
        label = model_visualization.make_layer_label(
            layer,
            show_layer_names=True,
            show_layer_activations=False,
            show_dtype=False,
            show_shapes=False,
            show_trainable=False,
        )
        self.assertNotIn(name, label)
        self.assertNotIn('tooltip="x', label)
        self.assertIn("bad&gt;, tooltip=&quot;x", label)
