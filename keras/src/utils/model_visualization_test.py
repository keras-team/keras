"""Tests for keras.src.utils.model_visualization."""

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.utils.model_visualization import check_pydot
from keras.src.utils.model_visualization import get_layer_activation_name
from keras.src.utils.model_visualization import make_layer_label
from keras.src.utils.model_visualization import model_to_dot


class CheckPydotTest(testing.TestCase):
    def test_check_pydot_returns_bool(self):
        result = check_pydot()
        self.assertIsInstance(result, bool)

    def test_pydot_available(self):
        # We know pydot is installed in this environment
        self.assertTrue(check_pydot())


class GetLayerActivationNameTest(testing.TestCase):
    def test_relu_activation(self):
        layer = layers.Dense(10, activation="relu")
        layer.build((None, 5))
        name = get_layer_activation_name(layer)
        self.assertEqual(name, "relu")

    def test_sigmoid_activation(self):
        layer = layers.Dense(10, activation="sigmoid")
        layer.build((None, 5))
        name = get_layer_activation_name(layer)
        self.assertEqual(name, "sigmoid")

    def test_linear_activation(self):
        layer = layers.Dense(10, activation="linear")
        layer.build((None, 5))
        name = get_layer_activation_name(layer)
        self.assertEqual(name, "linear")


class MakeLayerLabelTest(testing.TestCase):
    def _build_dense_layer(self):
        """Build a Dense layer with known input/output."""
        inp = layers.Input(shape=(5,))
        dense = layers.Dense(10, name="test_dense")
        dense(inp)
        return dense

    def test_label_contains_class_name(self):
        layer = self._build_dense_layer()
        label = make_layer_label(
            layer,
            show_layer_names=False,
            show_layer_activations=False,
            show_dtype=False,
            show_shapes=False,
            show_trainable=False,
        )
        self.assertIn("Dense", label)

    def test_label_with_layer_names(self):
        layer = self._build_dense_layer()
        label = make_layer_label(
            layer,
            show_layer_names=True,
            show_layer_activations=False,
            show_dtype=False,
            show_shapes=False,
            show_trainable=False,
        )
        self.assertIn("test_dense", label)

    def test_label_with_activations(self):
        inp = layers.Input(shape=(5,))
        dense = layers.Dense(10, activation="relu", name="relu_dense")
        dense(inp)
        label = make_layer_label(
            dense,
            show_layer_names=True,
            show_layer_activations=True,
            show_dtype=False,
            show_shapes=False,
            show_trainable=False,
        )
        self.assertIn("relu", label)

    def test_label_with_shapes(self):
        layer = self._build_dense_layer()
        label = make_layer_label(
            layer,
            show_layer_names=True,
            show_layer_activations=False,
            show_dtype=False,
            show_shapes=True,
            show_trainable=False,
        )
        self.assertIn("Output shape", label)

    def test_label_with_trainable(self):
        layer = self._build_dense_layer()
        label = make_layer_label(
            layer,
            show_layer_names=True,
            show_layer_activations=False,
            show_dtype=False,
            show_shapes=False,
            show_trainable=True,
        )
        self.assertIn("Trainable", label)

    def test_invalid_kwargs_raises(self):
        layer = self._build_dense_layer()
        with self.assertRaisesRegex(ValueError, "Invalid kwargs"):
            make_layer_label(
                layer,
                show_layer_names=True,
                show_layer_activations=False,
                show_dtype=False,
                show_shapes=False,
                show_trainable=False,
                invalid_arg=True,
            )


class ModelToDotTest(testing.TestCase):
    def test_functional_model_returns_dot(self):
        import pydot

        inp = layers.Input(shape=(10,))
        out = layers.Dense(5)(inp)
        model = models.Model(inputs=inp, outputs=out)
        dot = model_to_dot(model)
        self.assertIsInstance(dot, pydot.Dot)

    def test_sequential_model_returns_dot(self):
        import pydot

        model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(4)])
        dot = model_to_dot(model)
        self.assertIsInstance(dot, pydot.Dot)

    def test_unbuilt_model_raises(self):
        model = models.Sequential([layers.Dense(4)])
        with self.assertRaisesRegex(ValueError, "not yet been built"):
            model_to_dot(model)

    def test_show_shapes(self):
        inp = layers.Input(shape=(10,))
        out = layers.Dense(5)(inp)
        model = models.Model(inputs=inp, outputs=out)
        dot = model_to_dot(model, show_shapes=True)
        dot_str = dot.to_string()
        self.assertIn("Output shape", dot_str)

    def test_show_layer_names(self):
        inp = layers.Input(shape=(10,), name="my_input")
        out = layers.Dense(5, name="my_dense")(inp)
        model = models.Model(inputs=inp, outputs=out)
        dot = model_to_dot(model, show_layer_names=True)
        dot_str = dot.to_string()
        self.assertIn("my_dense", dot_str)

    def test_rankdir_lr(self):
        inp = layers.Input(shape=(10,))
        out = layers.Dense(5)(inp)
        model = models.Model(inputs=inp, outputs=out)
        dot = model_to_dot(model, rankdir="LR")
        # Check the rank direction is set
        dot_str = dot.to_string()
        self.assertIn("LR", dot_str)

    def test_subgraph_returns_cluster(self):
        import pydot

        inp = layers.Input(shape=(10,))
        out = layers.Dense(5)(inp)
        model = models.Model(inputs=inp, outputs=out)
        dot = model_to_dot(model, subgraph=True)
        self.assertIsInstance(dot, pydot.Cluster)

    def test_deprecated_layer_range_raises(self):
        inp = layers.Input(shape=(10,))
        out = layers.Dense(5)(inp)
        model = models.Model(inputs=inp, outputs=out)
        with self.assertRaisesRegex(ValueError, "layer_range"):
            model_to_dot(model, layer_range=["dense"])

    def test_multi_input_model(self):
        import pydot

        inp1 = layers.Input(shape=(5,))
        inp2 = layers.Input(shape=(3,))
        out = layers.Concatenate()([inp1, inp2])
        model = models.Model(inputs=[inp1, inp2], outputs=out)
        dot = model_to_dot(model)
        self.assertIsInstance(dot, pydot.Dot)


if __name__ == "__main__":
    testing.run_tests()
