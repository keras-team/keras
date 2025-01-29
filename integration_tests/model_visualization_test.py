import re

import keras
from keras.src import testing
from keras.src.utils import model_to_dot
from keras.src.utils import plot_model


def parse_text_from_html(html):
    pattern = r"<font[^>]*>(.*?)</font>"
    matches = re.findall(pattern, html)

    for match in matches:
        clean_text = re.sub(r"<[^>]*>", "", match)
        return clean_text
    return ""


def get_node_text(node):
    attributes = node.get_attributes()

    if "label" in attributes:
        html = node.get_attributes()["label"]
        return parse_text_from_html(html)
    else:
        return None


def get_edge_dict(dot):
    node_dict = dict()
    for node in dot.get_nodes():
        node_dict[node.get_name()] = get_node_text(node)

    edge_dict = dict()
    for edge in dot.get_edges():
        edge_dict[node_dict[edge.get_source()]] = node_dict[
            edge.get_destination()
        ]

    return edge_dict


class ModelVisualizationTest(testing.TestCase):
    def test_plot_sequential_model(self):
        model = keras.Sequential(
            [
                keras.Input((3,), name="input"),
                keras.layers.Dense(4, activation="relu", name="dense"),
                keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
            ]
        )

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(edge_dict["dense (Dense)"], "dense_1 (Dense)")

        file_name = "sequential.png"
        plot_model(model, file_name)
        self.assertFileExists(file_name)

        file_name = "sequential-show_shapes.png"
        plot_model(model, file_name, show_shapes=True)
        self.assertFileExists(file_name)

        file_name = "sequential-show_shapes-show_dtype.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
        )
        self.assertFileExists(file_name)

        file_name = "sequential-show_shapes-show_dtype-show_layer_names.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
        )
        self.assertFileExists(file_name)

        file_name = "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
        )
        self.assertFileExists(file_name)

        file_name = "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

        file_name = "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            rankdir="LR",
        )
        self.assertFileExists(file_name)

        file_name = "sequential-show_layer_activations-show_trainable.png"
        plot_model(
            model,
            file_name,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

    def test_plot_functional_model(self):
        inputs = keras.Input((3,), name="input")
        x = keras.layers.Dense(
            4, activation="relu", trainable=False, name="dense"
        )(inputs)
        residual = x
        x = keras.layers.Dense(4, activation="relu", name="dense_1")(x)
        x = keras.layers.Dense(4, activation="relu", name="dense_2")(x)
        x = keras.layers.Dense(4, activation="relu", name="dense_3")(x)
        x += residual
        residual = x
        x = keras.layers.Dense(4, activation="relu", name="dense_4")(x)
        x = keras.layers.Dense(4, activation="relu", name="dense_5")(x)
        x = keras.layers.Dense(4, activation="relu", name="dense_6")(x)
        x += residual
        x = keras.layers.Dropout(0.5, name="dropout")(x)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_7")(x)

        model = keras.Model(inputs, outputs)

        edge_dict = get_edge_dict(model_to_dot(model))

        self.assertEqual(edge_dict["input (InputLayer)"], "dense (Dense)")
        self.assertEqual(edge_dict["dense (Dense)"], "add (Add)")
        self.assertEqual(edge_dict["dense_1 (Dense)"], "dense_2 (Dense)")
        self.assertEqual(edge_dict["dense_2 (Dense)"], "dense_3 (Dense)")
        self.assertEqual(edge_dict["dense_3 (Dense)"], "add (Add)")
        self.assertEqual(edge_dict["add (Add)"], "add_1 (Add)")
        self.assertEqual(edge_dict["dense_4 (Dense)"], "dense_5 (Dense)")
        self.assertEqual(edge_dict["dense_5 (Dense)"], "dense_6 (Dense)")
        self.assertEqual(edge_dict["dense_6 (Dense)"], "add_1 (Add)")
        self.assertEqual(edge_dict["add_1 (Add)"], "dropout (Dropout)")
        self.assertEqual(edge_dict["dropout (Dropout)"], "dense_7 (Dense)")

        file_name = "functional.png"
        plot_model(model, file_name)
        self.assertFileExists(file_name)

        file_name = "functional-show_shapes.png"
        plot_model(model, file_name, show_shapes=True)
        self.assertFileExists(file_name)

        file_name = "functional-show_shapes-show_dtype.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
        )
        self.assertFileExists(file_name)

        file_name = "functional-show_shapes-show_dtype-show_layer_names.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
        )
        self.assertFileExists(file_name)

        file_name = (
            "functional-show_shapes-show_dtype-show_layer_activations.png"
        )
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
        )
        self.assertFileExists(file_name)

        file_name = "functional-show_shapes-show_dtype-show_layer_activations-show_trainable.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

        file_name = "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            rankdir="LR",
        )
        self.assertFileExists(file_name)

        file_name = "functional-show_layer_activations-show_trainable.png"
        plot_model(
            model,
            file_name,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

        file_name = (
            "functional-show_shapes-show_layer_activations-show_trainable.png"
        )
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

    def test_plot_subclassed_model(self):
        class MyModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense_1 = keras.layers.Dense(3, activation="relu")
                self.dense_2 = keras.layers.Dense(1, activation="sigmoid")

            def call(self, x):
                return self.dense_2(self.dense_1(x))

        model = MyModel()
        model.build((None, 3))

        file_name = "subclassed.png"
        plot_model(model, file_name)
        self.assertFileExists(file_name)

        file_name = "subclassed-show_shapes.png"
        plot_model(model, file_name, show_shapes=True)
        self.assertFileExists(file_name)

        file_name = "subclassed-show_shapes-show_dtype.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
        )
        self.assertFileExists(file_name)

        file_name = "subclassed-show_shapes-show_dtype-show_layer_names.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
        )
        self.assertFileExists(file_name)

        file_name = (
            "subclassed-show_shapes-show_dtype-show_layer_activations.png"
        )
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
        )
        self.assertFileExists(file_name)

        file_name = "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

        file_name = "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            rankdir="LR",
        )
        self.assertFileExists(file_name)

        file_name = "subclassed-show_layer_activations-show_trainable.png"
        plot_model(
            model,
            file_name,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

        file_name = (
            "subclassed-show_shapes-show_layer_activations-show_trainable.png"
        )
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        self.assertFileExists(file_name)

    def test_plot_nested_functional_model(self):
        inputs = keras.Input((3,), name="input")
        x = keras.layers.Dense(4, activation="relu", name="dense")(inputs)
        x = keras.layers.Dense(4, activation="relu", name="dense_1")(x)
        outputs = keras.layers.Dense(3, activation="relu", name="dense_2")(x)
        inner_model = keras.Model(inputs, outputs, name="inner_model")

        inputs = keras.Input((3,), name="input_1")
        x = keras.layers.Dense(
            3, activation="relu", trainable=False, name="dense_3"
        )(inputs)
        residual = x
        x = inner_model(x)
        x = keras.layers.Add(name="add")([x, residual])
        residual = x
        x = keras.layers.Dense(4, activation="relu", name="dense_4")(x)
        x = keras.layers.Dense(4, activation="relu", name="dense_5")(x)
        x = keras.layers.Dense(3, activation="relu", name="dense_6")(x)
        x = keras.layers.Add(name="add_1")([x, residual])
        x = keras.layers.Dropout(0.5, name="dropout")(x)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_7")(x)
        model = keras.Model(inputs, outputs)

        edge_dict = get_edge_dict(model_to_dot(model))

        self.assertEqual(edge_dict["input_1 (InputLayer)"], "dense_3 (Dense)")
        self.assertEqual(edge_dict["dense_3 (Dense)"], "add (Add)")
        self.assertEqual(edge_dict["inner_model (Functional)"], "add (Add)")
        self.assertEqual(edge_dict["add (Add)"], "add_1 (Add)")
        self.assertEqual(edge_dict["dense_4 (Dense)"], "dense_5 (Dense)")
        self.assertEqual(edge_dict["dense_5 (Dense)"], "dense_6 (Dense)")
        self.assertEqual(edge_dict["dense_6 (Dense)"], "add_1 (Add)")
        self.assertEqual(edge_dict["add_1 (Add)"], "dropout (Dropout)")
        self.assertEqual(edge_dict["dropout (Dropout)"], "dense_7 (Dense)")

        file_name = "nested-functional.png"
        plot_model(model, file_name, expand_nested=True)
        self.assertFileExists(file_name)

        file_name = "nested-functional-show_shapes.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = "nested-functional-show_shapes-show_dtype.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = (
            "nested-functional-show_shapes-show_dtype-show_layer_names.png"
        )
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            rankdir="LR",
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = (
            "nested-functional-show_layer_activations-show_trainable.png"
        )
        plot_model(
            model,
            file_name,
            show_layer_activations=True,
            show_trainable=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = "nested-functional-show_shapes-show_layer_activations-show_trainable.png"  # noqa: E501
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_layer_activations=True,
            show_trainable=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

    def test_plot_functional_model_with_splits_and_merges(self):
        class SplitLayer(keras.Layer):
            def call(self, x):
                return list(keras.ops.split(x, 2, axis=1))

        class ConcatLayer(keras.Layer):
            def call(self, xs):
                return keras.ops.concatenate(xs, axis=1)

        inputs = keras.Input((2,), name="input")
        a, b = SplitLayer()(inputs)

        a = keras.layers.Dense(2, name="dense")(a)
        b = keras.layers.Dense(2, name="dense_1")(b)

        outputs = ConcatLayer(name="concat_layer")([a, b])
        model = keras.Model(inputs, outputs)

        edge_dict = get_edge_dict(model_to_dot(model))

        self.assertEqual(
            edge_dict["input (InputLayer)"], "split_layer (SplitLayer)"
        )
        self.assertEqual(
            edge_dict["split_layer (SplitLayer)"], "dense_1 (Dense)"
        )
        self.assertEqual(
            edge_dict["dense (Dense)"], "concat_layer (ConcatLayer)"
        )
        self.assertEqual(
            edge_dict["dense_1 (Dense)"], "concat_layer (ConcatLayer)"
        )

        file_name = "split-functional.png"
        plot_model(model, file_name, expand_nested=True)
        self.assertFileExists(file_name)

        file_name = "split-functional-show_shapes.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)

        file_name = "split-functional-show_shapes-show_dtype.png"
        plot_model(
            model,
            file_name,
            show_shapes=True,
            show_dtype=True,
            expand_nested=True,
        )
        self.assertFileExists(file_name)
