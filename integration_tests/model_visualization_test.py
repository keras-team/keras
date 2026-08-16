import re

import keras
from keras.src import testing
from keras.src.utils import model_to_dot
from keras.src.utils import plot_model


class SubclassModel(keras.models.Model):
    def __init__(self, name):
        super().__init__(name=name)

    def call(self, x):
        return x


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
    def get_node_dict(graph, path=""):
        nodes = {
            node.get_name(): path + get_node_text(node)
            for node in graph.get_nodes()
            if node.get_name() != "node"  # Dummy node inserted by pydot?
        }

        for subgraph in graph.get_subgraphs():
            sub_nodes = get_node_dict(
                subgraph, path=f"{path}{subgraph.get_label()} > "
            )
            nodes.update(sub_nodes)

        return nodes

    node_dict = get_node_dict(dot)

    def get_edges(graph):
        edges = list(graph.get_edges())
        for subgraph in graph.get_subgraphs():
            edges.extend(get_edges(subgraph))
        return edges

    edge_dict = dict()
    dangling_edges = []
    for edge in get_edges(dot):
        source_node = node_dict.get(edge.get_source(), None)
        destination_node = node_dict.get(edge.get_destination(), None)
        if source_node is None or destination_node is None:
            dangling_edges.append(
                f"from '{source_node}'/'{edge.get_source()}' "
                f"to '{destination_node}'/'{edge.get_destination()}'"
            )
        if source_node in edge_dict:
            destination_nodes = edge_dict[source_node]
            if not isinstance(destination_nodes, set):
                destination_nodes = set([destination_nodes])
                edge_dict[source_node] = destination_nodes
            destination_nodes.add(destination_node)
        else:
            edge_dict[source_node] = destination_node

    if dangling_edges:
        raise ValueError(f"Dangling edges found: {dangling_edges}")
    return edge_dict


class ModelVisualizationTest(testing.TestCase):
    def multi_plot_model(self, model, name, expand_nested=False):
        if expand_nested:
            name = f"{name}-expand_nested"

        TEST_CASES = [
            {},
            {
                "show_shapes": True,
            },
            {
                "show_shapes": True,
                "show_dtype": True,
            },
            {
                "show_shapes": True,
                "show_dtype": True,
                "show_layer_names": True,
            },
            {
                "show_shapes": True,
                "show_dtype": True,
                "show_layer_names": True,
                "show_layer_activations": True,
            },
            {
                "show_shapes": True,
                "show_dtype": True,
                "show_layer_names": True,
                "show_layer_activations": True,
                "show_trainable": True,
            },
            {
                "show_shapes": True,
                "show_dtype": True,
                "show_layer_names": True,
                "show_layer_activations": True,
                "show_trainable": True,
                "rankdir": "LR",
            },
            {
                "show_layer_activations": True,
                "show_trainable": True,
            },
        ]

        for test_case in TEST_CASES:
            tags = [v if k == "rankdir" else k for k, v in test_case.items()]
            file_name = f"{'-'.join([name] + tags)}.png"
            plot_model(
                model, file_name, expand_nested=expand_nested, **test_case
            )
            self.assertFileExists(file_name)

    def test_plot_sequential_model(self):
        model = keras.Sequential(
            [
                keras.Input((3,), name="input"),
                keras.layers.Dense(4, activation="relu", name="dense"),
                keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
            ]
        )

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense (Dense)": "dense_1 (Dense)",
            },
        )
        self.multi_plot_model(model, "sequential")

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
        self.assertEqual(
            edge_dict,
            {
                "input (InputLayer)": "dense (Dense)",
                "dense (Dense)": {"dense_1 (Dense)", "add (Add)"},
                "dense_1 (Dense)": "dense_2 (Dense)",
                "dense_2 (Dense)": "dense_3 (Dense)",
                "dense_3 (Dense)": "add (Add)",
                "add (Add)": {"dense_4 (Dense)", "add_1 (Add)"},
                "dense_4 (Dense)": "dense_5 (Dense)",
                "dense_5 (Dense)": "dense_6 (Dense)",
                "dense_6 (Dense)": "add_1 (Add)",
                "add_1 (Add)": "dropout (Dropout)",
                "dropout (Dropout)": "dense_7 (Dense)",
            },
        )
        self.multi_plot_model(model, "functional")

    def test_plot_subclassed_model(self):
        model = SubclassModel(name="subclass")
        model.build((None, 3))

        self.multi_plot_model(model, "subclassed")

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
        self.assertEqual(
            edge_dict,
            {
                "input_1 (InputLayer)": "dense_3 (Dense)",
                "dense_3 (Dense)": {"inner_model (Functional)", "add (Add)"},
                "inner_model (Functional)": "add (Add)",
                "add (Add)": {"dense_4 (Dense)", "add_1 (Add)"},
                "dense_4 (Dense)": "dense_5 (Dense)",
                "dense_5 (Dense)": "dense_6 (Dense)",
                "dense_6 (Dense)": "add_1 (Add)",
                "add_1 (Add)": "dropout (Dropout)",
                "dropout (Dropout)": "dense_7 (Dense)",
            },
        )
        self.multi_plot_model(model, "nested-functional")

        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "input_1 (InputLayer)": "dense_3 (Dense)",
                "dense_3 (Dense)": {
                    "inner_model > input (InputLayer)",
                    "add (Add)",
                },
                "inner_model > input (InputLayer)": "inner_model > dense (Dense)",  # noqa: E501
                "inner_model > dense (Dense)": "inner_model > dense_1 (Dense)",  # noqa: E501
                "inner_model > dense_1 (Dense)": "inner_model > dense_2 (Dense)",  # noqa: E501
                "inner_model > dense_2 (Dense)": "add (Add)",
                "add (Add)": {"dense_4 (Dense)", "add_1 (Add)"},
                "dense_4 (Dense)": "dense_5 (Dense)",
                "dense_5 (Dense)": "dense_6 (Dense)",
                "dense_6 (Dense)": "add_1 (Add)",
                "add_1 (Add)": "dropout (Dropout)",
                "dropout (Dropout)": "dense_7 (Dense)",
            },
        )
        self.multi_plot_model(model, "nested-functional", expand_nested=True)

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
            edge_dict,
            {
                "input (InputLayer)": "split_layer (SplitLayer)",
                "split_layer (SplitLayer)": {
                    "dense (Dense)",
                    "dense_1 (Dense)",
                },
                "dense (Dense)": "concat_layer (ConcatLayer)",
                "dense_1 (Dense)": "concat_layer (ConcatLayer)",
            },
        )
        self.multi_plot_model(model, "split-functional")

    def test_plot_sequential_in_sequential(self):
        inner_model = keras.models.Sequential(
            [
                keras.layers.Dense(10, name="dense2"),
                keras.layers.Dense(10, name="dense3"),
            ],
            name="sub",
        )
        model = keras.models.Sequential(
            [
                keras.layers.Dense(10, name="dense1"),
                inner_model,
            ],
        )
        model.build((1, 10))

        #
        #  +-------------------------+
        #  |     dense1 (Dense)      |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |    sub (Sequential)     |
        #  +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "sub (Sequential)",
            },
        )
        self.multi_plot_model(model, "sequential_in_sequential")

        #
        #    +-------------------------+
        #    |     dense1 (Dense)      |
        #    +-------------------------+
        #                 |
        #  +--------------|--------------+
        #  | sub          v              |
        #  | +-------------------------+ |
        #  | |     dense2 (Dense)      | |
        #  | +-------------------------+ |
        #  |              |              |
        #  |              v              |
        #  | +-------------------------+ |
        #  | |     dense3 (Dense)      | |
        #  | +-------------------------+ |
        #  +-----------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "sub > dense2 (Dense)",
                "sub > dense2 (Dense)": "sub > dense3 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "sequential_in_sequential", expand_nested=True
        )

    def test_plot_functional_in_functional(self):
        inner_input = keras.layers.Input((10,), name="inner_input")
        x = keras.layers.Dense(10, name="dense1")(inner_input)
        x = keras.layers.Dense(10, name="dense2")(x)
        inner_model = keras.models.Model(inner_input, x, name="inner")

        outer_input = keras.layers.Input((10,), name="outer_input")
        model = keras.models.Model(outer_input, inner_model(outer_input))

        #
        #  +-------------------------+
        #  |outer_input (InputLayer) |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |   inner (Functional)    |
        #  +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "inner (Functional)",
            },
        )
        self.multi_plot_model(model, "functional_in_functional")

        #
        #    +-------------------------+
        #    |outer_input (InputLayer) |
        #    +-------------------------+
        #                 |
        #  +--------------|--------------+
        #  | inner        v              |
        #  | +-------------------------+ |
        #  | |inner_input (InputLayer) | |
        #  | +-------------------------+ |
        #  |              |              |
        #  |              v              |
        #  | +-------------------------+ |
        #  | |     dense1 (Dense)      | |
        #  | +-------------------------+ |
        #  |              |              |
        #  |              v              |
        #  | +-------------------------+ |
        #  | |     dense2 (Dense)      | |
        #  | +-------------------------+ |
        #  +-----------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "inner > inner_input (InputLayer)",
                "inner > inner_input (InputLayer)": "inner > dense1 (Dense)",
                "inner > dense1 (Dense)": "inner > dense2 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "functional_in_functional", expand_nested=True
        )

    def test_plot_sequential_in_sequential_in_sequential(self):
        inner_model = keras.models.Sequential(
            [
                keras.layers.Dense(10, name="dense2"),
                keras.layers.Dense(10, name="dense3"),
            ],
            name="inner",
        )
        mid_model = keras.models.Sequential(
            [
                inner_model,
            ],
            name="mid",
        )
        model = keras.models.Sequential(
            [
                keras.layers.Dense(10, name="dense1"),
                mid_model,
                keras.layers.Dense(10, name="dense4"),
            ],
        )
        model.build((1, 10))

        #
        #  +-------------------------+
        #  |     dense1 (Dense)      |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |    mid (Sequential)     |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |     dense4 (Dense)      |
        #  +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid (Sequential)",
                "mid (Sequential)": "dense4 (Dense)",
            },
        )
        self.multi_plot_model(model, "sequential_in_sequential_in_sequential")

        #
        #      +-------------------------+
        #      |     dense1 (Dense)      |
        #      +-------------------------+
        #                   |
        #  +----------------|----------------+
        #  | mid            |                |
        #  | +--------------|--------------+ |
        #  | | inner        v              | |
        #  | | +-------------------------+ | |
        #  | | |     dense2 (Dense)      | | |
        #  | | +-------------------------+ | |
        #  | |              |              | |
        #  | |              v              | |
        #  | | +-------------------------+ | |
        #  | | |     dense3 (Dense)      | | |
        #  | | +-------------------------+ | |
        #  | +--------------|--------------+ |
        #  +----------------|----------------+
        #                   v
        #      +-------------------------+
        #      |     dense4 (Dense)      |
        #      +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid > inner > dense2 (Dense)",
                "mid > inner > dense2 (Dense)": "mid > inner > dense3 (Dense)",
                "mid > inner > dense3 (Dense)": "dense4 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "sequential_in_sequential_in_sequential", expand_nested=True
        )

    def test_plot_functional_in_sequential_in_sequential(self):
        input1 = keras.layers.Input((10,), name="input1")
        x = keras.layers.Dense(10, name="dense2")(input1)
        inner_model = keras.models.Model(input1, x, name="inner")

        mid_model = keras.models.Sequential(
            [
                inner_model,
            ],
            name="mid",
        )
        model = keras.models.Sequential(
            [
                keras.layers.Dense(10, name="dense1"),
                mid_model,
                keras.layers.Dense(10, name="dense3"),
            ],
        )
        model.build((1, 10))

        #
        #  +-------------------------+
        #  |     dense1 (Dense)      |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |    mid (Sequential)     |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |     dense3 (Dense)      |
        #  +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid (Sequential)",
                "mid (Sequential)": "dense3 (Dense)",
            },
        )
        self.multi_plot_model(model, "functional_in_sequential_in_sequential")

        #
        #      +-------------------------+
        #      |     dense1 (Dense)      |
        #      +-------------------------+
        #                   |
        #  +----------------|----------------+
        #  | mid            |                |
        #  | +--------------|--------------+ |
        #  | | inner        v              | |
        #  | | +-------------------------+ | |
        #  | | |   input1 (Inputlayer)   | | |
        #  | | +-------------------------+ | |
        #  | |              |              | |
        #  | |              v              | |
        #  | | +-------------------------+ | |
        #  | | |     dense2 (Dense)      | | |
        #  | | +-------------------------+ | |
        #  | +--------------|--------------+ |
        #  +----------------|----------------+
        #                   v
        #      +-------------------------+
        #      |     dense3 (Dense)      |
        #      +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid > inner > input1 (InputLayer)",
                "mid > inner > input1 (InputLayer)": "mid > inner > dense2 (Dense)",  # noqa: E501
                "mid > inner > dense2 (Dense)": "dense3 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "functional_in_sequential_in_sequential", expand_nested=True
        )

    def test_plot_functional_in_functional_in_functional(self):
        # From https://github.com/keras-team/keras/issues/21119
        inner_input = keras.layers.Input((10,), name="inner_input")
        x = keras.layers.Dense(10, name="dense1")(inner_input)
        inner_model = keras.models.Model(inner_input, x, name="inner")

        mid_input = keras.layers.Input((10,), name="mid_input")
        mid_output = inner_model(mid_input)
        mid_model = keras.models.Model(mid_input, mid_output, name="mid")

        outer_input = keras.layers.Input((10,), name="outer_input")
        x = mid_model(outer_input)
        x = keras.layers.Dense(10, name="dense2")(x)
        model = keras.models.Model(outer_input, x)

        #
        #  +-------------------------+
        #  |outer_input (InputLayer) |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |    mid (Functional)     |
        #  +-------------------------+
        #               |
        #               v
        #  +-------------------------+
        #  |     dense2 (Dense)      |
        #  +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "mid (Functional)",
                "mid (Functional)": "dense2 (Dense)",
            },
        )
        self.multi_plot_model(model, "functional_in_functional_in_functional")

        #
        #      +-------------------------+
        #      |outer_input (InputLayer) |
        #      +-------------------------+
        #                   |
        #  +----------------|----------------+
        #  | mid            |                |
        #  |   +-------------------------+   |
        #  |   | mid_input (Inputlayer)  |   |
        #  |   +-------------------------+   |
        #  | +--------------|--------------+ |
        #  | | inner        v              | |
        #  | | +-------------------------+ | |
        #  | | |inner_input (Inputlayer) | | |
        #  | | +-------------------------+ | |
        #  | |              |              | |
        #  | |              v              | |
        #  | | +-------------------------+ | |
        #  | | |     dense1 (Dense)      | | |
        #  | | +-------------------------+ | |
        #  | +--------------|--------------+ |
        #  +----------------|----------------+
        #                   v
        #      +-------------------------+
        #      |     dense2 (Dense)      |
        #      +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "mid > mid_input (InputLayer)",
                "mid > mid_input (InputLayer)": "mid > inner > inner_input (InputLayer)",  # noqa: E501
                "mid > inner > inner_input (InputLayer)": "mid > inner > dense1 (Dense)",  # noqa: E501
                "mid > inner > dense1 (Dense)": "dense2 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "functional_in_functional_in_functional", expand_nested=True
        )

    def test_plot_complex(self):
        # Note: this test exercises the case when `output_index` is not 0 and
        # changes when going deeply in nested models to resolve the destination
        # of an edge.
        inner_inpt1 = keras.layers.Input(shape=(10,), name="inner_inpt1")
        inner_inpt2 = keras.layers.Input(shape=(10,), name="inner_inpt2")
        inner_model = keras.models.Model(
            [inner_inpt1, inner_inpt2],
            [
                keras.layers.Dense(10, name="dense1")(inner_inpt1),
                keras.layers.Dense(10, name="dense2")(inner_inpt2),
            ],
            name="inner",
        )

        input0 = keras.layers.Input(shape=(10,), name="input0")
        input1 = keras.layers.Input(shape=(10,), name="input1")
        input2 = keras.layers.Input(shape=(10,), name="input2")
        input3 = keras.layers.Input(shape=(10,), name="input3")

        mid_sequential = keras.models.Sequential(
            [
                keras.layers.Dense(10, name="dense0"),
                SubclassModel(name="subclass0"),
            ],
            name="seq",
        )
        mid_subclass = SubclassModel(name="subclass3")
        mid_model = keras.models.Model(
            [input0, input1, input2, input3],
            [
                mid_sequential(input0),
                *inner_model([input1, input2]),
                mid_subclass(input3),
            ],
            name="mid",
        )

        outer_input = keras.layers.Input((10,), name="outer_input")
        mid_outputs = mid_model(
            [outer_input, outer_input, outer_input, outer_input]
        )
        model = keras.models.Model(
            outer_input,
            [
                keras.layers.Add(name="add1")([mid_outputs[0], mid_outputs[1]]),
                keras.layers.Add(name="add2")([mid_outputs[2], mid_outputs[3]]),
            ],
        )

        #
        #                 +-------------------------+
        #                 |outer_input (InputLayer) |
        #                 +-------------------------+
        #                              |
        #                              v
        #                 +-------------------------+
        #                 |    mid (Functional)     |
        #                 +-------------------------+
        #                          |      |
        #                          v      v
        #  +-------------------------+  +-------------------------+
        #  |       add1 (Add)        |  |       add2 (Add)        |
        #  +-------------------------+  +-------------------------+
        #
        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "mid (Functional)",
                "mid (Functional)": {"add1 (Add)", "add2 (Add)"},
            },
        )
        self.multi_plot_model(model, "complex")

        #
        #                               +-----------+
        #            +------------------|outer_input|-----------------+
        #            |                  +-----------+                 |
        #            |                   |         |                  |
        #  +---------|-------------------|---------|------------------|-------+
        #  | mid     v                   v         v                  v       |
        #  |   +-----------+     +-----------+ +-----------+    +-----------+ |
        #  |   |  input0   |     |  input1   | |  input2   |    |  input3   | |
        #  |   +-----------+     +-----------+ +-----------+    +-----------+ |
        #  | +-------|-------+ +-------|-------------|-------+        |       |
        #  | | seq   v       | | inner v             v       |        |       |
        #  | | +-----------+ | | +-----------+ +-----------+ |  +-----------+ |
        #  | | |  dense0   | | | |inner_inp1t| |inner_inp2t| |  | subclass3 | |
        #  | | +-----------+ | | +-----------+ +-----------+ |  +-----------+ |
        #  | |       |       | |       |             |       |    |           |
        #  | |       v       | |       v             v       |    |           |
        #  | | +-----------+ | | +-----------+ +-----------+ |    |           |
        #  | | | subclass0 | | | |  dense1   | |  dense2   | |    |           |
        #  | | +-----------+ | | +-----------+ +-----------+ |    |           |
        #  | +-----------|---+ +---|---------------------|---+    |           |
        #  +-------------|---------|---------------------|--------|-----------+
        #                v         v                     v        v
        #               +-----------+                   +-----------+
        #               |    add1   |                   |   add2    |
        #               +-----------+                   +-----------+
        #
        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                # 1st row
                "outer_input (InputLayer)": {
                    "mid > input0 (InputLayer)",
                    "mid > input1 (InputLayer)",
                    "mid > input2 (InputLayer)",
                    "mid > input3 (InputLayer)",
                },
                # 2nd row
                "mid > input0 (InputLayer)": "mid > seq > dense0 (Dense)",
                "mid > input1 (InputLayer)": "mid > inner > inner_inpt1 (InputLayer)",  # noqa: E501
                "mid > input2 (InputLayer)": "mid > inner > inner_inpt2 (InputLayer)",  # noqa: E501
                "mid > input3 (InputLayer)": "mid > subclass3 (SubclassModel)",
                # 3rd row
                "mid > seq > dense0 (Dense)": "mid > seq > subclass0 (SubclassModel)",  # noqa: E501
                "mid > inner > inner_inpt1 (InputLayer)": "mid > inner > dense1 (Dense)",  # noqa: E501
                "mid > inner > inner_inpt2 (InputLayer)": "mid > inner > dense2 (Dense)",  # noqa: E501
                # 4th row
                "mid > seq > subclass0 (SubclassModel)": "add1 (Add)",
                "mid > inner > dense1 (Dense)": "add1 (Add)",
                "mid > inner > dense2 (Dense)": "add2 (Add)",
                "mid > subclass3 (SubclassModel)": "add2 (Add)",
            },
        )
        self.multi_plot_model(model, "complex", expand_nested=True)
