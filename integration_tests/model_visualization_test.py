"""
Optimized and restructured model visualization test suite for Keras.

This module provides comprehensive testing for Keras model visualization functionality,
including support for sequential, functional, subclassed, and nested model architectures.
"""

import re
from typing import Dict, Set, Union, List, Optional, Any
from dataclasses import dataclass

import keras
from keras.src import testing
from keras.src.utils import model_to_dot, plot_model


@dataclass
class PlotConfig:
    """Configuration class for plot model parameters."""
    show_shapes: bool = False
    show_dtype: bool = False
    show_layer_names: bool = False
    show_layer_activations: bool = False
    show_trainable: bool = False
    rankdir: str = "TB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            key: value for key, value in self.__dict__.items()
            if value is not False and value != "TB"
        }
    
    def get_filename_suffix(self) -> str:
        """Generate filename suffix based on configuration."""
        active_options = []
        if self.show_shapes:
            active_options.append("shapes")
        if self.show_dtype:
            active_options.append("dtype")
        if self.show_layer_names:
            active_options.append("layer_names")
        if self.show_layer_activations:
            active_options.append("activations")
        if self.show_trainable:
            active_options.append("trainable")
        if self.rankdir != "TB":
            active_options.append(self.rankdir)
        
        return "-".join(active_options) if active_options else "default"


class SubclassModel(keras.models.Model):
    """Simple subclassed model for testing purposes."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
    
    def call(self, x):
        return x


class HTMLParser:
    """Utility class for parsing HTML content from DOT graph labels."""
    
    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """
        Extract clean text from HTML font tags.
        
        Args:
            html: HTML string containing font tags
            
        Returns:
            Cleaned text content or empty string if no matches found
        """
        pattern = r"<font[^>]*>(.*?)</font>"
        matches = re.findall(pattern, html)
        
        if matches:
            # Remove any remaining HTML tags from the first match
            clean_text = re.sub(r"<[^>]*>", "", matches[0])
            return clean_text
        return ""


class GraphAnalyzer:
    """Utility class for analyzing DOT graph structures."""
    
    @staticmethod
    def extract_node_text(node) -> Optional[str]:
        """
        Extract text content from a graph node.
        
        Args:
            node: Graph node object
            
        Returns:
            Node text content or None if no label found
        """
        attributes = node.get_attributes()
        if "label" in attributes:
            html_content = attributes["label"]
            return HTMLParser.extract_text_from_html(html_content)
        return None
    
    @staticmethod
    def build_node_dictionary(graph, path_prefix: str = "") -> Dict[str, str]:
        """
        Build a dictionary mapping node names to their full paths.
        
        Args:
            graph: Graph object to analyze
            path_prefix: Prefix for nested graph paths
            
        Returns:
            Dictionary mapping node names to their hierarchical paths
        """
        nodes = {}
        
        # Process direct nodes
        for node in graph.get_nodes():
            node_name = node.get_name()
            if node_name != "node":  # Skip dummy nodes inserted by pydot
                node_text = GraphAnalyzer.extract_node_text(node)
                if node_text:
                    nodes[node_name] = path_prefix + node_text
        
        # Process subgraphs recursively
        for subgraph in graph.get_subgraphs():
            subgraph_label = subgraph.get_label() or ""
            sub_prefix = f"{path_prefix}{subgraph_label} > " if subgraph_label else path_prefix
            sub_nodes = GraphAnalyzer.build_node_dictionary(subgraph, sub_prefix)
            nodes.update(sub_nodes)
        
        return nodes
    
    @staticmethod
    def collect_all_edges(graph) -> List:
        """
        Recursively collect all edges from graph and subgraphs.
        
        Args:
            graph: Graph object to analyze
            
        Returns:
            List of all edges in the graph hierarchy
        """
        edges = list(graph.get_edges())
        
        for subgraph in graph.get_subgraphs():
            edges.extend(GraphAnalyzer.collect_all_edges(subgraph))
        
        return edges
    
    @staticmethod
    def build_edge_dictionary(dot_graph) -> Dict[str, Union[str, Set[str]]]:
        """
        Build a dictionary representing the edge relationships in the graph.
        
        Args:
            dot_graph: DOT graph object
            
        Returns:
            Dictionary mapping source nodes to their destination nodes
            
        Raises:
            ValueError: If dangling edges are found in the graph
        """
        node_dict = GraphAnalyzer.build_node_dictionary(dot_graph)
        all_edges = GraphAnalyzer.collect_all_edges(dot_graph)
        
        edge_dict = {}
        dangling_edges = []
        
        for edge in all_edges:
            source_name = edge.get_source()
            dest_name = edge.get_destination()
            
            source_node = node_dict.get(source_name)
            dest_node = node_dict.get(dest_name)
            
            # Track dangling edges for error reporting
            if source_node is None or dest_node is None:
                dangling_edges.append(
                    f"from '{source_node}'/'{source_name}' "
                    f"to '{dest_node}'/'{dest_name}'"
                )
                continue
            
            # Handle multiple destinations for the same source
            if source_node in edge_dict:
                existing_dest = edge_dict[source_node]
                if isinstance(existing_dest, set):
                    existing_dest.add(dest_node)
                else:
                    edge_dict[source_node] = {existing_dest, dest_node}
            else:
                edge_dict[source_node] = dest_node
        
        if dangling_edges:
            raise ValueError(f"Dangling edges found: {dangling_edges}")
        
        return edge_dict


def parse_text_from_html(html):
    """Legacy function for backward compatibility."""
    return HTMLParser.extract_text_from_html(html)


def get_node_text(node):
    """Legacy function for backward compatibility."""
    return GraphAnalyzer.extract_node_text(node)


def get_edge_dict(dot):
    """Legacy function for backward compatibility."""
    return GraphAnalyzer.build_edge_dictionary(dot)


class ModelVisualizationTest(testing.TestCase):
    """Comprehensive test suite for Keras model visualization."""
    
    # Predefined test configurations for comprehensive testing
    STANDARD_TEST_CONFIGS = [
        {},
        {"show_shapes": True},
        {"show_shapes": True, "show_dtype": True},
        {"show_shapes": True, "show_dtype": True, "show_layer_names": True},
        {"show_shapes": True, "show_dtype": True, "show_layer_names": True, 
         "show_layer_activations": True},
        {"show_shapes": True, "show_dtype": True, "show_layer_names": True,
         "show_layer_activations": True, "show_trainable": True},
        {"show_shapes": True, "show_dtype": True, "show_layer_names": True,
         "show_layer_activations": True, "show_trainable": True, "rankdir": "LR"},
        {"show_layer_activations": True, "show_trainable": True},
    ]
    
    def multi_plot_model(self, model, name, expand_nested=False):
        """
        Run comprehensive plotting tests with various configurations.
        
        Args:
            model: Keras model to test
            name: Base name for output files
            expand_nested: Whether to expand nested models in visualization
        """
        if expand_nested:
            name = name + "-expand_nested"

        for test_case in self.STANDARD_TEST_CONFIGS:
            tags = [v if k == "rankdir" else k for k, v in test_case.items()]
            file_name = "-".join([name] + tags) + ".png"
            plot_model(
                model, file_name, expand_nested=expand_nested, **test_case
            )
            self.assertFileExists(file_name)

    def test_plot_sequential_model(self):
        """Test visualization of sequential models."""
        model = keras.Sequential([
            keras.Input((3,), name="input"),
            keras.layers.Dense(4, activation="relu", name="dense"),
            keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
        ])

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense (Dense)": "dense_1 (Dense)",
            },
        )
        self.multi_plot_model(model, "sequential")

    def test_plot_functional_model(self):
        """Test visualization of functional models with residual connections."""
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
        """Test visualization of subclassed models."""
        model = SubclassModel(name="subclass")
        model.build((None, 3))

        self.multi_plot_model(model, "subclassed")

    def test_plot_nested_functional_model(self):
        """Test visualization of nested functional models."""
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
                "inner_model > input (InputLayer)": "inner_model > dense (Dense)",
                "inner_model > dense (Dense)": "inner_model > dense_1 (Dense)",
                "inner_model > dense_1 (Dense)": "inner_model > dense_2 (Dense)",
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
        """Test visualization of functional models with split and merge operations."""
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
        """Test visualization of sequential models nested within sequential models."""
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

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "sub (Sequential)",
            },
        )
        self.multi_plot_model(model, "sequential_in_sequential")

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
        """Test visualization of functional models nested within functional models."""
        inner_input = keras.layers.Input((10,), name="inner_input")
        x = keras.layers.Dense(10, name="dense1")(inner_input)
        x = keras.layers.Dense(10, name="dense2")(x)
        inner_model = keras.models.Model(inner_input, x, name="inner")

        outer_input = keras.layers.Input((10,), name="outer_input")
        model = keras.models.Model(outer_input, inner_model(outer_input))

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "inner (Functional)",
            },
        )
        self.multi_plot_model(model, "functional_in_functional")

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
        """Test visualization of deeply nested sequential models."""
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

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid (Sequential)",
                "mid (Sequential)": "dense4 (Dense)",
            },
        )
        self.multi_plot_model(model, "sequential_in_sequential_in_sequential")

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
        """Test visualization of functional models nested within sequential models."""
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

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid (Sequential)",
                "mid (Sequential)": "dense3 (Dense)",
            },
        )
        self.multi_plot_model(model, "functional_in_sequential_in_sequential")

        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "dense1 (Dense)": "mid > inner > input1 (InputLayer)",
                "mid > inner > input1 (InputLayer)": "mid > inner > dense2 (Dense)",
                "mid > inner > dense2 (Dense)": "dense3 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "functional_in_sequential_in_sequential", expand_nested=True
        )

    def test_plot_functional_in_functional_in_functional(self):
        """Test visualization of deeply nested functional models."""
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

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "mid (Functional)",
                "mid (Functional)": "dense2 (Dense)",
            },
        )
        self.multi_plot_model(model, "functional_in_functional_in_functional")

        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "mid > mid_input (InputLayer)",
                "mid > mid_input (InputLayer)": "mid > inner > inner_input (InputLayer)",
                "mid > inner > inner_input (InputLayer)": "mid > inner > dense1 (Dense)",
                "mid > inner > dense1 (Dense)": "dense2 (Dense)",
            },
        )
        self.multi_plot_model(
            model, "functional_in_functional_in_functional", expand_nested=True
        )

    def test_plot_complex(self):
        """Test visualization of complex models with multiple inputs and outputs."""
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

        edge_dict = get_edge_dict(model_to_dot(model))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": "mid (Functional)",
                "mid (Functional)": {"add1 (Add)", "add2 (Add)"},
            },
        )
        self.multi_plot_model(model, "complex")

        edge_dict = get_edge_dict(model_to_dot(model, expand_nested=True))
        self.assertEqual(
            edge_dict,
            {
                "outer_input (InputLayer)": {
                    "mid > input0 (InputLayer)",
                    "mid > input1 (InputLayer)",
                    "mid > input2 (InputLayer)",
                    "mid > input3 (InputLayer)",
                },
                "mid > input0 (InputLayer)": "mid > seq > dense0 (Dense)",
                "mid > input1 (InputLayer)": "mid > inner > inner_inpt1 (InputLayer)",
                "mid > input2 (InputLayer)": "mid > inner > inner_inpt2 (InputLayer)",
                "mid > input3 (InputLayer)": "mid > subclass3 (SubclassModel)",
                "mid > seq > dense0 (Dense)": "mid > seq > subclass0 (SubclassModel)",
                "mid > inner > inner_inpt1 (InputLayer)": "mid > inner > dense1 (Dense)",
                "mid > inner > inner_inpt2 (InputLayer)": "mid > inner > dense2 (Dense)",
                "mid > seq > subclass0 (SubclassModel)": "add1 (Add)",
                "mid > inner > dense1 (Dense)": "add1 (Add)",
                "mid > inner > dense2 (Dense)": "add2 (Add)",
                "mid > subclass3 (SubclassModel)": "add2 (Add)",
            },
        )
        self.multi_plot_model(model, "complex", expand_nested=True)
