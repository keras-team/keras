from bs4 import BeautifulSoup

import keras
from keras.src import testing
from keras.src.utils import model_to_dot


class ModelVisualizationTest(testing.TestCase):
    def _get_node_text(self, node):
        attributes = node.get_attributes()

        if "label" in attributes:
            html = node.get_attributes()["label"]
            parsed_html = BeautifulSoup(html)
            return parsed_html.getText()
        else:
            return None

    def _get_edge_dict(self, dot):
        node_dict = dict()
        for node in dot.get_nodes():
            node_dict[node.get_name()] = self._get_node_text(node)

        edge_dict = dict()
        for edge in dot.get_edges():
            edge_dict[node_dict[edge.get_source()]] = node_dict[
                edge.get_destination()
            ]

        return edge_dict

    def test_model_to_dot_sequential_model(self):
        model = keras.Sequential(
            [
                keras.Input((3,)),
                keras.layers.Dense(4, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        dot = model_to_dot(model)
        edge_dict = self._get_edge_dict(dot)

        self.assertEqual(edge_dict["<dense (Dense)>"], "<dense_1 (Dense)>")
