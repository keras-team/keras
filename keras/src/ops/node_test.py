import numpy as np

from keras.src import Layer
from keras.src import testing
from keras.src.backend import KerasTensor
from keras.src.ops.node import Node


class DummyLayer(Layer):
    pass


class NodeTest(testing.TestCase):
    # Testing a simple node and layer combination **a**
    def test_simple_case(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        a_layer = DummyLayer()
        node = Node(a_layer, outputs=a, call_args=(), call_kwargs={})

        self.assertEqual(node.is_input, True)

        self.assertEqual(node.output_tensors[0], a)
        self.assertEqual(node.output_tensors[0].shape, shape)

    # Testing a simple node connection with args and kwargs **a** --> **b**
    def test_single_wired_layers(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        a_layer = DummyLayer()
        node1 = Node(a_layer, outputs=a, call_args=(), call_kwargs={})

        b = KerasTensor(shape=shape)
        x = KerasTensor(shape=shape)
        kwargs = {"x": x}
        args = (a,)
        b_layer = DummyLayer()
        node2 = Node(b_layer, outputs=b, call_args=args, call_kwargs=kwargs)

        self.assertEqual(node1.is_input, True)
        self.assertEqual(node2.is_input, False)

        self.assertEqual(node1.operation, a_layer)
        self.assertEqual(node2.operation, b_layer)

        self.assertEqual(node1.output_tensors[0], a)
        self.assertEqual(node1.output_tensors[0].shape, shape)

        self.assertEqual(a_layer._inbound_nodes[0], node1)
        self.assertEqual(a_layer._outbound_nodes[0], node2)

        self.assertEqual(b_layer._inbound_nodes[0], node2)
        self.assertEqual(node2.parent_nodes[0], node1)

        self.assertEqual(node2.input_tensors, [a, x])
        self.assertEqual(node2.arguments.kwargs, kwargs)
        self.assertEqual(node2.arguments.args, args)

    # Testing when output tensor is not Keras Tensor
    def test_output_tensor_error(self):
        a = np.random.rand(2, 3, 4)
        a_layer = DummyLayer()
        with self.assertRaisesRegex(
            ValueError, "operation outputs must be tensors."
        ):
            Node(a_layer, outputs=a, call_args=(), call_kwargs={})
