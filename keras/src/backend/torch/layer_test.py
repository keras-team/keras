import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend.common import global_state

if backend.backend() == "torch":
    import torch


@pytest.mark.skipif(backend.backend() != "torch", reason="Torch only test.")
class LayerTest(testing.TestCase):

    def get_torch_parameter_from_variable(self, layer, variable):
        parameters = [
            (pname, p)
            for pname, p in layer.named_parameters()
            if id(p) == id(variable.value)
        ]
        if len(parameters) > 1:
            raise ValueError(
                "Found more than one paramter match with variable."
            )
        if len(parameters) == 0:
            return None
        return parameters[0]

    def test_torch_params_create_deterministic(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight(initializer="zeros")
                self.w3 = self.add_weight(dtype="bool", trainable=False)
                self.w4 = self.add_weight(
                    initializer="ones",
                    dtype="int32",
                    shape=(2, 2),
                    trainable=False,
                )
                self.w5 = self.add_weight(initializer="ones", shape=(2, 2))

        params = []
        for _ in range(5):
            global_state.clear_session()
            layer = MyLayer()
            layer.build(None)
            layer_params = list(
                (pname, np.copy(backend.convert_to_numpy(p)))
                for pname, p in layer.named_parameters()
            )
            self.assertEqual(len(layer_params), 4)
            params.append(layer_params)

        for idx in range(len(params) - 1):
            param_a = params[idx]
            param_b = params[idx + 1]
            self.assertEqual(len(param_a), len(param_b))
            for i in range(len(param_a)):
                self.assertEqual(param_a[i][0], param_b[i][0])
                np.testing.assert_array_equal(param_a[i][1], param_b[i][1])

    def test_nested_modification_propagate(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.child = ChildLayer()

            def call(self, input):
                return self.child(input)

        class ChildLayer(layers.Layer):

            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight()
                self.w2 = self.add_weight(dtype="int32", trainable=False)
                self.grand_child = GrandChildLayer()

            def call(self, input):
                return self.grand_child(input)

        class GrandChildLayer(layers.Layer):

            def __init__(self):
                super().__init__()
                self.w3 = self.add_weight(name="w3", trainable=False)
                self.w4 = self.add_weight(shape=(2, 2), name="w4")
                self.w5 = self.add_weight(
                    initializer="ones", shape=(2, 2), name="w5"
                )

            def call(self, input):
                return input

        layer = MyLayer()
        layer(backend.KerasTensor((1,)))

        self.assertEqual(len(list(layer.parameters())), 5)
        w5 = self.get_torch_parameter_from_variable(
            layer, layer.child.grand_child.w5
        )
        self.assertIsNotNone(w5)
        # the order is trainable -> non_trainable -> seed, so in torch list it
        # is w4->w5->w3.
        self.assertEqual(w5[0], "child.grand_child.torch_params.1")
        w5_value = w5[1]

        np.testing.assert_array_equal(
            backend.convert_to_numpy(w5_value), np.ones((2, 2))
        )

        layer.child.grand_child.w5.assign_sub(1)

        np.testing.assert_array_equal(
            backend.convert_to_numpy(w5_value), np.zeros((2, 2))
        )

    def test_trainable_modification_propagates(self):
        class Layer(layers.Layer):

            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight(name="w1")
                self.w2 = self.add_weight(name="w2")

        layer = Layer()
        layer.build(None)

        torch_w1 = self.get_torch_parameter_from_variable(layer, layer.w1)
        self.assertIsNotNone(torch_w1)
        torch_w1_value = torch_w1[1]

        self.assertTrue(layer.w1.trainable)
        self.assertTrue(torch_w1_value.requires_grad)

        layer.w1.trainable = False
        self.assertFalse(layer.w1.trainable)
        self.assertFalse(torch_w1_value.requires_grad)

        # the order of parameter dict is w.r.t to the creation time trainable
        # non-trainable assignment, flipping later doesn't affect parameter
        # list order.
        def untrack_torch_params(target_layer):
            for t in target_layer._layers:
                untrack_torch_params(t)
            del layer.torch_params

        untrack_torch_params(layer)

        torch_w1 = self.get_torch_parameter_from_variable(layer, layer.w1)
        self.assertIsNotNone(torch_w1)
        torch_w1_value = torch_w1[1]
        self.assertEqual(torch_w1[0], "torch_params.0")
        np.testing.assert_array_equal(
            backend.convert_to_numpy(torch_w1_value), layer.w1.numpy()
        )

        torch_w2 = self.get_torch_parameter_from_variable(layer, layer.w2)
        self.assertIsNotNone(torch_w2)
        torch_w2_value = torch_w2[1]
        self.assertEqual(torch_w2[0], "torch_params.1")
        np.testing.assert_array_equal(
            backend.convert_to_numpy(torch_w2_value), layer.w2.numpy()
        )

    def test_load_dict_store_restore(self):

        class Layer(layers.Layer):

            def __init__(self):
                super().__init__()
                self.w = self.add_weight(shape=(2, 2))
                self.child = ChildLayer()

            def call(self, input):
                return input + self.child(input)

        class ChildLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.stat = self.add_weight(
                    shape=(2, 2),
                    dtype="int32",
                    trainable=False,
                    initializer="zero",
                )
                self.w = self.add_weight(shape=(2, 2))
                self.seed_gen = backend.random.SeedGenerator(seed=1337)

            def call(self, input):
                return input

        layer1 = Layer()
        layer1(backend.KerasTensor((1,)))
        layer1.child.seed_gen.next()
        layer1.child.stat.assign_add(1)

        state_dict = layer1.state_dict()
        self.assertEqual(len(state_dict), 4)
        global_state.clear_session()

        layer2 = Layer()
        layer2(backend.KerasTensor((1,)))

        layer2_initial_state_dict = layer2.state_dict()

        for key, v in state_dict.items():
            self.assertTrue(key in layer2_initial_state_dict)
            self.assertFalse(torch.equal(v, layer2_initial_state_dict[key]))

        for layer1_var, layer2_var in zip(layer1.variables, layer2.variables):
            self.assertTrue(layer1_var.path, layer2_var.path)
            self.assertFalse(
                np.array_equal(layer1_var.numpy(), layer2_var.numpy())
            )

        layer2.load_state_dict(state_dict)
        for key, v in state_dict.items():
            self.assertTrue(key in layer2_initial_state_dict)
            self.assertTrue(torch.equal(v, layer2_initial_state_dict[key]))

        for layer1_var, layer2_var in zip(layer1.variables, layer2.variables):
            self.assertTrue(layer1_var.path, layer2_var.path)
            np.testing.assert_array_equal(
                layer1_var.numpy(), layer2_var.numpy()
            )

    def test_list_of_layers_tracked_properly(self):

        class Layer(layers.Layer):

            def __init__(self):
                super().__init__()
                self.cells = [
                    Cell(),
                    Cell(),
                    Cell(),
                ]

            def call(self, input):
                o = input
                for c in self.cells:
                    o = c(o)
                return o

        class Cell(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(shape=(2, 2))

            def call(self, input):
                return input

        layer = Layer()
        layer(backend.KerasTensor((1,)))
        self.assertEqual(len(list(layer.parameters())), 3)

    def test_throw_error_when_build_not_called(self):
        class Layer(layers.Layer):

            def __init__(self):
                super().__init__()
                self.w = self.add_weight(name="w")

        layer = Layer()
        with self.assertRaisesRegex(
            RuntimeError, "Did you forget to call model once?"
        ):
            list(layer.named_parameters())
