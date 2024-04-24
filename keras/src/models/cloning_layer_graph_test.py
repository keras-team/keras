import collections
import unittest

import numpy as np

import keras
from keras import Input
from keras import layers
from keras.src import tree
from keras.src.models import Functional
from keras.src.models import Model
from keras.src.models import Sequential
from keras.src.models.cloning_layer_graph import _handle_input_node
from keras.src.models.cloning_layer_graph import _walkback_one_tensor
from keras.src.models.cloning_layer_graph import clone_layer_graph
from keras.src.models.functional import is_input_keras_tensor
from keras.src.ops.node import Node

# Running these tests with "channels_first" will not test anything useful.
# Forcing "channels_last" for this test module only, otherwise some test
# cases in this module fail to create a valid test model.
_image_format = None


def setUpModule():
    global _image_format
    _image_format = keras.config.image_data_format()
    keras.config.set_image_data_format("channels_last")


def moduleCleanUp():
    # restore original image format
    image_format = keras.config.set_image_data_format(_image_format)


unittest.addModuleCleanup(moduleCleanUp)


def _gather_nested_node(node, visited, enter_nested):
    nested = isinstance(node.operation, Functional) or isinstance(
        node.operation, Sequential
    )
    # enter nested layers only once, after that,
    # just run the layer (i.e. do nothing here)
    if nested and enter_nested and id(node.operation) not in visited:
        _gather_tensors(
            node.operation.outputs, visited, enter_nested
        )  # jump into the nested layer
        visited.update(
            {id(node.operation): node.operation}
        )  # enter nested layers only once


def _gather_node(node, visited):
    for tensor in node.output_tensors:
        visited.update({id(tensor): tensor})


def _gather_tensors(tensor_struct, visited, enter_nested):
    tensors = tree.flatten(tensor_struct)
    for tensor in tensors:

        # retrieve node that produced tensor
        node = _walkback_one_tensor(tensor)

        # handle already visited node
        if id(node) in visited:
            continue

        # mark the node as visited
        visited.update({id(node): node})

        # handle input node
        if _handle_input_node(node, visited):
            continue

        # handle nested node
        _gather_nested_node(node, visited, enter_nested)

        # run the node, i.e. place its outputs in visited
        _gather_node(node, visited)

        # recursively continue iterating on inputs
        _gather_tensors(node.input_tensors, visited, enter_nested)  # flattened

    return visited


def gather_node_graph(input, output, enter_nested=True):
    for tensor in tree.flatten(input):
        if not (
            isinstance(tensor, keras.KerasTensor)
            and is_input_keras_tensor(tensor)
        ):
            raise ValueError(
                f"All input values must be KerasTensors. "
                f"Received {tensor} of type {type(tensor)}."
            )
    # visited will store nodes, output tensors and
    # nested operations (Finctional or Sequential)
    visited = collections.OrderedDict()
    visited = _gather_tensors(output, visited, enter_nested)
    return visited


def is_same_layer_graph(node_list1, node_list2, equivalent_nodes):
    if not equivalent_nodes:
        equivalent_nodes = []
    # handle special case of Sequential, which is a
    # Functional model but is not of type Functional.
    equivalent_nodes.append((Functional, Sequential))

    def is_equivalent_operation(op1, op2):
        equivalent = []
        for type1, type2 in equivalent_nodes:
            equivalent.append(isinstance(op1, type1) and isinstance(op2, type2))
            equivalent.append(isinstance(op2, type1) and isinstance(op1, type2))
        return any(equivalent)

    if len(node_list1) != len(node_list2):
        return False

    for item1, item2 in zip(node_list1.values(), node_list2.values()):
        if type(item1) != type(item2):
            if not is_equivalent_operation(item1, item2):
                return False

        if isinstance(item1, Node):
            if type(item1.operation) != type(item2.operation):
                if not is_equivalent_operation(
                    item1.operation, item2.operation
                ):
                    return False

        if isinstance(item1, keras.KerasTensor):
            if item1.shape != item2.shape or item1.dtype != item2.dtype:
                return False

    return True


def are_all_nodes_cloned(node_list1, node_list2):
    # all nodes apart from inputs which are never cloned
    for item1, item2 in zip(node_list1.values(), node_list2.values()):
        skip_input_tensor = isinstance(
            item1, keras.KerasTensor
        ) and is_input_keras_tensor(item1)
        skip_input_layer = isinstance(item1, Node) and isinstance(
            item1.operation, keras.layers.InputLayer
        )
        if skip_input_tensor or skip_input_layer:
            continue
        if id(item1) == id(item2):
            return False

    return True


def are_results_identical(model1, model2):
    # create a compatible input value for both models
    batch = 4
    inp_values = []
    for inp in tree.flatten(model1.input):
        inp_values.append(np.random.uniform(size=(batch,) + inp.shape[1:]))
    x = tree.pack_sequence_as(model1.input, inp_values)
    # run the values through both models: outputs should be identical
    result1 = model1(x)
    result2 = model2(x)
    res = [
        not np.any(
            keras.ops.convert_to_numpy(r1) - keras.ops.convert_to_numpy(r2)
        )
        for r1, r2 in zip(tree.flatten(result1), tree.flatten(result2))
    ]
    return all(res)


def are_all_weights_identical(model1, model2):
    same_trainable_weights = [
        not np.any(v.numpy() - w.numpy())
        for v, w in zip(model1.trainable_weights, model2.trainable_weights)
    ]
    same_non_trainable_weights = [
        not np.any(v.numpy() - w.numpy())
        for v, w in zip(
            model1.non_trainable_weights, model2.non_trainable_weights
        )
    ]
    return all(same_trainable_weights + same_non_trainable_weights)


# clone_fn that returns the same node
def identity_clone_fn(layer, *args, **kwargs):
    return layer(*args, **kwargs)  # identity


# Custom layers for tests


# custom wrapper around a Dense layer
class XLinearWrapper(layers.Layer):
    def __init__(self, dense_layer):
        super().__init__()
        self.units = dense_layer.units
        self.w = dense_layer.kernel
        self.b = dense_layer.bias

    def call(self, inputs):
        result = keras.ops.matmul(inputs, self.w) + self.b
        return result


# custom layer with dictionary output
class XLinearDictOutput(layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        result = keras.ops.matmul(inputs, self.w) + self.b
        return {"bias": self.b, "result": result}


# custom layer with multiple inputs
class XLinearMultiInput(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs, external_bias=0):
        result = keras.ops.matmul(inputs, self.w) + external_bias
        return result


# custom layer with dictionary input
class XLinearDictInput(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        shape = input_shape["x"]
        self.w = self.add_weight(
            shape=(shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        inp = inputs["x"]
        external_bias = inputs["external_bias"]
        result = keras.ops.matmul(inp, self.w) + external_bias
        return result


class CloningNodegraphTest(unittest.TestCase):

    def compare_models(self, model1, model2, equivalent_nodes=None):
        visited1 = gather_node_graph(
            model1.input, model1.output, enter_nested=True
        )
        visited2 = gather_node_graph(
            model2.input, model2.output, enter_nested=True
        )
        self.assertTrue(
            is_same_layer_graph(visited1, visited2, equivalent_nodes)
        )
        self.assertTrue(are_all_nodes_cloned(visited1, visited2))
        self.assertTrue(are_results_identical(model1, model2))
        self.assertTrue(are_all_weights_identical(model1, model2))

    def test_simplest_model(self):
        #   x:input, D:Dense, y:output
        #   model: x—D—D—y
        x = Input(shape=(3,), batch_size=2, name="input_a")
        y = layers.Dense(5)(x)
        y = layers.Dense(4)(y)

        model = Model(x, y, name="simplest")

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_sub1_model(self):
        #   x:input, C:Conv2D, y:output
        #              ╭╌╌╌╌╌╮
        #   model: x—C—┊—C—C—┊—y
        #              ╰╌╌╌╌╌╯
        x = Input(shape=(28, 28, 3), name="input")
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11c")(x)
        z = layers.Conv2D(8, (3, 3), padding="same", name="cnv33")(y)
        t = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")(z)
        sub = keras.Model(y, t)
        v = sub(y)
        model = keras.Model(x, v)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_sub2_model(self):
        #   x:input, C:Conv2D, y:output, a: instance of a submodel
        #              ╭╌╌╌╌╌╮  ╭╌╌╌╌╌╮
        #   model: x—C—┊—C—C—┊——┊—C—C—┊——y
        #              ╰╌╌╌╌╌╯a ╰╌╌╌╌╌╯a
        x = Input((28, 28, 3), name="input")
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11c")(x)
        z = layers.Conv2D(8, (3, 3), padding="same", name="cnv33")(y)
        t = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")(z)
        sub = keras.Model(y, t)
        v = sub(y)
        w = sub(v)
        model = keras.Model(x, w)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_diamond_model(self):
        # x:input, C:Conv2D, y:output,
        # a: instance of a submodel, CAT: concatenate
        #               ╭╌╌╌╌╌╮a
        #              ╱┊—C—C—┊╲
        #   model: x—C⟨ ╰╌╌╌╌╌╯ ⟩CAT—y
        #              ╲╭╌╌╌╌╌╮╱
        #               ┊—C—C—┊
        #               ╰╌╌╌╌╌╯a
        x = Input((28, 28, 3), name="input")
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11c")(x)
        z = layers.Conv2D(8, (3, 3), padding="same", name="cnv33")(y)
        t = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")(z)
        sub = keras.Model(y, t)
        v = sub(y)
        w = sub(y)
        vw = keras.layers.Concatenate()([v, w])
        model = keras.Model(x, vw)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_diamond_in_sub_model(self):
        #   x:input, C:Conv2D, y:output, CAT: concatenate
        #               ╭╌╌╌╌╌╌╌╌╌╌╌╮
        #   sub:        ┊    C      ┊
        #              —┊—C⟨   ⟩CAT—┊—
        #               ┊    C      ┊
        #               ╰╌╌╌╌╌╌╌╌╌╌╌╯
        #
        #   model:  x—sub—sub—y
        #
        x = Input((28, 28, 3), name="input")
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11c")(x)
        z = layers.Conv2D(4, (3, 3), padding="same", name="cnv33")(y)
        t = layers.Conv2D(4, (1, 1), padding="same", name="cnv11")(y)
        v = keras.layers.Concatenate()([z, t])
        sub = keras.Model(y, v)
        w = sub(y)
        w = sub(w)

        model = keras.Model(x, w)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_diamond_with_subs_model(self):
        #   x:input, C:Conv2D, y:output,
        #   a: instance of a submodel, CAT: concatenate
        #               ╭╌╌╌╌╌╮a
        #              ╱┊—C—C—┊╲
        #   model: x—C⟨ ╰╌╌╌╌╌╯ C
        #              ╲╭╌╌╌╌╌╮  ⟩CAT—y
        #               ┊—C—C—┊╱
        #               ╰╌╌╌╌╌╯a
        #
        x = Input((28, 28, 3), name="input")
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11c")(x)
        z = layers.Conv2D(8, (3, 3), padding="same", name="cnv33")(y)
        t = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")(z)
        sub = keras.Model(y, t)
        v = sub(y)
        v = layers.Conv2D(8, (1, 1), padding="same", name="cnv11b")(v)
        w = sub(y)
        vw = keras.layers.Concatenate()([v, w])
        model = keras.Model(x, vw)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_sub_in_sub_model(self):
        #   x:input, C:Conv2D, y:output, CAT: concatenate
        #               ╭╌╌╌╌╌╌╌╌╌╌╌╮
        #   sub:        ┊    C      ┊
        #              —┊—C⟨   ⟩CAT—┊—
        #               ┊    C      ┊
        #               ╰╌╌╌╌╌╌╌╌╌╌╌╯
        #
        #               ╭╌╌╌╌╌╌╌╌╌╌╮
        #   sub2:       ┊  sub     ┊
        #              —┊—⟨   ⟩CAT—┊—
        #               ┊  sub     ┊
        #               ╰╌╌╌╌╌╌╌╌╌╌╯
        #
        #   model: x—C—sub2—C—sub2—y
        #
        x = Input((28, 28, 3), name="input")
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11c")(x)
        z = layers.Conv2D(8, (3, 3), padding="same", name="cnv33")(y)
        t = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")(z)
        sub = keras.Model(y, t)
        v = sub(y)
        w = sub(y)
        vw = keras.layers.Concatenate()([v, w])
        sub2 = keras.Model(y, w, name="sub2")
        vw = layers.Conv2D(8, (1, 1), padding="same", name="cnv11b")(vw)
        vw = sub2(vw)
        model = keras.Model(x, vw)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_sequential_model(self):
        model = keras.Sequential(
            [
                Input(shape=(28, 28, 3)),
                layers.Conv2D(8, (3, 3), padding="same", name="cnv33"),
                layers.Conv2D(8, (1, 1), padding="same", name="cnv11"),
            ]
        )

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_sequential_sub(self):
        sub = keras.Sequential(
            [
                Input(shape=(28, 28, 8)),
                layers.Conv2D(8, (3, 3), padding="same", name="cnv33"),
                layers.Conv2D(8, (1, 1), padding="same", name="cnv11"),
            ]
        )

        x = Input((28, 28, 8), name="input")
        y = sub(x)
        y = sub(y)
        y = layers.Conv2D(8, (1, 1), padding="same", name="cnv11a")(y)
        y = sub(y)
        model = Model(x, y)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)
        self.compare_models(model, new_model)

    def test_insert_layer(self):
        # create layers beforehand so that we can use the same layers
        # in the reference model constructed by hand for comparison
        cnv33a = layers.Conv2D(8, (3, 3), padding="same", name="cnv33a")
        cnv33b = layers.Conv2D(8, (3, 3), padding="same", name="cnv33b")
        dense = layers.Dense(4)

        x = Input((28, 28, 3), name="input")
        y = cnv33a(x)
        y = cnv33b(y)
        y = dense(y)
        model = Model(x, y, name="simplest")

        # create layers to be inserted beforehand so that we can use the same
        # layers in the reference model constructed by hand for comparison
        cnt = 0
        cnv11layers = [
            layers.Conv2D(8, (1, 1), padding="same", name="cnv11a"),
            layers.Conv2D(8, (1, 1), padding="same", name="cnv11b"),
        ]

        def insert_layer_after_conv(layer, *args, **kwargs):
            nonlocal cnv11layers, cnt
            if isinstance(layer, layers.Conv2D):
                y = layer(*args, **kwargs)
                y = cnv11layers[cnt](y)
                cnt += 1
                return y
            else:
                return layer(*args, **kwargs)

        output = clone_layer_graph(
            model.input, model.output, insert_layer_after_conv
        )
        new_model = Model(model.input, output)

        # The resulting model should be equivalent to:
        x = Input((28, 28, 3), name="input")
        y = cnv33a(x)
        y = cnv11layers[0](y)
        y = cnv33b(y)
        y = cnv11layers[1](y)
        y = dense(y)
        ref_model = Model(x, y, name="simplest")

        self.compare_models(ref_model, new_model)

    def test_insert_layer_in_sub(self):
        # create layers beforehand so that we can use the same layers
        # in the reference model constructed by hand for comparison
        cnv110 = layers.Conv2D(8, (1, 1), padding="same", name="cnv110")
        cnv33a = layers.Conv2D(8, (3, 3), padding="same", name="cnv33a")
        cnv33b = layers.Conv2D(16, (3, 3), padding="same", name="cnv33b")

        x = Input((28, 28, 3), name="input")
        y = cnv110(x)
        z = cnv33a(y)
        z = y + z
        sub = Model(y, z)

        t = sub(y)
        t = layers.Concatenate()([y, t])
        t = cnv33b(t)
        model = Model(x, t)

        # Note: in this test, cnv33a is called twice in the layer graph
        # but since the second call is through sub, which is re-run only once
        # cnv33a will be invoked only once in clone_fn
        # (here insert_layer_after_conv). As a result, a single layer, cnv11a,
        # will be inserted after cnv33a. In normal usage, users can instantiate
        # the layer to insert directly in clone_fn.

        # create layers to be inserted beforehand so that we can use the same
        # layers in the reference model constructed by hand for comparison
        cnt = 0
        cnv11layers = [
            layers.Conv2D(8, (1, 1), padding="same", name="cnv11a"),
            layers.Conv2D(8, (1, 1), padding="same", name="cnv11b"),
        ]

        def insert_layer_after_conv(layer, *args, **kwargs):
            nonlocal cnv11layers, cnt
            if isinstance(layer, layers.Conv2D) and layer.kernel_size == (3, 3):
                y = layer(*args, **kwargs)
                y = cnv11layers[cnt](y)
                cnt += 1
                return y
            else:
                return layer(*args, **kwargs)

        output = clone_layer_graph(
            model.input, model.output, insert_layer_after_conv
        )
        new_model = Model(model.input, output)

        # The resulting model should be equivalent to:
        x = Input((28, 28, 3), name="input")
        y = cnv110(x)
        z = cnv33a(y)
        z = cnv11layers[0](z)
        z = y + z
        sub = Model(y, z)

        t = sub(y)
        t = layers.Concatenate()([y, t])
        t = cnv33b(t)
        t = cnv11layers[1](t)
        ref_model = Model(x, t)

        self.compare_models(ref_model, new_model)

    def test_insert_layer_in_sequential_sub(self):
        # create layers beforehand so that we can use the same layers
        # in the reference model constructed by hand for comparison
        cnv33a = layers.Conv2D(8, (3, 3), padding="same", name="cnv33a")
        cnv33b = layers.Conv2D(8, (3, 3), padding="same", name="cnv33b")
        cnv11 = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")

        sub = keras.Sequential(
            [
                # Input(shape=(28, 28, 8)),
                cnv33a,
                cnv33b,
            ]
        )

        x = Input((28, 28, 8), name="input")
        y = sub(x)
        y = sub(y)
        y = cnv11(y)
        y = sub(y)
        model = Model(x, y)

        # create layers to be inserted beforehand so that we can use the same
        # layers in the reference model constructed by hand for comparison
        cnv11_ins = [
            layers.Conv2D(8, (1, 1), padding="same", name="cnv11a"),
            layers.Conv2D(8, (1, 1), padding="same", name="cnv11b"),
        ]
        cnt = 0

        def insert_layer_after_conv(layer, *args, **kwargs):
            nonlocal cnv11_ins, cnt
            if isinstance(layer, layers.Conv2D) and layer.kernel_size == (3, 3):
                y = layer(*args, **kwargs)
                y = cnv11_ins[cnt](y)
                cnt += 1
                return y
            else:
                return layer(*args, **kwargs)

        output = clone_layer_graph(
            model.input, model.output, insert_layer_after_conv
        )
        new_model = Model(model.input, output)

        # The resulting model should be equivalent to:
        sub = keras.Sequential(
            [
                # Input(shape=(28, 28, 8)),
                cnv33a,
                cnv11_ins[0],
                cnv33b,
                cnv11_ins[1],
            ]
        )

        x = Input((28, 28, 8), name="input")
        y = sub(x)
        y = sub(y)
        y = cnv11(y)
        y = sub(y)
        ref_model = Model(x, y)

        self.compare_models(ref_model, new_model)

    def test_insert_layer_in_diamond(self):
        cnv11 = layers.Conv2D(8, (1, 1), padding="same", name="cnv11")
        cnv33a = layers.Conv2D(8, (3, 3), padding="same", name="cnv33a")
        cnv33b = layers.Conv2D(8, (3, 3), padding="same", name="cnv33b")

        x = Input((28, 28, 3), name="input")
        y = cnv33a(x)
        z = cnv33a(x)
        t = y + z
        t = cnv11(t)
        t = cnv33b(t)
        model = Model(x, t)

        # Note: in this test, cnv33 is called twice in the layer graph, as a
        # shared layer. Since this layer has two nodes in the graph, clone_fn
        # will be called twice on it. If users want to handle this case and
        # insert the same shared cnv11 layer after cnv33, they can do so with
        # the clone_fn code provided here. If users simply instantiate the
        # cnv11 to insert in clone_fn, there will be two (non-shared) instances.

        # create layers to be inserted beforehand so that we can use the same
        # layers in the reference model constructed by hand for comparison
        cnv11layers = {
            id(cnv33a):
                layers.Conv2D(8, (1, 1), padding="same", name="cnv11a"),
            id(cnv33b):
                layers.Conv2D(8, (1, 1), padding="same", name="cnv11b"),
        }

        def insert_layer_after_conv(layer, *args, **kwargs):
            nonlocal cnv11layers
            if isinstance(layer, layers.Conv2D) and layer.kernel_size == (3, 3):
                y = layer(*args, **kwargs)
                y = cnv11layers[id(layer)](y)
                return y
            else:
                return layer(*args, **kwargs)

        output = clone_layer_graph(
            model.input, model.output, insert_layer_after_conv
        )
        new_model = Model(model.input, output)

        # The resulting model should be equivalent to:
        x = Input((28, 28, 3), name="input")
        cnv11a = cnv11layers[id(cnv33a)]
        cnv11b = cnv11layers[id(cnv33b)]
        y = cnv33a(x)
        y = cnv11a(y)
        z = cnv33a(x)
        z = cnv11a(z)
        t = y + z
        t = cnv11(t)
        t = cnv33b(t)
        t = cnv11b(t)
        ref_model = Model(x, t)

        self.compare_models(ref_model, new_model)

    def test_pytree_output(self):
        x = Input(shape=(12,))
        y = layers.Dense(8)(x)
        z = layers.Dense(12)(y)
        t = layers.Concatenate()([y, z])
        output = {"direct": x, "small": y, "large": z, "concatenated": t}
        model = Model(x, output)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)

        self.compare_models(model, new_model)

    def test_pytree_input(self):
        x = {"in1": Input(shape=(12,)), "in2": Input(shape=(12,))}
        y = layers.Dense(8)(x["in1"])
        z = layers.Dense(12)(x["in2"])
        t = layers.Concatenate()([y, z])
        model = Model(x, t)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)

        self.compare_models(model, new_model)

    def test_pytree_between_layers(self):
        x = Input(shape=(12,))
        dicty = XLinearDictOutput(8)(x)
        z = layers.Dense(8)(dicty["result"])
        t = z + dicty["bias"]
        model = Model(x, t)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)

        self.compare_models(model, new_model)

    def test_pytree_partial(self):
        x = Input(shape=(12,))
        z = layers.Dense(8)(x)
        dicty = XLinearDictOutput(8)(z)
        # only one selected output of the layer becomes a model output
        output = dicty["result"]
        model = Model(x, output)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)

        self.compare_models(model, new_model)

    def test_pytree_anything(self):
        x = Input(shape=(12,))
        dicty = XLinearDictOutput(8)(x)
        y = XLinearMultiInput(8)(dicty["result"], external_bias=dicty["bias"])
        y = XLinearDictInput(8)({"x": y, "external_bias": dicty["bias"]})
        dictz = XLinearDictOutput(8)(y)
        output = [dictz["result"], dicty["bias"], dictz["bias"]]
        model = Model(x, output)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)

        self.compare_models(model, new_model)

    def test_pytree_sub(self):
        x = Input(shape=(8,))
        y = layers.Dense(8)(x)
        dicty = XLinearDictOutput(8)(x)
        z = XLinearMultiInput(8)(dicty["result"], external_bias=dicty["bias"])
        z = XLinearDictInput(8)({"x": z, "external_bias": dicty["bias"]})
        sub = Model(x, {"result": y, "extra_bias": dicty["bias"]})
        dictz = sub(y)
        dictt = sub(dictz["result"])
        dictw = sub(dictt["result"])
        output = [
            dictw["result"],
            dicty["bias"],
            dictz["extra_bias"],
            dictt["extra_bias"],
            dictw["extra_bias"],
        ]
        model = Model(x, output)

        output = clone_layer_graph(model.input, model.output, identity_clone_fn)
        new_model = Model(model.input, output)

        self.compare_models(model, new_model)

    def test_return_different_pytree(self):
        x = Input(shape=(8,))
        y = layers.Dense(8)(x)
        dicty = XLinearDictOutput(8)(y)
        y = XLinearDictInput(8)(
            {"x": dicty["result"], "external_bias": dicty["bias"]}
        )
        model = Model(x, y)

        def swap_layer(layer, *args, **kwargs):
            if isinstance(layer, XLinearDictOutput):
                # Try to replace XLinearDictOutput with regular Dense
                # which has a different output structure.
                y = layers.Dense(layer.units)(*args, **kwargs)
                return y
            else:
                return layer(*args, **kwargs)

        try:
            output = clone_layer_graph(model.input, model.output, swap_layer)
            assert False, (
                "A clone_fn function returning a structure that is different"
                "from the expected structure at that point in the graph should"
                "error out."
            )
        except TypeError:
            # OK, this is the expected error in this case
            return

    def test_wrapper(self):
        x = Input(shape=(8,))
        y = layers.Dense(8)(x)
        y = layers.Dense(16)(y)
        y = layers.Dense(32)(y)
        model = Model(x, y)

        def swap_layer(layer, *args, **kwargs):
            if isinstance(layer, layers.Dense):
                # Replace the layer with a wrapper
                wrapper_layer = XLinearWrapper(layer)
                return wrapper_layer(*args, **kwargs)
            else:
                return layer(*args, **kwargs)

        output = clone_layer_graph(model.input, model.output, swap_layer)
        new_model = Model(model.input, output)

        self.compare_models(
            model, new_model, equivalent_nodes=[(layers.Dense, XLinearWrapper)]
        )
