import os
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

from keras.models import clone_model
from keras.src import applications
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src import tree
from keras.src.layers.core.composite_layer import CompositeLayer
from keras.src.layers.core.dense import Dense
from keras.src.layers.core.input_layer import Input
from keras.src.layers.input_spec import InputSpec
from keras.src.models.model import Model
from keras.src.models.sequential import Sequential


class CompositeLayerTest(testing.TestCase):
    def test_basic_flow(self):
        def my_layer_fn(inputs):
            x = layers.Dense(5, name="dense1")(inputs)
            return layers.Dense(4, name="dense2")(x)

        layer = CompositeLayer(my_layer_fn, name="basic")

        self.assertEqual(layer.name, "basic")
        self.assertIsInstance(layer, CompositeLayer)
        self.assertIsInstance(layer, layers.Layer)
        self.assertFalse(layer.built)  # Should be lazily built

        # Eager call - should trigger build
        in_val = np.random.random((2, 3))
        out_val = layer(in_val)
        self.assertEqual(out_val.shape, (2, 4))
        self.assertTrue(layer.built)  # Should be built now

        # weights
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(layer.weights[0].path, "basic/dense1/kernel")
        self.assertEqual(layer.weights[0].shape, (3, 5))
        self.assertEqual(layer.weights[1].path, "basic/dense1/bias")
        self.assertEqual(layer.weights[1].shape, (5,))
        self.assertEqual(layer.weights[2].path, "basic/dense2/kernel")
        self.assertEqual(layer.weights[2].shape, (5, 4))
        self.assertEqual(layer.weights[3].path, "basic/dense2/bias")
        self.assertEqual(layer.weights[3].shape, (4,))

        # variables
        self.assertEqual(len(layer.variables), 4)
        self.assertEqual(layer.variables[0].path, "basic/dense1/kernel")
        self.assertEqual(layer.variables[0].shape, (3, 5))
        self.assertEqual(layer.variables[1].path, "basic/dense1/bias")
        self.assertEqual(layer.variables[1].shape, (5,))
        self.assertEqual(layer.variables[2].path, "basic/dense2/kernel")
        self.assertEqual(layer.variables[2].shape, (5, 4))
        self.assertEqual(layer.variables[3].path, "basic/dense2/bias")
        self.assertEqual(layer.variables[3].shape, (4,))

        # Symbolic call
        test_input = Input(shape=(3,), batch_size=2)
        test_output = layer(test_input)
        self.assertEqual(test_output.shape, (2, 4))
        self.assertTrue(layer.built)  # Should be built now

    def test_basic_flow_as_a_sublayer(self):
        # Build sublayer
        sublayer = CompositeLayer([layers.Flatten()])

        inputs = Input((None, 4, 5))
        outputs = layers.TimeDistributed(sublayer)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        x = np.random.random((2, 3, 4, 5))
        y = model(x)
        self.assertEqual(y.shape, (2, 3, 4 * 5))

    def test_basic_class_flow(self):
        class MyCompositeLayer(CompositeLayer):
            @staticmethod
            def my_layer_fn(inputs):
                x = layers.Dense(5)(inputs)
                return layers.Dense(4)(x)

            def __init__(self, **kwargs):
                super().__init__(MyCompositeLayer.my_layer_fn, **kwargs)

        layer = MyCompositeLayer(name="func_subclass")

        self.assertEqual(layer.name, "func_subclass")
        self.assertIsInstance(layer, CompositeLayer)
        self.assertIsInstance(layer, layers.Layer)
        self.assertFalse(layer.built)  # Should be lazily built

        # Eager call - should trigger build
        in_val = np.random.random((2, 3))
        out_val = layer(in_val)
        self.assertEqual(out_val.shape, (2, 4))
        self.assertTrue(layer.built)  # Should be built now

        # Symbolic call
        test_input = Input(shape=(3,), batch_size=2)
        test_output = layer(test_input)
        self.assertEqual(test_output.shape, (2, 4))
        self.assertTrue(layer.built)  # Should be built now

    def test_scalar_handling(self):
        def scalar_layer_fn(inputs):
            # Handle scalar input
            return inputs + 1.0

        layer = CompositeLayer(scalar_layer_fn)

        # Test with scalar added to tensor
        in_val = np.zeros((2, 3))
        out_val = layer(in_val)
        self.assertAllClose(out_val, np.ones((2, 3)))

    def test_non_mutable_state(self):
        def layer_fn(inputs):
            x = layers.Dense(5)(inputs)
            outputs = layers.Dense(5)(x)
            return outputs

        layer = CompositeLayer(layer_fn)
        layer.build([2, 3])
        with self.assertRaisesRegex(
            ValueError, "You cannot add new elements of state*"
        ):
            layer.extra_layer = layers.Dense(5)

    def test_multi_output(self):
        def multi_output_fn(inputs):
            x = layers.Dense(5)(inputs)
            output_a = layers.Dense(4)(x)
            output_b = layers.Dense(5)(x)
            return [output_a, output_b]

        layer = CompositeLayer(multi_output_fn)

        # Eager call
        in_val = np.random.random((2, 3))
        out_val = layer(in_val)
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

        # Symbolic call
        out_val = layer(Input(shape=(3,), batch_size=2))
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].shape, (2, 4))
        self.assertEqual(out_val[1].shape, (2, 5))

    def test_dict_io(self):
        def dict_io_fn(inputs):
            # Inputs is expected to be a dict with keys 'a' and 'b'
            x = inputs["a"] + inputs["b"]
            x = layers.Dense(5)(x)
            return layers.Dense(4)(x)

        layer = CompositeLayer(dict_io_fn)

        # Test with dictionary input
        in_val = {"a": np.random.random((2, 3)), "b": np.random.random((2, 3))}
        out_val = layer(in_val)
        self.assertEqual(out_val.shape, (2, 4))

    def test_layer_fn_init(self):
        # Test initialization with a layer function
        def my_layer_fn(inputs):
            x = layers.Dense(64, activation="relu")(inputs)
            return layers.Dense(32)(x)

        layer = CompositeLayer(my_layer_fn, name="layer_fn_composite")

        self.assertEqual(layer.name, "layer_fn_composite")
        self.assertIsInstance(layer, CompositeLayer)
        self.assertIsInstance(layer, layers.Layer)
        self.assertFalse(layer.built)  # Should be lazily built

        # Eager call - should trigger build
        in_val = np.random.random((2, 32))
        out_val = layer(in_val)
        self.assertEqual(out_val.shape, (2, 32))
        self.assertTrue(layer.built)  # Should be built now

        # Check that the layers are properly created after building
        self.assertEqual(len(layer.layers), 3)  # Exactly 3 layers, incl. Input

        # Symbolic call
        test_input = Input(shape=(32,), batch_size=2)
        test_output = layer(test_input)
        self.assertEqual(test_output.shape, (2, 32))

    def test_sequential_init(self):
        # Test initialization with a list of layers
        layer = CompositeLayer(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(32),
            ],
            name="sequential_composite",
        )

        self.assertEqual(layer.name, "sequential_composite")
        self.assertIsInstance(layer, CompositeLayer)
        self.assertIsInstance(layer, layers.Layer)

        # Check that the layers are properly stored
        layer.build(input_shape=(2, 32))
        self.assertEqual(len(layer.layers), 3)  # 3 layers incl. Input
        self.assertIsInstance(layer.layers[0], layers.InputLayer)
        self.assertIsInstance(layer.layers[1], layers.Dense)
        self.assertIsInstance(layer.layers[2], layers.Dense)

        # Eager call
        in_val = np.random.random((2, 32))
        out_val = layer(in_val)
        self.assertEqual(out_val.shape, (2, 32))

        # Symbolic call
        test_input = Input(shape=(32,), batch_size=2)
        test_output = layer(test_input)
        self.assertEqual(test_output.shape, (2, 32))

    def test_multi_layer_sequential_init(self):
        # Test with more complex sequential architecture
        layer = CompositeLayer(
            [
                layers.Dense(64, activation="relu", input_shape=(32,)),
                layers.Dropout(0.5),
                layers.BatchNormalization(),
                layers.Dense(32, activation="relu"),
                layers.Dense(16),
            ]
        )

        # Test forward pass
        in_val = np.random.random((4, 32))
        out_val = layer(in_val)
        self.assertEqual(out_val.shape, (4, 16))

        # Check layer composition, incl. InputLayer
        self.assertEqual(len(layer.layers), 6)

    def test_initialization_errors(self):
        # Test with invalid layers parameter type
        with self.assertRaisesRegex(
            ValueError, "is either a function .* or a .* list of layers."
        ):
            CompositeLayer("not_valid")

        # Test error when layers list is empty
        with self.assertRaisesRegex(
            ValueError, "is either a function .* or a .* list of layers."
        ):
            CompositeLayer([])

        x = y = 0
        def layer_fn():
            return x + y

        with self.assertRaisesRegex(
            ValueError, "must take the layer's inputs as its first argument"
        ):
            CompositeLayer(layer_fn)

        # This is allowed
        def layer_fn(inputs, constant_bias=0.1):
            return inputs + constant_bias

        CompositeLayer(layer_fn)(np.random.random((2, 32)))

    def test_serialization(self):
        def run_compsite_layer_serialization_test(layer):
            # this torch-specific attribute is not supposed
            # to be serialized / deserialized so delete it
            # before checking serialization / deserialization.
            if backend.backend() == "torch":
                if hasattr(layer, "_torch_params"):
                    del layer._torch_params
            self.run_class_serialization_test(layer)

        # Test basic model
        def layer_fn(x):
            return layers.Dense(3)(x)

        layer = CompositeLayer(layer_fn, trainable=False)
        inputs = Input(shape=(3,), batch_size=2)
        layer(inputs)  # build the layer
        if backend.backend() == "torch":
            del layer._torch_params
        run_compsite_layer_serialization_test(layer)

        # Test multi-io model
        def layer_fn(inputs):
            input_a, input_b = inputs
            xa = layers.Dense(5, name="middle_a")(input_a)
            xb = layers.Dense(5, name="middle_b")(input_b)
            output_a = layers.Dense(4, name="output_a")(xa)
            output_b = layers.Dense(4, name="output_b")(xb)
            return (output_a, output_b)

        layer = CompositeLayer(layer_fn, name="func")
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        layer([input_a, input_b])  # build the layer
        if backend.backend() == "torch":
            del layer._torch_params
        run_compsite_layer_serialization_test(layer)

        # Test model that includes floating ops
        def layer_fn(inputs):
            input_a, input_b = inputs
            x = input_a + input_b
            x = layers.Dense(5, name="middle")(x)
            output_a = layers.Dense(4, name="output_a")(x)
            output_b = layers.Dense(4, name="output_b")(x)
            return (output_a, output_b)

        layer = CompositeLayer(layer_fn, name="func")
        input_a = Input(shape=(3,), batch_size=2, name="input_a")
        input_b = Input(shape=(3,), batch_size=2, name="input_b")
        layer([input_a, input_b])  # build the layer
        if backend.backend() == "torch":
            del layer._torch_params
        run_compsite_layer_serialization_test(layer)

        # Test model with dict i/o
        def layer_fn(inputs):
            input_a = inputs["a"]
            input_b = inputs["b"]
            x = input_a + input_b
            x = layers.Dense(5)(x)
            return layers.Dense(4)(x)

        layer = CompositeLayer(layer_fn, name="func")
        input_a = Input(shape=(3,), batch_size=2, name="a")
        input_b = Input(shape=(3,), batch_size=2, name="b")
        layer({"a": input_a, "b": input_b})  # build the layer
        if backend.backend() == "torch":
            del layer._torch_params
        run_compsite_layer_serialization_test(layer)

    def test_config_serialization(self):
        # Test serialization of sequential initialization
        original_layer = CompositeLayer(
            layers=[
                layers.Dense(64, activation="relu", input_shape=(32,)),
                layers.Dense(32),
            ]
        )
        original_layer.build([None, 16])

        config = original_layer.get_config()
        self.assertEqual(config["name"], "composite_layer")
        self.assertEqual(len(config["layers"]), 3)

        # Recreate from config
        recreated_layer = CompositeLayer.from_config(config)
        self.assertEqual(config["name"], "composite_layer")
        self.assertEqual(len(recreated_layer.layers), 3)

        # Test the recreated layer works
        # Eager call
        in_val = np.random.random((2, 16))
        out_val = recreated_layer(in_val)
        self.assertEqual(out_val.shape, (2, 32))

        # Symbolic call
        input = Input(shape=(16,), batch_size=5)
        out_val = recreated_layer(input)
        self.assertEqual(out_val.shape, (5, 32))

        # Test serialization of layer_fn initialization
        def test_layer_fn(inputs):
            x = layers.Dense(64)(inputs)
            return layers.Dense(10)(x)

        composite = CompositeLayer(test_layer_fn)
        # Build it first by calling it
        in_val = np.random.random((2, 20))
        composite(in_val)

        # Save and recreate from config
        config = composite.get_config()
        recreated_layer = CompositeLayer.from_config(config)
        self.assertTrue(recreated_layer.built)

        # Test the recreated layer works
        # Eager call
        out_val = recreated_layer(in_val)
        self.assertEqual(out_val.shape, (2, 10))

        # Symbolic call
        input = Input(shape=(20,), batch_size=8)
        out_val = recreated_layer(input)
        # Test with a different batch size:
        # The layer was first called with batch_size=2
        # and is now called with batch_size=8
        self.assertEqual(out_val.shape, (8, 10))

    def test_class_serialization(self):
        class MyCompositeLayer(CompositeLayer):
            @staticmethod
            def my_layer_fn(inputs):
                x = layers.Dense(5)(inputs)
                return layers.Dense(4)(x)

            def __init__(self, **kwargs):
                super().__init__(MyCompositeLayer.my_layer_fn, **kwargs)

        layer = MyCompositeLayer(name="func_subclass")
        layer(Input(shape=(3,), batch_size=2))

        self.assertTrue(layer.built)

        config = layer.get_config()
        restored_layer = MyCompositeLayer.from_config(config)

        self.assertEqual(restored_layer.name, "func_subclass")
        self.assertIsInstance(restored_layer, MyCompositeLayer)
        self.assertIsInstance(restored_layer, CompositeLayer)
        self.assertIsInstance(restored_layer, layers.Layer)
        self.assertTrue(restored_layer.built)

        # Eager call - should trigger build
        in_val = np.random.random((2, 3))
        out_val = restored_layer(in_val)
        self.assertEqual(out_val.shape, (2, 4))

        # Symbolic call
        test_input = Input(shape=(3,), batch_size=2)
        test_output = layer(test_input)
        self.assertEqual(test_output.shape, (2, 4))

    def test_input_dict_with_extra_field(self):
        def layer_fn(inputs):
            x = inputs["a"] * 5
            outputs = x + 2
            return outputs

        layer = CompositeLayer(layer_fn)
        layer({"a": Input((3,))})  # build the layer

        # Eager call with extra value in dict
        in_val = {
            "a": np.random.random((2, 3)),
            "b": np.random.random((2, 1)),
        }
        with pytest.warns() as record:
            out_val = layer(in_val)
            self.assertLen(record, 1)
            self.assertStartsWith(
                str(record[0].message),
                "The structure of `inputs` doesn't match "
                "the expected structure",
            )
            self.assertEqual(out_val.shape, (2, 3))

    def test_warning_for_mismatched_inputs_structure(self):
        def is_input_warning(w):
            return str(w.message).startswith(
                "The structure of `inputs` doesn't match the expected structure"
            )

        def layer_fn(inputs):
            i1 = inputs["i1"]
            i2 = inputs["i2"]
            return layers.Add()([i1, i2])

        composite_layer = CompositeLayer(layer_fn)
        composite_layer(
            {"i1": Input((2,)), "i2": Input((2,))}
        )  # build the layer
        with pytest.warns() as warning_logs:
            composite_layer([np.ones((2, 2)), np.ones((2, 2))])
            self.assertLen(list(filter(is_input_warning, warning_logs)), 1)

        # No warning for mismatched tuples and lists.
        def layer_fn2(inputs):
            i1, i2 = inputs
            return layers.Add()([i1, i2])

        composite_layer = CompositeLayer(layer_fn2)
        composite_layer(
            [Input((2,)), Input((2,))]
        )  # build the layer with a list
        with warnings.catch_warnings(record=True) as warning_logs:
            # call the layer with a tuple
            composite_layer((np.ones((2, 2)), np.zeros((2, 2))))
            self.assertLen(list(filter(is_input_warning, warning_logs)), 0)

    @parameterized.named_parameters(
        ("list", list),
        ("tuple", tuple),
        ("dict", dict),
    )
    def test_restored_multi_output_type(self, out_type):
        def layer_fn(inputs):
            x = layers.Dense(5)(inputs)
            output_a = layers.Dense(4)(x)
            output_b = layers.Dense(5)(x)
            if out_type is dict:
                outputs = {"a": output_a, "b": output_b}
            else:
                outputs = out_type([output_a, output_b])
            return outputs

        layer = CompositeLayer(layer_fn)
        layer.build(input_shape=(2, 3))

        config = layer.get_config()
        layer_restored = CompositeLayer.from_config(config)

        # Eager call
        in_val = np.random.random((2, 3))
        out_val = layer_restored(in_val)
        self.assertIsInstance(out_val, out_type)

        # Symbolic call
        out_val = layer_restored(Input(shape=(3,), batch_size=2))
        self.assertIsInstance(out_val, out_type)

    def test_layer_getters(self):
        def layer_fn(inputs):
            # Test mixing ops and layers
            input_a = inputs["a"]
            input_b = inputs["b"]
            x = input_a + input_b
            x = layers.Dense(5, name="dense_1")(x)
            outputs = layers.Dense(4, name="dense_2")(x)
            return outputs

        layer = CompositeLayer(layer_fn)

        layer.build({"a": (2, 3), "b": (2, 3)})

        # Check layer composition, incl. InputLayer(s)
        self.assertEqual(len(layer.layers), 4)
        self.assertEqual(len(layer._function._operations), 5)
        self.assertEqual(layer.get_layer(index=2).name, "dense_1")
        self.assertEqual(layer.get_layer(index=3).name, "dense_2")
        self.assertEqual(layer.get_layer(name="dense_1").name, "dense_1")

    def test_training_arg(self):
        class Canary(layers.Layer):
            def call(self, x, training=False):
                assert training
                return x

            def compute_output_spec(self, x, training=False):
                return ops.KerasTensor(x.shape, dtype=x.dtype)

        # Test with layer_fn initialization
        def layer_fn(inputs):
            return Canary()(inputs)

        layer_fn_layer = CompositeLayer(layer_fn)
        layer_fn_layer(np.random.random((2, 3)), training=True)

        # Test with sequential initialization
        sequential_layer = CompositeLayer([Canary()])
        sequential_layer(np.random.random((2, 3)), training=True)

    def test_mask_arg(self):
        # TODO (same as in functional_test.py)
        pass

    def test_rank_standardization(self):
        def layer_fn(x):
            return layers.Dense(4)(x)

        # Downranking
        layer = CompositeLayer(layer_fn)
        layer.build((8, 10))
        out_val = layer(np.random.random((8, 10, 1)))
        self.assertEqual(out_val.shape, (8, 4))

        # Upranking
        layer = CompositeLayer(layer_fn)
        layer.build((8, 10, 1))
        out_val = layer(np.random.random((8, 10)))
        self.assertEqual(out_val.shape, (8, 10, 4))

    @pytest.mark.requires_trainable_backend
    def test_dtype_standardization(self):
        def layer_fn(x):
            float_input = x["float"]
            int_input = x["int"]
            float_output = float_input + 2
            int_output = int_input + 2
            return (float_output, int_output)

        # Contrary to a Functional Model, a CompositeLayer has
        # only one input dtype. All of its inputs will be created
        # in build() with the same dtype. If multiple inputs with
        # different dtypes are needed, use a Functional Model.

        # layer with dtype: forces inputs to that dtype
        layer = CompositeLayer(layer_fn, dtype="float16")

        # symbilic call
        float_data, int_data = layer(
            {
                "float": Input((2, 2), dtype="float32"),
                "int": Input((2, 2), dtype="int32"),
            }
        )

        self.assertEqual(backend.standardize_dtype(float_data.dtype), "float16")
        self.assertEqual(backend.standardize_dtype(int_data.dtype), "float16")

        # eager call
        float_data, int_data = layer(
            {
                "float": np.ones((8, 2, 2), dtype="float32"),
                "int": np.ones((8, 2, 2), dtype="int32"),
            }
        )

        self.assertEqual(backend.standardize_dtype(float_data.dtype), "float16")
        self.assertEqual(backend.standardize_dtype(int_data.dtype), "float16")

    def test_bad_input_spec(self):
        # Single input
        def layer_fn(x):
            return layers.Dense(2)(x)

        layer = CompositeLayer(layer_fn)
        layer.build((None, 4))
        with self.assertRaisesRegex(
            ValueError,
            r"Input .* is incompatible .* "
            r"expected shape=\(None, 4\), found shape=\(2, 3\)",
        ):
            layer(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "expects 1 input"):
            layer([np.zeros((2, 4)), np.zeros((2, 4))])

        # List input
        def layer_fn(inputs):
            input_a, input_b = inputs
            x = input_a + input_b
            return layers.Dense(2)(x)

        layer = CompositeLayer(layer_fn)
        input_a = Input(shape=(4,), name="a")
        input_b = Input(shape=(4,), name="b")
        layer([input_a, input_b])  # build the layer
        with self.assertRaisesRegex(ValueError, r"expects 2 input\(s\)"):
            layer(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            layer([np.zeros((2, 3)), np.zeros((2, 4))])

        # Dict input
        def layer_fn(inputs):
            input_a = inputs["a"]
            input_b = inputs["b"]
            y = input_a + input_b
            return layers.Dense(2)(y)

        layer = CompositeLayer(layer_fn)
        input_a = Input(shape=(4,), name="a")
        input_b = Input(shape=(4,))
        layer({"a": input_a, "b": input_b})  # build the layer
        with self.assertRaisesRegex(
            ValueError, r"expects 2 input\(s\), but it received 1 input"
        ):
            layer(np.zeros((2, 3)))
        with self.assertRaisesRegex(
            ValueError, r"expected shape=\(None, 4\), found shape=\(2, 3\)"
        ):
            layer({"a": np.zeros((2, 3)), "b": np.zeros((2, 4))})

    def test_manual_input_spec(self):
        def layer_fn(x):
            return layers.Dense(2)(x)

        layer = CompositeLayer(layer_fn)
        layer.input_spec = InputSpec(shape=(None, 4, 3))
        with self.assertRaisesRegex(
            ValueError,
            r"expected shape=\(None, 4, 3\), found shape=\(8, 3, 3\)",
        ):
            layer(np.zeros((8, 3, 3)))
        layer(np.zeros((8, 4, 3)))

    def test_deeply_nested_composite_layer(self):
        def layer_fn(x):
            # input x is: {"1": i1, "others": {"2": i2, "3": i3}}
            i1 = x["1"]
            i2 = x["others"]["2"]
            i3 = x["others"]["3"]
            o1, o2, o3 = (
                layers.Dense(1)(i1),
                layers.Dense(2)(i2),
                layers.Dense(3)(i3),
            )
            return {"1": o1, "others": {"2": o2, "3": o3}}

        composite_layer = CompositeLayer(layer_fn)
        out_eager = composite_layer(
            {
                "1": np.ones((8, 1)),
                "others": {"2": np.ones((8, 2)), "3": np.ones((8, 3))},
            }
        )
        out_symbolic = composite_layer(
            {
                "1": Input((1,), batch_size=8),
                "others": {
                    "2": Input((2,), batch_size=8),
                    "3": Input((3,), batch_size=8),
                },
            }
        )
        for out in [out_eager, out_symbolic]:
            self.assertIsInstance(out, dict)
            self.assertEqual(set(out.keys()), {"1", "others"})
            self.assertEqual(out["1"].shape, (8, 1))
            self.assertIsInstance(out["others"], dict)
            self.assertEqual(set(out["others"].keys()), {"2", "3"})
            self.assertEqual(out["others"]["2"].shape, (8, 2))
            self.assertEqual(out["others"]["3"].shape, (8, 3))

    @pytest.mark.requires_trainable_backend
    def test_model_with_composite_layers_serialization(self):
        def layer_fn(x):
            # input x is: {"1": i1, "others": {"2": i2, "3": i3}}
            i1 = x["1"]
            i2 = x["others"]["2"]
            i3 = x["others"]["3"]
            o1, o2, o3 = (
                layers.Dense(1)(i1),
                layers.Dense(2)(i2),
                layers.Dense(3)(i3),
            )
            return {"1": o1, "others": {"2": o2, "3": o3}}

        composite_layer = CompositeLayer(layer_fn)
        symbolic_input = {
            "1": Input((1,)),
            "others": {"2": Input((2,)), "3": Input((3,))},
        }
        y = composite_layer(symbolic_input)
        y = layers.Concatenate()([y["1"], y["others"]["2"], y["others"]["3"]])
        output = layers.Dense(4)(y)
        model = Model(symbolic_input, output)

        temp_filepath = os.path.join(self.get_temp_dir(), "deeply_nested.keras")
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        num_input = {
            "1": np.ones((8, 1)),
            "others": {"2": np.ones((8, 2)), "3": np.ones((8, 3))},
        }
        out_eager = model(num_input)
        new_out_eager = loaded_model(num_input)
        self.assertAllClose(out_eager, new_out_eager)

    def test_for_composite_layer_in_sequential(self):
        if backend.image_data_format() == "channels_first":
            image_size = (3, 256, 256)
        else:
            image_size = (256, 256, 3)
        base_model = applications.mobilenet.MobileNet(
            include_top=False, weights=None
        )
        layer = CompositeLayer(
            [
                layers.Conv2D(32, (3, 3)),
                layers.Conv2D(64, (3, 3)),
                layers.Conv2D(128, (3, 3)),
            ]
        )
        model = Sequential()
        model.add(layers.Input(shape=image_size))
        model.add(base_model)
        model.add(layer)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(7, activation="softmax"))
        # eager call
        model(np.random.random((4,) + image_size))
        # symbolic call
        model(Input(shape=image_size))
        # serialization
        config = model.get_config()
        model = Sequential.from_config(config)
        # eager call
        model(np.random.random((4,) + image_size))
        # symbolic call
        model(Input(shape=image_size))

    def test_add_loss(self):
        # TODO (same as in functional_test.py)
        pass

    def test_layers_setter(self):
        layer = CompositeLayer([layers.Dense(4)])

        with self.assertRaisesRegex(AttributeError, "attribute is reserved"):
            layer.layers = [layers.Dense(5)]

    def test_list_input_with_dict_build(self):
        def layer_fn(inputs):
            x1 = inputs["IT"]
            x2 = inputs["IS"]
            return layers.subtract([x1, x2])

        layer = CompositeLayer(layer_fn)
        x1 = Input((10,))
        x2 = Input((10,))
        layer({"IT": x1, "IS": x2})  # build the layer
        x1 = ops.ones((1, 10))
        x2 = ops.zeros((1, 10))
        # eager call works
        layer({"IT": x1, "IS": x2})
        # Note: the test fails here only because the order of dict
        # keys "IT", "IS" is different from the sorted order of the
        # keys "IS", "IT". Otherwise, passing a list of inputs to
        # a model expecting a dictionary of inputs seems to be allowed,
        # as long as flattening the dict does not result in reordering.
        with self.assertRaisesRegex(
            ValueError,
            "The structure of `inputs` doesn't match the expected structure",
        ):
            layer([x1, x2])

    @pytest.mark.requires_trainable_backend
    def test_functional_subclass_serialization(self):
        class FuncSubclass(CompositeLayer):
            @staticmethod
            def layer_fn(x):
                y = layers.Dense(8)(x)
                return layers.Dense(4)(y)

            def __init__(self, name=None, **kwargs):
                super().__init__(FuncSubclass.layer_fn, name, **kwargs)

        inputs = Input((4,), name="input")
        y = FuncSubclass()(inputs)
        model = Model(inputs, y)
        data = ops.ones((8, 4))
        output1 = model(data)  # build the model
        temp_filepath = os.path.join(self.get_temp_dir(), "comp_subclass.keras")
        model.save(temp_filepath)

        # Note: this recreates the layer by calling FuncSubclass.__init__
        # and does *not* test the functional.function_from_config method.

        loaded_model = saving.load_model(
            temp_filepath, custom_objects={"FuncSubclass": FuncSubclass}
        )

        output2 = loaded_model(data)
        self.assertAllClose(output1, output2)

    def test_functional_in_functional_with_reuse_serialization(self):
        ini = initializers.Ones()

        # sub-functional
        def layer_fn(inputs):
            y = layers.Dense(6, kernel_initializer=ini)(inputs)
            return layers.Dense(8, kernel_initializer=ini)(y)

        sub_layer1 = CompositeLayer(layer_fn)

        comp_layer = CompositeLayer(
            [
                sub_layer1,
                sub_layer1,  # reuse
            ]
        )

        data = ops.ones((2, 8))
        output1 = comp_layer(data)

        # this recreates the model from the saved functional graph
        # and *does* test the functional.function_from_config method.

        config = comp_layer.get_config()
        loaded_model = CompositeLayer.from_config(config)

        # check the model works
        output2 = loaded_model(data)
        # check both models return the same results
        # (weights were initialized deterministically)
        self.assertAllClose(output1, output2)

    @pytest.mark.requires_trainable_backend
    def test_functional_in_functional_with_reuse_saving(self):
        # sub-functional
        def layer_fn(inputs):
            y = layers.Dense(6)(inputs)
            return layers.Dense(8)(y)

        sub_layer1 = CompositeLayer(layer_fn)

        inputs = Input((8,))
        y1 = sub_layer1(inputs)
        outputs = sub_layer1(y1)  # reuse

        model = Model(inputs, outputs)
        data = ops.ones((2, 8))
        output1 = model(data)  # build the model

        # this recreates the model from the saved functional graph
        # and *does* test the functional.function_from_config method.

        temp_filepath = os.path.join(
            self.get_temp_dir(), "func_subclass_reuse.keras"
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # check the model works
        output2 = loaded_model(data)
        # check both models return the same results
        # (weights were initialized deterministically)
        self.assertAllClose(output1, output2)

    @pytest.mark.requires_trainable_backend
    def test_composite_in_functional_model(self):
        class ConvStack(CompositeLayer):
            def __init__(self, **kwargs):
                @staticmethod
                def layer_fn(x):
                    y = layers.Conv2D(
                        12,
                        kernel_size=(3, 3),
                        padding="same",
                        activation="relu",
                        name="c1",
                    )(x)
                    y = layers.Conv2D(
                        16, (3, 3), padding="same", activation="relu", name="c2"
                    )(y)
                    z = layers.Conv2D(
                        16, kernel_size=(1, 1), activation="relu", name="c3"
                    )(x)
                    return y + z

                super().__init__(layer_fn, **kwargs)

        class RegulStack(CompositeLayer):
            def __init__(self, **kwargs):
                super().__init__(
                    [layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten()],
                    **kwargs,
                )

        class DenseStack(CompositeLayer):
            def __init__(self, **kwargs):
                super().__init__(
                    [
                        layers.Dense(128, activation="relu"),
                        layers.Dropout(0.25),
                    ],
                    **kwargs,
                )

        if backend.config.image_data_format() == "channels_first":
            input = Input(shape=(1, 28, 28))
            data = np.ones((2, 1, 28, 28))
        else:
            input = Input(shape=(28, 28, 1))
            data = np.ones((2, 28, 28, 1))
        composite_layer1 = ConvStack(name="c1")
        x = composite_layer1(input)
        composite_layer2 = ConvStack(name="c2")
        x = composite_layer2(x)
        composite_layer3 = RegulStack(name="r1")
        x = composite_layer3(x)
        composite_layer4 = DenseStack(name="d1")
        x = composite_layer4(x)
        output = layers.Dense(10, activation="softmax", name="head")(x)
        model = Model(input, output, name="func_model")
        model(input)  # check the model builds

        def is_spec_equal(spec1, spec2):
            tree.assert_same_structure(spec1, spec2)

            def compare_spec(s1, s2):
                return s1.shape == s2.shape and s1.dtype == s2.dtype

            result = tree.map_structure(compare_spec, spec1, spec2)
            return all(tree.flatten(result))

        def get_image_format_shape(channeles_first_shape, channels_last_shape):
            if backend.config.image_data_format() == "channels_first":
                return channeles_first_shape
            else:
                return channels_last_shape

        # check the layers were built correctly
        shape = get_image_format_shape((None, 1, 28, 28), (None, 28, 28, 1))
        self.assertTrue(
            is_spec_equal(composite_layer1.input_spec, InputSpec(shape=shape))
        )
        shape = get_image_format_shape((None, 16, 28, 28), (None, 28, 28, 16))
        self.assertTrue(
            is_spec_equal(composite_layer2.input_spec, InputSpec(shape=shape))
        )
        shape = get_image_format_shape((None, 16, 28, 28), (None, 28, 28, 16))
        self.assertTrue(
            is_spec_equal(composite_layer3.input_spec, InputSpec(shape=shape))
        )
        self.assertTrue(
            is_spec_equal(
                composite_layer4.input_spec,
                InputSpec(shape=(None, 28 * 28 * 16 // 4)),
            )
        )

        # test model serialization with weights
        output1 = model(data)
        temp_filepath = os.path.join(self.get_temp_dir(), "func_nested.keras")
        model.save(temp_filepath)
        loaded_model = saving.load_model(
            temp_filepath,
            custom_objects={
                "ConvStack": ConvStack,
                "RegulStack": RegulStack,
                "DenseStack": DenseStack,
            },
        )

        output2 = loaded_model(data)
        self.assertAllClose(output1, output2)

    def test_optional_inputs(self):
        class OptionalInputLayer(layers.Layer):
            def call(self, x, y=None):
                if y is not None:
                    return x + y
                return x

            def compute_output_shape(self, x_shape):
                return x_shape

        def layer_fn(x):
            x1 = x[0]
            x2 = x[1]
            return OptionalInputLayer()(x2, x1)

        layer = CompositeLayer(layer_fn)

        # declare the first arg as optional
        input_spec = [
            InputSpec(shape=(None, 2), optional=True),
            InputSpec(shape=(None, 2)),
        ]
        layer.input_spec = input_spec

        # symbolic test
        x1 = Input((2,))
        x2 = Input((2,))
        out = layer([x1, x2])
        # Eager test
        data = np.ones((8, 2))
        out = layer([None, data])
        self.assertAllClose(out, data)
        out = layer([data, data])
        self.assertAllClose(out, data * 2)

        # Error message when passing None to a non-optional input
        with self.assertRaisesRegex(
            ValueError, "Optional inputs must be declared"
        ):
            out = layer([data, None])

    # This is still problematic because of spurius
    # wrapping of _manual_input_spec in ListWrapper etc...
    # def test_list_vs_tuple_input_spec2(self):
    #     def layer_fn(inputs):
    #         a, l = inputs
    #         b, c = l
    #         a = Dense(8)(a)
    #         b = Dense(8)(b)
    #         c = Dense(8)(c)
    #         return a+b+c

    #     layer = CompositeLayer(layer_fn)
    #     layer.input_spec = (InputSpec(shape=(None, 8)),
    #                         [InputSpec(shape=(None, 8)),
    #                          InputSpec(shape=(None, 8))])
    #     sym_inputs =  (Input(shape=(None, 8)),
    #                    [Input(shape=(None, 8)),
    #                    Input(shape=(None, 8))])
    #     layer(sym_inputs)

    def test_list_vs_tuple_input_spec(self):
        def layer_fn(inputs):
            a, b = inputs
            a = Dense(8)(a)
            b = Dense(8)(b)
            return a+b

        layer = CompositeLayer(layer_fn)
        layer.input_spec = (InputSpec(shape=(None, 8)), # tuple
                             InputSpec(shape=(None, 8)))
        sym_inputs =  [Input(batch_shape=(None, 8)), # list
                       Input(batch_shape=(None, 8))]
        layer(sym_inputs)
        layer.input_spec = [InputSpec(shape=(None, 8)), # list
                             InputSpec(shape=(None, 8))]
        sym_inputs =  (Input(batch_shape=(None, 8)), # tuple
                       Input(batch_shape=(None, 8)))
        layer(sym_inputs)

    def test_no_batch_input(self):

        def layer_fn(inputs):
            x, y, idx = inputs # idx does not have a batch size
                               # this happens in LLM text generation
                               # with cache_update_idx
            start = [0, idx, 0, 0]
            return ops.slice_update(x, start, y)

        layer = CompositeLayer(layer_fn)

        spec = (InputSpec(shape=(None, 16, 4, 8)),
                InputSpec(shape=(None, 1, 4, 8)),
                InputSpec(shape=(), dtype="int32")) # no batch size
        # input with no batch size defined through manual input_spec
        layer.input_spec = spec

        #layer(inputs)
        layer((np.zeros((32, 16, 4, 8)),
              np.zeros((32, 1, 4, 8)),
              backend.convert_to_tensor(0)))

        def layer_fn(inputs):
            x, mul = inputs # mul does not have a batch size
            return x * mul

        layer = CompositeLayer(layer_fn)

        # input with no batch size defined by calling the layer
        inputs = (Input(shape=(16,)),
                 Input(batch_shape=()))

        layer(inputs) # symbolic call
        # numerical call
        result = layer((np.ones((32, 16)),
                        backend.convert_to_tensor(2)))
        self.assertAllClose(result, np.ones((32, 16)) * 2)        

    # Keeping this as a manual test so that pydot
    # and graphvizare not required for testing.
    # def test_plot(self):
    #     def layer_fn(x):
    #         y = layers.Dense(8)(x)
    #         y = layers.Dense(4)(y)
    #         return y

    #     layer = CompositeLayer(layer_fn)

    #     x = Input((4,))
    #     y = layers.Dense(8)(x)
    #     y = layers.Dense(4)(y)
    #     model = Model(x,y)

    #     model2 = Sequential([
    #         layers.Dense(8),
    #         layers.Dense(8)
    #     ])

    #     x = Input((12,))
    #     y = layer(x)
    #     y = model(y)
    #     y = model2(y)
    #     model = Model(x,y)

    #     data = np.random.uniform(size=(4, 12))
    #     model(data)

    #     utils.plot_model(model, expand_nested=True)

    def test_clone_model(self):
        const_init = initializers.Ones()
        zero_init = initializers.Zeros()

        # alternative dense implementation with dict output
        class AltDense(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer=const_init,
                    trainable=True,
                )
                self.b = self.add_weight(
                    shape=(self.units,),
                    initializer=zero_init,
                    trainable=True,
                )

            def call(self, inputs):
                return ops.matmul(inputs, self.w) + self.b

        data = np.random.uniform(size=(4, 12))

        # CompositeLayer using regular dense layers
        def layer_fn(x):
            y = layers.Dense(
                8, kernel_initializer=const_init, name="original1"
            )(x)
            y = layers.Dense(
                8, kernel_initializer=const_init, name="original2"
            )(y)
            return y

        layer = CompositeLayer(layer_fn)

        # Composite layer instatiated as a subclass of CompositeLayer
        # It is cloned as a vanilla CompositeLayer for now.
        class FuncSub(CompositeLayer):
            def __init__(self, name=None, **kwargs):
                super().__init__(layer_fn, name, **kwargs)

        flayer = FuncSub()

        x = Input((12,))
        y = layers.Dense(8, kernel_initializer=const_init, name="original3")(x)
        y = layer(y)
        y = flayer(y)  # subclass layer
        y = layer(y)  # shared layer
        model = Model(x, y)
        # build the model
        model(data)

        for variable in model.variables:
            self.assertContainsSubsequence(variable.path, "original")

        # replace regular dense layers with alternative
        # dense layers and rewire dict output
        def replace_fn(layer, *args, **kwargs):
            if isinstance(layer, layers.Dense):
                return AltDense(layer.units)(*args, **kwargs)
            else:
                return layer(*args, **kwargs)  # pass-through

        # clone function thas does not do any layer cloning
        # but only creates a new layer graph.
        model2 = clone_model(
            model,
            input_tensors=x,
            # everyhting is done in call_function
            clone_function=lambda x: x,
            call_function=replace_fn,
            recursive=True,
        )

        model2(data)

        # original model is unchanged
        for variable in model.variables:
            self.assertContainsSubsequence(variable.path, "original")

        # new model has new AltDense layers
        for variable in model2.variables:
            self.assertContainsSubsequence(variable.path, "alt_dense")

        self.assertEqual(len(model.layers), len(model2.layers))
        for layer1, layer2 in zip(model.layers, model2.layers):
            if isinstance(layer1, layers.Dense):
                self.assertTrue(layer2.__class__ is AltDense)
            # A subclass of CompositeLayer is cloned as CompositeLayer for now
            elif isinstance(layer1, FuncSub):
                self.assertTrue(
                    layer2.__class__ is CompositeLayer
                    or layer2.__class__ is FuncSub
                )
            else:
                self.assertEqual(layer1.__class__, layer2.__class__)

        self.assertAllClose(model(data), model2(data))

    def test_clone_recursive(self):
        def layer_fn1(inputs):
            return layers.Dense(32, name="dense_2")(inputs)

        layer1 = CompositeLayer(layer_fn1, name="sub")

        def layer_fn2(inputs):
            return layer1(inputs)

        slayer = CompositeLayer(layer_fn2, name="subfunc")

        inputs = layers.Input(shape=(16, 32))
        outputs = slayer(inputs)
        model = Model(inputs, outputs)

        def call_fn(layer, *args, **kwargs):
            if isinstance(layer, layers.Dense):
                new_layer = layers.Dense(layer.units, name="dense_modified")
                return new_layer(*args, **kwargs)
            return layer(*args, **kwargs)

        new_model = clone_model(model, call_function=call_fn, recursive=True)
        sub = new_model.get_layer("subfunc").get_layer("sub")
        self.assertEqual(sub.layers[1].name, "dense_modified")

    def test_clone_with_input_spec(self):
        def layer_fn(inputs):
            return layers.Dense(12)(inputs)

        layer1 = CompositeLayer(layer_fn)
        layer1.input_spec = InputSpec(shape=(None, 12), optional=True)

        def layer_fn2(inputs):
            x = layers.Dense(12)(inputs)
            return layer1(x)
        
        layer2 = CompositeLayer(layer_fn2)
        layer2.input_spec = InputSpec(shape=(None, 12), optional=True)

        layer2(ops.ones(shape=(8,12))) # build it

        def call_fn(layer, *args, **kwargs):
          return layer(*args, **kwargs)
        
        new_layer = clone_model(layer2, clone_function=lambda x:x,
                                        call_function=call_fn,
                                        recursive=True)
        self.assertEqual(new_layer.input_spec, layer2.input_spec)
        self.assertEqual(new_layer.layers[2].input_spec, layer1.input_spec)

    def test_clone_before_build(self):
        def layer_fn(inputs):
            return layers.Dense(12)(inputs)
        
        layer = CompositeLayer(layer_fn)

        def call_fn(layer, *args, **kwargs):
          return layer(*args, **kwargs)
        
        with self.assertRaisesRegex(
            ValueError, "model has no graph of layers. It is probably not built"
        ):
            clone_model(layer, clone_function=lambda x:x,
                               call_function=call_fn,
                               recursive=True)


    def test_build_twice(self):
        def layer_fn(inputs):
            return layers.Dense(5)(inputs)

        layer = CompositeLayer(layer_fn)
        layer.build([2, 3])

        id1 = id(layer.layers[0])
        id2 = id(layer.layers[1])

        # calling build a second time should do nothing
        layer.build([2, 3])

        self.assertEqual(id1, id(layer.layers[0]))
        self.assertEqual(id2, id(layer.layers[1]))
