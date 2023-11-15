import numpy as np
import pytest

from keras import backend
from keras import layers
from keras import testing
from keras.layers.core.input_layer import Input
from keras.models.functional import Functional
from keras.models.sequential import Sequential


@pytest.mark.requires_trainable_backend
class SequentialTest(testing.TestCase):
    def test_basic_flow_with_input(self):
        model = Sequential(name="seq")
        model.add(Input(shape=(2,), batch_size=3))
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        model.summary()

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)

        # Test eager call
        x = np.random.random((3, 2))
        y = model(x)

        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(y.shape, (3, 5))

        # Test symbolic call
        x = backend.KerasTensor((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test `layers` constructor arg
        model = Sequential(
            layers=[
                Input(shape=(2,), batch_size=3),
                layers.Dense(4),
                layers.Dense(5),
            ]
        )
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)

        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test pop
        model.pop()
        self.assertEqual(len(model.layers), 1)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 2)

        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 4))

    def test_legacy_flow_with_input_shape(self):
        model = Sequential(name="seq")
        model.add(layers.Dense(4, input_shape=(2,)))
        model.add(layers.Dense(5))

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)
        self.assertEqual(type(model._functional), Functional)

        # Input_dim works too
        model = Sequential(name="seq")
        model.add(layers.Dense(4, input_dim=2))
        model.add(layers.Dense(5))

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)
        self.assertEqual(type(model._functional), Functional)

        # Subsequent input_shapes are ignored
        model = Sequential(name="seq")
        model.add(layers.Dense(4, input_shape=(2,)))
        model.add(layers.Dense(5, input_shape=(3, 4)))

        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 4)
        self.assertEqual(type(model._functional), Functional)

    def test_basic_flow_deferred(self):
        model = Sequential(name="seq")
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        model.summary()

        self.assertEqual(len(model.layers), 2)

        # Test eager call
        x = np.random.random((3, 2))
        y = model(x)
        self.assertTrue(model.built)
        model.summary()

        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(y.shape, (3, 5))

        # Test symbolic call
        x = backend.KerasTensor((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test `layers` constructor arg
        model = Sequential(
            layers=[
                layers.Dense(4),
                layers.Dense(5),
            ]
        )
        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test pop
        model.pop()
        self.assertEqual(len(model.layers), 1)
        self.assertTrue(model.built)
        self.assertEqual(len(model.weights), 2)

        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 4))

    def test_dict_inputs(self):
        class DictLayer(layers.Layer):
            def call(self, inputs):
                assert isinstance(inputs, dict)
                return inputs

        model = Sequential([DictLayer()])
        x = {"a": np.random.random((3, 2)), "b": np.random.random((3, 2))}
        y = model(x)
        self.assertEqual(type(y), dict)
        model.summary()

    def test_list_inputs(self):
        class ListLayer(layers.Layer):
            def call(self, inputs):
                assert isinstance(inputs, list)
                return inputs

        model = Sequential([ListLayer()])
        x = [np.random.random((3, 2)), np.random.random((3, 2))]
        y = model(x)
        self.assertEqual(type(y), list)
        model.summary()

    def test_errors(self):
        # Trying to pass 2 Inputs
        model = Sequential()
        model.add(Input(shape=(2,), batch_size=3))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            model.add(Input(shape=(2,), batch_size=3))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            model.add(layers.InputLayer(shape=(2,), batch_size=3))

        # Same name 2x
        model = Sequential()
        model.add(layers.Dense(2, name="dense"))
        with self.assertRaisesRegex(ValueError, "should have unique names"):
            model.add(layers.Dense(2, name="dense"))

        # No layers
        model = Sequential()
        x = np.random.random((3, 2))
        with self.assertRaisesRegex(ValueError, "no layers"):
            model(x)

        # Build conflict
        model = Sequential()
        model.add(Input(shape=(2,), batch_size=3))
        model.add(layers.Dense(2))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            model.build((3, 4))
        # But this works
        model.build((3, 2))

    def test_shape_inference_failure(self):
        class DynamicLayer(layers.Layer):
            def call(self, inputs):
                return inputs + 1.0

            def compute_output_spec(self, *args, **kwargs):
                raise NotImplementedError

        model = Sequential([DynamicLayer()])
        x = np.random.random((3, 2))
        y = model(x)
        self.assertAllClose(y, x + 1)
        model.summary()

    def test_serialization(self):
        # Unbuilt deferred
        model = Sequential(name="seq")
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        revived = self.run_class_serialization_test(model)
        self.assertLen(revived.layers, 2)

        # Built deferred
        model.build((2, 3))
        revived = self.run_class_serialization_test(model)
        self.assertLen(revived.layers, 2)

        # Regular
        model = Sequential(name="seq")
        model.add(Input(shape=(2,), batch_size=3))
        model.add(layers.Dense(4))
        model.add(layers.Dense(5))
        model.add(layers.Dense(6))
        revived = self.run_class_serialization_test(model)
        self.assertLen(revived.layers, 3)

        # Weird
        class DictLayer(layers.Layer):
            def call(self, inputs):
                assert isinstance(inputs, dict)
                return inputs

        model = Sequential([DictLayer()])
        revived = self.run_class_serialization_test(
            model, custom_objects={"DictLayer": DictLayer}
        )
        self.assertLen(revived.layers, 1)

    def test_functional_properties(self):
        model = Sequential(name="seq")
        inputs = Input(shape=(2,))
        model.add(inputs)
        model.add(layers.Dense(4))

        self.assertEqual(model.inputs, [inputs])
        self.assertEqual(model.outputs, [model.layers[-1].output])
        self.assertEqual(model.input_shape, (None, 2))
        self.assertEqual(model.output_shape, (None, 4))

    def test_bad_layer(self):
        model = Sequential(name="seq")
        with self.assertRaisesRegex(ValueError, "Only instances of"):
            model.add({})

        model = Sequential(name="seq")

        class BadLayer(layers.Layer):
            def call(self, inputs, training):
                return inputs

        model.add(BadLayer())
        with self.assertRaisesRegex(
            ValueError, "can only have a single positional"
        ):
            model.build((None, 2))
