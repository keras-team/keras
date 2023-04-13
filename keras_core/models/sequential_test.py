import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import testing
from keras_core.layers.core.input_layer import Input
from keras_core.models.functional import Functional
from keras_core.models.sequential import Sequential


class SequentialTest(testing.TestCase):
    def test_basic_flow_with_input(self):
        model = Sequential(name="seq")
        model.add(Input(shape=(2,), batch_size=3))
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
                Input(shape=(2,), batch_size=3),
                layers.Dense(4),
                layers.Dense(5),
            ]
        )
        x = np.random.random((3, 2))
        y = model(x)
        self.assertEqual(y.shape, (3, 5))

        # Test pop
        model.pop()
        self.assertFalse(model.built)
        self.assertEqual(model._functional, None)
        x = np.random.random((3, 2))
        y = model(x)
        self.assertTrue(model.built)
        self.assertEqual(type(model._functional), Functional)
        self.assertEqual(y.shape, (3, 4))

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
        self.assertFalse(model.built)
        self.assertEqual(model._functional, None)
        x = np.random.random((3, 2))
        y = model(x)
        self.assertTrue(model.built)
        self.assertEqual(type(model._functional), Functional)
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

    def test_serialization(self):
        # TODO
        pass
