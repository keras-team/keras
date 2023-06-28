import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import ops
from keras_core import testing


class LayerTest(testing.TestCase):
    def test_compute_output_spec(self):
        # Test that implementing compute_output_shape
        # is enough to make compute_output_spec work.

        # Case: single output
        class TestLayer(layers.Layer):
            def call(self, x):
                assert False  # Should never be called.

            def compute_output_shape(self, input_shape):
                return input_shape

        layer = TestLayer()
        self.assertEqual(
            layer.compute_output_spec(backend.KerasTensor((2, 3))).shape, (2, 3)
        )

        # Case: tuple output
        class TestLayer(layers.Layer):
            def call(self, x):
                assert False  # Should never be called.

            def compute_output_shape(self, input_shape):
                return (input_shape, input_shape)

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertTrue(isinstance(out, tuple))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))

        # Case: list output
        class TestLayer(layers.Layer):
            def call(self, x):
                assert False  # Should never be called.

            def compute_output_shape(self, input_shape):
                return [input_shape, input_shape]

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertTrue(isinstance(out, list))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))

        # Case: dict output
        class TestLayer(layers.Layer):
            def call(self, x):
                assert False  # Should never be called.

            def compute_output_shape(self, input_shape):
                return {"1": input_shape, "2": input_shape}

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(len(out), 2)
        self.assertEqual(out["1"].shape, (2, 3))
        self.assertEqual(out["2"].shape, (2, 3))

        # Case: nested tuple output
        class TestLayer(layers.Layer):
            def call(self, x):
                assert False  # Should never be called.

            def compute_output_shape(self, input_shape):
                return (
                    input_shape,
                    (input_shape, input_shape),
                    (input_shape, input_shape),
                )

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertTrue(isinstance(out, tuple))
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertTrue(isinstance(out[1], tuple))
        self.assertEqual(len(out[1]), 2)
        self.assertEqual(out[1][0].shape, (2, 3))
        self.assertEqual(out[1][1].shape, (2, 3))
        self.assertTrue(isinstance(out[2], tuple))
        self.assertEqual(len(out[2]), 2)
        self.assertEqual(out[2][0].shape, (2, 3))
        self.assertEqual(out[2][1].shape, (2, 3))

        # Case: nested dict output
        class TestLayer(layers.Layer):
            def call(self, x):
                assert False  # Should never be called.

            def compute_output_shape(self, input_shape):
                return {
                    "1": input_shape,
                    "2": {"11": input_shape, "22": input_shape},
                }

        layer = TestLayer()
        out = layer.compute_output_spec(backend.KerasTensor((2, 3)))
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(len(out), 2)
        self.assertEqual(out["1"].shape, (2, 3))
        self.assertTrue(isinstance(out["2"], dict))
        self.assertEqual(len(out["2"]), 2)
        self.assertEqual(out["2"]["11"].shape, (2, 3))
        self.assertEqual(out["2"]["22"].shape, (2, 3))

    def test_positional_arg_error(self):
        class SomeLayer(layers.Layer):
            def call(self, x, bool_arg):
                if bool_arg:
                    return x
                return x + 1

        x = backend.KerasTensor(shape=(2, 3), name="x")
        with self.assertRaisesRegex(
            ValueError, "Only input tensors may be passed as"
        ):
            SomeLayer()(x, True)

        # This works
        SomeLayer()(x, bool_arg=True)

    def test_rng_seed_tracking(self):
        class RNGLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.seed_gen = backend.random.SeedGenerator(seed=1337)

            def call(self, x):
                return backend.random.dropout(x, rate=0.5, seed=self.seed_gen)

        layer = RNGLayer()
        self.assertEqual(layer.variables, [layer.seed_gen.state])
        self.assertAllClose(layer.variables[0], [1337, 0])
        layer(np.ones((3, 4)))
        self.assertAllClose(layer.variables[0], [1337, 1])

        # Test tracking in list attributes.
        class RNGListLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.seed_gens = []
                self.seed_gens.append(backend.random.SeedGenerator(seed=1))
                self.seed_gens.append(backend.random.SeedGenerator(seed=10))

            def call(self, x):
                x = backend.random.dropout(x, rate=0.5, seed=self.seed_gens[0])
                x = backend.random.dropout(x, rate=0.5, seed=self.seed_gens[1])
                return x

        layer = RNGListLayer()
        self.assertEqual(
            layer.variables,
            [layer.seed_gens[0].state, layer.seed_gens[1].state],
        )
        self.assertAllClose(layer.variables[0], [1, 0])
        self.assertAllClose(layer.variables[1], [10, 0])
        layer(np.ones((3, 4)))
        self.assertAllClose(layer.variables[0], [1, 1])
        self.assertAllClose(layer.variables[1], [10, 1])

    def test_layer_tracking(self):
        class NestedLayer(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)
                self.layer_dict = {
                    "dense2": layers.Dense(units),
                }
                self.layer_list = [layers.Dense(units)]
                self.units = units

            def build(self, input_shape):
                self.layer_list.append(layers.Dense(self.units))

            def call(self, x):
                x = self.dense1(x)
                x = self.layer_dict["dense2"](x)
                x = self.layer_list[0](x)
                x = self.layer_list[1](x)
                return x

        class DoubleNestedLayer(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.inner_layer = NestedLayer(units)

            def call(self, x):
                return self.inner_layer(x)

        layer = NestedLayer(3)
        layer.build((1, 3))
        self.assertLen(layer._layers, 4)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.weights, 8)

        layer = DoubleNestedLayer(3)
        self.assertLen(layer._layers, 1)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.inner_layer.weights, 8)
        self.assertLen(layer.weights, 8)

    def test_build_on_call(self):
        class LayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)

            def call(self, x):
                return self.dense1(x)

        layer = LayerWithUnbuiltState(2)
        layer(backend.KerasTensor((3, 4)))
        self.assertLen(layer.weights, 2)

        class KwargsLayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)
                self.dense2 = layers.Dense(units)

            def call(self, x1, x2):
                return self.dense1(x1) + self.dense2(x2)

        layer = KwargsLayerWithUnbuiltState(2)
        layer(backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4)))
        self.assertLen(layer.weights, 4)

        layer = KwargsLayerWithUnbuiltState(2)
        layer(x1=backend.KerasTensor((3, 4)), x2=backend.KerasTensor((3, 4)))
        self.assertLen(layer.weights, 4)

    def test_activity_regularization(self):
        class ActivityRegularizer(layers.Layer):
            def call(self, x):
                return x

        layer = ActivityRegularizer(activity_regularizer="l1")
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 0.01)

        # losses are reset upon call
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 0.01)

        # KerasTensors are no op
        layer = ActivityRegularizer(activity_regularizer="l1")
        layer(layers.Input(batch_shape=(2, 2)))
        self.assertLen(layer.losses, 0)

    def test_add_loss(self):
        class LossLayer(layers.Layer):
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x

        layer = LossLayer()
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 1.0)

        # losses are reset upon call
        layer = LossLayer()
        layer(np.ones((1,)))
        self.assertLen(layer.losses, 1)
        self.assertAllClose(layer.losses[0], 1.0)

        # It works inside a model
        model = models.Sequential([layer])
        model(np.ones((1,)))
        self.assertLen(model.losses, 1)
        self.assertAllClose(model.losses[0], 1.0)

    def test_training_arg_value_resolution(self):
        # Check that even if `training` is not passed
        # to an inner layer, the outer value gets propagated
        # in __call__.
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.dp = layers.Dropout(0.9)

            def call(self, x, training=False):
                return self.dp(x)

        layer = TrainingLayer()
        x = np.ones((4, 4))
        y = layer(x)
        self.assertEqual(ops.min(y), 1)
        y = layer(x, training=True)
        self.assertEqual(ops.min(y), 0)

        # Check that it still works one level deeper.
        class WrappedTrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.dp = TrainingLayer()

            def call(self, x, training=False):
                return self.dp(x)

        layer = WrappedTrainingLayer()
        x = np.ones((4, 4))
        y = layer(x)
        self.assertEqual(ops.min(y), 1)
        y = layer(x, training=True)
        self.assertEqual(ops.min(y), 0)

        # Check that if `training` is passed
        # to an inner layer in call(), the explicitly
        # passed value is what the layer sees.
        class TrainingLayerExplicit(layers.Layer):
            def __init__(self):
                super().__init__()
                self.dp = layers.Dropout(0.9)

            def call(self, x, training=False):
                return self.dp(x, training=True)

        layer = TrainingLayerExplicit()
        x = np.ones((4, 4))
        y = layer(x, training=False)
        self.assertEqual(ops.min(y), 0)

        # Test that layer interruption does not cause
        # the call context to linger
        class BadLayer(layers.Layer):
            def call(self, x, training=False):
                raise RuntimeError("oops!")

        x = np.ones((4, 4))
        layer = BadLayer()
        try:
            # training=True will be recorded
            # in the call context
            layer(x, training=True)
        except RuntimeError:
            pass
        layer = TrainingLayer()
        # But this layer call should not see it
        y = layer(x)
        self.assertEqual(ops.min(y), 1)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Some torch ops not implemented for float16 on CPU.",
    )
    def test_mixed_precision(self):
        print("Run with backend.backend()", backend.backend())
        x = np.ones((4, 4))

        layer = layers.Dense(2, dtype="float16")
        y = layer(x)
        self.assertEqual(layer.compute_dtype, "float16")
        self.assertEqual(layer.variable_dtype, "float16")
        self.assertEqual(backend.standardize_dtype(y.dtype), "float16")

        layer = layers.Dense(2, dtype="mixed_float16")
        y = layer(x)
        self.assertEqual(layer.compute_dtype, "float16")
        self.assertEqual(layer.variable_dtype, "float32")
        self.assertEqual(backend.standardize_dtype(y.dtype), "float16")
        self.assertEqual(layer.kernel.dtype, "float32")

    def test_masking(self):
        class BasicMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                assert mask is not None
                return x

        layer = BasicMaskedLayer()
        x = backend.numpy.ones((4, 4))
        x._keras_mask = backend.numpy.ones((4,))
        layer(x)

        layer(backend.numpy.ones((4, 4)), mask=backend.numpy.ones((4,)))

        class NestedInputMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                assert isinstance(x, list)
                assert len(x) == 2
                assert isinstance(mask, list)
                assert len(mask) == 2
                return x

        layer = NestedInputMaskedLayer()
        x1 = backend.numpy.ones((4, 4))
        x1._keras_mask = backend.numpy.ones((4,))
        x2 = backend.numpy.ones((4, 4))
        x2._keras_mask = backend.numpy.ones((4,))
        layer([x1, x2])

        layer(
            [backend.numpy.ones((4, 4)), backend.numpy.ones((4, 4))],
            mask=[backend.numpy.ones((4,)), backend.numpy.ones((4,))],
        )

        class PositionalInputsMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x1, x2, x1_mask=None, x2_mask=None):
                assert x1_mask is not None
                assert x2_mask is not None
                return x1 + x2

        layer = PositionalInputsMaskedLayer()
        layer(x1, x2)
        layer(x1=x1, x2=x2)

        class PositionalNestedInputsMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x1, x2, x1_mask=None, x2_mask=None):
                assert isinstance(x1, tuple)
                assert x1_mask is not None
                assert x2_mask is not None
                assert isinstance(x1_mask, tuple)
                return x1[0] + x1[1] + x2

        layer = PositionalNestedInputsMaskedLayer()
        x1_1 = backend.numpy.ones((4, 4))
        x1_1._keras_mask = backend.numpy.ones((4,))
        x1_2 = backend.numpy.ones((4, 4))
        x1_2._keras_mask = backend.numpy.ones((4,))
        x2 = backend.numpy.ones((4, 4))
        x2._keras_mask = backend.numpy.ones((4,))
        layer((x1_1, x1_2), x2)
        layer(x1=(x1_1, x1_2), x2=x2)

    def test_stateless_call(self):
        class TestLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self._seed_generator = backend.random.SeedGenerator(1337)
                self.ntw = self.add_weight(
                    shape=(),
                    initializer="zeros",
                    trainable=False,
                )
                self.tw = self.add_weight(
                    shape=(),
                    initializer="zeros",
                    trainable=True,
                )
                self.built = True

            def call(self, x):
                x = backend.convert_to_tensor(x, dtype="float32")
                self.add_loss(ops.sum(x))
                self.ntw.assign(ops.sum(x))
                x = x + backend.random.normal(
                    shape=(), seed=self._seed_generator
                )
                return x + self.tw + self.ntw

        data = np.random.random((3, 4))
        layer = TestLayer()
        out = layer(data)
        layer1 = TestLayer()
        out1 = layer1(data)
        # Check that the layer is in fact deterministic
        self.assertAllClose(out, out1)

        # Test stateless_call correctness
        layer2 = TestLayer()
        trainable_variables = layer2.trainable_variables
        non_trainable_variables = layer2.non_trainable_variables
        out2, non_trainable_variables = layer2.stateless_call(
            trainable_variables, non_trainable_variables, data
        )
        self.assertAllClose(out1, out2)
        self.assertEqual(
            len(layer1.non_trainable_variables), len(non_trainable_variables)
        )
        for ref_v, v in zip(
            layer1.non_trainable_variables, non_trainable_variables
        ):
            self.assertAllClose(ref_v, v)

        # Test with loss collection
        layer3 = TestLayer()
        trainable_variables = layer3.trainable_variables
        non_trainable_variables = layer3.non_trainable_variables
        out3, non_trainable_variables, losses = layer3.stateless_call(
            trainable_variables,
            non_trainable_variables,
            data,
            return_losses=True,
        )
        self.assertAllClose(out1, out3)
        for ref_v, v in zip(
            layer1.non_trainable_variables, non_trainable_variables
        ):
            self.assertAllClose(ref_v, v)
        for ref_loss, loss in zip(layer1.losses, losses):
            self.assertAllClose(ref_loss, loss)

    def test_trainable_setting(self):
        class NonTrainableWeightsLayer(layers.Layer):
            def build(self, _):
                self.w1 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.w2 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=False,
                )
                self.seed = backend.random.SeedGenerator(123)

            def call(self, inputs):
                return inputs

        class NestedNonTrainableWeightsLayer(layers.Layer):
            def build(self, _):
                self.w1 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.w2 = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=False,
                )
                self.nested = NonTrainableWeightsLayer()
                self.nested.build(None)

            def call(self, inputs):
                return inputs

        layer = NestedNonTrainableWeightsLayer()
        layer.build(None)
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.trainable_variables), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_variables), 3)

        layer.trainable = False
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.trainable_variables), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)
        self.assertEqual(len(layer.non_trainable_variables), 5)
        self.assertFalse(layer.w1.trainable)
        self.assertFalse(layer.nested.w1.trainable)

        layer.trainable = True
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.trainable_variables), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_variables), 3)
        self.assertTrue(layer.w1.trainable)
        self.assertTrue(layer.nested.w1.trainable)

        layer = NestedNonTrainableWeightsLayer(trainable=False)
        layer.build(None)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.trainable_variables), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)
        self.assertEqual(len(layer.non_trainable_variables), 5)

        layer.trainable = True
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.trainable_variables), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_variables), 3)

    def test_build_signature_errors(self):
        class NoShapeSuffix(layers.Layer):
            def build(self, foo_shape, bar):
                self.built = True

            def call(self, foo, bar):
                return foo + bar

        class NonMatchingArgument(layers.Layer):
            def build(self, foo_shape, baz_shape):
                self.built = True

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        class MatchingArguments(layers.Layer):
            def build(self, bar_shape, foo_shape):
                self.foo_shape = foo_shape
                self.bar_shape = bar_shape
                self.built = True

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        class SubsetArguments(layers.Layer):
            def build(self, baz_shape, foo_shape):
                self.foo_shape = foo_shape
                self.baz_shape = baz_shape
                self.built = True

            def call(self, foo, bar=None, baz=None):
                return foo[:, 0] + bar[:, 0] + baz[:, 0]

        class SingleArgument(layers.Layer):
            def build(self, anything_whatsoever):
                self.foo_shape = anything_whatsoever
                self.built = True

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        foo = backend.numpy.ones((4, 1))
        bar = backend.numpy.ones((4, 2))
        baz = backend.numpy.ones((4, 3))
        with self.assertRaisesRegex(
            ValueError,
            r"argument `bar`, which does not end in `_shape`",
        ):
            layer = NoShapeSuffix()
            layer(foo, bar)

        with self.assertRaisesRegex(
            ValueError,
            r"`baz_shape`, but `call\(\)` does not have argument `baz`",
        ):
            layer = NonMatchingArgument()
            layer(foo, bar)

        # Align by name when build and call arguments match.
        layer = MatchingArguments()
        layer(foo, bar)
        self.assertEqual(layer.foo_shape, foo.shape)
        self.assertEqual(layer.bar_shape, bar.shape)

        # Align by name when build supports a subset of call arguments.
        layer = SubsetArguments()
        layer(foo, bar, baz)
        self.assertEqual(layer.foo_shape, foo.shape)
        self.assertEqual(layer.baz_shape, baz.shape)

        # When build has only one argument, match the first call argument.
        layer = SingleArgument()
        layer(foo, bar)
        self.assertEqual(layer.foo_shape, foo.shape)

    def test_training_arg_not_specified(self):
        class NoTrainingSpecified(layers.Layer):
            def __init__(self):
                super().__init__()

            def build(self, input_shape):
                self.activation = layers.Activation("linear")

            def call(self, inputs):
                return self.activation(inputs)

        layer = NoTrainingSpecified()
        inputs = ops.random.uniform(shape=(1, 100, 100, 3))
        layer(inputs, training=True)

    def test_tracker_locking(self):
        class BadLayer(layers.Layer):
            def call(self, x):
                self.w = self.add_weight(initializer="zeros", shape=())
                return x

        layer = BadLayer()
        with self.assertRaisesRegex(
            ValueError,
            "cannot add new elements of state",
        ):
            layer(np.random.random((3, 2)))

    def test_init_after_state_tracking(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                self.some_attr = True
                self.w = backend.Variable(np.random.random((2,)))
                super().__init__()

        layer = MyLayer()
        self.assertEqual(len(layer.weights), 1)
