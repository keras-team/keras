import pickle

import numpy as np
import pytest

from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import global_state


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
        self.assertIsInstance(out, tuple)
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
        self.assertIsInstance(out, list)
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
        self.assertIsInstance(out, dict)
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
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertIsInstance(out[1], tuple)
        self.assertEqual(len(out[1]), 2)
        self.assertEqual(out[1][0].shape, (2, 3))
        self.assertEqual(out[1][1].shape, (2, 3))
        self.assertIsInstance(out[2], tuple)
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
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["1"].shape, (2, 3))
        self.assertIsInstance(out["2"], dict)
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
                return x * backend.random.normal(x.shape, seed=self.seed_gen)

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
                x = x * backend.random.normal(x.shape, seed=self.seed_gens[0])
                x = x * backend.random.normal(x.shape, seed=self.seed_gens[1])
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
        class LayerWithDenseLayers(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense1 = layers.Dense(units)
                self.layer_dict = {
                    "dense2": layers.Dense(units),
                }
                self.layer_list = [layers.Dense(units)]
                self.units = units
                self.seed_generator = backend.random.SeedGenerator(seed=1)

            def build(self, input_shape):
                self.layer_list.append(layers.Dense(self.units))

            def call(self, x):
                x = self.dense1(x)
                x = self.layer_dict["dense2"](x)
                x = self.layer_list[0](x)
                x = self.layer_list[1](x)
                return x

        class ParentLayer(layers.Layer):
            def __init__(self, inner_layer):
                super().__init__()
                self.inner_layer = inner_layer

            def call(self, x):
                return self.inner_layer(x)

        layer = LayerWithDenseLayers(3)
        layer.build((1, 3))
        self.assertLen(layer._layers, 4)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.variables, 9)
        self.assertLen(layer.weights, 8)

        layer = ParentLayer(LayerWithDenseLayers(3))
        self.assertLen(layer._layers, 1)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.variables, 9)
        self.assertLen(layer.weights, 8)

        layer = ParentLayer(ParentLayer(LayerWithDenseLayers(3)))
        self.assertLen(layer._layers, 1)
        layer(np.zeros((1, 3)))
        self.assertLen(layer.variables, 9)
        self.assertLen(layer.weights, 8)

    def test_metric_tracking(self):
        class LayerWithMetric(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)
                self.metric = metrics.MeanSquaredError(name="my_metric")

            def build(self, input_shape):
                self.dense.build(input_shape)

            def call(self, x):
                return self.dense(x)

        class ParentLayerWithMetric(layers.Layer):
            def __init__(self, inner_layer):
                super().__init__()
                self.inner_layer = inner_layer
                self.metric = metrics.MeanSquaredError(name="my_metric")

            def build(self, input_shape):
                self.inner_layer.build(input_shape)

            def call(self, x):
                return self.inner_layer(x)

        layer = LayerWithMetric(3)
        layer.build((1, 3))

        self.assertLen(layer.metrics, 1)
        self.assertLen(layer.metrics_variables, 2)
        self.assertLen(layer.trainable_variables, 2)
        self.assertLen(layer.non_trainable_variables, 0)

        layer = ParentLayerWithMetric(LayerWithMetric(3))
        layer.build((1, 3))

        self.assertLen(layer.metrics, 2)
        self.assertLen(layer.metrics_variables, 4)
        self.assertLen(layer.trainable_variables, 2)
        self.assertLen(layer.non_trainable_variables, 0)

        layer = ParentLayerWithMetric(ParentLayerWithMetric(LayerWithMetric(3)))
        layer.build((1, 3))

        self.assertLen(layer.metrics, 3)
        self.assertLen(layer.metrics_variables, 6)
        self.assertLen(layer.trainable_variables, 2)
        self.assertLen(layer.non_trainable_variables, 0)

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

    @pytest.mark.requires_trainable_backend
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

        # It works recursively in nested models
        model = models.Sequential([model])
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
        x = np.ones((4, 4))

        layer = layers.Dense(2, dtype="float16")
        y = layer(x)
        self.assertEqual(layer.compute_dtype, "float16")
        self.assertEqual(layer.variable_dtype, "float16")
        self.assertDType(y, "float16")

        layer = layers.Dense(2, dtype="mixed_float16")
        y = layer(x)
        self.assertEqual(layer.compute_dtype, "float16")
        self.assertEqual(layer.variable_dtype, "float32")
        self.assertDType(y, "float16")
        self.assertEqual(layer.kernel.dtype, "float32")

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Some torch ops not implemented for float16 on CPU.",
    )
    def test_autocast(self):
        assertDType = self.assertDType

        # A layer with a int dtype (some preprocessing layers do this).
        class InnerLayerOne(layers.Layer):
            def __init__(self):
                super().__init__(dtype="int")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                    dtype="float32",
                )
                self.built = True

            def call(self, x):
                # Should not autocast.
                assertDType(self.v, "float32")
                return ops.cast(x, "float32") + self.v

        # A layer that is explicitly full precision.
        class InnerLayerTwo(layers.Layer):
            def __init__(self):
                super().__init__(dtype="float32")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.built = True

            def call(self, x):
                # Should not autocast.
                assertDType(self.v, "float32")
                return x + self.v

        # A layer that is explicitly mixed precision but with autocast=False
        # weight.
        class InnerLayerThree(layers.Layer):
            def __init__(self):
                super().__init__(dtype="mixed_float16")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                    autocast=False,
                )
                self.built = True

            def call(self, x):
                # Should not autocast `self.v`.
                assertDType(self.v, "float32")
                return ops.add(x, self.v)

        # A layer that is explicitly mixed precision with inner layers.
        class MixedPrecisionLayer(layers.Layer):
            def __init__(self):
                super().__init__(dtype="mixed_float16")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self.inner_one = InnerLayerOne()
                self.inner_two = InnerLayerTwo()
                self.inner_three = InnerLayerThree()
                self.built = True

            def call(self, x):
                # Should autocast.
                assertDType(self.v, "float16")
                return self.inner_three(
                    self.inner_two(self.inner_one(x + self.v))
                )

        layer = MixedPrecisionLayer()
        y = layer(np.array(0.0))
        self.assertEqual(y, 4.0)

    def test_autocast_with_np_array(self):
        assertDType = self.assertDType

        class CustomLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, x):
                # Here are the assertions.
                assertDType(x[0], "float32")  # Cast to compute_dtype
                assertDType(x[1], "int32")  # Untouched

        x = [np.zeros(1, dtype="float64"), np.zeros(1, dtype="int32")]
        CustomLayer()(x)

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
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
                    regularizer="l1",
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
        self.assertLen(losses, 2)
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

    def test_add_weight_defaults(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight()
                self.w2 = self.add_weight(dtype="int32", trainable=False)
                self.w3 = self.add_weight(dtype="bool", trainable=False)
                self.w4 = self.add_weight(
                    dtype="int32", shape=(2, 2), trainable=False
                )
                self.w5 = self.add_weight(initializer="ones", shape=(2, 2))

        layer = MyLayer()
        self.assertEqual(layer.w1.shape, ())
        self.assertEqual(layer.w1.dtype, "float32")

        self.assertEqual(layer.w2.shape, ())
        self.assertEqual(layer.w2.dtype, "int32")
        self.assertAllClose(backend.convert_to_numpy(layer.w2), 0)

        self.assertEqual(layer.w3.shape, ())
        self.assertEqual(layer.w3.dtype, "bool")
        self.assertAllClose(backend.convert_to_numpy(layer.w3), False)

        self.assertEqual(layer.w4.shape, (2, 2))
        self.assertEqual(layer.w4.dtype, "int32")
        self.assertAllClose(
            backend.convert_to_numpy(layer.w4), np.zeros((2, 2))
        )

        self.assertEqual(layer.w5.shape, (2, 2))
        self.assertEqual(layer.w5.dtype, "float32")
        self.assertAllClose(backend.convert_to_numpy(layer.w5), np.ones((2, 2)))

    def test_remove_weight(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight()

            def custom_remove_w(self):
                self.w = self._untrack_variable(self.w)

            def custom_change_dtype(self):
                self.w = self._untrack_variable(self.w)
                self.w = self.add_weight(
                    initializer="zeros", dtype="int8", trainable=False
                )

        layer = MyLayer()
        self.assertEqual(len(layer.weights), 1)
        layer.custom_remove_w()
        self.assertEqual(len(layer.weights), 0)
        self.assertEqual(layer.w, None)

        layer = MyLayer()
        self.assertEqual(layer.w.dtype, "float32")
        self.assertEqual(layer.w.trainable, True)
        layer.custom_change_dtype()
        self.assertEqual(layer.w.dtype, "int8")
        self.assertEqual(layer.w.trainable, False)

    def test_trainable_init_arg(self):
        inputs = layers.Input(shape=(1,))
        layer = layers.Dense(2, trainable=False)
        outputs = layer(inputs)
        model = models.Model(inputs, outputs)

        self.assertFalse(layer.trainable)
        self.assertLen(layer._trainable_variables, 2)
        self.assertLen(layer._non_trainable_variables, 0)
        self.assertLen(layer.trainable_weights, 0)
        self.assertLen(model.trainable_weights, 0)
        self.assertLen(model.non_trainable_weights, 2)

        layer.trainable = True
        self.assertTrue(layer.trainable)
        self.assertLen(layer._trainable_variables, 2)
        self.assertLen(layer._non_trainable_variables, 0)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(model.trainable_weights, 2)
        self.assertLen(model.non_trainable_weights, 0)

    def test_dtype_policy_setter(self):
        layer = layers.Dense(2)
        # Set by string
        layer.dtype_policy = "mixed_bfloat16"
        self.assertEqual(layer.dtype_policy.name, "mixed_bfloat16")
        self.assertEqual(layer.dtype_policy.compute_dtype, "bfloat16")
        self.assertEqual(layer.dtype_policy.variable_dtype, "float32")
        # Set by DTypePolicy
        layer.dtype_policy = dtype_policies.DTypePolicy("mixed_float16")
        self.assertEqual(layer.dtype_policy.name, "mixed_float16")
        self.assertEqual(layer.dtype_policy.compute_dtype, "float16")
        self.assertEqual(layer.dtype_policy.variable_dtype, "float32")
        # Set with DTypePolicyMap
        dtype_policy_map = dtype_policies.DTypePolicyMap()
        layer = layers.Dense(2, dtype=dtype_policy_map)
        layer.build([None, 1])
        layer.dtype_policy = "mixed_bfloat16"
        self.assertIsInstance(
            layer._dtype_policy, dtype_policies.DTypePolicyMap
        )
        self.assertEqual(
            layer._dtype_policy[layer.path],
            dtype_policies.DTypePolicy("mixed_bfloat16"),
        )

    def test_pickle_layer(self):
        layer = layers.Dense(2)
        reloaded = pickle.loads(pickle.dumps(layer))
        self.assertEqual(layer.get_config(), reloaded.get_config())

    def test_serialize_dtype(self):
        assertIsNone = self.assertIsNone
        assertIsNotNone = self.assertIsNotNone

        class AssertionDense(layers.Dense):
            def __init__(self, *args, **kwargs):
                dtype = kwargs["dtype"]
                if isinstance(dtype, str):
                    # `dtype` is a plain string, it should be the `name` from a
                    # `DTypePolicy`
                    dtype = dtype_policies.get(dtype)
                    assertIsNone(dtype.quantization_mode)
                else:
                    # `dtype` is a DTypePolicy instance, it should be an
                    # instance of `QuantizedDTypePolicy`
                    assertIsNotNone(dtype.quantization_mode)
                super().__init__(*args, **kwargs)

        # Test floating dtype serialization
        layer = layers.Dense(2, dtype="bfloat16")
        config = layer.get_config()
        self.assertIn("dtype", config)
        self.assertEqual(
            config["dtype"],
            dtype_policies.serialize(dtype_policies.DTypePolicy("bfloat16")),
        )
        AssertionDense.from_config(config)  # Assertion inside

        # Test quantized dtype serialization
        layer = layers.Dense(2, dtype="int8_from_bfloat16")
        config = layer.get_config()
        self.assertIn("dtype", config)
        self.assertEqual(
            config["dtype"],
            dtype_policies.serialize(dtype_policies.get("int8_from_bfloat16")),
        )
        AssertionDense.from_config(config)  # Assertion inside

    def test_serialize_activity_regularizer(self):
        layer = layers.Dense(2, activity_regularizer="l2")
        config = layer.get_config()
        self.assertIn("activity_regularizer", config)
        new_layer = layers.Dense.from_config(config)
        self.assertEqual(
            new_layer.activity_regularizer.__class__.__name__, "L2"
        )

        layer = layers.Dense(2)
        config = layer.get_config()
        self.assertNotIn("activity_regularizer", config)

    def test_custom_layer_add_weight_in_init_name(self):
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = InnerLayer()

        class InnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )
                self.inner = InnerInnerLayer()

        class InnerInnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )

        layer = TrainingLayer()
        layer.build(None)
        self.assertEqual(len(layer.variables), 2)
        variable_paths = set(v.path for v in layer.variables)
        self.assertTrue("inner_layer/inner" in variable_paths)
        self.assertTrue("inner_inner_layer/inner" in variable_paths)
        if backend.backend() == "torch":
            parameter_names = set(
                param_name.replace("torch_params.", "")
                for param_name, _ in layer.named_parameters()
            )
            self.assertSetEqual(variable_paths, parameter_names)

    def test_custom_layer_add_weight_in_build_name(self):
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = InnerLayer()

            def call(self, input):
                return self.inner(input)

        class InnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = InnerInnerLayer()

            def build(self, _):
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )

            def call(self, input):
                return self.var + self.inner(input)

        class InnerInnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()

            def build(self, _):
                self.var = self.add_weight(
                    shape=(1,),
                    name="inner",
                )

            def call(self, input):
                return self.var + input

        layer = TrainingLayer()
        output = layer(
            backend.KerasTensor(
                (4, 1),
            )
        )
        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(len(layer.variables), 2)
        variable_paths = set(v.path for v in layer.variables)
        self.assertTrue("training_layer/inner_layer/inner" in variable_paths)
        self.assertTrue(
            "training_layer/inner_layer/inner_inner_layer/inner"
            in variable_paths
        )
        if backend.backend() == "torch":
            parameter_names = set(
                param_name.replace("torch_params.", "")
                for param_name, _ in layer.named_parameters()
            )
            self.assertSetEqual(variable_paths, parameter_names)

    def test_layer_variable_tracking_correct(self):
        class TrainingLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.post_build_modify_layer = PostBuildModifyLayer()

            def call(self, input):
                return self.post_build_modify_layer(input)

        class PostBuildModifyLayer(layers.Layer):

            def call(self, input):
                return self.var + input

            def build(self, _):
                self.var = self.add_weight(
                    shape=(2,),
                    name="var",
                )

            def post_build_add(self):
                self._tracker.unlock()
                self.additional_var = self.add_weight(
                    shape=(2,),
                    name="var2",
                )
                self._tracker.lock()

            def post_build_remove(self):
                self._tracker.unlock()
                self._untrack_variable(self.var)
                del self.var
                self._tracker.lock()

        layer = TrainingLayer()
        output = layer(backend.KerasTensor((4, 2)))

        self.assertEqual(output.shape, (4, 2))
        self.assertEqual(len(layer.variables), 1)
        self.assertEqual(
            layer.variables[0].path,
            "training_layer/post_build_modify_layer/var",
        )
        if backend.backend() == "torch":
            parameter_names = [pname for pname, _ in layer.named_parameters()]
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "torch_params.training_layer/post_build_modify_layer/var",
            )

        layer.post_build_modify_layer.post_build_add()
        self.assertEqual(len(layer.variables), 2)
        self.assertEqual(
            layer.variables[0].path,
            "training_layer/post_build_modify_layer/var",
        )
        self.assertEqual(
            layer.variables[1].path,
            "training_layer/post_build_modify_layer/var2",
        )
        if backend.backend() == "torch":
            # TODO (haohuanw, fchollet): Needs further discussion on how to
            # properly manage torch params. Post build modification cannot
            # propagate to parent torch params.
            parameter_names = [pname for pname, _ in layer.named_parameters()]
            # Below check should have 2 parameters instead of 1.
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "torch_params.training_layer/post_build_modify_layer/var",
            )

            parameter_names = [
                pname
                for pname, _ in layer.post_build_modify_layer.named_parameters()
            ]
            self.assertEqual(len(parameter_names), 2)
            self.assertEqual(
                parameter_names[0],
                "torch_params.training_layer/post_build_modify_layer/var",
            )
            self.assertEqual(
                parameter_names[1],
                "torch_params.training_layer/post_build_modify_layer/var2",
            )

        layer.post_build_modify_layer.post_build_remove()
        self.assertEqual(len(layer.variables), 1)
        self.assertEqual(
            layer.variables[0].path,
            "training_layer/post_build_modify_layer/var2",
        )
        if backend.backend() == "torch":
            # TODO (haohuanw, fchollet): Needs further discussion on how to
            # properly manage torch params. Post build modification cannot
            # propagate to parent torch params.
            parameter_names = [pname for pname, _ in layer.named_parameters()]
            # Below check should have 1 parameters instead of 2, torch_params
            # in parent layer is wrong.
            self.assertEqual(len(parameter_names), 2)
            self.assertEqual(
                parameter_names[0],
                "post_build_modify_layer.torch_params.training_layer/"
                "post_build_modify_layer/var2",
            )
            self.assertEqual(
                parameter_names[1],
                "torch_params.training_layer/post_build_modify_layer/var",
            )

            parameter_names = [
                pname
                for pname, _ in layer.post_build_modify_layer.named_parameters()
            ]
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "torch_params.training_layer/post_build_modify_layer/var2",
            )

    @pytest.mark.skipif(backend.backend() != "torch", reason="Torch only test.")
    def test_torch_params_create_deterministic(self):
        class MyLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w1 = self.add_weight()
                self.w2 = self.add_weight(dtype="int32", trainable=False)
                self.w3 = self.add_weight(dtype="bool", trainable=False)
                self.w4 = self.add_weight(
                    dtype="int32", shape=(2, 2), trainable=False
                )
                self.w5 = self.add_weight(initializer="ones", shape=(2, 2))

        layer1 = MyLayer()
        layer1.build(None)
        layer1_names = list(pname for pname, _ in layer1.named_parameters())
        global_state.clear_session()
        layer2 = MyLayer()
        layer2.build(None)
        layer2_names = list(pname for pname, _ in layer2.named_parameters())
        self.assertListEqual(layer1_names, layer2_names)
