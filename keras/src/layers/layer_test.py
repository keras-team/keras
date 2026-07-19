import inspect
import pickle
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import Input
from keras.src import backend
from keras.src import dtype_policies
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.backend.common import global_state
from keras.src.backend.common import tensor_attributes
from keras.src.backend.common.remat import RematScope
from keras.src.layers.layer import CallSpec
from keras.src.layers.layer import _bind_call_arguments
from keras.src.layers.layer import any_symbolic_tensors
from keras.src.models import Model


def _build_f1_functional_model(input_dim=8):
    """Small Dense + LayerNormalization + Embedding + Dropout functional
    model matching the shape of the F1 CallSpec-fast-path report's workload:
    a mix of layers that do and do not accept `training`, called the way
    `fit()`/`evaluate()`/`predict()` do (a single positional tensor input
    plus `training=<bool>` passed explicitly by the trainer)."""
    inputs = Input(shape=(input_dim,))
    x = layers.Dense(input_dim)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(input_dim)(x)
    return Model(inputs, x)


# Representative `call()`-style signatures for
# `test_bind_call_arguments_matches_signature_bind`, which checks
# `_bind_call_arguments` against `inspect.Signature.bind` on these.
def _binder_single(inputs):
    pass


def _binder_training_mask(inputs, training=None, mask=None):
    pass


def _binder_mha(
    query,
    value,
    key=None,
    query_mask=None,
    value_mask=None,
    key_mask=None,
    attention_mask=None,
    return_attention_scores=False,
    training=None,
    use_causal_mask=False,
):
    pass


def _binder_var_args_kwargs(*args, **kwargs):
    pass


def _binder_mixed_varargs(inputs, *args, training=None, **kwargs):
    pass


def _binder_kw_only(a, b=2, *, c, d=4):
    pass


def _binder_mask(inputs, mask=None):
    pass


def _binder_pos_only_var_kw(x, /, **kwargs):
    pass


def _binder_pos_only(x, /):
    pass


class MockRemat:
    """Mock remat by returning a wrapper Mock calling the original function"""

    def __init__(self):
        self.rematted_functions = {}

    def __call__(self, func):
        if func in self.rematted_functions:
            return self.rematted_functions[func]

        wrapped_func = mock.Mock(wraps=func)
        self.rematted_functions[func] = wrapped_func
        return wrapped_func


class LayerTest(testing.TestCase):
    def test_compute_output_spec(self):
        # Test that implementing compute_output_shape
        # is enough to make compute_output_spec work.

        # Case: single output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

            def compute_output_shape(self, input_shape):
                return input_shape

        layer = TestLayer()
        self.assertEqual(
            layer.compute_output_spec(backend.KerasTensor((2, 3))).shape, (2, 3)
        )

        # Case: tuple output
        class TestLayer(layers.Layer):
            def call(self, x):
                raise RuntimeError("Should never be called.")

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
                raise RuntimeError("Should never be called.")

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
                raise RuntimeError("Should never be called.")

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
                raise RuntimeError("Should never be called.")

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
                raise RuntimeError("Should never be called.")

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

    @parameterized.named_parameters(
        ("call", "call", None),
        ("compute_output_shape", "compute_output_shape", None),
        (
            "quantized_build",
            "quantized_build",
            {"input_shape": None, "mode": None},
        ),
        ("quantize", "quantize", {"mode": "int8"}),
        ("_int8_call", "_int8_call", None),
        ("_float8_call", "_float8_call", None),
    )
    def test_not_implemented_error(self, method, args):
        layer = layers.Layer()
        layer.built = True

        with self.assertRaisesRegex(
            NotImplementedError,
            f"does not have a `{method}` method implemented.",
        ):
            if isinstance(args, dict):
                getattr(layer, method)(**args)
            else:
                getattr(layer, method)(args)

    def test_layer_with_remat(self):
        """Test rematerialization on a simple layer."""
        # Create a mock to track calls to remat
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class SomeLayer(layers.Layer):
                def call(self, x):
                    return x + 1

            input_tensor = backend.random.uniform((2, 4))
            layer = SomeLayer()
            # Case 1: Without rematerialization
            output_no_remat = layer(input_tensor)

            # Case 2: With rematerialization
            with RematScope(mode="full"):
                layer = SomeLayer()
                output_with_remat = layer(input_tensor)

            # Assert outputs are the same
            self.assertAllClose(output_no_remat, output_with_remat)

        # Ensure remat was applied in the second case
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_rematerialized_call_none(self):
        layer = layers.Dense(4)
        layer.rematerialized_call(layer.call, ops.ones((2, 3)))

    def test_quantized_layer_with_remat(self):
        """Test rematerialization on a quantized layer."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):
            input_tensor = backend.random.uniform((2, 4))

            # Case 2: With rematerialization
            with RematScope(mode="full"):
                layer = layers.Dense(3)
                layer.build((2, 4))
                layer.quantize("float8")
                layer(input_tensor)

        # Ensure remat was applied
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_gptq_quantization_by_setting_dtype(self):
        """Tests error being raised when dtype is set to GPTQ."""
        with self.assertRaisesRegex(
            ValueError,
            "Implicitly enabling GPTQ quantization.*is not supported",
        ):
            layer = layers.Dense(3)
            layer.build((2, 4))
            layer.dtype_policy = "gptq/4/-1_from_float32"

    @pytest.mark.skipif(
        backend.backend() in ("openvino", "numpy"),
        reason="remat not supported on OpenVino and Numpy",
    )
    def test_functional_model_with_remat(self):
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):
            # Define model inputs
            inputs = Input(shape=(32, 32, 3))

            # just one layer in remat scope
            with RematScope(mode="activations"):
                layer = layers.Dense(64, activation="relu")
                output = layer(layers.Flatten()(inputs))

            # Build the functional model
            model = Model(inputs=inputs, outputs=output)

            # Compile the model
            model.compile(optimizer="adam", loss="mse")

            # Generate dummy data for testing
            x_train = np.random.random((10, 32, 32, 3)).astype(np.float32)
            y_train = np.random.random((10, 64)).astype(np.float32)

            # Run training to ensure `RematScope` is applied correctly
            model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)

        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_remat_wrapper_list_of_layers(self):
        """Test rematerialization using list_of_layers mode."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class TestLayer(layers.Layer):
                def call(self, x):
                    return x + 1

            class OtherLayer(layers.Layer):
                def call(self, x):
                    return x * 2

            remat_layers = ["test_layer"]
            input_tensor = backend.random.uniform((4, 4))

            with RematScope(mode="list_of_layers", layer_names=remat_layers):
                test_layer = TestLayer(name="test_layer")
                other_layer = OtherLayer(name="other_layer")
                output_test = test_layer(input_tensor)
                output_other = other_layer(input_tensor)

            self.assertAllClose(output_test, input_tensor + 1)
            self.assertAllClose(output_other, input_tensor * 2)

        # Ensure remat was applied to the correct layer
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_remat_larger_than_mode(self):
        """Test rematerialization using larger_than mode."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class TestLayer(layers.Layer):
                def compute_output_shape(self, input_shape):
                    return input_shape

                def call(self, x):
                    return x + 1

            input_tensor = backend.random.uniform((100, 100))  # Large tensor

            with RematScope(mode="larger_than", output_size_threshold=5000):
                layer = TestLayer()
                output = layer(input_tensor)

            self.assertAllClose(output, input_tensor + 1)

        # Ensure remat was applied
        self.assertLen(mock_remat.rematted_functions, 1)
        next(iter(mock_remat.rematted_functions.values())).assert_called()

    def test_remat_larger_than_mode_high_threshold(self):
        """Test rematerialization using larger_than mode."""
        mock_remat = MockRemat()
        with mock.patch(
            "keras.src.backend.common.remat.remat", wraps=mock_remat
        ):

            class TestLayer(layers.Layer):
                def compute_output_shape(self, input_shape):
                    return input_shape

                def call(self, x):
                    return x + 1

            input_tensor = backend.random.uniform((100, 100))  # Large tensor

            with RematScope(mode="larger_than", output_size_threshold=50000):
                layer = TestLayer()
                output = layer(input_tensor)

            self.assertAllClose(output, input_tensor + 1)

        # Ensure remat was not applied
        self.assertLen(mock_remat.rematted_functions, 0)

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

        class DictLayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)

            def call(self, xs):
                result = self.dense(xs["x1"])
                if xs.get("x2", None) is not None:
                    result += self.dense(xs["x2"])
                return result

        layer = DictLayerWithUnbuiltState(2)
        layer(
            {
                "x1": backend.KerasTensor((3, 4)),
                "x2": backend.KerasTensor((3, 4)),
            }
        )
        self.assertLen(layer.weights, 2)

        layer = DictLayerWithUnbuiltState(2)
        layer({"x1": backend.KerasTensor((3, 4)), "x2": None})
        self.assertLen(layer.weights, 2)

        class ListLayerWithUnbuiltState(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)

            def call(self, xs):
                result = self.dense(xs[0])
                if xs[1] is not None:
                    result += self.dense(xs[1])
                return result

        layer = ListLayerWithUnbuiltState(2)
        layer([backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))])
        self.assertLen(layer.weights, 2)

        layer = ListLayerWithUnbuiltState(2)
        layer([backend.KerasTensor((3, 4)), None])
        self.assertLen(layer.weights, 2)

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

    @parameterized.named_parameters(
        ("batch_size_0", 0),
        ("batch_size_1", 1),
        ("batch_size_5", 5),
        ("batch_size_10", 10),
    )
    def test_activity_regularization_batch_normalization(self, batch_size):
        class SimpleLayer(layers.Layer):
            def call(self, x):
                return x

        layer = SimpleLayer(activity_regularizer="l2")
        layer(ops.ones((batch_size, 5)) * 2.0)
        self.assertLen(layer.losses, 1)
        expected_loss = 0.0 if batch_size == 0 else 0.2
        self.assertAllClose(layer.losses[0], expected_loss)

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

    def test_signature_default_training_does_not_leak(self):
        """A `training=True` call() signature default (as on `Resizing`/
        `CenterCrop`) stays local to that layer: it is not propagated
        through the shared call context to sibling or downstream layers.
        """

        class DefaultsTrainingTrue(layers.Layer):
            def call(self, x, training=True):  # mirrors `Resizing.call`
                return x

        x = np.ones((4, 4))

        # Functional graph: the downstream layer defaults training=None so
        # any value propagated from the upstream layer is directly visible.
        seen = {}

        class RecordTraining(layers.Layer):
            def call(self, x, training=None):
                seen["training"] = training
                return x

        inp = Input((4,))
        out = RecordTraining()(DefaultsTrainingTrue()(inp))
        model = Model(inp, out)
        model(x)
        self.assertIsNone(seen["training"])
        model(x, training=True)
        self.assertTrue(seen["training"])

        # Imperative call: same invariant inside another layer's call().
        class Wrapper(layers.Layer):
            def __init__(self):
                super().__init__()
                self.pre = DefaultsTrainingTrue()
                self.dp = layers.Dropout(0.9)

            def call(self, x):
                return self.dp(self.pre(x))

        layer = Wrapper()
        self.assertEqual(ops.min(layer(x)), 1)
        self.assertEqual(ops.min(layer(x, training=True)), 0)

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
                self._build_at_init()

            def call(self, x):
                # Should not autocast.
                assertDType(self.v, "float32")
                return ops.add(ops.cast(x, "float32"), self.v)

        # A layer that is explicitly full precision.
        class InnerLayerTwo(layers.Layer):
            def __init__(self):
                super().__init__(dtype="float32")
                self.v = self.add_weight(
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                self._build_at_init()

            def call(self, x):
                # Should not autocast.
                assertDType(self.v, "float32")
                return ops.add(x, self.v)

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
                self._build_at_init()

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
                self._build_at_init()

            def call(self, x):
                # Should autocast.
                assertDType(self.v, "float16")
                return self.inner_three(
                    self.inner_two(self.inner_one(ops.add(x, self.v)))
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

    def test_keras_mask_with_autocast(self):
        test_obj = self

        class CustomLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsNotNone(mask)
                test_obj.assertDType(x, "float16")
                return x

        x = ops.zeros((1, 2), dtype="float32")
        mask = ops.array([True, False])
        backend.set_keras_mask(x, mask)
        y = CustomLayer(dtype="float16")(x)
        self.assertAllEqual(
            backend.get_keras_mask(y),
            mask,
            msg="Masking is not propagated by Autocast",
        )

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="compute_output_spec not supported with numpy",
    )
    def test_end_to_end_masking(self):
        # Check that masking survives compilation
        model = models.Sequential(
            [
                layers.Embedding(
                    2, 2, mask_zero=True, embeddings_initializer="ones"
                ),
            ]
        )
        model.compile(loss="mse")
        targets = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]])
        loss = model.evaluate(np.array([[1, 0, 0, 1]]), targets, verbose=0)
        self.assertAllClose(loss, 0.0)

    def test_masking(self):
        test_obj = self

        class BasicMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsNotNone(mask)
                return x

        layer = BasicMaskedLayer()
        x = backend.numpy.ones((4, 4))
        mask = backend.numpy.ones((4,))
        backend.set_keras_mask(x, mask)
        layer(x)

        layer(backend.numpy.ones((4, 4)), mask=backend.numpy.ones((4,)))

        class NestedInputMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsInstance(x, list)
                test_obj.assertLen(x, 2)
                test_obj.assertIsInstance(mask, list)
                test_obj.assertLen(mask, 2)
                return x

        layer = NestedInputMaskedLayer()
        x1 = backend.numpy.ones((4, 4))
        mask1 = backend.numpy.ones((4,))
        backend.set_keras_mask(x1, mask1)
        x2 = backend.numpy.ones((4, 4))
        mask2 = backend.numpy.ones((4,))
        backend.set_keras_mask(x2, mask2)
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
                test_obj.assertIsNotNone(x1_mask)
                test_obj.assertIsNotNone(x2_mask)
                return x1 + x2

        layer = PositionalInputsMaskedLayer()
        layer(x1, x2)
        layer(x1=x1, x2=x2)

        class PositionalNestedInputsMaskedLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x1, x2, x1_mask=None, x2_mask=None):
                test_obj.assertIsInstance(x1, tuple)
                test_obj.assertIsNotNone(x1_mask)
                test_obj.assertIsNotNone(x2_mask)
                test_obj.assertIsInstance(x1_mask, tuple)
                return x1[0] + x1[1] + x2

        layer = PositionalNestedInputsMaskedLayer()
        x1_1 = backend.numpy.ones((4, 4))
        mask1 = backend.numpy.ones((4,))
        backend.set_keras_mask(x1_1, mask1)
        x1_2 = backend.numpy.ones((4, 4))
        mask2 = backend.numpy.ones((4,))
        backend.set_keras_mask(x1_2, mask2)
        x2 = backend.numpy.ones((4, 4))
        mask2 = backend.numpy.ones((4,))
        backend.set_keras_mask(x2, mask2)
        layer((x1_1, x1_2), x2)
        layer(x1=(x1_1, x1_2), x2=x2)

        class MaskUnsetDuringCallLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                test_obj.assertIsNotNone(mask)
                backend.set_keras_mask(x, None)  # Unset mask
                return x

        layer = MaskUnsetDuringCallLayer()
        x = backend.numpy.ones((4, 4))
        mask = backend.numpy.ones((4,))
        backend.set_keras_mask(x, mask)
        y = layer(x)
        self.assertAllClose(backend.get_keras_mask(y), mask)

    def test_masking_with_explicit_kwarg_propagation(self):
        """This test validates that an explicit `mask` kwarg is correctly
        used to compute the output mask.
        """

        class PassthroughMaskLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.supports_masking = True

            def call(self, x, mask=None):
                # The layer itself can use the mask.
                self.used_mask = mask is not None
                return x

        layer = PassthroughMaskLayer()
        # Create an input tensor WITHOUT an attached mask.
        x = backend.numpy.ones((4, 4))
        self.assertIsNone(backend.get_keras_mask(x))

        # Create a mask to be passed explicitly.
        explicit_mask = backend.numpy.array([True, True, False, False])

        # Call the layer, passing the mask as a keyword argument.
        y = layer(x, mask=explicit_mask)

        # Assert that the layer's internal call received the mask.
        self.assertTrue(layer.used_mask)

        # Assert that the output tensor 'y' now has the explicit mask attached
        # for propagation to the next layer.
        self.assertAllClose(backend.get_keras_mask(y), explicit_mask)

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
                self._build_at_init()

            def call(self, x):
                x = backend.convert_to_tensor(x, dtype="float32")
                self.add_loss(ops.sum(x))
                self.ntw.assign(ops.sum(x))
                x = x + backend.random.normal(
                    shape=(), seed=self._seed_generator
                )
                return ops.add(x, ops.add(self.tw, self.ntw))

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
            self.assertAllClose(v, ref_v)

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
            self.assertAllClose(v, ref_v)
        self.assertLen(losses, 2)
        for ref_loss, loss in zip(layer1.losses, losses):
            self.assertAllClose(loss, ref_loss)

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

            def call(self, foo, bar):
                return foo[:, 0] + bar[:, 0]

        class SubsetArguments(layers.Layer):
            def build(self, baz_shape, foo_shape):
                self.foo_shape = foo_shape
                self.baz_shape = baz_shape

            def call(self, foo, bar=None, baz=None):
                return foo[:, 0] + bar[:, 0] + baz[:, 0]

        class SingleArgument(layers.Layer):
            def build(self, anything_whatsoever):
                self.foo_shape = anything_whatsoever

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

    def test_add_weight_string_as_first_positional_arg(self):
        """Test that passing a string as first positional arg to add_weight
        raises a clear error guiding users to use name= keyword."""

        # Case 1: String as only positional arg (e.g. add_weight("matrix"))
        class MyLayer1(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight("my_weight")

        with self.assertRaisesRegex(
            ValueError,
            "name.*keyword argument",
        ):
            MyLayer1()

        # Case 2: String positional + shape kwarg — the exact bug from
        # https://github.com/keras-team/keras/issues/22265
        # In Keras 2 this was valid: add_weight("matrix", shape=(3, 4))
        class MyLayer2(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(
                    "matrix", shape=(3, 4), initializer="zeros"
                )

        with self.assertRaisesRegex(
            ValueError,
            "name.*keyword argument",
        ):
            MyLayer2()

        # Case 3: shape passed both positionally and as keyword
        class MyLayer3(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight((3, 4), shape=(3, 4))

        with self.assertRaisesRegex(
            ValueError,
            "`shape` was passed both positionally and as a keyword argument",
        ):
            MyLayer3()

        # Case 4: positional shape / initializer / dtype must remain valid.
        class MyLayer4(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight((3, 4), "zeros", "float32")

        layer = MyLayer4()
        self.assertEqual(layer.w.shape, (3, 4))
        self.assertEqual(layer.w.dtype, "float32")
        self.assertAllClose(backend.convert_to_numpy(layer.w), np.zeros((3, 4)))

        # Case 5: too many positional arguments
        class MyLayer5(layers.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight((3, 4), "zeros", "float32", "name")

        with self.assertRaisesRegex(
            TypeError,
            "takes at most 3 positional arguments",
        ):
            MyLayer5()

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

    def test_trainable_init_arg_validation(self):
        with self.assertRaisesRegex(ValueError, "to be a boolean"):
            layers.Dense(2, trainable="yes")
        with self.assertRaisesRegex(ValueError, "to be a boolean"):
            layers.Dense(2, trainable=1)
        with self.assertRaisesRegex(ValueError, "to be a boolean"):
            layers.Dense(2, trainable=None)

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
                param_name.replace("_torch_params.", "")
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
                param_name.replace("_torch_params.", "")
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
                "_torch_params.training_layer/post_build_modify_layer/var",
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
                "_torch_params.training_layer/post_build_modify_layer/var",
            )

            parameter_names = [
                pname
                for pname, _ in layer.post_build_modify_layer.named_parameters()
            ]
            self.assertEqual(len(parameter_names), 2)
            self.assertEqual(
                parameter_names[0],
                "_torch_params.training_layer/post_build_modify_layer/var",
            )
            self.assertEqual(
                parameter_names[1],
                "_torch_params.training_layer/post_build_modify_layer/var2",
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
                "post_build_modify_layer._torch_params.training_layer/"
                "post_build_modify_layer/var2",
            )
            self.assertEqual(
                parameter_names[1],
                "_torch_params.training_layer/post_build_modify_layer/var",
            )

            parameter_names = [
                pname
                for pname, _ in layer.post_build_modify_layer.named_parameters()
            ]
            self.assertEqual(len(parameter_names), 1)
            self.assertEqual(
                parameter_names[0],
                "_torch_params.training_layer/post_build_modify_layer/var2",
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

    @pytest.mark.skipif(
        not backend.SUPPORTS_COMPLEX_DTYPES,
        reason=f"{backend.backend()} backend doesn't support complex dtypes.",
    )
    def test_complex_dtype_support(self):
        class MyDenseLayer(layers.Layer):
            def __init__(self, num_outputs):
                super(MyDenseLayer, self).__init__()
                self.num_outputs = num_outputs

            def build(self, input_shape):
                self.kernel = self.add_weight(
                    shape=[int(input_shape[-1]), self.num_outputs],
                )

            def call(self, inputs):
                kernel = ops.cast(self.kernel, "complex64")
                return ops.matmul(inputs, kernel)

        inputs = ops.zeros([10, 5], dtype="complex64")
        layer = MyDenseLayer(10)
        output = layer(inputs)
        self.assertEqual(output.shape, (10, 10))

    def test_call_context_args_with_custom_layers(self):
        class Inner(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=None):
                return x + (1 if foo_mode else 0)

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")
                self.inner = Inner()

            def call(self, x):
                # Outer doesn’t even need to re‑inject explicitly:
                # our base class will propagate foo_mode automatically
                return self.inner(x)

        layer = Outer()
        self.assertEqual(int(layer(np.array(0), foo_mode=True)), 1)
        self.assertEqual(int(layer(np.array(0))), 0)

    def test_register_call_context_arguments(self):
        """Validate that registering call-context args works as expected."""

        class MyLayer(layers.Layer):
            def call(self, x):
                return x

        layer = MyLayer()

        layer._register_call_context_args("foo_mode")

        self.assertCountEqual(
            layer._call_context_args, ("foo_mode", "training")
        )

    def test_register_call_context_arguments_after_call(self):
        """Validate that registering call-context args after the layer has
        been called raises an error."""

        class MyLayer(layers.Layer):
            def call(self, x):
                return x

        layer = MyLayer()
        layer(np.array(0))
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot add call-context args after the layer has been called.",
        ):
            layer._register_call_context_args("foo_mode")

    def test_context_args_with_triple_nesting_and_priority(self):
        """Validate that call-context args are propagated correctly
        through multiple layers, and that the most specific value is used
        when multiple values are passed down the call-stack.
        """

        class Inner(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=None):
                return x + (1 if foo_mode else 0)

        class Middle(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                return self.inner(x)

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.middle = Middle()

            def call(self, x):
                # Outer explicitly sets foo_mode=False when calling Inner,
                # so the value being passed here should be ignored.
                return self.middle(x)

        layer = Outer()
        layer._register_call_context_args("foo_mode")

        # The value of foo_mode is set to True in the call to Outer,
        # so it should automatically propagate to Inner through Middle.
        self.assertEqual(int(layer(np.array(0), foo_mode=True)), 1)
        self.assertEqual(int(layer(np.array(0))), 0)

    def test_context_arg_propagation_without_declaration(self):
        """Validate that layer does not resolve a propagated arg if it is not
        declared as a call-context arg in the layer itself."""

        class Inner(layers.Layer):
            def call(self, x, foo_mode=None):
                return x + (1 if foo_mode else 0)

        class Wrapper(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                return self.inner(x)

        layer = Wrapper()
        layer._register_call_context_args("foo_mode")

        # The value of foo_mode is set to True in the call to Wrapper,
        # However, it is not declared as a call-context arg in Inner,
        # so it should not resolve to True inside Inner (and instead
        # default to False).
        self.assertEqual(int(layer(np.array(0), foo_mode=True)), 0)

    def test_call_context_args_with_func_seq_models_as_layers(self):
        """Validate that call-context args are propagated correctly
        through functional and sequential models when used as layers.
        """

        class Inner(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=False):
                # If foo_mode=True add 1, otherwise add 0
                add_val = ops.where(foo_mode, 1.0, 0.0)
                return x + add_val

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                # We don’t explicitly pass foo_mode here—Base Layer.__call__
                # should inject it into `self.inner`
                return self.inner(x)

        sample_input = np.array([[1.0], [2.0]])

        # Sequential model
        seq = models.Sequential([layers.Identity(), Outer()])
        # Tell the Sequential model to propagate foo_mode down
        # the call-stack
        seq._register_call_context_args("foo_mode")

        # foo_mode=True -> input + 1
        out_true = seq(sample_input, foo_mode=True)
        self.assertAllClose(out_true, sample_input + 1.0)

        # foo_mode omitted -> foo_mode defaults to False -> no change
        out_false = seq(sample_input)
        self.assertAllClose(out_false, sample_input)

        # Functional model
        inp = Input(shape=(1,))
        out = layers.Identity()(inp)
        out = Outer()(out)
        model = models.Model(inp, out)
        # Tell the Functional model to propagate foo_mode down
        # the call-stack
        model._register_call_context_args("foo_mode")

        # foo_mode=True -> input + 1
        y1 = model(sample_input, foo_mode=True)
        self.assertAllClose(y1, sample_input + 1.0)

        # foo_mode omitted -> foo_mode defaults to False -> no change
        y2 = model(sample_input)
        self.assertAllClose(y2, sample_input)

    def test_layer_build_with_attention_mask_arg(self):
        test = self

        class CustomLayer(layers.Layer):
            def call(self, inputs, attention_mask=None):
                if attention_mask is not None:
                    return inputs * ops.cast(attention_mask, x.dtype)
                return inputs

            def build(self, inputs_shape, attention_mask_shape=None):
                test.assertIsNotNone(attention_mask_shape)
                self.built = True

        layer = CustomLayer()
        x = np.ones((2, 3), dtype="float32")
        mask = np.ones((2, 1), dtype="float32")
        y = layer(x, attention_mask=mask)
        self.assertEqual(y.shape, (2, 3))

    def _assert_callspec_dict_equal(self, fast_dict, slow_dict):
        # Compare CallSpec arg dicts. Tensor/symbolic values are the same
        # object in both specs (compare by identity); everything else by
        # value. This avoids elementwise tensor comparisons.
        self.assertEqual(list(fast_dict.keys()), list(slow_dict.keys()))
        for key in fast_dict:
            fast_value = fast_dict[key]
            slow_value = slow_dict[key]
            if backend.is_tensor(fast_value) or isinstance(
                fast_value, backend.KerasTensor
            ):
                self.assertIs(fast_value, slow_value)
            else:
                self.assertEqual(fast_value, slow_value)

    def test_callspec_fast_path_matches_slow_path(self):
        class OneArg(layers.Layer):
            def call(self, inputs):
                return inputs

        class TrainingArg(layers.Layer):
            def call(self, inputs, training=None):
                return inputs

        class TrainingMaskArg(layers.Layer):
            def call(self, inputs, training=None, mask=None):
                return inputs

        class ScalarDefaultArg(layers.Layer):
            def call(self, inputs, scale=1.0):
                return inputs

        eager_input = ops.convert_to_tensor(
            np.random.rand(2, 3).astype("float32")
        )
        symbolic_input = backend.KerasTensor((2, 3))

        for layer_cls in (
            OneArg,
            TrainingArg,
            TrainingMaskArg,
            ScalarDefaultArg,
        ):
            layer = layer_cls()
            self.assertTrue(
                layer._call_spec_fast_path,
                msg=f"{layer_cls.__name__} should be fast-path eligible",
            )
            for x in (eager_input, symbolic_input):
                fast = CallSpec._fast(
                    layer._call_arg_names,
                    layer._call_arg_defaults,
                    layer._call_first_arg_name,
                    x,
                )
                slow = CallSpec(
                    layer._call_param_specs,
                    layer._call_context_args,
                    (x,),
                    {},
                )
                self.assertEqual(fast.argument_names, slow.argument_names)
                self.assertEqual(
                    fast.tensor_arguments_names, slow.tensor_arguments_names
                )
                self.assertEqual(
                    fast.nested_tensor_argument_names,
                    slow.nested_tensor_argument_names,
                )
                self.assertEqual(fast.eager, slow.eager)
                self.assertIs(fast.first_arg, slow.first_arg)
                self._assert_callspec_dict_equal(
                    fast.user_arguments_dict, slow.user_arguments_dict
                )
                self._assert_callspec_dict_equal(
                    fast.arguments_dict, slow.arguments_dict
                )
                self._assert_callspec_dict_equal(
                    fast.tensor_arguments_dict, slow.tensor_arguments_dict
                )

    @parameterized.named_parameters(
        ("single", _binder_single, ("t0",), {}),
        ("single_kw", _binder_single, (), {"inputs": "t0"}),
        ("unexpected_kw", _binder_single, ("t0",), {"training": False}),
        ("std_training", _binder_training_mask, ("t0",), {"training": False}),
        (
            "training_and_mask",
            _binder_training_mask,
            ("t0",),
            {"training": True, "mask": None},
        ),
        ("positional_training", _binder_training_mask, ("t0", False), {}),
        ("defaults_only", _binder_training_mask, ("t0",), {}),
        (
            "mha_shape",
            _binder_mha,
            ("t0", "t1"),
            {"use_causal_mask": True, "training": False},
        ),
        ("mha_qkv_positional", _binder_mha, ("t0", "t1", "t2"), {}),
        (
            "varargs_varkw",
            _binder_var_args_kwargs,
            ("t0", 1, 2),
            {"x": 3, "training": False},
        ),
        ("varargs_empty", _binder_var_args_kwargs, (), {}),
        (
            "mixed_varargs",
            _binder_mixed_varargs,
            ("t0", 1, 2),
            {"training": True, "extra": 9},
        ),
        ("mixed_defaults", _binder_mixed_varargs, ("t0",), {}),
        ("kwonly_required", _binder_kw_only, (1,), {"c": 3}),
        ("kwonly_full", _binder_kw_only, (1, 2), {"c": 3, "d": 5}),
        ("kwonly_missing_raises", _binder_kw_only, (1,), {}),
        ("too_many_positional_raises", _binder_kw_only, (1, 2, 3), {"c": 1}),
        (
            "duplicate_arg_raises",
            _binder_training_mask,
            ("t0",),
            {"inputs": "t0"},
        ),
        ("missing_required_raises", _binder_single, (), {}),
        ("mask_kwarg", _binder_mask, ("t0",), {"mask": "t1"}),
        # Positional-only: the parameter's name is still usable as a
        # **kwargs key, so binding must not report it as a duplicate.
        (
            "pos_only_name_into_var_kw",
            _binder_pos_only_var_kw,
            ("t0",),
            {"x": 5},
        ),
        ("pos_only_unexpected_kw_raises", _binder_pos_only, ("t0",), {"x": 5}),
    )
    def test_bind_call_arguments_matches_signature_bind(self, fn, args, kwargs):
        # `_bind_call_arguments` must reproduce `inspect.Signature.bind` +
        # `apply_defaults` exactly: same raise/no-raise outcome, same
        # bound values, and same ordering for both the provided-only dict
        # (`bind().arguments`) and the defaults-applied dict.
        sig = inspect.signature(fn)
        param_specs = tuple(
            (p.name, p.kind, p.default) for p in sig.parameters.values()
        )
        ref_raised = None
        try:
            bound = sig.bind(*args, **dict(kwargs))
            ref_provided = dict(bound.arguments)
            ref_provided_order = list(bound.arguments)
            bound.apply_defaults()
            ref_full = dict(bound.arguments)
            ref_full_order = list(bound.arguments)
        except TypeError as e:
            ref_raised = e
        try:
            provided, full = _bind_call_arguments(
                param_specs, args, dict(kwargs)
            )
        except TypeError as e:
            if ref_raised is None:
                raise AssertionError(
                    f"_bind_call_arguments raised {e!r} but "
                    "inspect.Signature.bind bound successfully"
                )
            return
        self.assertIsNone(
            ref_raised,
            msg=(
                "inspect.Signature.bind raised but _bind_call_arguments "
                f"bound successfully: provided={provided!r}"
            ),
        )
        self.assertEqual(provided, ref_provided)
        self.assertEqual(list(provided), ref_provided_order)
        self.assertEqual(full, ref_full)
        self.assertEqual(list(full), ref_full_order)

    def test_callspec_fast_path_matches_slow_path_training_kwarg(self):
        # Extends test_callspec_fast_path_matches_slow_path to the
        # training=<bool> fast-path shape (F1): one accepting layer (like
        # Dense) and one non-accepting layer (like LayerNormalization), for
        # both training=True and training=False, and both eager and
        # symbolic inputs. Also checks the caller-kwargs pop side effect
        # for the non-accepting case.
        class TrainingArg(layers.Layer):
            def call(self, inputs, training=None):
                return inputs

        class NoTrainingArg(layers.Layer):
            def call(self, inputs, scale=1.0):
                return inputs

        eager_input = ops.convert_to_tensor(
            np.random.rand(2, 3).astype("float32")
        )
        symbolic_input = backend.KerasTensor((2, 3))

        for layer_cls in (TrainingArg, NoTrainingArg):
            layer = layer_cls()
            self.assertTrue(
                layer._call_spec_fast_path,
                msg=f"{layer_cls.__name__} should be fast-path eligible",
            )
            for x in (eager_input, symbolic_input):
                for training_value in (True, False):
                    fast = CallSpec._fast(
                        layer._call_arg_names,
                        layer._call_arg_defaults,
                        layer._call_first_arg_name,
                        x,
                        training=training_value,
                    )
                    slow_kwargs = {"training": training_value}
                    slow = CallSpec(
                        layer._call_param_specs,
                        layer._call_context_args,
                        (x,),
                        slow_kwargs,
                    )
                    self.assertEqual(fast.argument_names, slow.argument_names)
                    self.assertEqual(
                        fast.tensor_arguments_names,
                        slow.tensor_arguments_names,
                    )
                    self.assertEqual(
                        fast.nested_tensor_argument_names,
                        slow.nested_tensor_argument_names,
                    )
                    self.assertEqual(fast.eager, slow.eager)
                    self.assertIs(fast.first_arg, slow.first_arg)
                    self._assert_callspec_dict_equal(
                        fast.user_arguments_dict, slow.user_arguments_dict
                    )
                    self._assert_callspec_dict_equal(
                        fast.arguments_dict, slow.arguments_dict
                    )
                    self._assert_callspec_dict_equal(
                        fast.tensor_arguments_dict,
                        slow.tensor_arguments_dict,
                    )
                    # Slow path pops non-accepted context args out of the
                    # caller's kwargs dict; NoTrainingArg does not accept
                    # `training`, so it must be popped.
                    if layer_cls is NoTrainingArg:
                        self.assertNotIn("training", slow_kwargs)
                    else:
                        self.assertIn("training", slow_kwargs)

    def test_call_fast_path_training_kwarg_mirrors_kwargs_pop(self):
        # Layer.__call__'s fast-path gate must mirror CallSpec.__init__'s
        # side effect of popping non-accepted context args from the
        # caller-supplied kwargs dict, since downstream code (mask
        # population, the eventual self.call(**kwargs), Node construction)
        # reads that same dict.
        class NoTrainingArg(layers.Layer):
            def call(self, inputs, scale=1.0):
                return inputs

        layer = NoTrainingArg()
        self.assertTrue(layer._call_spec_fast_path)
        x = ops.convert_to_tensor(np.random.rand(2, 3).astype("float32"))
        # Calling with training=<bool> must not raise (it would if the pop
        # did not happen and training leaked into self.call(**kwargs), since
        # NoTrainingArg.call() does not accept `training`).
        y = layer(x, training=True)
        self.assertEqual(tuple(y.shape), (2, 3))

    def test_call_skips_conversion_map_structure_for_training_kwarg(self):
        # N2: `Layer.__call__`'s step-1 conversion-skip gate must not be
        # defeated merely because the caller passed the well-known
        # `training=<bool-or-None>` kwarg (the shape fit()/evaluate()/
        # predict() always use). When the input tensor already matches the
        # layer's dtype, the training=<bool-or-None> call must trigger zero
        # `maybe_convert`-driven `tree.map_structure` calls, same as the
        # plain no-kwargs call, whereas pre-fix it cost 2 extra calls
        # (args + kwargs) from step 1 alone. This counts only step 1's own
        # conversion calls (identified by the `fn` argument passed to
        # `map_structure`), not any unrelated `map_structure` calls a later
        # step may make for other reasons (e.g. N1's mask-pipeline gate
        # applies to a narrower kwargs shape than N2's conversion gate:
        # `training=None` deliberately still takes the general mask path,
        # since `None` needs slow-path call-context inheritance, so it
        # legitimately costs one non-conversion `map_structure` call that
        # `training=<bool>` and the no-kwargs call do not).
        class Identity(layers.Layer):
            def call(self, inputs, training=None):
                return inputs

        layer = Identity()
        x = ops.convert_to_tensor(np.random.rand(2, 3).astype("float32"))
        # Warm up (build, dtype policy resolution) outside the assertion.
        layer(x)

        real_map_structure = tree.map_structure

        def count_conversion_map_structure_calls(fn):
            call_count = 0

            def counting_map_structure(map_fn, *args, **kwargs):
                nonlocal call_count
                if getattr(map_fn, "__name__", "") == "maybe_convert":
                    call_count += 1
                return real_map_structure(map_fn, *args, **kwargs)

            with mock.patch.object(
                tree, "map_structure", counting_map_structure
            ):
                fn()
            return call_count

        baseline_count = count_conversion_map_structure_calls(lambda: layer(x))
        self.assertEqual(baseline_count, 0)

        for training_value in (True, False, None):
            training_count = count_conversion_map_structure_calls(
                lambda: layer(x, training=training_value)
            )
            self.assertEqual(
                training_count,
                baseline_count,
                msg=(
                    "training="
                    f"{training_value!r} should cost the same "
                    "conversion-driven tree.map_structure call count as "
                    "the no-kwargs call (step 1's conversion walk must be "
                    "skipped in both)"
                ),
            )

        y = layer(x, training=False)
        self.assertIs(y, x)

    def test_call_still_converts_with_training_kwarg_dtype_mismatch(self):
        # Negative case: even with the well-known training=<bool> kwarg
        # shape, a dtype-mismatched input must still be converted. Uses
        # float16 (not float64) because JAX disables x64 precision by
        # default, which would silently downcast a float64 input to
        # float32 on `convert_to_tensor` and defeat the mismatch premise
        # this test relies on; float16 vs. float32 is a real mismatch on
        # every backend regardless of precision configuration.
        class Identity(layers.Layer):
            def call(self, inputs, training=None):
                return inputs

        layer = Identity(dtype="float32")
        x16 = ops.convert_to_tensor(np.random.rand(2, 3).astype("float16"))
        layer(ops.convert_to_tensor(np.random.rand(2, 3).astype("float32")))

        real_map_structure = tree.map_structure
        call_count = 0

        def counting_map_structure(map_fn, *args, **kwargs):
            nonlocal call_count
            if getattr(map_fn, "__name__", "") == "maybe_convert":
                call_count += 1
            return real_map_structure(map_fn, *args, **kwargs)

        with mock.patch.object(tree, "map_structure", counting_map_structure):
            y = layer(x16, training=False)
        self.assertGreater(
            call_count,
            0,
            msg="dtype-mismatched input must still take the conversion path",
        )
        self.assertEqual(backend.standardize_dtype(y.dtype), "float32")

    def test_call_still_converts_with_extra_kwargs(self):
        # Negative case: a second kwarg alongside training= must still take
        # the general (conversion-eligible) path.
        class MaskArg(layers.Layer):
            def call(self, inputs, training=None, mask=None):
                return inputs

        layer = MaskArg()
        x = ops.convert_to_tensor(np.random.rand(2, 3).astype("float32"))
        layer(x)

        real_map_structure = tree.map_structure
        call_count = 0

        def counting_map_structure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_map_structure(*args, **kwargs)

        with mock.patch.object(tree, "map_structure", counting_map_structure):
            layer(x, training=False, mask=None)
        self.assertGreater(
            call_count,
            0,
            msg="multi-kwarg calls must still take the conversion path",
        )

    def test_call_training_kwarg_skip_output_matches_slow_path(self):
        # Numeric-equivalence proof: with the skip gate widened, the output
        # of a training=<bool> call on a dtype-matching input must be
        # identical (same object, since conversion is skipped) to what the
        # pre-fix conversion path would have produced (a value-equal, if not
        # identical, tensor).
        class Identity(layers.Layer):
            def call(self, inputs, training=None):
                return inputs

        layer = Identity()
        x = ops.convert_to_tensor(np.random.rand(2, 3).astype("float32"))
        layer(x)

        for training_value in (True, False, None):
            fast_path_output = layer(x, training=training_value)
            # Directly exercise the conversion function the slow path would
            # have applied, to confirm it is a value-preserving no-op for
            # this dtype-matching shape.
            converted = layer.dtype_policy.convert_input(
                x, layer.autocast, layer.input_dtype
            )
            self.assertIs(fast_path_output, x)
            np.testing.assert_array_equal(
                ops.convert_to_numpy(fast_path_output),
                ops.convert_to_numpy(converted),
            )

    def test_callspec_fast_path_excludes_unsupported_signatures(self):
        class MultiArg(layers.Layer):
            def call(self, inputs, other):
                return inputs

        class VarArgs(layers.Layer):
            def call(self, inputs, *args, **kwargs):
                return inputs

        class RequiredKwOnly(layers.Layer):
            def call(self, inputs, *, state):
                return inputs

        self.assertFalse(MultiArg()._call_spec_fast_path)
        self.assertFalse(VarArgs()._call_spec_fast_path)
        self.assertFalse(RequiredKwOnly()._call_spec_fast_path)

        # A tensor-valued default would land in tensor_arguments_dict in the
        # slow path, so it must fall through.
        tensor_default = ops.convert_to_tensor(np.zeros((3,), dtype="float32"))

        class TensorDefault(layers.Layer):
            def call(self, inputs, bias=tensor_default):
                return inputs

        self.assertFalse(TensorDefault()._call_spec_fast_path)

    def test_call_fast_path_training_kwarg_negative_cases(self):
        # Only kwargs == {"training": <exact Python bool>} may take the
        # training-kwarg fast path. Everything else (training=None, a
        # non-bool "truthy" training value, a tensor training value, a
        # second kwarg alongside training, and a subclass-registered
        # context arg) must fall through to the slow path unchanged. We
        # verify this indirectly: patch CallSpec._fast with a counter and
        # confirm it is *not* invoked for these shapes.
        class TrainingArg(layers.Layer):
            def call(self, inputs, training=None, mask=None):
                return inputs

        class FooModeArg(layers.Layer):
            def __init__(self):
                super().__init__()
                self._register_call_context_args("foo_mode")

            def call(self, inputs, foo_mode=None):
                return inputs

        x = ops.convert_to_tensor(np.random.rand(2, 3).astype("float32"))
        calls = []
        original_fast = CallSpec._fast.__func__

        def counting_fast(cls, *args, **kwargs):
            calls.append(1)
            return original_fast(cls, *args, **kwargs)

        with mock.patch.object(CallSpec, "_fast", classmethod(counting_fast)):
            layer = TrainingArg()
            self.assertTrue(layer._call_spec_fast_path)

            calls.clear()
            layer(x, training=None)
            self.assertEqual(len(calls), 0)

            calls.clear()
            layer(x, training=np.bool_(True))
            self.assertEqual(len(calls), 0)

            calls.clear()
            layer(x, training=ops.convert_to_tensor(True))
            self.assertEqual(len(calls), 0)

            calls.clear()
            layer(x, training=True, mask=None)
            self.assertEqual(len(calls), 0)

            foo_layer = FooModeArg()
            self.assertTrue(foo_layer._call_spec_fast_path)
            calls.clear()
            foo_layer(x, foo_mode=True)
            self.assertEqual(len(calls), 0)

            # Sanity: the eligible shape does take the fast path.
            calls.clear()
            layer(x, training=True)
            self.assertEqual(len(calls), 1)

    def test_callspec_fast_path_dense_numeric(self):
        x = ops.convert_to_tensor(np.random.rand(4, 8).astype("float32"))
        dense = layers.Dense(16)
        # First call builds and runs through the fast path.
        self.assertTrue(dense._call_spec_fast_path)
        y_fast = dense(x)
        # Force the slow path on the same built layer and rerun.
        dense._call_spec_fast_path = False
        y_slow = dense(x)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(y_fast), ops.convert_to_numpy(y_slow)
        )

    def test_callspec_fast_path_dense_numeric_training_kwarg(self):
        # Dense doesn't accept `training`, so this exercises the
        # kwargs-pop mirror on an accepting-vs-non-accepting boundary case
        # via a layer that does accept it (BatchNorm-like) is covered by
        # the end-to-end functional-model test below; here we confirm the
        # fast vs. slow numeric result matches for a non-accepting layer
        # called with training=<bool>.
        x = ops.convert_to_tensor(np.random.rand(4, 8).astype("float32"))
        dense = layers.Dense(16)
        self.assertTrue(dense._call_spec_fast_path)
        y_fast = dense(x, training=True)
        dense._call_spec_fast_path = False
        y_slow = dense(x, training=True)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(y_fast), ops.convert_to_numpy(y_slow)
        )

    def test_f1_functional_model_training_kwarg_fast_path_numeric(self):
        # End-to-end: model(x, training=False), the fit()/evaluate()/
        # predict() call shape, must produce the same numeric output on
        # the fast path as with the fast path force-disabled on every
        # layer, and Dropout must still see training=True when called
        # with training=True (behavioral delivery of the context value,
        # not just a numeric-parity coincidence).
        model = _build_f1_functional_model()
        x = np.random.rand(4, 8).astype("float32")

        y_fast = model(x, training=False)

        for layer in model._flatten_layers(include_self=False):
            layer._call_spec_fast_path = False
        y_slow = model(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(y_fast),
            ops.convert_to_numpy(y_slow),
            atol=1e-5,
        )

        # Dropout with training=True actually drops (behavioral check that
        # the context value, not just its numeric shape, is delivered).
        model2 = _build_f1_functional_model()
        y_train = ops.convert_to_numpy(model2(x, training=True))
        y_infer = ops.convert_to_numpy(model2(x, training=False))
        self.assertFalse(np.allclose(y_train, y_infer))

    def test_f1_functional_model_training_kwarg_bind_count_drops(self):
        # Deterministic proof (the series' standard proof style): calling
        # a functional model the way fit()/evaluate()/predict() do --
        # model(x, training=<bool>) -- must not cost more
        # inspect.Signature.bind calls than the bare model(x) case, now
        # that the training-kwarg shape is fast-path eligible too.
        model = _build_f1_functional_model()
        x = np.random.rand(4, 8).astype("float32")

        # Warm up (build all layers) outside of the count.
        model(x)

        original_bind = inspect.Signature.bind
        counts = {"n": 0}

        def counting_bind(self, *args, **kwargs):
            counts["n"] += 1
            return original_bind(self, *args, **kwargs)

        with mock.patch.object(inspect.Signature, "bind", counting_bind):
            counts["n"] = 0
            model(x)
            bare_count = counts["n"]

            counts["n"] = 0
            model(x, training=False)
            training_kwarg_count = counts["n"]

        self.assertEqual(training_kwarg_count, bare_count)

    def test_n1_single_tensor_call_skips_tree_ops(self):
        # Deterministic proof (N1): a mask-free, eager, single-tensor,
        # no-kwargs call must not invoke `tree.map_structure`,
        # `tree.flatten`, or `any_symbolic_tensors` at all -- the mask
        # pipeline (step 2's positional-arg check, step 6's mask
        # population, `previous_mask`, `_set_mask_metadata`, and step 8's
        # symbolic-call detection) must take the O(1) leaf branch
        # end-to-end. Uses a no-`input_spec`, trivial-`call()` layer so
        # `_assert_input_compatibility` (step 3, untouched by N1) and the
        # layer's own numeric ops (both backend-dependent in their
        # `tree.flatten` usage, e.g. the numpy backend's
        # `convert_to_tensor`) contribute zero calls of their own,
        # keeping the assertion exact and backend-agnostic.
        class Plain(layers.Layer):
            def call(self, x):
                return x

        x = ops.convert_to_tensor(np.random.rand(4, 8).astype("float32"))
        plain = Plain()
        plain(x)  # Warm up: build the layer outside of the count.
        self.assertTrue(plain._call_spec_fast_path)

        orig_map_structure = tree.map_structure
        orig_flatten = tree.flatten

        def counting_map_structure(*args, **kwargs):
            counting_map_structure.calls += 1
            return orig_map_structure(*args, **kwargs)

        def counting_flatten(*args, **kwargs):
            counting_flatten.calls += 1
            return orig_flatten(*args, **kwargs)

        counting_map_structure.calls = 0
        counting_flatten.calls = 0

        with (
            mock.patch.object(tree, "map_structure", counting_map_structure),
            mock.patch.object(tree, "flatten", counting_flatten),
            mock.patch(
                "keras.src.layers.layer.any_symbolic_tensors",
                wraps=any_symbolic_tensors,
            ) as mock_any_symbolic,
        ):
            plain(x)

        self.assertEqual(counting_map_structure.calls, 0)
        self.assertEqual(counting_flatten.calls, 0)
        mock_any_symbolic.assert_not_called()

    def test_n1_symbolic_single_tensor_call_skips_tree_ops(self):
        # Same proof as above, for the symbolic (KerasTensor) call shape:
        # `any_symbolic_tensors` must not be invoked even though the call
        # *is* symbolic -- `call_spec.eager` (already computed by
        # `CallSpec._fast`) is reused instead of re-walking the args. Note:
        # `Node.__init__` -> `SymbolicArguments.__init__` still calls
        # `tree.map_structure` for every symbolic call (that machinery is
        # unrelated to, and untouched by, N1), so this test only asserts
        # on `any_symbolic_tensors`, not on `tree.map_structure` calls.
        x = backend.KerasTensor((4, 8))
        dense = layers.Dense(16)
        self.assertTrue(dense._call_spec_fast_path)

        with mock.patch(
            "keras.src.layers.layer.any_symbolic_tensors"
        ) as mock_any_symbolic:
            y = dense(x)

        self.assertIsInstance(y, backend.KerasTensor)
        mock_any_symbolic.assert_not_called()
        # A symbolic call must still create exactly one inbound Node (the
        # whole point of step 8 -- functional-model graph wiring must be
        # unaffected by skipping `any_symbolic_tensors`), and the output's
        # `KerasHistory` must point back at it.
        self.assertLen(dense._inbound_nodes, 1)
        self.assertIs(y._keras_history.operation, dense)

    def test_n1_mask_pipeline_numeric_parity(self):
        # Numeric/behavioral parity between the N1 leaf fast path and the
        # general (fast-path force-disabled) path, across the mask
        # scenarios the mechanism touches: mask-free, mask-propagating
        # (supports_masking=True), and mask-destroying
        # (supports_masking=False, with the accompanying warning).
        mask_value = ops.convert_to_tensor(
            np.array([[True, True, False], [True, False, False]])
        )
        x = ops.convert_to_tensor(np.random.rand(2, 3, 4).astype("float32"))
        backend.set_keras_mask(x, mask_value)

        class MaskFree(layers.Layer):
            def call(self, inputs):
                return inputs * 2

        class MaskPropagating(layers.Layer):
            def call(self, inputs):
                return inputs * 2

            def compute_mask(self, inputs, mask=None):
                return mask

            @property
            def supports_masking(self):
                return True

        for layer_cls in (MaskFree, MaskPropagating):
            fast_layer = layer_cls()
            y_fast = fast_layer(x)
            mask_fast = backend.get_keras_mask(y_fast)

            slow_layer = layer_cls()
            slow_layer._call_spec_fast_path = False
            y_slow = slow_layer(x)
            mask_slow = backend.get_keras_mask(y_slow)

            np.testing.assert_array_equal(
                ops.convert_to_numpy(y_fast), ops.convert_to_numpy(y_slow)
            )
            if layer_cls is MaskPropagating:
                self.assertIsNotNone(mask_fast)
                np.testing.assert_array_equal(
                    ops.convert_to_numpy(mask_fast),
                    ops.convert_to_numpy(mask_slow),
                )
            else:
                self.assertIsNone(mask_fast)
                self.assertIsNone(mask_slow)

        # Mask-destroying case: both paths must warn identically and drop
        # the mask.
        class MaskDestroying(layers.Layer):
            def call(self, inputs):
                return inputs * 2

        fast_layer = MaskDestroying()
        with self.assertWarnsRegex(UserWarning, "does not support masking"):
            y_fast = fast_layer(x)
        self.assertIsNone(backend.get_keras_mask(y_fast))

        slow_layer = MaskDestroying()
        slow_layer._call_spec_fast_path = False
        with self.assertWarnsRegex(UserWarning, "does not support masking"):
            y_slow = slow_layer(x)
        self.assertIsNone(backend.get_keras_mask(y_slow))

    def test_n1_set_mask_metadata_leaf_fast_path(self):
        # Direct unit test of `_set_mask_metadata`'s leaf branch: a
        # non-nested (single-tensor) output must get the mask set/skipped
        # identically to the general (nested) path's outcome for the
        # equivalent single-element structure.
        class Passthrough(layers.Layer):
            def call(self, inputs):
                return inputs

            def compute_mask(self, inputs, previous_mask=None):
                return previous_mask

        layer = Passthrough()
        x = ops.convert_to_tensor(np.zeros((2, 3), dtype="float32"))
        mask = ops.convert_to_tensor(np.array([True, False]))

        # Leaf output: mask gets set via the fast branch.
        out_leaf = ops.convert_to_tensor(np.zeros((2, 3), dtype="float32"))
        layer._set_mask_metadata(x, out_leaf, mask)
        result_mask = backend.get_keras_mask(out_leaf)
        self.assertIsNotNone(result_mask)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(result_mask), ops.convert_to_numpy(mask)
        )

        # Leaf output that already has a mask: fast branch must return
        # early without overwriting it (mirrors `mask_already_computed`
        # in the general path).
        out_already_masked = ops.convert_to_tensor(
            np.zeros((2, 3), dtype="float32")
        )
        preexisting = ops.convert_to_tensor(np.array([False, False]))
        backend.set_keras_mask(out_already_masked, preexisting)
        layer._set_mask_metadata(x, out_already_masked, mask)
        preserved_mask = backend.get_keras_mask(out_already_masked)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(preserved_mask),
            ops.convert_to_numpy(preexisting),
        )

        # `compute_mask` returning None: fast branch must not set a mask.
        class NoMask(layers.Layer):
            def call(self, inputs):
                return inputs

            def compute_mask(self, inputs, previous_mask=None):
                return None

        no_mask_layer = NoMask()
        out_none = ops.convert_to_tensor(np.zeros((2, 3), dtype="float32"))
        no_mask_layer._set_mask_metadata(x, out_none, mask)
        self.assertIsNone(backend.get_keras_mask(out_none))

        # Nested (tuple) output: must fall through to the general path
        # unchanged (this is what the RNN return_state=True shape hits).
        class TupleOut(layers.Layer):
            def call(self, inputs):
                return inputs, inputs

            def compute_mask(self, inputs, previous_mask=None):
                return previous_mask, None

        tuple_layer = TupleOut()
        out_tuple = (
            ops.convert_to_tensor(np.zeros((2, 3), dtype="float32")),
            ops.convert_to_tensor(np.zeros((2, 3), dtype="float32")),
        )
        tuple_layer._set_mask_metadata(x, out_tuple, mask)
        mask0 = backend.get_keras_mask(out_tuple[0])
        mask1 = backend.get_keras_mask(out_tuple[1])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(mask0), ops.convert_to_numpy(mask)
        )
        self.assertIsNone(mask1)

        # Contract-violation guard: a leaf output whose `compute_mask`
        # (incorrectly) returns a nested structure must still get the
        # *first* flattened mask -- matching what the general/nested path
        # would have done via `zip(flat_outputs, tree.flatten(masks))` --
        # not the raw nested structure itself.
        class LeafOutputNestedMask(layers.Layer):
            def call(self, inputs):
                return inputs

            def compute_mask(self, inputs, previous_mask=None):
                return (previous_mask, None)

        malformed_layer = LeafOutputNestedMask()
        out_malformed = ops.convert_to_tensor(np.zeros((2, 3), dtype="float32"))
        malformed_layer._set_mask_metadata(x, out_malformed, mask)
        malformed_result = backend.get_keras_mask(out_malformed)
        self.assertNotIsInstance(malformed_result, tuple)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(malformed_result), ops.convert_to_numpy(mask)
        )

        # Contract-violation guard, empty case: `compute_mask` returning an
        # empty nested structure must not set a mask (must not raise
        # either) -- matches `zip(flat_outputs, [])` silently producing
        # zero pairs in the general path.
        class LeafOutputEmptyMask(layers.Layer):
            def call(self, inputs):
                return inputs

            def compute_mask(self, inputs, previous_mask=None):
                return ()

        empty_layer = LeafOutputEmptyMask()
        out_empty = ops.convert_to_tensor(np.zeros((2, 3), dtype="float32"))
        empty_layer._set_mask_metadata(x, out_empty, mask)
        self.assertIsNone(backend.get_keras_mask(out_empty))

    def test_n1_multi_tensor_mask_path_unaffected(self):
        # Multi-tensor calls (e.g. `k_mask`/`v_mask`-style population) are
        # never `single_tensor_call`-eligible, so they must always take
        # the general `CallSpec.__init__` + `tree.map_structure` path,
        # unchanged by N1.
        class TwoArgMasking(layers.Layer):
            def call(self, q, v, q_mask=None, v_mask=None):
                return q + v

        q = ops.convert_to_tensor(np.ones((2, 3), dtype="float32"))
        v = ops.convert_to_tensor(np.ones((2, 3), dtype="float32"))
        q_mask_value = ops.convert_to_tensor(np.array([True, False]))
        backend.set_keras_mask(q, q_mask_value)

        layer = TwoArgMasking()
        self.assertFalse(layer._call_spec_fast_path)

        orig_map_structure = tree.map_structure
        counting = {"n": 0}

        def counting_map_structure(*args, **kwargs):
            counting["n"] += 1
            return orig_map_structure(*args, **kwargs)

        with mock.patch.object(tree, "map_structure", counting_map_structure):
            layer(q, v)

        # The multi-tensor mask-population branch (step 6's `elif` arm)
        # must still run through `tree.map_structure`.
        self.assertGreater(counting["n"], 0)

    def test_n1_tensor_attr_dict_name_cache(self):
        # The `_DICT_NAME_CACHE` in `tensor_attributes.py` must return the
        # same interned key string on every call for a given `attr` (not
        # just an equal string), and lookups/writes through the cache must
        # still behave identically to uncached f-string keys.
        first = tensor_attributes._dict_name("_some_test_attr")
        second = tensor_attributes._dict_name("_some_test_attr")
        self.assertEqual(first, "_some_test_attr_dict")
        self.assertIs(first, second)

        class Plain:
            pass

        obj = Plain()
        self.assertIsNone(
            tensor_attributes.get_tensor_attr(obj, "_some_test_attr")
        )
        tensor_attributes.set_tensor_attr(obj, "_some_test_attr", "value")
        self.assertEqual(
            tensor_attributes.get_tensor_attr(obj, "_some_test_attr"),
            "value",
        )
        tensor_attributes.set_tensor_attr(obj, "_some_test_attr", None)
        self.assertIsNone(
            tensor_attributes.get_tensor_attr(obj, "_some_test_attr")
        )

    def test_construction_with_custom_pytree_default_without_len(self):
        # Regression test: a call() default that is a custom pytree-
        # registered object with no `__len__` must not crash the
        # fast-path eligibility check at layer construction time, and
        # such a layer must fall through to the slow call path rather
        # than being (incorrectly) treated as fast-path eligible, since
        # the fast path cannot safely introspect a value it cannot take
        # the length of.
        class NoLenPytree:
            def __init__(self, value):
                self.value = value

            # optree (jax/tensorflow/numpy/openvino) interface.
            def tree_flatten(self):
                return (self.value,), None

            @classmethod
            def tree_unflatten(cls, aux_data, children):
                return cls(children[0])

            # torch interface (torch backend uses its own tree dispatch,
            # not optree, so both interfaces are needed for this test to
            # be meaningful on every backend).
            def torchtree_flatten(self):
                return [self.value], None

            def torchtree_flatten_with_keys(self):
                return [("value", self.value)], None

            @classmethod
            def torchtree_unflatten(cls, children, context):
                return cls(children[0])

        tree.register_tree_node_class(NoLenPytree)
        if backend.backend() == "torch":
            from torch.utils import _pytree as torch_tree

            self.addCleanup(torch_tree._deregister_pytree_node, NoLenPytree)
        else:
            optree = pytest.importorskip("optree")
            self.addCleanup(
                optree.unregister_pytree_node, NoLenPytree, namespace="keras"
            )

        self.assertTrue(tree.is_nested(NoLenPytree(1.0)))
        with self.assertRaises(TypeError):
            len(NoLenPytree(1.0))

        class CustomDefaultLayer(layers.Layer):
            def call(self, inputs, config=NoLenPytree(1.0)):
                return inputs

        # Must not raise, e.g. TypeError: object of type 'NoLenPytree'
        # has no len().
        layer = CustomDefaultLayer()
        self.assertIsInstance(layer, CustomDefaultLayer)
        # A default whose length can't be determined must not be treated
        # as fast-path eligible.
        self.assertFalse(layer._call_spec_fast_path)
