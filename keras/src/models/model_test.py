import os
import pickle
from collections import namedtuple

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import testing
from keras.src import tree
from keras.src.layers.core.input_layer import Input
from keras.src.models.functional import Functional
from keras.src.models.model import Model
from keras.src.models.model import model_from_json


def _get_model():
    input_a = Input(shape=(3,), batch_size=2, name="input_a")
    input_b = Input(shape=(3,), batch_size=2, name="input_b")
    x = input_a + input_b
    x = layers.Dense(5)(x)
    outputs = layers.Dense(4)(x)
    model = Model([input_a, input_b], outputs)
    return model


def _get_model_multi_outputs_list():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    output_b = layers.Dense(1, name="output_b", activation="sigmoid")(x)
    model = Model(x, [output_a, output_b])
    return model


def _get_model_multi_outputs_list_no_output_names():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1)(x)
    output_b = layers.Dense(1, activation="sigmoid")(x)
    model = Model(x, [output_a, output_b])
    return model


def _get_model_single_output():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    model = Model(x, output_a)
    return model


def _get_model_single_output_list():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    model = Model(x, [output_a])
    return model


def _get_model_single_output_dict():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    model = Model(x, {"output_a": output_a})
    return model


def _get_model_multi_outputs_dict():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    output_b = layers.Dense(1, name="output_b", activation="sigmoid")(x)
    model = Model(x, {"output_a": output_a, "output_b": output_b})
    return model


def _get_model_multi_outputs_struct_list_like(_type):
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    model = Model(x, _type([y1, y2]))
    return model


def _get_model_multi_outputs_struct_namedtuple():
    Y = namedtuple("Y", ["y1", "y2"])
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    model = Model(x, Y(y1, y2))
    return model, Y


def _get_model_multi_outputs_struct_dict():
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    model = Model(x, {"a": y1, "b": y2})
    return model


def _get_model_multi_outputs_struct():
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    y3 = layers.Dense(1, name="y3", activation="sigmoid")(x)
    model = Model(
        x,
        {
            "a": (y1, y2),
            "b": {"b1": y1, "b2": y2},
            "c": {"c1": (y1, y2), "c2": y2},
            "d": y3,
        },
    )
    return model


def _get_model_multi_outputs_dict_with_single_tensor():
    x = Input(shape=(3,), name="input_a")
    output = layers.Dense(1, name="output_a")(x)
    model = Model(x, {"output_a": output, "output_b": output})
    return model


def _get_model_with_custom_compute_loss():
    class MyModel(Model):
        def __init__(self):
            inputs = Input(shape=(3,), name="inputs")
            outputs = layers.Dense(1, name="a")(inputs)
            super().__init__(inputs=inputs, outputs=outputs)

        def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
            y_pred = [y_pred, y_pred]  # To list
            return super().compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, **kwargs
            )

    model = MyModel()
    return model


def _get_model_with_duplicate_variable_path():
    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.dense1 = layers.Dense(4, activation="relu", name="layer1")
            self.dense2 = layers.Dense(4, activation="relu", name="layer1")
            self.dense3 = layers.Dense(2)

        def call(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            return self.dense3(x)

    model = MyModel()
    x = np.random.random((1, 16))
    model(x)
    return model


def _get_model_optional_inputs():
    class OptionalInputLayer(layers.Layer):
        def __init__(self):
            super().__init__()
            self.dense = layers.Dense(2)

        def call(self, x, o=None):
            z = x if o is None else x + o
            return self.dense(z)

    x = Input((2,), name="x")
    o = Input((2,), name="o", optional=True)
    y = OptionalInputLayer()(x, o)
    model = Model({"x": x, "o": o}, y)
    return model


def _get_variable_value_by_path(variables, path):
    for v in variables:
        if v.path == path:
            return v.value
    raise ValueError(f"No variable was find with path = {path}")


@pytest.mark.requires_trainable_backend
class ModelTest(testing.TestCase):
    def test_functional_rerouting(self):
        model = _get_model()
        self.assertIsInstance(model, Functional)

    def test_json_serialization(self):
        model = _get_model()
        json_string = model.to_json()
        new_model = model_from_json(json_string)
        self.assertEqual(json_string, new_model.to_json())

    def test_tuple_input_model_subclass(self):
        # https://github.com/keras-team/keras/issues/324

        class MultiInputModel(Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense1 = layers.Dense(4)

            def call(self, inputs):
                a, b = inputs
                r = self.dense1(a)
                return layers.concatenate([r, b])

        model = MultiInputModel()
        x1 = np.random.rand(3, 3)
        x2 = np.random.rand(3, 2)
        out = model((x1, x2))
        self.assertEqual(out.shape, (3, 6))

    def test_reviving_functional_from_config_custom_layer(self):
        class CustomDense(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.dense = layers.Dense(units)

            def call(self, x):
                return self.dense(x)

        inputs = layers.Input((4,))
        outputs = CustomDense(10)(inputs)
        model = Model(inputs, outputs)
        config = model.get_config()

        new_model = Model.from_config(
            config, custom_objects={"CustomDense": CustomDense}
        )
        self.assertIsInstance(new_model, Functional)

    def test_reviving_functional_from_config_custom_model(self):
        class CustomModel(Model):
            def __init__(self, *args, param=1, **kwargs):
                super().__init__(*args, **kwargs)
                self.param = param

            def get_config(self):
                base_config = super().get_config()
                config = {"param": self.param}
                return base_config | config

        inputs = layers.Input((3,))
        outputs = layers.Dense(5)(inputs)
        model = CustomModel(inputs=inputs, outputs=outputs, param=3)

        new_model = CustomModel.from_config(model.get_config())
        self.assertEqual(new_model.param, 3)

    @parameterized.named_parameters(
        ("single_output_1", _get_model_single_output),
        ("single_output_2", _get_model_single_output),
        ("single_output_3", _get_model_single_output),
        ("single_output_4", _get_model_single_output),
        ("single_list_output_1", _get_model_single_output_list),
        ("single_list_output_2", _get_model_single_output_list),
        ("single_list_output_3", _get_model_single_output_list),
        ("single_list_output_4", _get_model_single_output_list),
    )
    def test_functional_pickling(self, model_fn):
        model = model_fn()
        self.assertIsInstance(model, Functional)
        model.compile()
        x = np.random.rand(8, 3)

        reloaded_pickle = pickle.loads(pickle.dumps(model))

        pred_reloaded = reloaded_pickle.predict(x)
        pred = model.predict(x)

        self.assertAllClose(np.array(pred_reloaded), np.array(pred))

    @parameterized.named_parameters(
        ("single_output_1", _get_model_single_output, None),
        ("single_output_2", _get_model_single_output, "list"),
        ("single_output_3", _get_model_single_output, "dict"),
        ("single_output_4", _get_model_single_output, "dict_list"),
        ("single_list_output_1", _get_model_single_output_list, None),
        ("single_list_output_2", _get_model_single_output_list, "list"),
        ("single_list_output_3", _get_model_single_output_list, "dict"),
        ("single_list_output_4", _get_model_single_output_list, "dict_list"),
        ("single_dict_output_1", _get_model_single_output_dict, None),
        ("single_dict_output_2", _get_model_single_output_dict, "list"),
        ("single_dict_output_3", _get_model_single_output_dict, "dict"),
        ("single_dict_output_4", _get_model_single_output_dict, "dict_list"),
    )
    def test_functional_single_output(self, model_fn, loss_type):
        model = model_fn()
        self.assertIsInstance(model, Functional)
        loss = "mean_squared_error"
        if loss_type == "list":
            loss = [loss]
        elif loss_type == "dict":
            loss = {"output_a": loss}
        elif loss_type == "dict_list":
            loss = {"output_a": [loss]}
        model.compile(
            optimizer="sgd",
            loss=loss,
            metrics={
                "output_a": ["mean_squared_error", "mean_absolute_error"],
            },
            weighted_metrics={
                "output_a": "mean_squared_error",
            },
        )
        # Fit the model to make sure compile_metrics are built
        x = np.random.rand(8, 3)
        y = np.random.rand(8, 1)
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "mean_absolute_error",
                "mean_squared_error",
                "weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_list_losses(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=["mean_squared_error", "binary_crossentropy"],
            metrics=[
                "mean_squared_error",
                ["mean_squared_error", "accuracy"],
            ],
            loss_weights=[0.1, 2],
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_list_losses_abbr(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=["mse", "bce"],
            metrics=[
                ["bce", "mse", "mae"],
                ["mse", "acc"],
            ],
            loss_weights=[0.1, 2],
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_bce",
                "output_a_mae",
                "output_a_mse",
                "output_b_acc",
                "output_b_loss",
                "output_b_mse",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_nested_list_losses(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=["mean_squared_error", ["binary_crossentropy"]],
            metrics=[
                "mean_squared_error",
                ["mean_squared_error", "accuracy"],
            ],
            loss_weights=[0.1, 2],
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_dict_outputs_dict_losses(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": ["binary_crossentropy"],
            },
            metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
            weighted_metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs["output_a"].shape, (8, 1))
        self.assertEqual(outputs["output_b"].shape, (8, 1))
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            {"output_a": y1, "output_b": y2},
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
                "output_b_weighted_accuracy",
                "output_b_weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_dict_outputs_dict_losses_with_undefined_loss(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_b": ["binary_crossentropy"],
            },
            metrics={
                "output_b": ["mean_squared_error", "accuracy"],
            },
            weighted_metrics={
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs["output_a"].shape, (8, 1))
        self.assertEqual(outputs["output_b"].shape, (8, 1))
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            {"output_a": y1, "output_b": y2},
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_b_accuracy",
                "output_b_mean_squared_error",
                "output_b_weighted_accuracy",
                "output_b_weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_metrics(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
            weighted_metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Check list outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs[0].shape, (8, 1))
        self.assertEqual(outputs[1].shape, (8, 1))
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
                "output_b_weighted_accuracy",
                "output_b_weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_metrics_uniq_weighted(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error"],
            },
            weighted_metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        # `output_b_accuracy` doesn't have `weighted_` in metric name.
        # When a metric is only in weighted metrics, it skips `weighted_`
        # prefix. This behavior matches`tf.keras`.
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_partial_metrics(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_dict_outputs_with_single_tensor(self):
        model = _get_model_multi_outputs_dict_with_single_tensor()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))

        # `model` has 2 outputs, but there is actually only 1 output tensor.
        self.assertLen(model.outputs, 2)
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
        )
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(["loss", "output_a_loss", "output_b_loss"])
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_with_custom_compute_loss(self):
        model = _get_model_with_custom_compute_loss()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))

        # `model` has 1 output, but in `compute_loss` it is separated into 2.
        self.assertLen(model.outputs, 1)
        model.compile(
            optimizer="sgd", loss=["mean_squared_error", "binary_crossentropy"]
        )
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            ["binary_crossentropy_loss", "loss", "mean_squared_error_loss"]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_invalid_keys(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_c": "binary_crossentropy",
            },
        )

        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "Expected keys",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_list_outputs_dict_losses_no_output_names(self):
        model = _get_model_multi_outputs_list_no_output_names()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={"output_a": "mean_squared_error"},
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "Expected keys",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_list_outputs_dict_metrics_invalid_keys(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_c": ["mean_squared_error", "accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "In the dict argument `metrics`, "
            "key 'output_c' does not correspond to any model output",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_dict_outputs_dict_losses_invalid_keys(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_c": "binary_crossentropy",
            },
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            KeyError,
            "in the `loss` argument, can't be found "
            "in either the model's output",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_dict_outputs_dict_metrics_invalid_keys(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_c": ["mean_squared_error", "accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "In the dict argument `metrics`, "
            "key 'output_c' does not correspond to any model output",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_list_outputs_invalid_nested_list_losses(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=[
                "mean_squared_error",
                ["mean_squared_error", "binary_crossentropy"],
            ],
        )
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(["loss", "output_a_loss", "output_b_loss"])
        self.assertListEqual(hist_keys, ref_keys)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
    )
    def test_quantize(self, mode):
        model = _get_model()
        x1 = np.random.rand(2, 3)
        x2 = np.random.rand(2, 3)
        model.quantize(mode)
        _ = model((x1, x2))

        for layer in model._flatten_layers():
            if isinstance(layer, (layers.Dense, layers.EinsumDense)):
                self.assertEqual(
                    layer.dtype_policy.name, f"{mode}_from_float32"
                )
                self.assertEqual(layer.dtype_policy.quantization_mode, mode)
        if mode == "int8":
            self.assertLen(model.variables, 6)
            if backend.backend() == "torch":
                self.assertLen(list(model.named_parameters()), 6)
        elif mode == "float8":
            self.assertLen(model.variables, 16)
            if backend.backend() == "torch":
                self.assertLen(list(model.named_parameters()), 16)

    @parameterized.named_parameters(
        ("regex_string", "dense_1", ["dense_1"]),
        ("list_of_regex", ["dense_1", "output"], ["dense_1", "output"]),
        ("callable", lambda l: "dense" in l.name, ["dense_1", "dense_2"]),
    )
    def test_quantize_with_filters(self, filters, expected_quantized_layers):
        mode = "int8"
        inputs = layers.Input([3])
        x = layers.Dense(32, name="dense_1")(inputs)
        x = layers.Dense(32, name="dense_2")(x)
        outputs = layers.Dense(32, name="output")(x)
        model = Model(inputs, outputs)

        model.quantize(mode, filters=filters)

        for layer in model._flatten_layers():
            if layer.name in expected_quantized_layers:
                self.assertEqual(
                    layer.dtype_policy.name, f"{mode}_from_float32"
                )
            elif isinstance(layer, layers.Dense):
                self.assertNotEqual(
                    layer.dtype_policy.name, f"{mode}_from_float32"
                )

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
    )
    def test_quantize_unbuilt(self, mode):
        class MyModel(Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(32, activation="relu")
                self.dense2 = layers.Dense(5, activation="softmax")
                self.dropout = layers.Dropout(0.5)

            def call(self, inputs, training=False):
                x = self.dense1(inputs)
                x = self.dropout(x, training=training)
                return self.dense2(x)

        model = MyModel()
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            model.quantize(mode)

        x = np.random.rand(2, 3)
        _ = model(x)
        model.quantize(mode)

    def test_quantize_invalid_args(self):
        model = _get_model()
        with self.assertRaisesRegex(
            ValueError, "Invalid quantization mode. Expected one of"
        ):
            model.quantize("abc")

        with self.assertRaisesRegex(
            ValueError, "Unrecognized keyword arguments"
        ):
            model.quantize("int8", unrecognized_kwargs=None)

        with self.assertRaisesRegex(ValueError, "Invalid quantization mode"):
            model.quantize("int7")

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
    )
    def test_quantize_nested_model(self, mode):
        class NestedLayer(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)

            def call(self, x):
                x = self.dense(x)
                return x

        class DoubleNestedLayer(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.nested_dense1 = NestedLayer(units)
                self.nested_dense2 = NestedLayer(units)
                self.dense = layers.Dense(units)

            def call(self, x):
                x = self.nested_dense1(x)
                x = self.nested_dense2(x)
                x = self.dense(x)
                return x

        inputs = layers.Input([3])
        outputs = DoubleNestedLayer(8)(inputs)
        model = Model(inputs, outputs)
        model.quantize(mode)

        if mode == "int8":
            kernel_count = 0
            for weight in model.weights:
                if weight.name == "kernel":
                    kernel_count += 1
                    self.assertEqual(
                        backend.standardize_dtype(weight.dtype), "int8"
                    )
            self.assertEqual(kernel_count, 3)
        if mode == "float8":
            # kernel + bias + scale * 3 + amax_history * 3 == 8
            self.assertEqual(len(model.weights), 3 * 8)

    def test_get_state_tree(self):
        model = _get_model_single_output()
        model.compile(loss="mse", optimizer="adam")
        state_tree = model.get_state_tree()
        self.assertAllClose(
            state_tree["trainable_variables"]["output_a"]["kernel"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/kernel"
            ),
        )
        self.assertAllClose(
            state_tree["trainable_variables"]["output_a"]["bias"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/bias"
            ),
        )
        self.assertEqual(
            state_tree["non_trainable_variables"],
            {},
        )
        self.assertEqual(
            state_tree["metrics_variables"]["loss"]["count"],
            _get_variable_value_by_path(model.metrics_variables, "loss/count"),
        )
        self.assertEqual(
            state_tree["metrics_variables"]["loss"]["total"],
            _get_variable_value_by_path(model.metrics_variables, "loss/total"),
        )
        self.assertEqual(
            state_tree["optimizer_variables"]["adam"]["iteration"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/iteration"
            ),
        )
        self.assertEqual(
            state_tree["optimizer_variables"]["adam"]["learning_rate"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/learning_rate"
            ),
        )

        # Test with numpy
        state_tree = model.get_state_tree(value_format="numpy_array")
        self.assertIsInstance(
            state_tree["trainable_variables"]["output_a"]["kernel"], np.ndarray
        )

    def test_set_state_tree(self):
        variables = {
            "optimizer_variables": {
                "adam": {
                    "iteration": 0,
                    "learning_rate": 0.00001,
                }
            },
            "trainable_variables": {
                "output_a": {
                    "bias": [0.5],
                    "kernel": [[0.6], [0.7], [1.8]],
                }
            },
        }

        model = _get_model_single_output()
        model.compile(optimizer="adam")
        model.set_state_tree(variables)

        self.assertEqual(
            variables["optimizer_variables"]["adam"]["iteration"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/iteration"
            ),
        )
        self.assertEqual(
            variables["optimizer_variables"]["adam"]["learning_rate"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/learning_rate"
            ),
        )
        self.assertAllClose(
            variables["trainable_variables"]["output_a"]["bias"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/bias"
            ),
        )
        self.assertAllClose(
            variables["trainable_variables"]["output_a"]["kernel"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/kernel"
            ),
        )

    def test_get_state_tree_with_duplicate_path(self):
        model = _get_model_with_duplicate_variable_path()
        with self.assertRaisesRegex(
            ValueError,
            "The following variable path is found twice in the model",
        ):
            model.get_state_tree()

    def test_layers_setter(self):
        model = Model()
        with self.assertRaisesRegex(
            AttributeError, "`Model.layers` attribute is reserved"
        ):
            model.layers = [layers.Dense(4)]

    def get_struct_loss(self, structure):
        def loss_fn(y_true, y_pred):
            tree.assert_same_structure(structure, y_true)
            tree.assert_same_structure(structure, y_pred)
            tree.map_structure(
                lambda spec, tensor: self.assertEqual(spec.ndim, tensor.ndim),
                structure,
                y_true,
            )
            tree.map_structure(
                lambda spec, tensor: self.assertEqual(spec.ndim, tensor.ndim),
                structure,
                y_pred,
            )
            flat_y_pred = tree.flatten(y_pred)
            flat_y_true = tree.flatten(y_true)
            diff = 0
            for y_p, y_t in zip(flat_y_pred, flat_y_true):
                diff += losses.mean_absolute_error(y_t, y_p)
            return diff

        return loss_fn

    @parameterized.product(
        _type=[tuple, list], other_type=[list, tuple], weighted=[False, True]
    )
    def test_functional_struct_outputs_struct_losses(
        self, _type, other_type, weighted
    ):
        model = _get_model_multi_outputs_struct_list_like(_type)
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)
        y = _type([y1, y2])
        loss = other_type(
            [
                self.get_struct_loss(model.output),
                _type(
                    [
                        self.get_struct_loss(model.output[0]),
                        self.get_struct_loss(model.output[1]),
                    ]
                ),
            ]
        )
        if weighted:
            loss_weights = tree.map_structure(lambda _: np.random.rand(), loss)
        else:
            loss_weights = None

        model.compile(
            optimizer="sgd",
            loss=loss,
            loss_weights=loss_weights,
        )

        if _type is other_type:
            with self.assertRaisesRegex(
                ValueError, f"[Ee]xpected.*{_type.__name__}"
            ):
                model.fit(x, y, batch_size=2, epochs=1, verbose=0)
        else:
            # Check dict outputs.
            outputs = model.predict(x)
            self.assertIsInstance(outputs, _type)
            # Fit the model to make sure compile_metrics are built
            hist = model.fit(
                x,
                y,
                batch_size=2,
                epochs=1,
                verbose=0,
            )
            hist_keys = sorted(hist.history.keys())
            ref_keys = sorted(
                [
                    "loss",
                    "y1_loss",
                    "y2_loss",
                    "y1_y2_loss",
                ]
            )
            self.assertListEqual(hist_keys, ref_keys)

    @parameterized.named_parameters(("weighted", True), ("not_weighted", False))
    def test_functional_struct_outputs_dict_struct_losses(self, weighted):
        model = _get_model_multi_outputs_struct_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)

        y = {"a": y1, "b": y2}
        loss = [
            self.get_struct_loss(model.output),
            {
                "a": self.get_struct_loss(model.output["a"]),
                "b": self.get_struct_loss(model.output["a"]),
            },
        ]
        if weighted:
            loss_weights = tree.map_structure(lambda _: np.random.rand(), loss)
        else:
            loss_weights = None

        model.compile(
            optimizer="sgd",
            loss=loss,
            loss_weights=loss_weights,
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)

        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "a_loss",
                "b_loss",
                "a_b_loss",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_struct_outputs_namedtuple_struct_losses(self):
        model, Y = _get_model_multi_outputs_struct_namedtuple()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)

        y = Y(y1, y2)
        model.compile(
            optimizer="sgd",
            loss=[
                self.get_struct_loss(model.output),
                Y(
                    self.get_struct_loss(model.output.y1),
                    self.get_struct_loss(model.output.y2),
                ),
            ],
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, tuple)

        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "y1_loss",
                "y2_loss",
                "y1_y2_loss",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_deeply_nested_outputs_struct_losses(self):
        model = _get_model_multi_outputs_struct()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)
        y3 = np.random.rand(8, 1)
        y = {
            "a": (y1, y2),
            "b": {"b1": y1, "b2": y2},
            "c": {"c1": (y1, y2), "c2": y2},
            "d": y3,
        }
        model.compile(
            optimizer="sgd",
            loss={
                "a": [
                    self.get_struct_loss(model.output["a"]),
                    (None, self.get_struct_loss(model.output["a"][1])),
                ],
                "b": [
                    self.get_struct_loss(model.output["b"]),
                    {"b1": self.get_struct_loss(model.output["b"]["b1"])},
                ],
                "c": [
                    self.get_struct_loss(model.output["c"]),
                    {"c1": self.get_struct_loss(model.output["c"]["c1"])},
                ],
                "d": self.get_struct_loss(model.output["d"]),
            },
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)

        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "a/y2_loss",
                "a_loss",
                "b/b1_loss",
                "b_loss",
                "c/c1_loss",
                "c_loss",
                "d_loss",
                "loss",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    @parameterized.named_parameters(
        ("optional_none", True), ("optional_tensor", False)
    )
    def test_functional_optional_inputs(self, is_optional_none):
        model = _get_model_optional_inputs()
        x = np.ones((2, 2))
        o = None if is_optional_none else np.ones((2, 2))
        y_true = np.ones((2, 2))

        model.compile(loss="mse", optimizer="adam")
        model.fit(x={"x": x, "o": o}, y=y_true)
        model.evaluate(x={"x": x, "o": o}, y=y_true)
        model.predict(x={"x": x, "o": o})

    @parameterized.named_parameters(
        ("optional_none", True), ("optional_tensor", False)
    )
    def test_functional_optional_inputs_generator(self, is_optional_none):
        model = _get_model_optional_inputs()
        x = np.ones((2, 2))
        o = None if is_optional_none else np.ones((2, 2))
        y_true = np.ones((2, 2))

        def data_generator(with_y=True):
            for _ in range(4):
                yield ({"x": x, "o": o},) + ((y_true,) if with_y else ())

        model.compile(loss="mse", optimizer="adam")
        model.fit(data_generator())
        model.evaluate(data_generator())
        model.predict(data_generator(with_y=False))

    def test_export_error(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = _get_model()

        # Bad format
        with self.assertRaisesRegex(ValueError, "Unrecognized format="):
            model.export(temp_filepath, format="bad_format")

        # Bad backend
        if backend.backend() not in ("tensorflow", "jax", "torch"):
            with self.assertRaisesRegex(
                NotImplementedError,
                (
                    r"`export_saved_model` only currently supports the "
                    r"tensorflow, jax and torch backends."
                ),
            ):
                model.export(temp_filepath, format="tf_saved_model")
