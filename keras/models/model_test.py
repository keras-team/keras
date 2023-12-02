import numpy as np
import pytest
from absl.testing import parameterized

from keras import layers
from keras import testing
from keras.layers.core.input_layer import Input
from keras.models.functional import Functional
from keras.models.model import Model
from keras.models.model import model_from_json


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


@pytest.mark.requires_trainable_backend
class ModelTest(testing.TestCase, parameterized.TestCase):
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
        elif loss_type == "dict_lsit":
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
        # TODO `tf.keras` also outputs individual losses for outputs
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_a_mean_squared_error",
                "output_b_accuracy",
                # "output_b_loss",
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
        # TODO `tf.keras` also outputs individual losses for outputs
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_a_bce",
                "output_a_mae",
                "output_a_mse",
                "output_b_acc",
                # "output_b_loss",
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
        # TODO `tf.keras` also outputs individual losses for outputs
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_a_mean_squared_error",
                "output_b_accuracy",
                # "output_b_loss",
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
        # TODO `tf.keras` also outputs individual losses for outputs
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                # "output_b_loss",
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
        # TODO `tf.keras` also outputs individual losses for outputs
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                # "output_b_loss",
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
        # TODO `tf.keras` also outputs individual losses for outputs
        # `output_b_accuracy` doesn't have `weighted_` in metric name.
        # When a metric is only in weighted metrics, it skips `weighted_`
        # prefix. This behavior matches`tf.keras`.
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                # "output_b_loss",
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
        # TODO `tf.keras` also outputs individual losses for outputs
        ref_keys = sorted(
            [
                "loss",
                # "output_a_loss",
                "output_b_accuracy",
                # "output_b_loss",
                "output_b_mean_squared_error",
            ]
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
            "In the dict argument `loss`, "
            "key 'output_c' does not correspond to any model output",
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
            "In the dict argument `loss`, "
            "key 'output_a' does not correspond to any model output",
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
            ValueError,
            "In the dict argument `loss`, "
            "key 'output_c' does not correspond to any model output",
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
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "when providing the `loss` argument as a list, "
            "it should have as many entries as the model has outputs",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
