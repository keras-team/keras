import numpy as np
import pytest
from absl.testing import parameterized

import keras_core
from keras_core import backend
from keras_core import initializers
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import ops
from keras_core import optimizers
from keras_core import testing
from keras_core.callbacks.callback import Callback

if backend.backend() == "jax":
    from keras_core.backend.jax.trainer import JAXTrainer as Trainer
elif backend.backend() == "torch":
    from keras_core.backend.torch.trainer import TorchTrainer as Trainer
elif backend.backend() == "tensorflow":
    from keras_core.backend.tensorflow.trainer import (
        TensorFlowTrainer as Trainer,
    )
elif backend.backend() == "numpy":
    from keras_core.backend.numpy.trainer import NumpyTrainer as Trainer
else:
    raise ImportError(f"Invalid backend: {backend.backend()}")


# A model is just a layer mixed in with a Trainer.
class ExampleModel(layers.Dense, Trainer):
    def __init__(self, units):
        layers.Dense.__init__(
            self,
            units=units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        Trainer.__init__(self)


class StructModel(layers.Layer, Trainer):
    def __init__(self, units):
        layers.Layer.__init__(self)
        Trainer.__init__(self)
        self.dense_1 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        self.dense_2 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )

    def call(self, x):
        return {
            "y_one": self.dense_1(x["x_one"]),
            "y_two": self.dense_2(x["x_two"]),
        }


class ListModel(layers.Layer, Trainer):
    def __init__(self, units):
        layers.Layer.__init__(self)
        Trainer.__init__(self)
        self.dense_1 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        self.dense_2 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )

    def call(self, x):
        assert isinstance(x, (list, tuple))
        return self.dense_1(x[0]) + self.dense_2(x[1])


class TrainingTestingLayer(layers.Layer, Trainer):
    def __init__(self):
        layers.Layer.__init__(self)
        Trainer.__init__(self)

    def call(self, x, training=False):
        if training:
            return x
        return x * 0


class TestTrainer(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_metric_tracking(self):
        class ModelWithMetric(layers.Dense, Trainer):
            def __init__(self, units):
                layers.Dense.__init__(
                    self,
                    units=units,
                    use_bias=False,
                    kernel_initializer=initializers.Ones(),
                )
                Trainer.__init__(self)
                self.my_metric = metrics.MeanSquaredError(name="my_metric")

        model = ModelWithMetric(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        x = np.ones((2, 4))
        y = np.zeros((2, 3))
        # Fit the model to make sure compile_metrics are built
        model.fit(x, y, batch_size=2, epochs=1)

        # The model should have 3 metrics: loss_tracker, compile_metrics,
        # my_metric.
        self.assertEqual(len(model.metrics), 3)
        self.assertEqual(model.metrics[0], model._loss_tracker)
        self.assertEqual(model.metrics[1], model.my_metric)
        self.assertEqual(model.metrics[2], model._compile_metrics)

        # All metrics should have their weights created
        self.assertEqual(len(model._loss_tracker.variables), 2)
        self.assertEqual(len(model._compile_metrics.variables), 2)
        self.assertEqual(len(model.my_metric.variables), 2)

        # And those weights are tracked at the model level
        self.assertEqual(len(model.metrics_variables), 6)

        # Models with only weighted_metrics should have the same 3 metrics
        model_weighted = ModelWithMetric(units=3)
        model_weighted.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            weighted_metrics=[metrics.MeanSquaredError()],
        )
        model_weighted.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            sample_weight=np.ones(2),
        )
        self.assertEqual(len(model_weighted.metrics), 3)

    @parameterized.named_parameters(
        [
            ("eager", True, False, False),
            ("graph_fn", False, False, False),
            ("jit", False, True, False),
            ("steps_per_epoch_eager", True, False, True),
            ("steps_per_epoch_graph_fn", False, False, True),
            ("steps_per_epoch_jit", False, True, True),
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_fit_flow(self, run_eagerly, jit_compile, use_steps_per_epoch):
        if not run_eagerly and not jit_compile and use_steps_per_epoch:
            if backend.backend() == "tensorflow":
                self.skipTest(
                    "TODO: Graph mode without XLA in TF backend leads to "
                    "unexpected logs, need further checks."
                )

        model = ExampleModel(units=3)
        epochs = 3
        batch_size = 20
        steps_per_epoch = 7
        dataset_size = batch_size * (steps_per_epoch - 2)
        x = np.ones((dataset_size, 4))
        y = np.zeros((dataset_size, 3))

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        history = model.fit(
            x,
            y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch if use_steps_per_epoch else None,
            epochs=epochs,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("mean_squared_error", history)
        self.assertAllClose(
            history["mean_squared_error"],
            [14.402393, 10.991339, 8.388159],
            atol=6.1051628e-1,
        )

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_evaluate_flow(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        output = model.evaluate(x, y, batch_size=batch_size)
        self.assertAllClose(output, [16.0, 16.0])
        output = model.evaluate(x, y, batch_size=batch_size, return_dict=True)
        self.assertTrue(isinstance(output, dict))
        self.assertIn("loss", output)
        self.assertIn("mean_squared_error", output)
        self.assertAllClose(output["mean_squared_error"], 16.0)

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_predict_flow(self, run_eagerly, jit_compile):
        # Test basic example
        model = ExampleModel(units=3)
        model.run_eagerly = run_eagerly
        model.jit_compile = jit_compile

        x = np.ones((100, 4))
        batch_size = 16
        outputs = model.predict(x, batch_size=batch_size)
        self.assertAllClose(outputs, 4 * np.ones((100, 3)))

        # Test with input/output structs
        model = StructModel(units=3)
        model.run_eagerly = run_eagerly
        model.jit_compile = jit_compile

        x = {
            "x_one": np.ones((100, 4)),
            "x_two": np.ones((100, 4)),
        }
        batch_size = 16
        outputs = model.predict(x, batch_size=batch_size)
        self.assertTrue(isinstance(outputs, dict))
        self.assertEqual(len(outputs), 2)
        self.assertAllClose(outputs["y_one"], 4 * np.ones((100, 3)))
        self.assertAllClose(outputs["y_two"], 4 * np.ones((100, 3)))

    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="`steps_per_execution` not implemented for torch yet",
    )
    def test_steps_per_execution_steps_count(self):
        class StepCount(Callback):
            def __init__(self):
                super().__init__()
                self.count = 0
                self.batches = [0, 3, 6]

            def on_batch_begin(self, batch, logs=None):
                assert batch == self.batches[self.count]
                self.count += 1

        x = np.ones((100, 4))
        y = np.ones((100, 1))
        batch_size = 16
        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="adam",
            steps_per_execution=3,
        )
        step_count = StepCount()
        model.fit(x=x, y=y, batch_size=16, callbacks=[step_count], verbose=0)
        self.assertEqual(step_count.count, 3)

        model_2 = ExampleModel(units=1)
        model_2.compile(loss="mse", optimizer="adam", steps_per_execution=1)
        model_2.fit(x=x, y=y, batch_size=batch_size, verbose=0)

        self.assertAllClose(model.get_weights(), model_2.get_weights())
        self.assertAllClose(
            model.predict(x, batch_size=batch_size),
            model_2.predict(x, batch_size=batch_size),
        )
        self.assertAllClose(model.evaluate(x, y), model_2.evaluate(x, y))

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="`steps_per_execution` not implemented for torch yet",
    )
    def test_steps_per_execution_steps_count_without_training(self):
        class StepCount(Callback):
            def __init__(self):
                super().__init__()
                self.test_count = 0
                self.predict_count = 0
                self.batches = [0, 3, 6]

            def on_test_batch_begin(self, batch, logs=None):
                assert batch == self.batches[self.test_count]
                self.test_count += 1

            def on_predict_batch_begin(self, batch, logs=None):
                assert batch == self.batches[self.predict_count]
                self.predict_count += 1

        x = np.ones((100, 4))
        y = np.ones((100, 1))
        batch_size = 16
        model = ExampleModel(units=1)
        model.compile(loss="mse", steps_per_execution=3)
        step_count = StepCount()
        model.predict(x, batch_size=batch_size, callbacks=[step_count])
        self.assertEqual(step_count.predict_count, 3)
        model.evaluate(x, y, batch_size=batch_size, callbacks=[step_count])
        self.assertEqual(step_count.test_count, 3)

    @pytest.mark.requires_trainable_backend
    def test_training_arg(self):
        model = TrainingTestingLayer()
        model.compile(optimizer="rmsprop", loss="mse")
        x = np.ones((128, 1))
        y = np.zeros((128, 1))
        history = model.fit(x, y, batch_size=32)
        self.assertAllClose(history.history["loss"], [1.0])
        val_loss = model.evaluate(x, y, batch_size=32)
        self.assertAllClose(val_loss, 0.0)
        preds = model.predict(x)
        self.assertAllClose(preds, np.zeros((128, 1)))

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_on_batch_methods(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        sw = np.arange(100).reshape((100,)).astype("float32") / 50.0

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        logs = model.train_on_batch(x, y)
        self.assertTrue(isinstance(logs, list))
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs[0], 16.0)

        logs = model.train_on_batch(x, y, return_dict=True)
        self.assertTrue(isinstance(logs, dict))
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs["loss"], 15.579)

        logs = model.test_on_batch(x, y)
        self.assertTrue(isinstance(logs, list))
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs[0], 15.173)

        logs = model.test_on_batch(x, y, return_dict=True)
        self.assertTrue(isinstance(logs, dict))
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs["loss"], 14.97)

        output = model.predict_on_batch(x)
        self.assertTrue(isinstance(output, np.ndarray))
        self.assertAllClose(output[0], np.array([3.789511, 3.789511, 3.789511]))

        # With sample weights
        logs = model.train_on_batch(x, y, sw)
        self.assertAlmostEqual(logs[0], 14.819)
        logs = model.test_on_batch(x, y, sw)
        self.assertAlmostEqual(logs[0], 14.595)
        output = model.predict_on_batch(x)
        self.assertAllClose(output[0], np.array([3.689468, 3.689468, 3.689468]))

        # With class weights
        logs = model.train_on_batch(x, y, class_weight={1: 0.3, 0: 0.2})
        self.assertAlmostEqual(logs[0], 12.899)

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_on_batch_methods_without_training(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))

        model.compile(
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        output = model.predict_on_batch(x)
        self.assertTrue(isinstance(output, np.ndarray))
        self.assertAllClose(output[0], np.array([4.0, 4.0, 4.0]))

        logs = model.test_on_batch(x, y)
        self.assertTrue(isinstance(logs, list))
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs[0], 16.0)

        logs = model.test_on_batch(x, y, return_dict=True)
        self.assertTrue(isinstance(logs, dict))
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs["loss"], 16.0)

    def test_nested_input_predict(self):
        # https://github.com/keras-team/keras-core/issues/325

        class TupleInputModel(keras_core.Model):
            def call(self, inputs):
                a, b = inputs
                return a + b

        model = TupleInputModel()
        x1, x2 = np.random.rand(2, 3, 4)
        out = model.predict((x1, x2))
        self.assertEqual(out.shape, (3, 4))

        class DictInputModel(keras_core.Model):
            def call(self, inputs):
                return inputs["a"] + inputs["b"]

        model = DictInputModel()
        x1, x2 = np.random.rand(2, 3, 4)
        out = model.predict({"a": x1, "b": x2})
        self.assertEqual(out.shape, (3, 4))

    @pytest.mark.requires_trainable_backend
    def test_callback_methods_keys(self):
        class CustomCallback(Callback):
            def on_train_begin(self, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_train_end(self, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == [
                    "loss",
                    "mean_absolute_error",
                    "val_loss",
                    "val_mean_absolute_error",
                ]

            def on_epoch_begin(self, epoch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_epoch_end(self, epoch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == [
                    "loss",
                    "mean_absolute_error",
                    "val_loss",
                    "val_mean_absolute_error",
                ]

            def on_test_begin(self, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_test_end(self, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == ["loss", "mean_absolute_error"]

            def on_predict_begin(self, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_predict_end(self, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_train_batch_begin(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_train_batch_end(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == ["loss", "mean_absolute_error"]

            def on_test_batch_begin(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_test_batch_end(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == ["loss", "mean_absolute_error"]

            def on_predict_batch_begin(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == []

            def on_predict_batch_end(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                assert keys == ["outputs"]

        model = ExampleModel(units=3)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        x = np.ones((16, 4))
        y = np.zeros((16, 3))
        x_test = np.ones((16, 4))
        y_test = np.zeros((16, 3))
        model.fit(
            x,
            y,
            callbacks=[CustomCallback()],
            batch_size=4,
            validation_data=(x_test, y_test),
        )
        model.evaluate(x_test, y_test, batch_size=4)
        model.predict(x_test, batch_size=4)

    @pytest.mark.requires_trainable_backend
    def test_internal_only_loss(self):
        class LossLayer(layers.Layer):
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x

        model = keras_core.Sequential(
            [
                layers.Dense(2),
                LossLayer(),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam")
        x = np.ones((16, 2))
        y = np.zeros((16, 1))
        model.fit(x, y, batch_size=4)

    def get_layer(self):
        class ExampleLayer(keras_core.Layer):
            def call(self, x):
                return x * 2

        return ExampleLayer

    def get_model(self):
        class ExampleModel(keras_core.Model):
            def call(self, x):
                return x * 2

        return ExampleModel

    def get_functional(self):
        ExampleLayer = self.get_layer()

        class ExampleFunctional(keras_core.Functional):
            def __init__(self, input_shape=(None,)):
                inputs = keras_core.Input(input_shape)
                outputs = ExampleLayer()(inputs)
                super().__init__(inputs=inputs, outputs=outputs)

        return ExampleFunctional

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "model",
                "model_class": "get_model",
            },
            {
                "testcase_name": "layer",
                "model_class": "get_layer",
            },
            {
                "testcase_name": "functional",
                "model_class": "get_functional",
            },
        ]
    )
    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        keras_core.backend.backend() != "tensorflow",
        reason="Only tensorflow supports raggeds",
    )
    def test_trainer_with_raggeds(self, model_class):
        from keras_core.utils.module_utils import tensorflow as tf

        def loss_fn(y, y_pred, sample_weight=None):
            return 0

        model = getattr(self, model_class)()()
        x = tf.ragged.constant([[1], [2, 3]])

        # test forward pass
        y = model(x)
        self.assertEqual(type(y), tf.RaggedTensor)

        # test training
        if model_class in ["get_model", "get_functional"]:
            model.compile(optimizer="adam", loss=loss_fn)
            model.fit(x, x)
            y = model.predict(x)
            self.assertEqual(type(y), tf.RaggedTensor)

        # test if everything works with the sequential model
        model = keras_core.Sequential([model])
        model.compile(optimizer="adam", loss=loss_fn)
        model.fit(x, x)
        y = model.predict(x)
        self.assertEqual(type(y), tf.RaggedTensor)

    def test_predict_dropout(self):
        # Test that `predict` with a dropout op
        # has nondeterministic behavior across batches.

        inputs = layers.Input((20,))
        outputs = layers.Dropout(0.5, seed=1337)(inputs, training=True)
        model = keras_core.Model(inputs, outputs)
        out1 = model.predict(np.ones((4, 20)), batch_size=2)
        self.assertGreater(5, np.sum(np.abs(out1[:2, :] - out1[2:4, :])))

        out2 = model.predict_on_batch(np.ones((2, 20)))
        out3 = model.predict_on_batch(np.ones((2, 20)))
        self.assertGreater(5, np.sum(np.abs(out2 - out3)))

    @pytest.mark.requires_trainable_backend
    def test_recompile(self):
        inputs = layers.Input((2,))
        outputs = layers.Dense(3)(inputs)
        model = keras_core.Model(inputs, outputs)
        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
        history_1 = model.fit(np.ones((3, 2)), np.ones((3, 3))).history
        eval_out_1 = model.evaluate(
            np.ones((3, 2)), np.ones((3, 3)), return_dict=True
        )
        model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
        history_2 = model.fit(np.ones((3, 2)), np.ones((3, 3))).history
        eval_out_2 = model.evaluate(
            np.ones((3, 2)), np.ones((3, 3)), return_dict=True
        )
        self.assertEqual(
            sorted(list(history_1.keys())), ["loss", "mean_squared_error"]
        )
        self.assertEqual(
            sorted(list(eval_out_1.keys())), ["loss", "mean_squared_error"]
        )
        self.assertEqual(
            sorted(list(history_2.keys())), ["loss", "mean_absolute_error"]
        )
        self.assertEqual(
            sorted(list(eval_out_2.keys())), ["loss", "mean_absolute_error"]
        )

    @pytest.mark.requires_trainable_backend
    def test_nested_inputs(self):
        model = ListModel(units=2)
        out = model([np.ones((3, 2)), np.ones((3, 3))])
        self.assertEqual(tuple(out.shape), (3, 2))
        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
        history = model.fit(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        ).history
        self.assertAllClose(history["loss"], 16.0)
        train_out = model.train_on_batch(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        )
        self.assertAllClose(train_out[0], 15.2200)
        eval_out = model.evaluate(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        )
        self.assertAllClose(eval_out[0], 13.0321)
        eval_out = model.test_on_batch(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        )
        self.assertAllClose(eval_out[0], 13.0321)
        predict_out = model.predict([np.ones((3, 2)), np.ones((3, 3))])
        self.assertEqual(predict_out.shape, (3, 2))
        predict_out = model.predict_on_batch([np.ones((3, 2)), np.ones((3, 3))])
        self.assertEqual(predict_out.shape, (3, 2))

    @pytest.mark.requires_trainable_backend
    def test_validation_data_infinite_generator(self):
        # Test that you can pass an infinite generator to `validation_data`
        # arg of fit() as well as a `validation_steps` argument and that
        # validation only runs for the correct number of steps.
        inputs = layers.Input((2,))
        outputs = layers.Dense(3)(inputs)
        model = keras_core.Model(inputs, outputs)
        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])

        class Recorder(keras_core.callbacks.Callback):
            def __init__(self):
                self.train_counter = 0
                self.val_counter = 0

            def on_train_batch_end(self, *args, **kwargs):
                self.train_counter += 1

            def on_test_batch_end(self, *args, **kwargs):
                self.val_counter += 1

        def infinite_gen():
            while True:
                yield np.ones((2, 2)), np.ones((2, 3))

        recorder = Recorder()

        model.fit(
            infinite_gen(),
            validation_data=infinite_gen(),
            steps_per_epoch=3,
            validation_steps=4,
            epochs=1,
            shuffle=False,
            callbacks=[recorder],
        )
        self.assertEqual(recorder.train_counter, 3)
        self.assertEqual(recorder.val_counter, 4)
