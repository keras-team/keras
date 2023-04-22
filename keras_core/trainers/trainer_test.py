import numpy as np

from keras_core import backend
from keras_core import initializers
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers
from keras_core import testing

if backend.backend() == "jax":
    from keras_core.backend.jax.trainer import JAXTrainer as Trainer
else:
    from keras_core.backend.tensorflow.trainer import (
        TensorFlowTrainer as Trainer,
    )


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


class OutputStructModel(layers.Layer, Trainer):
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
            "y_one": self.dense_1(x),
            "y_two": self.dense_2(x),
        }


class TestTrainer(testing.TestCase):
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

    def _test_fit_flow(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16
        epochs = 3

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        history = model.fit(x, y, batch_size=batch_size, epochs=epochs)
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("mean_squared_error", history)
        self.assertAllClose(
            history["mean_squared_error"], [13.938, 9.547, 6.539], atol=1e-2
        )

    def test_fit_flow_eager(self):
        self._test_fit_flow(run_eagerly=True, jit_compile=False)

    def test_fit_flow_graph_fn(self):
        self._test_fit_flow(run_eagerly=False, jit_compile=False)

    def test_fit_flow_jit(self):
        self._test_fit_flow(run_eagerly=False, jit_compile=True)

    def _test_evaluate_flow(self, run_eagerly, jit_compile):
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

    def test_evaluate_flow_eager(self):
        self._test_evaluate_flow(run_eagerly=True, jit_compile=False)

    def test_evaluate_flow_graph_fn(self):
        self._test_evaluate_flow(run_eagerly=False, jit_compile=False)

    def test_evaluate_flow_jit(self):
        self._test_evaluate_flow(run_eagerly=False, jit_compile=True)

    def _test_predict_flow(self, run_eagerly, jit_compile):
        # Test basic example
        model = ExampleModel(units=3)
        model.run_eagerly = run_eagerly
        model.jit_compile = jit_compile

        x = np.ones((100, 4))
        batch_size = 16
        outputs = model.predict(x, batch_size=batch_size)
        self.assertAllClose(outputs, 4 * np.ones((100, 3)))

        # Test with output struct
        model = OutputStructModel(units=3)
        model.run_eagerly = run_eagerly
        model.jit_compile = jit_compile

        x = np.ones((100, 4))
        batch_size = 16
        outputs = model.predict(x, batch_size=batch_size)
        self.assertTrue(isinstance(outputs, dict))
        self.assertEqual(len(outputs), 2)
        self.assertAllClose(outputs["y_one"], 4 * np.ones((100, 3)))
        self.assertAllClose(outputs["y_two"], 4 * np.ones((100, 3)))

    def test_predicte_flow_eager(self):
        self._test_predict_flow(run_eagerly=True, jit_compile=False)

    def test_predict_flow_graph_fn(self):
        self._test_predict_flow(run_eagerly=False, jit_compile=False)

    def test_predict_flow_jit(self):
        self._test_predict_flow(run_eagerly=False, jit_compile=True)
