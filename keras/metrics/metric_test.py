import numpy as np

from keras import backend
from keras import initializers
from keras import metrics as metrics_module
from keras import ops
from keras import testing
from keras.metrics.metric import Metric


class ExampleMetric(Metric):
    def __init__(self, name="mean_square_error", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.sum = self.add_variable(
            name="sum", shape=(), initializer=initializers.Zeros()
        )
        self.total = self.add_variable(
            name="total",
            shape=(),
            initializer=initializers.Zeros(),
            dtype="int32",
        )

    def update_state(self, y_true, y_pred):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        sum = ops.sum((y_true - y_pred) ** 2)
        self.sum.assign(self.sum + sum)
        batch_size = ops.shape(y_true)[0]
        self.total.assign(self.total + batch_size)

    def result(self):
        return self.sum / (ops.cast(self.total, dtype="float32") + 1e-7)

    def reset_state(self):
        self.sum.assign(0.0)
        self.total.assign(0)


class MetricTest(testing.TestCase):
    def test_end_to_end_flow(self):
        metric = ExampleMetric(name="mse")
        self.assertEqual(metric.name, "mse")
        self.assertEqual(len(metric.variables), 2)

        num_samples = 20
        y_true = np.random.random((num_samples, 3))
        y_pred = np.random.random((num_samples, 3))
        batch_size = 8
        for b in range(0, num_samples // batch_size + 1):
            y_true_batch = y_true[b * batch_size : (b + 1) * batch_size]
            y_pred_batch = y_pred[b * batch_size : (b + 1) * batch_size]
            metric.update_state(y_true_batch, y_pred_batch)

        self.assertAllClose(metric.total, 20)
        result = metric.result()
        self.assertAllClose(
            result, np.sum((y_true - y_pred) ** 2) / num_samples
        )
        metric.reset_state()
        self.assertEqual(metric.result(), 0.0)

    def test_stateless_update_state(self):
        metric = ExampleMetric(name="mse")
        self.assertEqual(len(metric.variables), 2)
        original_variable_values = (
            metric.variables[0].numpy(),
            metric.variables[1].numpy(),
        )

        num_samples = 20
        y_true = np.random.random((num_samples, 3))
        y_pred = np.random.random((num_samples, 3))
        batch_size = 8
        metric_variables = metric.variables
        for b in range(0, num_samples // batch_size + 1):
            y_true_batch = y_true[b * batch_size : (b + 1) * batch_size]
            y_pred_batch = y_pred[b * batch_size : (b + 1) * batch_size]
            metric_variables = metric.stateless_update_state(
                metric_variables, y_true_batch, y_pred_batch
            )

        self.assertAllClose(metric.variables[0], original_variable_values[0])
        self.assertAllClose(metric.variables[1], original_variable_values[1])
        metric.variables[0].assign(metric_variables[0])
        metric.variables[1].assign(metric_variables[1])
        self.assertAllClose(metric.total, 20)
        result = metric.result()
        self.assertAllClose(
            result, np.sum((y_true - y_pred) ** 2) / num_samples
        )

        if backend.backend() == "jax":
            # Check no side effects.
            import jax

            @jax.jit
            def update(metric_variables, y_true_batch, y_pred_batch):
                metric_variables = metric.stateless_update_state(
                    metric_variables, y_true_batch, y_pred_batch
                )

            update(metric_variables, y_true_batch, y_pred_batch)

    def test_stateless_result(self):
        metric = ExampleMetric(name="mse")
        res = metric.stateless_result([ops.ones(()) * 12, ops.ones(()) * 3])
        self.assertAllClose(res, 4.0)

    def test_variable_tracking(self):
        # In list
        metric = ExampleMetric(name="mse")
        metric.more_vars = [backend.Variable(0.0), backend.Variable(1.0)]
        self.assertEqual(len(metric.variables), 4)

        # In dict
        metric = ExampleMetric(name="mse")
        metric.more_vars = {
            "a": backend.Variable(0.0),
            "b": backend.Variable(1.0),
        }
        self.assertEqual(len(metric.variables), 4)

        # In nested structured
        metric = ExampleMetric(name="mse")
        metric.more_vars = {"a": [backend.Variable(0.0), backend.Variable(1.0)]}
        self.assertEqual(len(metric.variables), 4)

    def test_submetric_tracking(self):
        # Plain attr
        metric = ExampleMetric(name="mse")
        metric.submetric = ExampleMetric(name="submse")
        self.assertEqual(len(metric.variables), 4)

        # In list
        metric = ExampleMetric(name="mse")
        metric.submetrics = [
            ExampleMetric(name="submse1"),
            ExampleMetric(name="submse2"),
        ]
        self.assertEqual(len(metric.variables), 6)

        # In dict
        metric = ExampleMetric(name="mse")
        metric.submetrics = {
            "1": ExampleMetric(name="submse1"),
            "2": ExampleMetric(name="submse2"),
        }
        self.assertEqual(len(metric.variables), 6)

    def test_serialization(self):
        self.run_class_serialization_test(
            ExampleMetric(name="mse"),
            custom_objects={"ExampleMetric": ExampleMetric},
        )

    def test_get_method(self):
        metric = metrics_module.get("mse")
        self.assertIsInstance(metric, metrics_module.MeanSquaredError)

        metric = metrics_module.get("mean_squared_error")
        self.assertIsInstance(metric, metrics_module.MeanSquaredError)

        metric = metrics_module.get("categorical_accuracy")
        self.assertIsInstance(metric, metrics_module.CategoricalAccuracy)

        metric = metrics_module.get(None)
        self.assertEqual(metric, None)

        with self.assertRaises(ValueError):
            metrics_module.get("typo")
