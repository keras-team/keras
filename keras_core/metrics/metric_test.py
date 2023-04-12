from keras_core import testing
from keras_core import initializers
from keras_core import operations as ops
from keras_core import backend
from keras_core.metrics.metric import Metric
import numpy as np


class ExampleMetric(Metric):
    def __init__(self, name="mean_square_error", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.sum = self.add_variable(name="sum", initializer=initializers.Zeros())
        self.total = self.add_variable(
            name="total", initializer=initializers.Zeros(), dtype="int32"
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
            print(b * batch_size, (b + 1) * batch_size)
            y_true_batch = y_true[b * batch_size : (b + 1) * batch_size]
            y_pred_batch = y_pred[b * batch_size : (b + 1) * batch_size]
            metric.update_state(y_true_batch, y_pred_batch)

        self.assertAllClose(metric.total, 20)
        result = metric.result()
        self.assertAllClose(result, np.sum((y_true - y_pred) ** 2) / num_samples)
        metric.reset_state()
        self.assertEqual(metric.result(), 0.0)

    def test_variable_tracking(self):
        # In list
        metric = ExampleMetric(name="mse")
        metric.more_vars = [backend.Variable(0.0), backend.Variable(1.0)]
        self.assertEqual(len(metric.variables), 4)

        # In dict
        metric = ExampleMetric(name="mse")
        metric.more_vars = {"a": backend.Variable(0.0), "b": backend.Variable(1.0)}
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
        # TODO
        pass
