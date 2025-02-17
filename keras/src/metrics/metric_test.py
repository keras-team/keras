import pickle

import numpy as np

from keras.src import backend
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import testing
from keras.src.metrics.metric import Metric


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
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
        sum = ops.sum((y_true - y_pred) ** 2)
        self.sum.assign(self.sum + sum)
        batch_size = ops.shape(y_true)[0]
        self.total.assign(self.total + batch_size)

    def result(self):
        _sum = ops.cast(self.sum, dtype=self.dtype)
        _total = ops.cast(self.total, dtype=self.dtype)
        _epsilon = ops.cast(backend.epsilon(), dtype=self.dtype)
        return _sum / (_total + _epsilon)

    def reset_state(self):
        self.sum.assign(0)
        self.total.assign(0)


class MetricTest(testing.TestCase):
    def setUp(self):
        self._global_dtype_policy = dtype_policies.dtype_policy.dtype_policy()
        self._floatx = backend.floatx()
        return super().setUp()

    def tearDown(self):
        dtype_policies.dtype_policy.set_dtype_policy(self._global_dtype_policy)
        backend.set_floatx(self._floatx)
        return super().tearDown()

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

    def test_stateless_reset_state(self):
        metric = ExampleMetric(name="mse")
        num_samples = 20
        y_true = np.random.random((num_samples, 3))
        y_pred = np.random.random((num_samples, 3))
        metric.update_state(y_true, y_pred)
        vars = metric.stateless_reset_state()
        self.assertLen(vars, 2)
        self.assertEqual(vars[0], 0)
        self.assertEqual(vars[1], 0)

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

        # Two levels deep
        metric = ExampleMetric(name="mse")
        metric.submetric = ExampleMetric(name="submse")
        metric.submetric.submetric = ExampleMetric(name="subsubmse")
        self.assertEqual(len(metric.variables), 6)

    def test_serialization(self):
        self.run_class_serialization_test(
            ExampleMetric(name="mse"),
            custom_objects={"ExampleMetric": ExampleMetric},
        )

    def test_pickle(self):
        metric = metrics_module.get("mse")
        reloaded = pickle.loads(pickle.dumps(metric))
        self.assertIsInstance(reloaded, metrics_module.MeanSquaredError)

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

    def test_dtype_arg(self):
        metric = ExampleMetric(name="mse", dtype="float16")
        self.assertEqual(metric.name, "mse")
        self.assertEqual(len(metric.variables), 2)

        num_samples = 10
        y_true = np.random.random((num_samples, 3))
        y_pred = np.random.random((num_samples, 3))
        metric.update_state(y_true, y_pred)
        result = metric.result()
        self.assertAllClose(
            result, np.sum((y_true - y_pred) ** 2) / num_samples, atol=1e-3
        )
        self.assertDType(result, "float16")

        # Test DTypePolicy for `dtype` argument
        metric = ExampleMetric(
            dtype=dtype_policies.DTypePolicy("mixed_float16")
        )
        metric.update_state(y_true, y_pred)
        metric.update_state(y_true, y_pred)
        result = metric.result()
        self.assertAllClose(
            result, np.sum((y_true - y_pred) ** 2) / num_samples, atol=1e-3
        )
        self.assertDType(result, "float16")

        # `dtype` setter should raise AttributeError
        with self.assertRaises(AttributeError):
            metric.dtype = "bfloat16"

    def test_default_dtype(self):
        y_true = np.random.random((10, 3))
        y_pred = np.random.random((10, 3))

        # Defaults to `keras.config.floatx()` not global `dtype_policy`
        dtype_policies.dtype_policy.set_dtype_policy("mixed_float16")
        metric = ExampleMetric()
        metric.update_state(y_true, y_pred)
        result = metric.result()
        self.assertDType(result, "float32")

        backend.set_floatx("float16")
        metric = ExampleMetric()
        metric.update_state(y_true, y_pred)
        result = metric.result()
        self.assertDType(result, backend.floatx())
