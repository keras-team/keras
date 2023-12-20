import numpy as np
from absl.testing import parameterized

from keras import backend
from keras import metrics as losses_module
from keras import metrics as metrics_module
from keras import ops
from keras import testing
from keras.trainers.compile_utils import CompileLoss
from keras.trainers.compile_utils import CompileMetrics


class TestCompileMetrics(testing.TestCase):
    def test_single_output_case(self):
        compile_metrics = CompileMetrics(
            metrics=[metrics_module.MeanSquaredError()],
            weighted_metrics=[metrics_module.MeanSquaredError()],
        )
        # Test symbolic build
        y_true, y_pred = backend.KerasTensor((3, 4)), backend.KerasTensor(
            (3, 4)
        )
        compile_metrics.build(y_true, y_pred)
        # Test eager build
        y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_pred = np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]])
        sample_weight = np.array([1, 0.0, 1])
        compile_metrics.build(y_true, y_pred)

        # Test update / result / reset flow
        compile_metrics.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        y_pred = np.array([[0.3, 0.2], [0.1, 0.4], [0.2, 0.3]])
        compile_metrics.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertAllClose(result["mean_squared_error"], 0.055833336)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0725)

        compile_metrics.reset_state()
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertAllClose(result["mean_squared_error"], 0.0)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0)

    def test_list_output_case(self):
        compile_metrics = CompileMetrics(
            metrics=[
                [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
                [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
            ],
            weighted_metrics=[
                [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
                [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
            ],
        )
        # Test symbolic build
        y_true = [backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))]
        y_pred = [backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))]
        compile_metrics.build(y_true, y_pred)
        self.assertEqual(len(compile_metrics.metrics), 8)

        # Test eager build
        y_true = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        ]
        y_pred = [
            np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]]),
            np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]]),
        ]
        sample_weight = np.array([1, 0.0, 1])
        compile_metrics.build(y_true, y_pred)
        self.assertEqual(len(compile_metrics.metrics), 8)

        # Test update / result / reset flow
        compile_metrics.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        y_pred = [
            np.array([[0.3, 0.2], [0.1, 0.4], [0.2, 0.3]]),
            np.array([[0.3, 0.2], [0.1, 0.4], [0.2, 0.3]]),
        ]
        compile_metrics.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["mean_squared_error"], 0.055833336)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0725)

        compile_metrics.reset_state()
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["mean_squared_error"], 0.0)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0)

    def test_dict_output_case(self):
        compile_metrics = CompileMetrics(
            metrics={
                "output_1": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(name="mse"),
                ],
                "output_2": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(name="mse"),
                ],
            },
            weighted_metrics={
                "output_1": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(name="mse"),
                ],
                "output_2": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(name="mse"),
                ],
            },
        )
        # Test symbolic build
        y_true = {
            "output_1": backend.KerasTensor((3, 4)),
            "output_2": backend.KerasTensor((3, 4)),
        }
        y_pred = {
            "output_1": backend.KerasTensor((3, 4)),
            "output_2": backend.KerasTensor((3, 4)),
        }
        compile_metrics.build(y_true, y_pred)
        # Test eager build
        y_true = {
            "output_1": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "output_2": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        }
        y_pred = {
            "output_1": np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]]),
            "output_2": np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]]),
        }
        sample_weight = np.array([1, 0.0, 1])
        compile_metrics.build(y_true, y_pred)

        # Test update / result / reset flow
        compile_metrics.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        y_pred = {
            "output_1": np.array([[0.3, 0.2], [0.1, 0.4], [0.2, 0.3]]),
            "output_2": np.array([[0.3, 0.2], [0.1, 0.4], [0.2, 0.3]]),
        }
        compile_metrics.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 8)
        # Result values obtained from `tf.keras`
        # m = tf.keras.metrics.MeanSquaredError()
        # m.update_state(y_true, y_pred1, sample_weight=weight)
        # m.update_state(y_true, y_pred2, sample_weight=weight)
        # m.result().numpy()
        self.assertAllClose(result["output_1_mean_squared_error"], 0.055833336)
        self.assertAllClose(result["output_2_mean_squared_error"], 0.055833336)
        self.assertAllClose(result["output_1_mse"], 0.055833336)
        self.assertAllClose(result["output_2_mse"], 0.055833336)
        self.assertAllClose(
            result["output_1_weighted_mean_squared_error"], 0.0725
        )
        self.assertAllClose(
            result["output_2_weighted_mean_squared_error"], 0.0725
        )
        self.assertAllClose(result["output_1_weighted_mse"], 0.0725)
        self.assertAllClose(result["output_2_weighted_mse"], 0.0725)

        compile_metrics.reset_state()
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["output_1_mean_squared_error"], 0.0)
        self.assertAllClose(result["output_2_mean_squared_error"], 0.0)
        self.assertAllClose(result["output_1_weighted_mean_squared_error"], 0.0)
        self.assertAllClose(result["output_2_weighted_mean_squared_error"], 0.0)

    def test_name_conversions(self):
        compile_metrics = CompileMetrics(
            metrics=["acc", "accuracy", "mse"],
            weighted_metrics=[],
        )
        y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_pred = np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]])
        compile_metrics.build(y_true, y_pred)
        compile_metrics.update_state(y_true, y_pred, sample_weight=None)
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertAllClose(result["acc"], 0.333333)
        self.assertAllClose(result["accuracy"], 0.333333)
        self.assertTrue("mse" in result)

    def test_custom_metric_function(self):
        def my_custom_metric(y_true, y_pred):
            return ops.mean(ops.square(y_true - y_pred), axis=-1)

        compile_metrics = CompileMetrics(
            metrics=[my_custom_metric],
            weighted_metrics=[],
        )
        y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_pred = np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]])
        compile_metrics.build(y_true, y_pred)
        compile_metrics.update_state(y_true, y_pred, sample_weight=None)
        result = compile_metrics.result()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertTrue("my_custom_metric" in result)


class TestCompileLoss(testing.TestCase, parameterized.TestCase):
    def test_single_output_case(self):
        compile_loss = CompileLoss(
            loss=losses_module.MeanSquaredError(),
        )
        # Test symbolic build
        y_true, y_pred = backend.KerasTensor((3, 4)), backend.KerasTensor(
            (3, 4)
        )
        compile_loss.build(y_true, y_pred)
        # Test eager build
        y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_pred = np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]])
        compile_loss.build(y_true, y_pred)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 0.068333, atol=1e-5)

    def test_single_output_case_with_crossentropy_loss(self):
        compile_loss = CompileLoss(loss="crossentropy")

        # Test symbolic build
        y_true, y_pred = backend.KerasTensor((3, 4)), backend.KerasTensor(
            (3, 4)
        )
        compile_loss.build(y_true, y_pred)
        # Test eager build
        y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_pred = np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]])
        compile_loss.build(y_true, y_pred)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 0.706595, atol=1e-5)

    @parameterized.parameters(True, False)
    def test_list_output_case(self, broadcast):
        if broadcast:
            # Test broadcasting single loss to all outputs
            compile_loss = CompileLoss(
                loss="mse",
            )
        else:
            compile_loss = CompileLoss(
                loss=["mse", "mse"],
            )
        # Test symbolic build
        y_true = [backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))]
        y_pred = [backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))]
        compile_loss.build(y_true, y_pred)
        # Test eager build
        y_true = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        ]
        y_pred = [
            np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
            np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        ]
        compile_loss.build(y_true, y_pred)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 0.953333, atol=1e-5)

    @parameterized.parameters(True, False)
    def test_dict_output_case(self, broadcast):
        if broadcast:
            # Test broadcasting single loss to all outputs
            compile_loss = CompileLoss(
                loss="mse",
            )
        else:
            compile_loss = CompileLoss(
                loss={"a": "mse", "b": "mse"},
            )
        # Test symbolic build
        y_true = {
            "a": backend.KerasTensor((3, 4)),
            "b": backend.KerasTensor((3, 4)),
        }
        y_pred = {
            "a": backend.KerasTensor((3, 4)),
            "b": backend.KerasTensor((3, 4)),
        }
        compile_loss.build(y_true, y_pred)
        # Test eager build
        y_true = {
            "a": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        sample_weight = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([3.0, 2.0, 1.0]),
        }
        compile_loss.build(y_true, y_pred)
        value = compile_loss(y_true, y_pred, sample_weight)
        self.assertAllClose(value, 1.266666, atol=1e-5)

    def test_list_loss_dict_data(self):
        compile_loss = CompileLoss(loss=["mse", "mae"], output_names=["b", "a"])
        y_true = [backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))]
        y_pred = [backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))]
        compile_loss.build(y_true, y_pred)
        y_true = {
            "a": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 1.07666, atol=1e-5)
