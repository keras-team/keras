import numpy as np

from keras_core import backend
from keras_core import metrics as losses_module
from keras_core import metrics as metrics_module
from keras_core import testing
from keras_core.trainers.compile_utils import CompileLoss
from keras_core.trainers.compile_utils import CompileMetrics


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
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 2)
        self.assertAllClose(result["mean_squared_error"], 0.055833336)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0725)

        compile_metrics.reset_state()
        result = compile_metrics.result()
        self.assertTrue(isinstance(result, dict))
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
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["mean_squared_error"], 0.055833336)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0725)

        compile_metrics.reset_state()
        result = compile_metrics.result()
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["mean_squared_error"], 0.0)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0)

    def test_dict_output_case(self):
        compile_metrics = CompileMetrics(
            metrics={
                "output_1": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
                "output_2": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
            },
            weighted_metrics={
                "output_1": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
                ],
                "output_2": [
                    metrics_module.MeanSquaredError(),
                    metrics_module.MeanSquaredError(),
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
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["mean_squared_error"], 0.055833336)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0725)

        compile_metrics.reset_state()
        result = compile_metrics.result()
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 8)
        self.assertAllClose(result["mean_squared_error"], 0.0)
        self.assertAllClose(result["weighted_mean_squared_error"], 0.0)

    def test_name_conversions(self):
        compile_metrics = CompileMetrics(
            metrics=["acc", "accuracy"],
            weighted_metrics=[],
        )
        y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_pred = np.array([[0.4, 0.1], [0.2, 0.6], [0.6, 0.1]])
        compile_metrics.build(y_true, y_pred)
        compile_metrics.update_state(y_true, y_pred, sample_weight=None)
        result = compile_metrics.result()
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 2)
        self.assertAllClose(result["acc"], 0.333333)
        self.assertAllClose(result["accuracy"], 0.333333)


class TestCompileLoss(testing.TestCase):
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
