from collections import namedtuple

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import metrics as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.trainers.compile_utils import CompileLoss
from keras.src.trainers.compile_utils import CompileMetrics


class TestCompileMetrics(testing.TestCase):
    def test_single_output_case(self):
        compile_metrics = CompileMetrics(
            metrics=[metrics_module.MeanSquaredError()],
            weighted_metrics=[metrics_module.MeanSquaredError()],
        )
        # Test symbolic build
        y_true = backend.KerasTensor((3, 4))
        y_pred = backend.KerasTensor((3, 4))
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

    def test_dict_outputs_ignore_mismatched_output_names(self):
        """Tests that when output_names does not match dict keys, the correct
        keys are used."""

        # output_names represent internal op names that do not match dict keys.
        compile_metrics = CompileMetrics(
            metrics={
                "a": metrics_module.MeanSquaredError(),
                "b": metrics_module.MeanSquaredError(),
            },
            weighted_metrics=None,
            output_names=["dense", "dense_1"],
        )

        # Symbolic build with dict outputs keyed by user-facing names.
        y_true = {
            "a": backend.KerasTensor((3, 2)),
            "b": backend.KerasTensor((3, 2)),
        }
        y_pred = {
            "a": backend.KerasTensor((3, 2)),
            "b": backend.KerasTensor((3, 2)),
        }

        # The build method should correctly map metrics for outputs 'a' and 'b',
        # even when the op names do not match.
        compile_metrics.build(y_true, y_pred)

        # Make the two outputs produce different MSEs to verify mapping.
        y_true = {
            "a": np.zeros((3, 2), dtype="float32"),
            "b": np.zeros((3, 2), dtype="float32"),
        }
        y_pred = {
            # MSE(a) = 0.0
            "a": np.zeros((3, 2), dtype="float32"),
            # MSE(b) = 1.0
            "b": np.ones((3, 2), dtype="float32"),
        }
        compile_metrics.update_state(y_true, y_pred)

        result = compile_metrics.result()
        self.assertIsInstance(result, dict)

        # Should expose metrics under the dict keys ('a', 'b'),
        # and not the internal names.
        self.assertIn("a_mean_squared_error", result)
        self.assertIn("b_mean_squared_error", result)
        self.assertAllClose(result["a_mean_squared_error"], 0.0)
        self.assertAllClose(result["b_mean_squared_error"], 1.0, atol=1e-6)


class TestCompileLoss(testing.TestCase):
    def test_single_output_case(self):
        compile_loss = CompileLoss(
            loss=losses_module.MeanSquaredError(),
        )
        # Test symbolic build
        y_true = backend.KerasTensor((3, 4))
        y_pred = backend.KerasTensor((3, 4))
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
        y_true = backend.KerasTensor((3, 4))
        y_pred = backend.KerasTensor((3, 4))
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

    def test_struct_loss(self):
        y_true = {
            "a": {
                "c": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                "d": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            },
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": {
                "c": np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
                "d": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
            },
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        loss = {"a": {"c": "mse", "d": "mae"}}
        compile_loss = CompileLoss(loss=loss, output_names=["c", "d", "b"])
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_pred
        )
        compile_loss.build(y_true_symb, y_pred_symb)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 1.07666, atol=1e-5)

    def test_struct_loss_valid_weights(self):
        y_true = {
            "a": np.array([1, 2]),
            "b": np.array([1, 2]),
        }
        y_pred = {
            "a": np.array([3, 4]),
            "b": np.array([3, 4]),
        }
        loss = {"a": "mse", "b": "mse"}
        compile_loss = CompileLoss(
            loss=loss,
            output_names=["a", "b"],
            loss_weights={
                "a": np.ones((2,)),
                "b": np.zeros((2,)),
            },
        )
        compile_loss.build(y_true, y_pred)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 4)

        # Metrics still report unweighted loss.
        a_loss_mean, b_loss_mean = compile_loss.metrics
        self.assertEqual(a_loss_mean.result(), 4)
        self.assertEqual(b_loss_mean.result(), 4)

    def test_struct_loss_invalid_weights(self):
        y_true = {
            "a": {
                "c": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                "d": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            },
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": {
                "c": np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
                "d": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
            },
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        loss = {"a": {"c": "mse", "d": "mae"}}
        compile_loss = CompileLoss(
            loss=loss, output_names=["c", "d", "b"], loss_weights=[1]
        )
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_pred
        )
        with self.assertRaisesRegex(
            ValueError, "must match the number of losses"
        ):
            compile_loss.build(y_true_symb, y_pred_symb)

    def test_struct_loss_indice_path(self):
        y_true = {
            "a": (
                np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            ),
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": (
                np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
                np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
            ),
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        loss = {"a": ["mse", "mae"]}
        compile_loss = CompileLoss(loss=loss, output_names=["c", "d", "b"])
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_pred
        )
        compile_loss.build(y_true_symb, y_pred_symb)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 1.07666, atol=1e-5)

    def test_struct_loss_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])
        y_true = {
            "a": Point(
                np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            ),
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": Point(
                np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
                np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
            ),
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        loss = {"a": Point("mse", "mae")}
        compile_loss = CompileLoss(loss=loss, output_names=["c", "d", "b"])
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_pred
        )
        compile_loss.build(y_true_symb, y_pred_symb)
        value = compile_loss(y_true, y_pred)
        self.assertAllClose(value, 1.07666, atol=1e-5)

    def test_struct_loss_invalid_path(self):
        y_true = {
            "a": {
                "c": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                "d": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            },
            "b": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        }
        y_pred = {
            "a": {
                "c": np.array([[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]]),
                "d": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
            },
            "b": np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        }
        loss = {"a": {"c": "mse"}, "b": {"d": "mae"}}
        compile_loss = CompileLoss(loss=loss, output_names=["c", "d", "b"])
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((3, 4)), y_pred
        )
        with self.assertRaisesRegex(
            KeyError, "can't be found in the model's output"
        ):
            compile_loss.build(y_true_symb, y_pred_symb)

    def test_different_container_types(self):
        y1, y2, y3 = np.array([[1]]), np.array([[2]]), np.array([[3]])
        y_true = ([{"a": y1}, {"b": ([y2], y3)}],)
        y_pred = [({"a": y1}, {"b": [(y2,), y3]})]
        loss = "mse"
        compile_loss = CompileLoss(loss=loss, output_names=["a", "b", "c"])
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((1, 1)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((1, 1)), y_pred
        )
        compile_loss.build(y_true_symb, y_pred_symb)
        compile_loss(y_true, y_pred)

    def test_structure_mismatch(self):
        y_true = [np.array([[1]]), np.array([[1]])]
        y_pred = [np.array([[1]]), np.array([[1]])]
        loss = ["mse", "mse"]
        compile_loss = CompileLoss(loss=loss, output_names=["a", "b"])
        y_true_symb = tree.map_structure(
            lambda _: backend.KerasTensor((1, 1)), y_true
        )
        y_pred_symb = tree.map_structure(
            lambda _: backend.KerasTensor((1, 1)), y_pred
        )
        compile_loss.build(y_true_symb, y_pred_symb)
        with self.assertRaisesRegex(
            ValueError, "y_true and y_pred have different structures."
        ):
            wrong_struc_y_true = [np.array([[1]])]
            compile_loss(wrong_struc_y_true, y_pred)

    @parameterized.parameters(
        ["mse", None, None],
        [None, "mse", None],
        [None, None, "mse"],
        [None, "mse", "mse"],
        ["mse", None, "mse"],
    )
    def test_y_true_partial_y_pred_span(self, *loss_conf):
        loss_conf = list(loss_conf)
        ones = np.ones((320, 3))
        zeros = np.zeros((320, 3))
        twos = np.ones((320, 3)) * 2
        y_pred = [zeros, ones, twos]
        y_true = [y for y, loss in zip(y_pred, loss_conf) if loss is not None]
        y_true = y_true[0] if len(y_true) == 1 else y_true
        compile_loss = CompileLoss(loss=loss_conf, output_names=["a", "b", "c"])
        # build call
        compile_loss(y_true, y_pred)
        # built call
        loss = compile_loss(y_true, y_pred)
        self.assertEqual(loss, 0.0)
