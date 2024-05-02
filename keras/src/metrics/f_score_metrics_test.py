import numpy as np
from absl.testing import parameterized

from keras.src import testing
from keras.src.metrics import f_score_metrics


class FBetaScoreTest(parameterized.TestCase, testing.TestCase):
    def _run_test(
        self,
        y_true,
        y_pred,
        sample_weights,
        average,
        beta,
        threshold,
        reference_result,
    ):
        fbeta = f_score_metrics.FBetaScore(
            average, beta, threshold, dtype="float32"
        )
        fbeta.update_state(y_true, y_pred, sample_weights)
        result = fbeta.result()
        self.assertAllClose(result, reference_result, atol=1e-6)

    def test_config(self):
        fbeta_obj = f_score_metrics.FBetaScore(
            beta=0.5, threshold=0.3, average=None, dtype="float32"
        )
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.average, None)
        self.assertEqual(fbeta_obj.threshold, 0.3)
        self.assertEqual(fbeta_obj.dtype, "float32")

        # Check save and restore config
        fbeta_obj2 = f_score_metrics.FBetaScore.from_config(
            fbeta_obj.get_config()
        )
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.average, None)
        self.assertEqual(fbeta_obj2.threshold, 0.3)
        self.assertEqual(fbeta_obj2.dtype, "float32")

    @parameterized.parameters(
        ("micro", 0.5),
        ("micro", 1.0),
        ("micro", 2.0),
        ("macro", 0.5),
        ("macro", 1.0),
        ("macro", 2.0),
        ("weighted", 0.5),
        ("weighted", 1.0),
        ("weighted", 2.0),
    )
    def test_fbeta_perfect_score(self, average, beta):
        y_true = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
        y_pred = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        self._run_test(
            y_true,
            y_pred,
            None,
            average=average,
            beta=beta,
            threshold=0.66,
            reference_result=1.0,
        )

    @parameterized.parameters(
        ("micro", 0.5),
        ("micro", 1.0),
        ("micro", 2.0),
        ("macro", 0.5),
        ("macro", 1.0),
        ("macro", 2.0),
        ("weighted", 0.5),
        ("weighted", 1.0),
        ("weighted", 2.0),
    )
    def test_fbeta_worst_score(self, average, beta):
        y_true = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
        y_pred = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        self._run_test(
            y_true,
            y_pred,
            None,
            average=average,
            beta=beta,
            threshold=0.66,
            reference_result=0.0,
        )

    @parameterized.parameters(
        # average, beta, result
        (None, 0.5, [0.71428573, 0.5, 0.833334]),
        (None, 1.0, [0.8, 0.5, 0.6666667]),
        (None, 2.0, [0.9090904, 0.5, 0.555556]),
        ("micro", 0.5, 0.6666667),
        ("micro", 1.0, 0.6666667),
        ("micro", 2.0, 0.6666667),
        ("macro", 0.5, 0.6825397),
        ("macro", 1.0, 0.6555555),
        ("macro", 2.0, 0.6548822),
        ("weighted", 0.5, 0.6825397),
        ("weighted", 1.0, 0.6555555),
        ("weighted", 2.0, 0.6548822),
    )
    def test_fbeta_random_score(self, average, beta, result):
        y_pred = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        y_true = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
        self._run_test(
            y_true,
            y_pred,
            None,
            average=average,
            beta=beta,
            threshold=0.66,
            reference_result=result,
        )

    @parameterized.parameters(
        # average, beta, result
        (None, 0.5, [0.9090904, 0.555556, 1.0]),
        (None, 1.0, [0.8, 0.6666667, 1.0]),
        (None, 2.0, [0.71428573, 0.833334, 1.0]),
        ("micro", 0.5, 0.833334),
        ("micro", 1.0, 0.833334),
        ("micro", 2.0, 0.833334),
        ("macro", 0.5, 0.821549),
        ("macro", 1.0, 0.822222),
        ("macro", 2.0, 0.849206),
        ("weighted", 0.5, 0.880471),
        ("weighted", 1.0, 0.844445),
        ("weighted", 2.0, 0.829365),
    )
    def test_fbeta_random_score_none(self, average, beta, result):
        y_true = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
        y_pred = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        self._run_test(
            y_true,
            y_pred,
            None,
            average=average,
            beta=beta,
            threshold=None,
            reference_result=result,
        )

    @parameterized.parameters(
        # average, beta, sample_weights, result
        (None, 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.909091, 0.555556, 1.0]),
        (None, 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (None, 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.9375, 0.714286, 1.0]),
        (None, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.8, 0.666667, 1.0]),
        (None, 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (None, 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.857143, 0.8, 1.0]),
        (None, 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.714286, 0.833333, 1.0]),
        (None, 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (None, 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.789474, 0.909091, 1.0]),
        ("micro", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
        ("micro", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("micro", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
        ("micro", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
        ("micro", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("micro", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
        ("micro", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
        ("micro", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("micro", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
        ("macro", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.821549),
        ("macro", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
        ("macro", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.883929),
        ("macro", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.822222),
        ("macro", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
        ("macro", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.885714),
        ("macro", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.849206),
        ("macro", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
        ("macro", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.899522),
        ("weighted", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.880471),
        ("weighted", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("weighted", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.917857),
        ("weighted", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.844444),
        ("weighted", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("weighted", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.902857),
        ("weighted", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.829365),
        ("weighted", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("weighted", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.897608),
    )
    def test_fbeta_weighted_random_score_none(
        self, average, beta, sample_weights, result
    ):
        y_true = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
        y_pred = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        self._run_test(
            y_true,
            y_pred,
            sample_weights,
            average=average,
            beta=beta,
            threshold=None,
            reference_result=result,
        )

    def test_invalid_average_raises_value_error(self):
        expected_message = (
            "Invalid `average` argument value. Expected one of: "
            r"\{None, 'micro', 'macro', 'weighted'\}. "
            "Received: average=invalid_average"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="invalid_average",
                beta=1.0,
                threshold=None,
                dtype="float32",
            )

    def test_beta_integer_type_raises_value_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid `beta` argument value. It should be a Python float.",
        ):
            f_score_metrics.FBetaScore(
                average="macro", beta=1, threshold=None, dtype="float32"
            )

    def test_beta_string_type_raises_value_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid `beta` argument value. It should be a Python float.",
        ):
            f_score_metrics.FBetaScore(
                average="macro", beta="1.0", threshold=None, dtype="float32"
            )

    def test_beta_none_type_raises_value_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid `beta` argument value. It should be a Python float.",
        ):
            f_score_metrics.FBetaScore(
                average="macro", beta=None, threshold=None, dtype="float32"
            )

    def test_beta_zero_raises_value_error(self):
        expected_message = (
            "Invalid `beta` argument value. It should be > 0. "
            "Received: beta=0.0"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="macro", beta=0.0, threshold=None, dtype="float32"
            )

    def test_beta_negative_one_raises_value_error(self):
        expected_message = (
            "Invalid `beta` argument value. It should be > 0. "
            "Received: beta=-1.0"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="macro", beta=-1.0, threshold=None, dtype="float32"
            )

    def test_beta_negative_half_raises_value_error(self):
        expected_message = (
            "Invalid `beta` argument value. It should be > 0. "
            "Received: beta=-0.5"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="macro", beta=-0.5, threshold=None, dtype="float32"
            )

    def test_threshold_not_float_raises_value_error(self):
        expected_message_pattern = (
            "Invalid `threshold` argument value. "
            "It should be a Python float. "
            "Received: threshold=1 of type '<class 'int'>'"
        )
        with self.assertRaisesRegex(ValueError, expected_message_pattern):
            f_score_metrics.FBetaScore(
                average="macro", beta=1.0, threshold=1, dtype="float32"
            )

    def test_threshold_string_raises_value_error(self):
        expected_message_pattern = (
            "Invalid `threshold` argument value. "
            "It should be a Python float. "
            "Received: threshold=0.5 of type '<class 'str'>'"
        )
        with self.assertRaisesRegex(ValueError, expected_message_pattern):
            f_score_metrics.FBetaScore(
                average="macro", beta=1.0, threshold="0.5", dtype="float32"
            )

    def test_threshold_above_one_raises_value_error(self):
        expected_message = (
            "Invalid `threshold` argument value. "
            "It should verify 0 < threshold <= 1. "
            "Received: threshold=1.1"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="macro", beta=1.0, threshold=1.1, dtype="float32"
            )

    def test_threshold_zero_raises_value_error(self):
        expected_message = (
            "Invalid `threshold` argument value. "
            "It should verify 0 < threshold <= 1. "
            "Received: threshold=0.0"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="macro", beta=1.0, threshold=0.0, dtype="float32"
            )

    def test_threshold_negative_raises_value_error(self):
        expected_message = (
            "Invalid `threshold` argument value. "
            "It should verify 0 < threshold <= 1. "
            "Received: threshold=-0.5"
        )
        with self.assertRaisesRegex(ValueError, expected_message):
            f_score_metrics.FBetaScore(
                average="macro", beta=1.0, threshold=-0.5, dtype="float32"
            )

    def test_non_2d_input_shapes_raises_value_error(self):
        fbeta = f_score_metrics.FBetaScore(beta=1.0, dtype="float32")
        y_true_shape = (2, 3, 4)
        y_pred_shape = (2, 3, 4)
        expected_error_message = (
            "FBetaScore expects 2D inputs with shape "
            r"\(batch_size, output_dim\)\. Received input "
            r"shapes: y_pred\.shape=\(2, 3, 4\) and "
            r"y_true\.shape=\(2, 3, 4\)\."
        )
        with self.assertRaisesRegex(ValueError, expected_error_message):
            fbeta._build(y_true_shape, y_pred_shape)

    def test_undefined_output_dim_raises_value_error(self):
        fbeta = f_score_metrics.FBetaScore(beta=1.0, dtype="float32")
        y_true_shape = (2, None)
        y_pred_shape = (2, None)
        expected_error_message = (
            "FBetaScore expects 2D inputs with shape "
            r"\(batch_size, output_dim\), with output_dim fully "
            r"defined \(not None\)\. Received input "
            r"shapes: y_pred\.shape=\(2, None\) and "
            r"y_true\.shape=\(2, None\)\."
        )
        with self.assertRaisesRegex(ValueError, expected_error_message):
            fbeta._build(y_true_shape, y_pred_shape)


class F1ScoreTest(testing.TestCase):
    def test_config(self):
        f1_obj = f_score_metrics.F1Score(dtype="float32")
        config = f1_obj.get_config()
        self.assertNotIn("beta", config)

        # Check save and restore config
        f1_obj = f_score_metrics.F1Score.from_config(config)
        self.assertEqual(f1_obj.average, None)
        self.assertEqual(f1_obj.dtype, "float32")

    def test_correctness(self):
        f1 = f_score_metrics.F1Score()
        fbeta = f_score_metrics.FBetaScore(beta=1.0)

        y_true = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )
        y_pred = np.array(
            [
                [0.9, 0.1, 0],
                [0.2, 0.6, 0.2],
                [0, 0, 1],
                [0.4, 0.3, 0.3],
                [0, 0.9, 0.1],
                [0, 0, 1],
            ]
        )

        fbeta.update_state(y_true, y_pred)
        f1.update_state(y_true, y_pred)
        self.assertAllClose(fbeta.result(), f1.result(), atol=1e-6)
