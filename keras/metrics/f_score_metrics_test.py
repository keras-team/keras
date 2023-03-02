# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for F-score metrics."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.metrics import f_score_metrics
from keras.testing_infra import test_utils


@test_utils.run_v2_only
class FBetaScoreTest(parameterized.TestCase, tf.test.TestCase):
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
        y_true = tf.constant(y_true, dtype="float32")
        y_pred = tf.constant(y_pred, dtype="float32")
        fbeta = f_score_metrics.FBetaScore(average, beta, threshold)
        fbeta.update_state(y_true, y_pred, sample_weights)
        result = fbeta.result().numpy()
        self.assertAllClose(result, reference_result, atol=1e-6)

    def test_config(self):
        fbeta_obj = f_score_metrics.FBetaScore(
            beta=0.5, threshold=0.3, average=None
        )
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.average, None)
        self.assertEqual(fbeta_obj.threshold, 0.3)
        self.assertEqual(fbeta_obj.dtype, tf.float32)

        # Check save and restore config
        fbeta_obj2 = f_score_metrics.FBetaScore.from_config(
            fbeta_obj.get_config()
        )
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.average, None)
        self.assertEqual(fbeta_obj2.threshold, 0.3)
        self.assertEqual(fbeta_obj2.dtype, tf.float32)

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            average=["micro", "macro", "weighted"], beta=[0.5, 1.0, 2.0]
        )
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

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            average=["micro", "macro", "weighted"], beta=[0.5, 1.0, 2.0]
        )
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


@test_utils.run_v2_only
class F1ScoreTest(tf.test.TestCase):
    def test_config(self):
        f1_obj = f_score_metrics.F1Score()
        config = f1_obj.get_config()
        self.assertNotIn("beta", config)

        # Check save and restore config
        f1_obj = f_score_metrics.F1Score.from_config(config)
        self.assertEqual(f1_obj.average, None)
        self.assertEqual(f1_obj.dtype, tf.float32)

    def test_correctness(self):
        f1 = f_score_metrics.F1Score()
        fbeta = f_score_metrics.FBetaScore(beta=1.0)

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

        fbeta.update_state(y_true, y_pred)
        f1.update_state(y_true, y_pred)
        self.assertAllClose(
            fbeta.result().numpy(), f1.result().numpy(), atol=1e-6
        )


if __name__ == "__main__":
    tf.test.main()
