# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for metrics."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import metrics
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import test_util
from keras.utils import tf_utils


class MetricsTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        global_ids = test_util.create_device_ids_array((2, 2))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            "CPU": dtensor.Mesh(
                ["X", "Y"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2, 2), "CPU"),
            )
        }
        self.mesh = self.configTestMesh(mesh_dict)
        tf_utils.set_random_seed(1337)

    @parameterized.parameters(
        (metrics.Accuracy, {}),
        (metrics.AUC, {}),
        (metrics.BinaryAccuracy, {}),
        (metrics.BinaryCrossentropy, {}),
        (metrics.BinaryIoU, {}),
        (metrics.CategoricalAccuracy, {}),
        (metrics.CategoricalCrossentropy, {}),
        (metrics.CategoricalHinge, {}),
        (metrics.CosineSimilarity, {}),
        (metrics.FalseNegatives, {}),
        (metrics.FalsePositives, {}),
        (metrics.Hinge, {}),
        (metrics.IoU, {"num_classes": 3, "target_class_ids": [1]}),
        (metrics.KLDivergence, {}),
        (metrics.LogCoshError, {}),
        (metrics.Mean, {}),
        (metrics.MeanAbsoluteError, {}),
        (metrics.MeanAbsolutePercentageError, {}),
        (metrics.MeanIoU, {"num_classes": 3}),
        (metrics.MeanRelativeError, {"normalizer": [1, 3, 2, 3]}),
        (metrics.MeanSquaredError, {}),
        (metrics.MeanSquaredLogarithmicError, {}),
        (metrics.OneHotIoU, {"num_classes": 3, "target_class_ids": [1]}),
        (metrics.OneHotMeanIoU, {"num_classes": 3}),
        (metrics.Poisson, {}),
        (metrics.Precision, {}),
        (metrics.PrecisionAtRecall, {"recall": 0.5}),
        (metrics.Recall, {}),
        (metrics.RecallAtPrecision, {"precision": 0.5}),
        (metrics.RootMeanSquaredError, {}),
        (metrics.SensitivityAtSpecificity, {"specificity": 0.5}),
        (metrics.SparseCategoricalAccuracy, {}),
        (metrics.SparseCategoricalCrossentropy, {}),
        (metrics.SparseTopKCategoricalAccuracy, {}),
        (metrics.SpecificityAtSensitivity, {"sensitivity": 0.5}),
        (metrics.SquaredHinge, {}),
        (metrics.Sum, {}),
        (metrics.TopKCategoricalAccuracy, {}),
        (metrics.TrueNegatives, {}),
        (metrics.TruePositives, {}),
    )
    def test_metric_layout(self, metric_cls, init_args):
        metric = metric_cls(**init_args, mesh=self.mesh)

        for weight in metric.non_trainable_weights:
            self.assertIsInstance(weight, dtensor.DVariable)
            self.assertTrue(weight.layout.is_fully_replicated())


if __name__ == "__main__":
    tf.test.main()
