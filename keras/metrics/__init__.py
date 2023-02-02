# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""All Keras metrics."""

# isort: off
from tensorflow.python.util.tf_export import keras_export

# Base classes and utilities
from keras.metrics.base_metric import Mean
from keras.metrics.base_metric import MeanMetricWrapper
from keras.metrics.base_metric import MeanTensor
from keras.metrics.base_metric import Metric
from keras.metrics.base_metric import Reduce
from keras.metrics.base_metric import Sum
from keras.metrics.base_metric import SumOverBatchSize
from keras.metrics.base_metric import SumOverBatchSizeMetricWrapper
from keras.metrics.base_metric import clone_metric
from keras.metrics.base_metric import clone_metrics

from keras.saving.legacy import serialization as legacy_serialization
from keras.saving.legacy.serialization import deserialize_keras_object
from keras.saving.legacy.serialization import serialize_keras_object

from keras.metrics.py_metric import PyMetric

# Individual metric classes

# Accuracy metrics
from keras.metrics.accuracy_metrics import Accuracy
from keras.metrics.accuracy_metrics import BinaryAccuracy
from keras.metrics.accuracy_metrics import CategoricalAccuracy
from keras.metrics.accuracy_metrics import SparseCategoricalAccuracy
from keras.metrics.accuracy_metrics import SparseTopKCategoricalAccuracy
from keras.metrics.accuracy_metrics import TopKCategoricalAccuracy

from keras.metrics.accuracy_metrics import accuracy
from keras.metrics.accuracy_metrics import binary_accuracy
from keras.metrics.accuracy_metrics import categorical_accuracy
from keras.metrics.accuracy_metrics import sparse_categorical_accuracy
from keras.metrics.accuracy_metrics import sparse_top_k_categorical_accuracy
from keras.metrics.accuracy_metrics import top_k_categorical_accuracy

# Probabilistic metrics
from keras.metrics.probabilistic_metrics import BinaryCrossentropy
from keras.metrics.probabilistic_metrics import CategoricalCrossentropy
from keras.metrics.probabilistic_metrics import KLDivergence
from keras.metrics.probabilistic_metrics import Poisson
from keras.metrics.probabilistic_metrics import SparseCategoricalCrossentropy

from keras.metrics.probabilistic_metrics import binary_crossentropy
from keras.metrics.probabilistic_metrics import categorical_crossentropy
from keras.metrics.probabilistic_metrics import poisson
from keras.metrics.probabilistic_metrics import kullback_leibler_divergence
from keras.metrics.probabilistic_metrics import sparse_categorical_crossentropy

# Regression metrics
from keras.metrics.regression_metrics import CosineSimilarity
from keras.metrics.regression_metrics import LogCoshError
from keras.metrics.regression_metrics import MeanAbsoluteError
from keras.metrics.regression_metrics import MeanAbsolutePercentageError
from keras.metrics.regression_metrics import MeanRelativeError
from keras.metrics.regression_metrics import MeanSquaredError
from keras.metrics.regression_metrics import MeanSquaredLogarithmicError
from keras.metrics.regression_metrics import RootMeanSquaredError
from keras.metrics.regression_metrics import R2Score

from keras.metrics.regression_metrics import cosine_similarity
from keras.metrics.regression_metrics import logcosh
from keras.metrics.regression_metrics import mean_absolute_error
from keras.metrics.regression_metrics import mean_absolute_percentage_error
from keras.metrics.regression_metrics import mean_squared_error
from keras.metrics.regression_metrics import mean_squared_logarithmic_error

# Confusion metrics
from keras.metrics.confusion_metrics import AUC
from keras.metrics.confusion_metrics import FalseNegatives
from keras.metrics.confusion_metrics import FalsePositives
from keras.metrics.confusion_metrics import Precision
from keras.metrics.confusion_metrics import PrecisionAtRecall
from keras.metrics.confusion_metrics import Recall
from keras.metrics.confusion_metrics import RecallAtPrecision
from keras.metrics.confusion_metrics import SensitivityAtSpecificity
from keras.metrics.confusion_metrics import SensitivitySpecificityBase
from keras.metrics.confusion_metrics import SpecificityAtSensitivity
from keras.metrics.confusion_metrics import TrueNegatives
from keras.metrics.confusion_metrics import TruePositives

# F-Scores
from keras.metrics.f_score_metrics import FBetaScore
from keras.metrics.f_score_metrics import F1Score

# IoU metrics
from keras.metrics.iou_metrics import BinaryIoU
from keras.metrics.iou_metrics import IoU
from keras.metrics.iou_metrics import MeanIoU
from keras.metrics.iou_metrics import OneHotIoU
from keras.metrics.iou_metrics import OneHotMeanIoU

# Hinge metrics
from keras.metrics.hinge_metrics import CategoricalHinge
from keras.metrics.hinge_metrics import Hinge
from keras.metrics.hinge_metrics import SquaredHinge

from keras.metrics.hinge_metrics import categorical_hinge
from keras.metrics.hinge_metrics import squared_hinge
from keras.metrics.hinge_metrics import hinge

# Aliases
acc = ACC = accuracy
bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
log_cosh = logcosh
cosine_proximity = cosine_similarity


@keras_export("keras.metrics.serialize")
def serialize(metric, use_legacy_format=False):
    """Serializes metric function or `Metric` instance.

    Args:
      metric: A Keras `Metric` instance or a metric function.

    Returns:
      Metric configuration dictionary.
    """
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(metric)
    return serialize_keras_object(metric)


@keras_export("keras.metrics.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    """Deserializes a serialized metric class/function instance.

    Args:
      config: Metric configuration.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.

    Returns:
        A Keras `Metric` instance or a metric function.
    """
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=globals(),
            custom_objects=custom_objects,
            printable_module_name="metric function",
        )
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="metric function",
    )


@keras_export("keras.metrics.get")
def get(identifier):
    """Retrieves a Keras metric as a `function`/`Metric` class instance.

    The `identifier` may be the string name of a metric function or class.

    >>> metric = tf.keras.metrics.get("categorical_crossentropy")
    >>> type(metric)
    <class 'function'>
    >>> metric = tf.keras.metrics.get("CategoricalCrossentropy")
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    You can also specify `config` of the metric to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Metric` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> metric = tf.keras.metrics.get(identifier)
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    Args:
      identifier: A metric identifier. One of None or string name of a metric
        function/class or metric configuration dictionary or a metric function
        or a metric class instance

    Returns:
      A Keras metric as a `function`/ `Metric` class instance.

    Raises:
      ValueError: If `identifier` cannot be interpreted.
    """
    if isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(identifier, use_legacy_format=use_legacy_format)
    elif isinstance(identifier, str):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")
