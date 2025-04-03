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
"""All TF-Keras metrics."""

# isort: off
import warnings
from tensorflow.python.util.tf_export import keras_export

# Base classes and utilities
from tf_keras.src.metrics.base_metric import Mean
from tf_keras.src.metrics.base_metric import MeanMetricWrapper
from tf_keras.src.metrics.base_metric import MeanTensor
from tf_keras.src.metrics.base_metric import Metric
from tf_keras.src.metrics.base_metric import Reduce
from tf_keras.src.metrics.base_metric import Sum
from tf_keras.src.metrics.base_metric import SumOverBatchSize
from tf_keras.src.metrics.base_metric import SumOverBatchSizeMetricWrapper
from tf_keras.src.metrics.base_metric import clone_metric
from tf_keras.src.metrics.base_metric import clone_metrics

from tf_keras.src.saving.legacy import serialization as legacy_serialization
from tf_keras.src.saving.serialization_lib import deserialize_keras_object
from tf_keras.src.saving.serialization_lib import serialize_keras_object

from tf_keras.src.metrics.py_metric import PyMetric

# Individual metric classes

# Accuracy metrics
from tf_keras.src.metrics.accuracy_metrics import Accuracy
from tf_keras.src.metrics.accuracy_metrics import BinaryAccuracy
from tf_keras.src.metrics.accuracy_metrics import CategoricalAccuracy
from tf_keras.src.metrics.accuracy_metrics import SparseCategoricalAccuracy
from tf_keras.src.metrics.accuracy_metrics import SparseTopKCategoricalAccuracy
from tf_keras.src.metrics.accuracy_metrics import TopKCategoricalAccuracy

from tf_keras.src.metrics.accuracy_metrics import accuracy
from tf_keras.src.metrics.accuracy_metrics import binary_accuracy
from tf_keras.src.metrics.accuracy_metrics import categorical_accuracy
from tf_keras.src.metrics.accuracy_metrics import sparse_categorical_accuracy
from tf_keras.src.metrics.accuracy_metrics import sparse_top_k_categorical_accuracy
from tf_keras.src.metrics.accuracy_metrics import top_k_categorical_accuracy

# Probabilistic metrics
from tf_keras.src.metrics.probabilistic_metrics import BinaryCrossentropy
from tf_keras.src.metrics.probabilistic_metrics import CategoricalCrossentropy
from tf_keras.src.metrics.probabilistic_metrics import KLDivergence
from tf_keras.src.metrics.probabilistic_metrics import Poisson
from tf_keras.src.metrics.probabilistic_metrics import SparseCategoricalCrossentropy

from tf_keras.src.metrics.probabilistic_metrics import binary_crossentropy
from tf_keras.src.metrics.probabilistic_metrics import categorical_crossentropy
from tf_keras.src.metrics.probabilistic_metrics import poisson
from tf_keras.src.metrics.probabilistic_metrics import kullback_leibler_divergence
from tf_keras.src.metrics.probabilistic_metrics import (
    sparse_categorical_crossentropy,
)

# Regression metrics
from tf_keras.src.metrics.regression_metrics import CosineSimilarity
from tf_keras.src.metrics.regression_metrics import LogCoshError
from tf_keras.src.metrics.regression_metrics import MeanAbsoluteError
from tf_keras.src.metrics.regression_metrics import MeanAbsolutePercentageError
from tf_keras.src.metrics.regression_metrics import MeanRelativeError
from tf_keras.src.metrics.regression_metrics import MeanSquaredError
from tf_keras.src.metrics.regression_metrics import MeanSquaredLogarithmicError
from tf_keras.src.metrics.regression_metrics import RootMeanSquaredError
from tf_keras.src.metrics.regression_metrics import R2Score

from tf_keras.src.metrics.regression_metrics import cosine_similarity
from tf_keras.src.metrics.regression_metrics import logcosh
from tf_keras.src.metrics.regression_metrics import mean_absolute_error
from tf_keras.src.metrics.regression_metrics import mean_absolute_percentage_error
from tf_keras.src.metrics.regression_metrics import mean_squared_error
from tf_keras.src.metrics.regression_metrics import mean_squared_logarithmic_error

# Confusion metrics
from tf_keras.src.metrics.confusion_metrics import AUC
from tf_keras.src.metrics.confusion_metrics import FalseNegatives
from tf_keras.src.metrics.confusion_metrics import FalsePositives
from tf_keras.src.metrics.confusion_metrics import Precision
from tf_keras.src.metrics.confusion_metrics import PrecisionAtRecall
from tf_keras.src.metrics.confusion_metrics import Recall
from tf_keras.src.metrics.confusion_metrics import RecallAtPrecision
from tf_keras.src.metrics.confusion_metrics import SensitivityAtSpecificity
from tf_keras.src.metrics.confusion_metrics import SensitivitySpecificityBase
from tf_keras.src.metrics.confusion_metrics import SpecificityAtSensitivity
from tf_keras.src.metrics.confusion_metrics import TrueNegatives
from tf_keras.src.metrics.confusion_metrics import TruePositives

# F-Scores
from tf_keras.src.metrics.f_score_metrics import FBetaScore
from tf_keras.src.metrics.f_score_metrics import F1Score

# IoU metrics
from tf_keras.src.metrics.iou_metrics import BinaryIoU
from tf_keras.src.metrics.iou_metrics import IoU
from tf_keras.src.metrics.iou_metrics import MeanIoU
from tf_keras.src.metrics.iou_metrics import OneHotIoU
from tf_keras.src.metrics.iou_metrics import OneHotMeanIoU

# Hinge metrics
from tf_keras.src.metrics.hinge_metrics import CategoricalHinge
from tf_keras.src.metrics.hinge_metrics import Hinge
from tf_keras.src.metrics.hinge_metrics import SquaredHinge

from tf_keras.src.metrics.hinge_metrics import categorical_hinge
from tf_keras.src.metrics.hinge_metrics import squared_hinge
from tf_keras.src.metrics.hinge_metrics import hinge

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
      metric: A TF-Keras `Metric` instance or a metric function.

    Returns:
      Metric configuration dictionary.
    """
    if metric is None:
        return None
    if not isinstance(metric, Metric):
        warnings.warn(
            "The `keras.metrics.serialize()` API should only be used for "
            "objects of type `keras.metrics.Metric`. Found an instance of "
            f"type {type(metric)}, which may lead to improper serialization."
        )
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
        A TF-Keras `Metric` instance or a metric function.
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
    """Retrieves a TF-Keras metric as a `function`/`Metric` class instance.

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
      A TF-Keras metric as a `function`/ `Metric` class instance.

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

