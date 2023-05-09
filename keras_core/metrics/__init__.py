from keras_core.api_export import keras_core_export
from keras_core.metrics.accuracy_metrics import Accuracy
from keras_core.metrics.accuracy_metrics import BinaryAccuracy
from keras_core.metrics.accuracy_metrics import CategoricalAccuracy
from keras_core.metrics.accuracy_metrics import SparseCategoricalAccuracy
from keras_core.metrics.accuracy_metrics import SparseTopKCategoricalAccuracy
from keras_core.metrics.accuracy_metrics import TopKCategoricalAccuracy
from keras_core.metrics.confusion_metrics import AUC
from keras_core.metrics.confusion_metrics import FalseNegatives
from keras_core.metrics.confusion_metrics import FalsePositives
from keras_core.metrics.confusion_metrics import Precision
from keras_core.metrics.confusion_metrics import PrecisionAtRecall
from keras_core.metrics.confusion_metrics import Recall
from keras_core.metrics.confusion_metrics import RecallAtPrecision
from keras_core.metrics.confusion_metrics import SensitivityAtSpecificity
from keras_core.metrics.confusion_metrics import SpecificityAtSensitivity
from keras_core.metrics.confusion_metrics import TrueNegatives
from keras_core.metrics.confusion_metrics import TruePositives
from keras_core.metrics.f_score_metrics import F1Score
from keras_core.metrics.f_score_metrics import FBetaScore
from keras_core.metrics.hinge_metrics import CategoricalHinge
from keras_core.metrics.hinge_metrics import Hinge
from keras_core.metrics.hinge_metrics import SquaredHinge
from keras_core.metrics.metric import Metric
from keras_core.metrics.probabilistic_metrics import BinaryCrossentropy
from keras_core.metrics.probabilistic_metrics import CategoricalCrossentropy
from keras_core.metrics.probabilistic_metrics import KLDivergence
from keras_core.metrics.probabilistic_metrics import Poisson
from keras_core.metrics.probabilistic_metrics import (
    SparseCategoricalCrossentropy,
)
from keras_core.metrics.reduction_metrics import Mean
from keras_core.metrics.reduction_metrics import MeanMetricWrapper
from keras_core.metrics.reduction_metrics import Sum
from keras_core.metrics.regression_metrics import CosineSimilarity
from keras_core.metrics.regression_metrics import LogCoshError
from keras_core.metrics.regression_metrics import MeanAbsoluteError
from keras_core.metrics.regression_metrics import MeanAbsolutePercentageError
from keras_core.metrics.regression_metrics import MeanSquaredError
from keras_core.metrics.regression_metrics import MeanSquaredLogarithmicError
from keras_core.metrics.regression_metrics import R2Score
from keras_core.metrics.regression_metrics import RootMeanSquaredError
from keras_core.saving import serialization_lib

ALL_OBJECTS = {
    # Base
    Metric,
    Mean,
    Sum,
    MeanMetricWrapper,
    # Regression
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredLogarithmicError,
    CosineSimilarity,
    LogCoshError,
    R2Score,
    # Classification
    AUC,
    FalseNegatives,
    FalsePositives,
    Precision,
    PrecisionAtRecall,
    Recall,
    RecallAtPrecision,
    SensitivityAtSpecificity,
    SpecificityAtSensitivity,
    TrueNegatives,
    TruePositives,
    # Hinge
    Hinge,
    SquaredHinge,
    CategoricalHinge,
    # Probabilistic
    KLDivergence,
    Poisson,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    # Accuracy
    Accuracy,
    BinaryAccuracy,
    CategoricalAccuracy,
    SparseCategoricalAccuracy,
    TopKCategoricalAccuracy,
    SparseTopKCategoricalAccuracy,
    # F-Score
    F1Score,
    FBetaScore,
}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {
        "mse": MeanSquaredError,
        "MSE": MeanSquaredError,
    }
)


@keras_core_export("keras_core.metrics.serialize")
def serialize(metric):
    """Serializes metric function or `Metric` instance.

    Args:
        metric: A Keras `Metric` instance or a metric function.

    Returns:
        Metric configuration dictionary.
    """
    return serialization_lib.serialize_keras_object(metric)


@keras_core_export("keras_core.metrics.deserialize")
def deserialize(config, custom_objects=None):
    """Deserializes a serialized metric class/function instance.

    Args:
        config: Metric configuration.
        custom_objects: Optional dictionary mapping names (strings)
            to custom objects (classes and functions) to be
            considered during deserialization.

    Returns:
        A Keras `Metric` instance or a metric function.
    """
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_core_export("keras_core.metrics.get")
def get(identifier):
    """Retrieves a Keras metric as a `function`/`Metric` class instance.

    The `identifier` may be the string name of a metric function or class.

    >>> metric = metrics.get("categorical_crossentropy")
    >>> type(metric)
    <class 'function'>
    >>> metric = metrics.get("CategoricalCrossentropy")
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    You can also specify `config` of the metric to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Metric` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> metric = metrics.get(identifier)
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    Args:
        identifier: A metric identifier. One of None or string name of a metric
            function/class or metric configuration dictionary or a metric
            function or a metric class instance

    Returns:
        A Keras metric as a `function`/ `Metric` class instance.
    """
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")
