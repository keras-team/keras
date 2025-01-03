import inspect

from keras.src.api_export import keras_export
from keras.src.metrics.accuracy_metrics import (Accuracy, BinaryAccuracy,
                                                CategoricalAccuracy,
                                                SparseCategoricalAccuracy,
                                                SparseTopKCategoricalAccuracy,
                                                TopKCategoricalAccuracy)
from keras.src.metrics.confusion_metrics import (AUC, FalseNegatives,
                                                 FalsePositives, Precision,
                                                 PrecisionAtRecall, Recall,
                                                 RecallAtPrecision,
                                                 SensitivityAtSpecificity,
                                                 SpecificityAtSensitivity,
                                                 TrueNegatives, TruePositives)
from keras.src.metrics.correlation_metrics import (ConcordanceCorrelation,
                                                   PearsonCorrelation)
from keras.src.metrics.f_score_metrics import F1Score, FBetaScore
from keras.src.metrics.hinge_metrics import (CategoricalHinge, Hinge,
                                             SquaredHinge)
from keras.src.metrics.iou_metrics import (BinaryIoU, IoU, MeanIoU, OneHotIoU,
                                           OneHotMeanIoU)
from keras.src.metrics.metric import Metric
from keras.src.metrics.probabilistic_metrics import (
    BinaryCrossentropy, CategoricalCrossentropy, KLDivergence, Poisson,
    SparseCategoricalCrossentropy)
from keras.src.metrics.reduction_metrics import Mean, MeanMetricWrapper, Sum
from keras.src.metrics.regression_metrics import (CosineSimilarity,
                                                  LogCoshError,
                                                  MeanAbsoluteError,
                                                  MeanAbsolutePercentageError,
                                                  MeanSquaredError,
                                                  MeanSquaredLogarithmicError,
                                                  R2Score,
                                                  RootMeanSquaredError)
from keras.src.saving import serialization_lib
from keras.src.utils.naming import to_snake_case

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
    # Correlation
    ConcordanceCorrelation,
    PearsonCorrelation,
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
    # IoU
    IoU,
    BinaryIoU,
    MeanIoU,
    OneHotIoU,
    OneHotMeanIoU,
}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
# TODO: Align with `tf.keras` and set the name attribute of metrics
# with the key name. Currently it uses default name of class definitions.
ALL_OBJECTS_DICT.update(
    {
        "bce": BinaryCrossentropy,
        "BCE": BinaryCrossentropy,
        "mse": MeanSquaredError,
        "MSE": MeanSquaredError,
        "mae": MeanAbsoluteError,
        "MAE": MeanAbsoluteError,
        "mape": MeanAbsolutePercentageError,
        "MAPE": MeanAbsolutePercentageError,
        "msle": MeanSquaredLogarithmicError,
        "MSLE": MeanSquaredLogarithmicError,
    }
)


@keras_export("keras.metrics.serialize")
def serialize(metric):
    """Serializes metric function or `Metric` instance.

    Args:
        metric: A Keras `Metric` instance or a metric function.

    Returns:
        Metric configuration dictionary.
    """
    return serialization_lib.serialize_keras_object(metric)


@keras_export("keras.metrics.deserialize")
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


@keras_export("keras.metrics.get")
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
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier
    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")
