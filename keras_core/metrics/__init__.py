from keras_core.api_export import keras_core_export
from keras_core.metrics.metric import Metric
from keras_core.metrics.regression_metrics import MeanSquaredError


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
            function/class or metric configuration dictionary or a metric function
            or a metric class instance

    Returns:
        A Keras metric as a `function`/ `Metric` class instance.
    """
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        # TODO
        # return deserialize(str(identifier))
        return globals()[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")
