"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from abc import abstractmethod
from . import backend as K
from .losses import mean_squared_error
from .losses import mean_absolute_error
from .losses import mean_absolute_percentage_error
from .losses import mean_squared_logarithmic_error
from .losses import hinge
from .losses import logcosh
from .losses import squared_hinge
from .losses import categorical_crossentropy
from .losses import sparse_categorical_crossentropy
from .losses import binary_crossentropy
from .losses import kullback_leibler_divergence
from .losses import poisson
from .losses import cosine_proximity
from .utils.generic_utils import deserialize_keras_object
from .engine.topology import _to_snake_case


class GlobalMetric(object):
    """Base class for global metrics, which persist over epochs. Global Metrics
    must inherit from this class.

    # Properties:
        __name__

    # Methods:
        update_states(self. y_true, y_pred)
        reset_states(self)
    """
    def __init__(self):
        """Initialize the Global Metric and give it a name.
        """
        # Instance name is class name with underscores
        cls_name = self.__class__.__name__
        self.__name__ = _to_snake_case(cls_name)

    def __call__(self, y_true, y_pred):
        return self.update_states(y_true, y_pred)

    @abstractmethod
    def update_states(self, y_true, y_pred):
        """Update the state of the metric with the current values of y_true,
        y_pred.

        # Arguments
            y_true: the true labels, either binary or categorical.
            y_pred: the predictions, either binary or categorical.

        # Returns
            The current state of the metric.

        # Raises
            NotImplementedError: if the derived class fails to implement the method.
        """
        raise NotImplementedError("Method not implemented.")

    @abstractmethod
    def reset_states(self):
        """Resets the state of the metric at the beginning of training and validation for each epoch.

        # Raises
            NotImplementedError: if the derived class fails to implement the method.
        """
        raise NotImplementedError("Method not implemented.")


def reset_global_metrics(metrics):
    """Call reset_states() for all global metrics.

    # Arguments
        metrics: a list of metric instances.
    """
    if metrics is not None:
        for metric in metrics:
            if isinstance(metric, GlobalMetric):
                metric.reset_states()


def get_global_metrics(metrics):
    """ Return a list of global metrics.

    # Arguments
        metrics: a list of metric instances.
    """
    global_metrics = []
    global_metric_names = []

    if metrics is not None:
        for m in metrics:
            if isinstance(m, GlobalMetric):
                global_metrics.append(m)
                global_metric_names.append(serialize(m))

    return global_metrics, global_metric_names


class TruePositives(GlobalMetric):
    """Global Metric to count the total true positives over all batches.

    # Properties
        threshold: the lower limit on y_pred that counts as a positive class prediction
        state: the current state of the metric through the current batch
    """

    def __init__(self, threshold=None):
        """Set the threshold and name the metric

        # Keyword Arguments
            threshold: the lower limit on y_pred that counts as a postive class prediction.
                Defaults to 0.5
        """
        super(TruePositives, self).__init__()

        if threshold is None:
            self.threshold = K.variable(value=0.5)
        else:
            self.threshold = K.variable(value=threshold)

        self.state = K.variable(value=0.0)

    def reset_states(self):
        """Reset the state at the beginning of training and evaluation for each epoch.
        """
        K.set_value(self.state, 0)

    def update_states(self, y_true, y_pred):
        """Update the state at the completion of each batch.

        # Arguments:
            y_true: the batch_wise labels
            y_pred: the batch_wise predictions

        # Returns:
            The total number of true positives seen this epoch at the completion of the batch.
        """

        # Slice the positive score
        y_true = y_true[:, 1]
        y_pred = y_pred[:, 1]
        K.sum(y_true * y_pred, axis = -1)
        # Softmax -> probabilities
        y_pred = K.cast(y_pred >= self.threshold, 'float32')
        # c = correct classifications
        c = K.cast(K.equal(y_pred, y_true), 'float32')
        # tp_batch = number of true positives in a batch
        tp_batch = K.sum(c * y_true)
        return K.update_add(self.state, tp_batch)


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def serialize(metric):
    return metric.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
