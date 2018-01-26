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


class GlobalMetric(object):

    @abstractmethod
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Method not implemented.")

    @abstractmethod
    def update_states(self):
        raise NotImplementedError("Method not implemented.")

    @abstractmethod
    def reset_states(self):
        raise NotImplementedError("Method not implemented.")


def reset_global_metrics(metrics):
    ''' Call reset_states() for all global metrics. '''
    for metric in metrics:
        if isinstance(metric, GlobalMetric):
            metric.reset_states()


def get_global_metric_names(metrics):
    ''' Return a list of global metric names. '''
    global_metric_lst = []
    for m in metrics:
        if isinstance(m, GlobalMetric):
            global_metric_lst.append(six.text_type(m.__name__))
    return global_metric_lst

def get_global_metric_index(metrics, offset=0):
    ''' Return a list of global metric names. '''
    global_metric_idx = []
    for i, m in enumerate(Model.metrics):
        if isinstance(m, GlobalMetric):
            global_metric_idx.append(i+offset)
    return global_metric_idx

class TruePositives(GlobalMetric):

    def __init__(self, threshold=None):

        self.__name__ = "true_positives"
        if threshold is None:
            self.threshold = K.variable(value=0.5)
        else:
            self.threshold = K.variable(value=threshold)
        # tp = true positives
        self.tp = K.variable(value=0.0)

    def __call__(self, y_true, y_pred):
        return self.update_states(y_true, y_pred)

    def reset_states(self):
        K.set_value(self.tp, 0)

    def update_states(self, y_true, y_pred):

        # Slice the positive score
        y_true = y_true[:, 1]
        y_pred = y_pred[:, 1]

        # Softmax -> probabilities
        y_pred = K.cast(y_pred >= self.threshold, 'float32')
        # c = correct classifications
        c = K.cast(K.equal(y_pred, y_true), 'float32')
        # tp_batch = number of true positives in a batch
        tp_batch = K.sum(c * y_true)
        return K.update_add(self.tp, tp_batch)


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
