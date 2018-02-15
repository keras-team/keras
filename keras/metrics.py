"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from . import backend as K
from .engine import Layer
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
from .utils.generic_utils import serialize_keras_object


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


# Stateful Metrics

class Recall(Layer):
    '''Compute recall over all batches.

    # Arguments
        name: String, name for the metric.
        label_encoding: String, 'binary' or 'one_hot', label encoding format.
    '''

    def __init__(self, name='recall', label_encoding='binary'):
        super(Recall, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.total_positives = K.variable(value=0, dtype='float32')

        if label_encoding not in ['binary', 'one_hot']:
            raise ValueError('Label encoding must be "binary" or "one_hot"')
        self.label_encoding = label_encoding

    def reset_states(self):
        K.set_value(self.true_positives, 0.0)
        K.set_value(self.total_positives, 0.0)

    def __call__(self, y_true, y_pred):
        '''Update recall computation.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            Overall recall for the epoch at the completion of the batch.
        '''
        # Batch
        if self.label_encoding == 'one_hot':
           y_true, y_pred = _one_hot_to_binary(y_true, y_pred)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        # Current
        current_true_positives = self.true_positives * 1
        current_total_positives = self.total_positives * 1

        # Updates
        updates = [K.update_add(self.true_positives, true_positives),
                   K.update_add(self.total_positives, total_positives)]
        self.add_update(updates, inputs=[y_true, y_pred])

        # Compute recall
        return (current_true_positives + true_positives) / \
               (current_total_positives + total_positives + K.epsilon())


class Precision(Layer):
    '''Compute precision over all batches.

    # Arguments
        name: String, name for the metric.
        label_encoding: String, 'binary' or 'one_hot', label encoding format.
    '''

    def __init__(self, name='precision', label_encoding='binary'):
        super(Precision, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.pred_positives = K.variable(value=0, dtype='float32')

        if label_encoding not in ['binary', 'one_hot']:
            raise ValueError('Label encoding must be "binary" or "one_hot"')
        self.label_encoding = label_encoding

    def reset_states(self):
        K.set_value(self.true_positives, 0.0)
        K.set_value(self.pred_positives, 0.0)

    def __call__(self, y_true, y_pred):
        '''Update precision computation.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            Overall precision for the epoch at the completion of the batch.
        '''
        # Batch
        if self.label_encoding == 'one_hot':
           y_true, y_pred = _one_hot_to_binary(y_true, y_pred)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        # Current
        current_true_positives = self.true_positives * 1
        current_pred_positives = self.pred_positives * 1

        # Updates
        updates = [K.update_add(self.true_positives, true_positives),
                   K.update_add(self.pred_positives, pred_positives)]
        self.add_update(updates, inputs=[y_true, y_pred])

        # Compute recall
        return (current_true_positives + true_positives) / \
               (current_pred_positives + pred_positives + K.epsilon())


def _one_hot_to_binary(y_true, y_pred):
    '''Convert one hot encoded labels and predictions to binary encoding.

    #  Arguments:
        y_true: Tensor, batch_wise labels (one hot encoded).
        y_pred: Tensor, batch_wise predictions (one hot encoded).

    # Returns:
        y_true: Tensor, batch_wise labels (binary encoded).
        y_pred: Tensor,  batch_wise predictions (binary encoded).
    '''
    return y_true[...,1], y_pred[...,1]


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def serialize(metric):
    return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, dict):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif isinstance(identifier, six.string_types):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
