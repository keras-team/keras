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
        class_ind: Integer, class index.
    '''

    def __init__(self, name='recall', class_ind=1):
        super(Recall, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.total_positives = K.variable(value=0, dtype='float32')
        self.class_ind = class_ind

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
        y_true, y_pred = _slice_by_class(y_true, y_pred, self.class_ind)
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
        class_ind: Integer, class index.
    '''

    def __init__(self, name='precision', class_ind=1):
        super(Precision, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.pred_positives = K.variable(value=0, dtype='float32')
        self.class_ind = class_ind

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
        y_true, y_pred = _slice_by_class(y_true, y_pred, self.class_ind)
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


def _slice_by_class(y_true, y_pred, class_ind):
    ''' Slice the batch predictions and labels with respect to a given class
    that is encoded by a categorical or binary label.

    #  Arguments:
        y_true: Tensor, batch_wise labels.
        y_pred: Tensor, batch_wise predictions.
        class_ind: Integer, class index.

    # Returns:
        y_slice_true: Tensor, batch_wise label slice.
        y_slice_pred: Tensor,  batch_wise predictions, slice.
    '''
    # Binary encoded
    if y_pred.shape[-1] == 1:
        y_slice_true, y_slice_pred = y_true, y_pred
    # Categorical encoded
    else:
        y_slice_true, y_slice_pred = y_true[..., class_ind], y_pred[..., class_ind]
    return y_slice_true, y_slice_pred

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
